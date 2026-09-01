# OCR parity — smart pipeline for `paddle` / `easyocr` + a fine-tuning track

**Date:** 2026-09-01
**Status:** design approved in chat, pending spec review
**Branch context:** `feat/phase2-engines` has unrelated in-flight changes (new
cloud providers, deleted `render/` + `translate/`). All work here is scoped to
`providers/ocr/*`, a new `eval/`, and a new `training/`. Commits stay separate
from the in-flight work.

## Goal

Close the quality gap between the offline OCR engines (`paddle`, `easyocr`) and
`gpt` on Urdu, and make it a **measured** claim (CER/WER against GPT output as a
silver reference), the same way `providers/translit/rule.py` reached GPT-4o
parity on clean text.

Two phases:

- **Phase 1 (this pass):** wrap both engines in a shared, deterministic,
  offline pipeline — preprocessing, RTL reading-order reconstruction, Urdu text
  normalization, and a rule-transliteration fill so each engine emits
  `urdu + devanagari + roman` "like gpt". Plus an eval harness that scores every
  OCR engine against cached GPT silver references.
- **Phase 2 (spec now, GPU step run later):** a hybrid dataset pipeline
  (synthetic Nastaliq/Naskh renders + GPT-distilled real images) and recognizer
  fine-tunes for both engines. The CPU-side tooling (renderer, distiller, format
  converters) is built and unit-tested now; the actual training runs are a
  copy-paste Colab/RunPod runbook. Providers gain dormant "load custom model if
  present, else stock" hooks.

## Non-goals

- Not training a model in this pass. Phase 2 delivers the tooling + runbook; the
  GPU job is a later session.
- Not beating GPT. The merge gate is *tuned ≥ current baseline* (regression
  guard). Parity with GPT is the target metric, not a blocker.
- Not a handwriting OCR. The dataset tooling reserves a slot for handwritten
  fonts/sources; no handwriting fine-tune here. Printed Nastaliq + Naskh
  (screenshots / PDF / scans / signage) is the phase-1 and phase-2 target.
- Not touching `curation.py`, the runner, the UI, or the API surface. The OCR
  result contract already carries `text`, `boxes`, `meta` — `meta` gains
  `devanagari` / `roman` / `reading_order`, which the compare UI already knows
  how to show (the `gpt` column populates the same keys).
- Not changing `providers/translit/*`. Phase-1 reuses `transliterate.transliterate`
  as-is via a thin wrapper.

## Background — why the offline engines lose to GPT today

1. **No preprocessing.** `providers/_imgutil.py` only decodes to RGB. Small
   text, low contrast, skew, and JPEG noise go straight into the engine.
2. **Scrambled word order.** `paddle.py` and `easyocr_p.py` both
   `"\n".join(...)` boxes in detector order. For RTL Urdu that reorders words
   within a line and can interleave lines/columns.
3. **No text normalization.** Engines emit Arabic-form codepoints
   (`ي` U+064A, `ك` U+0643), tatweel, stray joiners, Arabic-Indic digits, and
   ragged spacing around `،` / `۔`. CER against clean references is inflated by
   pure encoding differences.
4. **Weak recognizer on Nastaliq.** Genuinely a model limitation — the reason
   the project added cloud vision. Phase 2 addresses this; phases 1–4 of the
   rollout do not.

Items 1–3 are the cheap, deterministic wins and are worth measuring on their own
before any fine-tune.

## Architecture

```
providers/ocr/
  _pipeline.py      orchestrator: preprocess -> recognize -> RTL -> normalize -> translit-fill
  _prep.py          preprocessing stages (numpy / OpenCV), each toggleable + pure
  _rtl.py           word -> line grouping, line ordering, inline-LTR handling
  _normalize.py     normalize_urdu(): codepoint folding, digits, spacing, optional spellfix
  _translit_fill.py rule_translit(urdu) -> (devanagari, roman); never raises
  _model_hooks.py   resolve custom fine-tuned model dir/name if present (phase 2, dormant)
  paddle.py         thin: OcrConfig + _recognize(img, cfg) + run(...)
  easyocr_p.py      thin: OcrConfig + _recognize(img, cfg) + run(...)
  _imgutil.py       unchanged (still used for the raw decode)

eval/
  __init__.py
  refs.py           generate + cache GPT silver references
  score.py          CER / WER, raw and post-normalize
  run_eval.py       CLI: python -m eval.run_eval [--engines ...] [--refresh] [--spellfix] [--check]
  baseline.json     committed per-engine CER/WER snapshot the regression test reads

tests/fixtures/ocr/
  <slug>.png            input image (committed, small)
  <slug>.gpt.json       {"urdu": "...", "model": "...", "generated_at": "..."} (committed)
  README.md             how to add a fixture

training/               (phase 2 — see "Fine-tuning track" below)
  README.md  corpus/  synth/  distill/  easyocr/  paddle/
```

### Component A — `_pipeline.py` (orchestrator)

```python
@dataclass(frozen=True)
class Word:
    text: str
    box: list[tuple[int, int]]   # 4-point polygon, PROCESSED-image coords from the engine
    conf: float

@dataclass(frozen=True)
class Line:
    words: list[Word]
    text: str
    box: list[tuple[int, int]]
    conf: float

@dataclass(frozen=True)
class OcrConfig:
    # preprocessing
    grayscale: bool = True
    upscale_min_text_px: int = 22      # estimated cap height; upscale until met
    max_upscale: float = 3.0
    denoise: bool = True
    clahe: bool = True
    binarize: str = "sauvola"          # "none" | "otsu" | "sauvola"
    deskew: bool = True
    deskew_max_deg: float = 15.0
    pad: int = 12
    # reading order
    line_y_tol_frac: float = 0.6       # fraction of median word height
    # normalization
    map_digits_to: str = "ascii"      # "ascii" | "urdu" | "keep"
    fold_arabic_heh: bool = False     # ه U+0647 -> ہ U+06C1; unsafe in general, see _normalize.py
    spellfix: bool = False
    # engine-specific knobs, consumed by _recognize
    engine: dict = field(default_factory=dict)

@dataclass
class PipelineOut:
    text: str
    lines: list[Line]
    meta: dict            # {"reading_order": "rtl-pipeline", "prep": [...applied stages]}

def run(image: bytes, recognize, cfg: OcrConfig) -> PipelineOut: ...
```

`run` sequence:

1. `img0 = _imgutil.to_ndarray(image)` (RGB uint8).
2. `proc, inv = _prep.preprocess(img0, cfg)` — `proc` is the engine input;
   `inv` is a callable mapping a processed-image point back to original-image
   coords (composed scale + rotation + pad offset).
3. `words = recognize(proc, cfg.engine)` — engine does detect + recognize,
   returns `list[Word]` in processed coords.
4. `words = [w with box mapped through inv]` — boxes now in original coords for
   the UI overlay.
5. `lines = _rtl.group_into_lines(words, cfg)` then `_rtl.order_lines(lines)`.
6. `text = "\n".join(l.text for l in lines)`; `text = _normalize.normalize_urdu(text, cfg)`.
7. return `PipelineOut(text, lines, meta)`.

Every stage is best-effort: a raising prep stage logs and passes its input
through unchanged; `_rtl` failure falls back to detector-order join; `recognize`
raising propagates (the provider's `self._timed` turns it into `ok=False`).

### Component B — `_prep.py` (preprocessing)

Pure functions, `np.ndarray` in / out, dependencies limited to `numpy` +
`opencv-python-headless` (+ optional `scikit-image` for Sauvola; falls back to
`cv2.adaptiveThreshold`).

| function | note |
|---|---|
| `to_gray(img)` | RGB -> single channel |
| `upscale(img, min_text_px, max_factor)` | estimate text height from a horizontal projection / connected components; Lanczos up to `max_factor`; return `(img, scale)` |
| `denoise(img)` | `cv2.fastNlMeansDenoising` (gray) or bilateral; mild |
| `clahe(img)` | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` |
| `binarize(img, mode)` | `otsu` -> `cv2.threshold(+OTSU)`; `sauvola` -> skimage or adaptive fallback; `none` -> passthrough |
| `deskew(img, max_deg)` | angle from `cv2.minAreaRect` of the text mask (or Hough); rotate with white border; clamp to `±max_deg`, skip if estimate is unstable; return `(img, angle)` |
| `pad(img, px)` | constant white border |

`preprocess(img, cfg) -> (np.ndarray, Callable[[float, float], tuple[float, float]])`
composes the enabled stages, accumulates `scale` and `angle` and `pad`, and
builds `inv` from them. `meta["prep"]` lists what actually ran.

### Component C — `_rtl.py` (reading order)

- `group_into_lines(words, cfg)`: compute each word's y-center and height; sort by
  y-center; greedily open a new line when the next word's y-center exceeds the
  current line's mean by `cfg.line_y_tol_frac * median_height`; within a line
  sort words by x-center **descending** (RTL). Line `text` = words joined by a
  single space; line `box` = union bbox; line `conf` = mean word conf.
- `order_lines(lines)`: cluster line x-ranges into columns by gap analysis
  (`> 1.5 * median_intercol_gap`). One column -> already top-to-bottom.
  Multi-column -> rightmost column first (Urdu), each column top-to-bottom.
- Inline LTR runs (Latin words, digits, URLs): if `python-bidi` is importable,
  run each line's word list through it (base direction RTL) to get visual order;
  else heuristic — reverse the word list, then within each maximal run of
  Latin/digit tokens reverse again so those read left-to-right. Intra-token
  character order is never touched.

### Component D — `_normalize.py`

`normalize_urdu(s: str, cfg: OcrConfig) -> str`:

1. `unicodedata.normalize("NFC", s)`.
2. Codepoint folding (Arabic presentation / Arabic-script -> Urdu):
   `ي`→`ی` (U+064A→U+06CC), `ك`→`ک` (U+0643→U+06A9), `ﻻ`/`ﻼ`→`لا`, drop tatweel
   `ـ` (U+0640). A `ه` (U+0647) → `ہ` (U+06C1) fold is available behind
   `cfg` but **default off** — Urdu legitimately uses U+06C1/U+06BE and the
   fold is only safe on sources known to emit Arabic heh; eval measures it.
3. Remove ZERO WIDTH JOINER / NON-JOINER except between letters where the source
   image clearly needs them — practically: strip all ZWJ, keep ZWNJ. (Documented
   as a known lossy choice.)
4. Digits per `cfg.map_digits_to`: Arabic-Indic `٠-٩` and Extended `۰-۹`
   -> ASCII `0-9`, or all -> Urdu `۰-۹`, or keep.
5. Whitespace: collapse runs to one space, strip spaces *before* `،` `۔` `؟` `!`
   and after opening brackets, trim line ends, drop blank lines.
6. If `cfg.spellfix`: token-level, only for tokens of length ≥ 3 that are (a) not
   in the frequency list and (b) at edit distance 1 from exactly one list entry
   whose frequency is ≥ 50× the token's. List = keys of
   `transliterate._RAW_COMMON_WORDS` plus a small bundled `eval` frequency file.
   Off by default; `run_eval --spellfix` measures its effect.

### Component E — provider refactor

`easyocr_p.py` and `paddle.py` keep their `ProviderInfo`, `available()`, and the
`PROVIDER = ...()` tail. `ocr()` becomes:

```python
from providers.ocr._pipeline import run, OcrConfig
from providers.ocr._translit_fill import rule_translit
from providers.ocr import _model_hooks

_CFG = OcrConfig(engine={ ...tuned knobs... })

def _recognize(img, engine_cfg) -> list["Word"]:
    # easyocr: reader.readtext(img, detail=1, paragraph=False, **engine_cfg)
    #          -> [Word(text, box(4pt), conf)]
    # paddle:  engine.predict(img) -> _extract() -> [Word(...)]

class EasyOcr(BaseProvider):
    def ocr(self, image: bytes) -> OcrResult:
        if easyocr is None:
            return OcrResult(provider_id=self.info.id, ok=False, error="easyocr not installed")
        t = self._timed(run, image, _recognize, _CFG)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        out = t["value"]
        deva, roman = rule_translit(out.text)
        boxes = [{"box": w.box, "text": w.text, "conf": w.conf}
                 for line in out.lines for w in line.words]
        return OcrResult(
            provider_id=self.info.id, ok=True, ms=t["ms"],
            text=out.text, boxes=boxes,
            meta={"devanagari": deva, "roman": roman, **out.meta},
        )
```

Tuned starting knobs (eval refines them; values live in each provider's `_CFG`):

- **EasyOCR** `engine=`: `decoder="beamsearch"`, `beamWidth=10`,
  `contrast_ths=0.1`, `adjust_contrast=0.5`, `text_threshold=0.6`,
  `low_text=0.3`, `link_threshold=0.4`, `mag_ratio=1.5`, `slope_ths=0.2`,
  `ycenter_ths=0.5`, `height_ths=0.7`, `width_ths=0.5`, `add_margin=0.1`.
  Reader still `["ur", "en"]`, `gpu=False`. `_prep` does the upscaling, so
  `mag_ratio` stays modest.
- **PaddleOCR** `engine=`: passed at construction where the installed version
  supports it — `text_det_thresh=0.3`, `text_det_box_thresh=0.5`,
  `text_det_unclip_ratio=1.8`, `drop_score=0.4`, `use_textline_orientation=True`,
  `enable_mkldnn=False` (kept — 3.3.x PIR/oneDNN crash). Unknown kwargs are
  filtered against the constructor signature so a version bump can't crash it.

`OcrConfig` for Paddle sets `binarize="none"` (its detector prefers grayscale
contrast to hard binarization); EasyOCR uses `binarize="sauvola"`. Both use
`upscale`, `clahe`, `deskew`, `denoise`.

### Component F — `_translit_fill.py`

```python
import transliterate as _rule

def rule_translit(urdu: str) -> tuple[str, str]:
    if not urdu.strip():
        return "", ""
    try:
        deva, roman = _rule.transliterate(urdu)
        return deva or "", roman or ""
    except Exception:
        return "", ""
```

Keeps the OCR engines self-contained and offline while matching the `gpt`
column's shape. No new dependency.

### Component G — eval harness (`eval/`)

- `refs.py`: `ensure_refs(fixtures_dir, refresh=False)` — for each `<slug>.png`
  lacking `<slug>.gpt.json` (or all, if `refresh`), call
  `providers.ocr.gpt.PROVIDER.ocr(bytes)`, write
  `{"urdu": result.text, "model": result.meta["model"], "generated_at": iso}`.
  If `OPENAI_API_KEY` is unset and a ref is missing -> print an actionable
  message and skip that fixture (no crash).
- `score.py`: `cer(ref, hyp)` = Levenshtein(ref, hyp) / len(ref) via the
  `Levenshtein` package; `wer(ref, hyp)` on whitespace tokens; both computed
  raw and after `normalize_urdu` applied to *both* sides. Aggregate = mean over
  fixtures, plus a per-fixture list.
- `run_eval.py`: `python -m eval.run_eval [--engines paddle,easyocr,gpt]
  [--refresh] [--spellfix] [--check]`. Runs each engine over the fixtures,
  prints a Markdown table `engine | CER | WER | CER(norm) | WER(norm) | ms/img`.
  `--check` compares `paddle`/`easyocr` `CER(norm)` against `baseline.json` and
  exits non-zero on regression beyond `+0.01` absolute.
- `baseline.json`: committed. Regenerated deliberately with
  `run_eval --write-baseline` when an improvement lands, mentioned in the commit.

### Component H — `tests/fixtures/ocr/`

6–10 small PNGs spanning the phase-1 domain: 3–4 digital Nastaliq screenshots,
2–3 scanned/print (Naskh + Nastaliq), 1–2 signage/photo. Each with a committed
`.gpt.json`. Images must be redistributable — synthetic renders (from the
phase-2 renderer) or user-supplied images the user confirms are shareable.
`README.md` documents the "drop png, run `run_eval --refresh`, commit both" loop.

## Fine-tuning track (phase 2)

```
training/
  README.md              the runbook: env, dataset build, train, export, wire-in, re-eval
  corpus/
    fetch_corpus.py      Urdu sentences from a permissively-licensed source
                         (CC-100 ur sample / Urdu Wikipedia dump extract) -> sentences.txt
                         + LICENCES.md with provenance
  synth/
    fonts/               git-ignored; fonts.json lists file -> licence (Noto Nastaliq
                         Urdu OFL, Noto Naskh Arabic OFL, user-added Nastaliq)
    render.py            sentence + font -> tight line PNG + label; Pillow with raqm
                         (libraqm) for correct Arabic shaping; varied size/weight
    augment.py           blur, JPEG recompress, gaussian noise, paper/screen bg
                         textures, ink bleed (morphology), <=3deg rotation, mild
                         perspective, brightness/contrast jitter
    build_synth.py       -> data/synth/{train,val}/*.png + gt.txt   (data/ git-ignored)
  distill/
    collect.py           user real images -> run the offline detector (reuse the
                         phase-1 engine's detect stage) -> per-line crops + boxes
    align.py             GPT page transcription (via eval/refs style call) split
                         across the crop sequence: match by line count, then by
                         relative length + rapidfuzz ratio; emit only pairs above
                         a confidence threshold, log the drops
    build_distill.py     -> data/distill/{train,val}/*.png + gt.txt
  easyocr/
    to_lmdb.py           crops + gt.txt -> LMDB in deep-text-recognition-benchmark
                         layout
    config.yaml          arch pinned to EasyOCR's Urdu/Arabic recognizer
                         (generation-2 / None-VGG-BiLSTM-CTC) + its character set
    train.md             clone JaidedAI/deep-text-recognition-benchmark, train
                         command, checkpoint -> EasyOCR custom model conversion,
                         files to drop in ~/.EasyOCR/user_network/urdu_ft.{py,yaml,pth}
  paddle/
    to_paddle_rec.py     crops + gt.txt -> PaddleOCR rec label files (train/val .txt)
    arabic_rec_ft.yml    fine-tune config from arabic_PP-OCRv4_rec, pretrained
                         weights, character_dict_path = arabic_dict.txt, low LR
    train.md             tools/train.py -c arabic_rec_ft.yml, tools/export_model.py,
                         resulting inference dir -> OCR_PADDLE_REC_DIR
```

**Built and unit-tested now (CPU, no large downloads in tests):**
`corpus/fetch_corpus.py` (network-guarded, cached sample committed for tests),
`synth/render.py` + `augment.py` + `build_synth.py`, `distill/align.py` core,
`easyocr/to_lmdb.py`, `paddle/to_paddle_rec.py`. Tests use 3–5 sentences and a
single bundled OFL font.

**Runbook only (needs GPU + multi-GB downloads):** the actual
`deep-text-recognition-benchmark` training run, the PaddleOCR `tools/train.py`
run, checkpoint export. `training/README.md` gives copy-paste Colab and RunPod
cells, expected wall-clock, and the acceptance step (`run_eval --engines
paddle,easyocr,gpt` before/after, custom model wired via env / `user_network`).

**Provider hooks — built now, dormant until a model exists:**
`_model_hooks.py` -
- `easyocr_recog_network()` -> `"urdu_ft"` if
  `~/.EasyOCR/user_network/urdu_ft.py` exists or `OCR_EASYOCR_RECOG_NETWORK`
  set, else `None`. `_get_reader()` passes `recog_network=` when non-None.
- `paddle_rec_dir()` -> `os.environ.get("OCR_PADDLE_REC_DIR")` if it points at a
  real dir, else `None`. `_get_engine()` passes the matching kwarg when set.
Monkeypatched tests assert the custom branch is taken when the marker is present
and the stock branch otherwise.

**Dataset strategy (hybrid, confirmed):** synthetic supplies volume
(target ~100–300k lines across fonts/augmentations); GPT-distilled real images
supply domain fit (however many the user provides). Public sets (UPTI, UTRSet)
are optional add-ons noted in the runbook with their licence caveats — not a
dependency. Handwriting: `synth/fonts/` accepts handwriting-style fonts and
`fonts.json` can tag them, but no handwriting run is part of this spec.

## Data flow (phase 1)

```
image bytes
  -> _imgutil.to_ndarray                       RGB uint8
  -> _prep.preprocess(cfg)                      -> (proc img, inv transform), meta.prep
  -> engine._recognize(proc, cfg.engine)        detect + recognize -> list[Word] (proc coords)
  -> map each Word.box through inv              -> original-image coords
  -> _rtl.group_into_lines / order_lines        -> list[Line], RTL-correct
  -> "\n".join(line.text)
  -> _normalize.normalize_urdu(cfg)             folding, digits, spacing, (spellfix)
  -> _translit_fill.rule_translit              -> devanagari, roman
  -> OcrResult(text, boxes, meta{devanagari, roman, reading_order, prep})
```

## Error handling

- `_prep` stage raises -> log at WARNING, pass input through unchanged, omit it
  from `meta.prep`.
- `_rtl` raises -> fall back to `"\n".join(w.text for w in words)` in engine
  order; `meta.reading_order = "fallback-detector-order"`.
- `_normalize` raises -> return the pre-normalization string.
- `_translit_fill` never raises -> `("", "")` on any failure.
- `_recognize` raises or the engine lib is absent -> existing behavior:
  `OcrResult(ok=False, error=...)` via `self._timed` / the `is None` guard.
- Optional deps (`scikit-image`, `python-bidi`) absent -> feature flag flips off
  with a one-time `logging` warning; pipeline still runs.
- `eval`: missing `OPENAI_API_KEY` and missing cached ref -> skip fixture with a
  message; never crash the run.
- `_model_hooks`: a broken/partial custom-model dir -> treat as absent, log once,
  use stock.

## Testing

New / changed test files (default suite stays fast; engine-dependent tests gated
`RUN_HEAVY=1`):

- `tests/test_ocr_prep.py` — each stage on hand-built arrays: `deskew` rotates a
  known-skewed black bar back to < 1deg; `upscale` reaches `min_text_px` and
  reports the scale; `binarize("otsu")` output has exactly 2 values; `preprocess`
  round-trips a known point through `inv` to within 1px.
- `tests/test_ocr_rtl.py` — synthetic `Word` lists: RTL within a line;
  two-line vertical split at the tolerance boundary; two-column page orders
  right column first; a line with an inline Latin token keeps that token LTR.
- `tests/test_ocr_normalize.py` — `(raw, expected)` table, one row per rule
  (U+064A fold, U+0643 fold, tatweel strip, digit map ascii/urdu/keep, space
  before `،`, ligature split); `spellfix` corrects `katab`→dictionary form only
  under the frequency ratio.
- `tests/test_ocr_pipeline.py` — fake `_recognize` returning fixed `Word`s;
  asserts end-to-end `text`, box coords in original space, `meta` keys.
- `tests/test_providers_smoke.py` — extend: `paddle` / `easyocr` still return an
  `OcrResult`; `meta` now has `devanagari`, `roman`, `reading_order`. Monkeypatch
  the engine so it runs without the heavy lib.
- `tests/test_provider_paddle.py` — keep the existing version-safety test; add
  one asserting unknown kwargs are filtered from the constructor call.
- `tests/test_ocr_model_hooks.py` — custom branch taken when a marker file / env
  var is present (tmp path + monkeypatch), stock branch otherwise.
- `tests/test_eval_score.py` — `cer` / `wer` on known pairs; normalization makes
  a pure-encoding diff score 0.
- `tests/test_eval_regression.py` (`RUN_HEAVY=1`) — run `paddle` / `easyocr`
  over `tests/fixtures/ocr/` with cached refs (no network), assert
  `CER(norm) <= baseline + 0.01`.
- `tests/test_synth_render.py` — render 3 sentences with the bundled font;
  PNGs are non-blank, `gt.txt` line count matches.
- `tests/test_distill_align.py` — synthetic crop sequence + a fake page
  transcription; alignment maps the right text to each crop and drops the
  planted mismatch.

Manual verification before calling phase 1 done: `python -m eval.run_eval
--engines gpt,paddle,easyocr` on a machine with the engines installed; record
the before/after table in the phase-1 completion note.

## Dependencies (`urdu-free-toolkit/requirements.txt`)

Add, grouped with comments:

```
# OCR pipeline
opencv-python-headless
python-Levenshtein            # eval scoring
# OCR pipeline — optional (code-guarded; pipeline runs without them)
scikit-image                  # Sauvola binarization; else cv2.adaptiveThreshold
python-bidi                   # inline-LTR ordering; else heuristic
# OCR fine-tuning tooling (phase 2, CPU side)
rapidfuzz                     # distill alignment
# Pillow already present; needs libraqm at runtime for Arabic shaping in synth/render.py
fonttools
```

Training-only heavyweights (`torch` already in tree for easyocr; PaddleOCR
training extras, `lmdb`) are documented in `training/README.md`, not added to the
app's `requirements.txt`.

## Rollout / implementation order (for the plan)

**Phase 1**
1. `_prep.py` + `_rtl.py` + `_normalize.py` + their unit tests. No provider
   changes yet.
2. `_pipeline.py` + `_translit_fill.py` + `test_ocr_pipeline.py` (fake engine).
3. Refactor `easyocr_p.py` and `paddle.py` onto `run(...)`, add tuned `_CFG`,
   `meta.devanagari` / `roman`; update `test_providers_smoke.py` and
   `test_provider_paddle.py`.
4. `eval/` (`refs.py`, `score.py`, `run_eval.py`) + `tests/fixtures/ocr/`
   (images + `.gpt.json`) + `baseline.json` + `test_eval_regression.py`.
   Run `run_eval` for real once; commit the baseline and the before/after table.

**Phase 2 scaffold (CPU, built + tested now)**
5. `_model_hooks.py` + dormant wiring in both providers + `test_ocr_model_hooks.py`.
6. `training/corpus/` + `training/synth/` + `test_synth_render.py`.
7. `training/distill/` + `test_distill_align.py`.
8. `training/easyocr/` + `training/paddle/` converters + configs +
   `training/README.md` runbook.

**Phase 2 execution (later session, not in this plan)**
9. Run the runbook on a cloud GPU; wire the resulting models via env /
   `user_network`; re-run `run_eval`; update `baseline.json` and README with the
   new numbers.

10. `urdu-free-toolkit/README.md` — OCR section: the pipeline, how to run the
    eval, link to `training/README.md`.

## Risks / open questions

- **EasyOCR fine-tune arch pinning.** The exact recognizer architecture +
  character set EasyOCR's Urdu model uses must be confirmed against the installed
  version before `easyocr/config.yaml` is final. Runbook task, flagged in
  `train.md`; use context7 / upstream source when writing the plan.
- **PaddleOCR API churn.** Which `text_det_*` / `drop_score` kwargs the installed
  version accepts varies. Mitigation: filter kwargs against the constructor
  signature (Component E) and cover it with a test.
- **`libraqm` on Windows** for `synth/render.py`. If Pillow lacks raqm, Arabic
  shaping is wrong. Runbook: install via conda-forge or use the WSL path;
  `render.py` asserts `features.check("raqm")` and fails loudly.
- **GPT silver refs are not ground truth.** GPT can misread too. Fixtures the
  user can hand-correct are better; the harness supports editing a `.gpt.json`
  by hand (it's just JSON) and notes this in the fixtures README.
- **Distill alignment noise.** Line-count mismatch between the offline detector
  and GPT's line breaks is common. `align.py` is conservative (drops
  low-confidence pairs); the runbook expects synthetic data to carry the bulk.
```
