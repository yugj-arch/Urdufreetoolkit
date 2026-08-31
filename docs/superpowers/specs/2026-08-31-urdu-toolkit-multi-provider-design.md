# Urdu Toolkit — multi-provider, free-first, compare + batch

**Date:** 2026-08-31
**Status:** design approved in chat, pending spec review

## Goal

Rework the current API-only app into a **free/offline-first** toolkit where
the user picks, at runtime in the UI, *which* engine(s) run each step — and
can run several at once and compare their output side by side. Paid APIs
(OpenAI / Anthropic / Google / Azure / OCR.space) stay available as *extra*
providers, never required. "No cost" must cover the large majority of the
workflow.

Four capabilities, each a pluggable provider list:

1. **OCR** — image → Urdu text
2. **Transliteration** — Urdu script → Devanagari + Roman
3. **Translation** — Urdu → English / Hindi *meaning* (new capability)
4. **In-place render** — erase the Urdu in the image, draw the transliteration where it was

Three modes: **Single image** (compare), **Batch** (100–200 images), **Paste text**.

## Non-goals

- Not a production/multi-user server. Local personal use; Flask dev server + a
  background worker thread is acceptable. No Redis/Celery/gunicorn now.
- Not a forgery-grade image editor. In-place render is "good enough to read".
- Not solving Urdu's missing-short-vowel ambiguity. No offline engine restores
  unwritten vowels reliably; that is a property of the abjad script. Mitigation
  is *comparison* + optional LLM tie-breaker, stated honestly in the UI/README.

## Architecture

```
providers/
  base.py        # Capability enum, Provider protocol, result dataclasses, BaseProvider
  registry.py    # auto-discover provider modules; availability gate; UI metadata
  ocr/           tesseract.py easyocr_p.py paddle.py rapidocr.py surya.py doctr.py
                 hf_trocr.py kraken.py                       # offline
                 gpt.py claude.py gemini.py ocrspace.py google_vision.py azure_vision.py  # API
  translit/      aksharamukha_p.py uroman_p.py icu.py hutrans.py indic_xlit.py
                 indic_translit.py rule.py                   # offline
                 gpt.py claude.py gemini.py                  # API
  translate/     indictrans2.py nllb.py m2m100.py argos.py deep_translator_p.py  # offline/no-key
                 gpt.py claude.py gemini.py                  # API
  render/        pil_box.py opencv_inpaint.py lama.py        # offline
                 gpt_image.py gemini_image.py                # API
runner.py        # run N providers of one capability in parallel: ThreadPoolExecutor,
                 # per-provider timeout, failures isolated to their own result row
batch.py         # job dir lifecycle, chunked upload, background worker, resume,
                 # export CSV / XLSX / per-file TXT zip; one active job at a time
warmup.py        # pre-download every offline model so the first real run isn't slow
settings.py      # read/write .env (API keys entered in the UI); provider config
app.py           # Flask: single-image SSE, batch polling, settings, engine list
templates/index.html
static/           app.css  app.js         # split out of index.html (it's grown large)
assets/fonts/     NotoSans-Regular.ttf  NotoSansDevanagari-Regular.ttf
                  NotoNastaliqUrdu-Regular.ttf   # OFL, bundled, for pil_box render
jobs/             # per-job: input/ , results.jsonl , meta.json , exports  (gitignored)
docs/superpowers/specs/2026-08-31-urdu-toolkit-multi-provider-design.md
```

Existing files:
- `ocr.py` — logic moves into `providers/ocr/gpt.py` + `providers/translit/gpt.py`;
  keep thin re-exports only if a test needs them, otherwise delete.
- `imgedit.py` — logic moves into `providers/render/gpt_image.py`.
- `transliterate.py` — stays, wrapped by `providers/translit/rule.py`.

### Provider contract (`providers/base.py`)

```python
class Capability(str, Enum):
    OCR = "ocr"; TRANSLIT = "translit"; TRANSLATE = "translate"; RENDER = "render"

@dataclass
class ProviderInfo:
    id: str                  # stable slug, e.g. "surya"
    label: str               # UI name, e.g. "Surya OCR"
    capability: Capability
    kind: str                # "offline" | "api"
    needs: list[str]         # e.g. ["OPENAI_API_KEY"] or ["tesseract-binary"]
    note: str                # short quality/cost hint shown under the checkbox

class Provider(Protocol):
    info: ProviderInfo
    def available(self) -> tuple[bool, str]:   # (usable_now?, why-not message)
    # capability-specific method, one of:
    def ocr(self, image: bytes) -> OcrResult
    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult
    def translate(self, text: str, opts: TranslateOpts) -> TranslateResult
    def render(self, image: bytes, lines: list[str]) -> RenderResult
```

All result dataclasses share: `provider_id`, `ok: bool`, `error: str`, `ms: int`, `meta: dict`.
- `OcrResult`: `text`, `boxes: list|None` (for render), `notes`
- `TranslitResult`: `devanagari`, `roman`
- `TranslateResult`: `english`, `hindi`
- `RenderResult`: `png: bytes`

`available()` gate combines: import succeeds + required env keys present +
required binary on PATH + (optional) model files/downloadable. The registry
caches this per process; `warmup.py` and the settings panel can bust the cache.

### runner.py

`run(capability, provider_ids, payload, timeout_s) -> list[Result]`
- `ThreadPoolExecutor(max_workers=len(provider_ids))`
- each provider call wrapped: catch everything → `ok=False, error=str(e)`
- per-provider soft timeout via `future.result(timeout=...)`; on timeout the row
  is `ok=False, error="timed out after Ns"`, the others still return
- providers lazy-load their model on first call and cache it module-global

## Providers (initial set — "use every one you can find")

Legend: **off** = offline/no network after install · **free-net** = free but needs
internet, no key · **key** = needs an API key (entered in Settings).

### OCR

| id | lib / model | mode | Urdu note |
|---|---|---|---|
| tesseract | `pytesseract` + `urd` traineddata | off | needs Tesseract binary (Windows: UB Mannheim). Weak on Nastaliq. Auto-disabled if binary absent. |
| easyocr | `easyocr` lang `ur` | off | PyTorch. Solid all-rounder. |
| paddle | `paddleocr` + `paddlepaddle` lang `arabic` | off | good on printed Naskh. |
| rapidocr | `rapidocr-onnxruntime` arabic models | off | ONNX, no torch, fastest light option. |
| surya | `surya-ocr` | off | transformer, native Urdu support, best free on Nastaliq. ~1 GB. |
| doctr | `python-doctr[torch]` | off | Arabic-script detect+recognise; clean scans. |
| hf_trocr | `transformers` + configurable HF checkpoint (default a community Urdu TrOCR/UTRNet-style model) | off | strong on clean printed lines; ships **disabled** if no checkpoint loads cleanly — never blocks. |
| kraken | `kraken` + Arabic/Urdu model | off | experimental; Windows install may fail → non-fatal, stays unavailable. |
| gpt | `openai` gpt-4o vision | key | best overall; restores short vowels from context. ~1–2¢/img. |
| claude | `anthropic` vision | key | comparable to gpt. |
| gemini | `google-generativeai` vision | key | free tier available. |
| ocrspace | `requests` → ocr.space, lang `urd` | free-net (free key) | quick cloud baseline. |
| google_vision | `google-cloud-vision` | key | enable when GCP creds present. |
| azure_vision | Azure AI Vision REST | key | enable when key+endpoint present. |

### Transliteration (Urdu → Devanagari + Roman)

| id | lib | mode | note |
|---|---|---|---|
| aksharamukha | `aksharamukha` | off | Urdu → Devanagari + Roman (Roman style: Natural / ISO 15919 / IAST). Primary default. |
| uroman | `uroman` | off | Roman only; universal romanizer, very robust. |
| icu | `PyICU` `Arabic-Latin` transform | off | Roman; ICU rules. Windows: prebuilt wheel. |
| hutrans | LTRC `python-hutrans` (from GitHub) | off | Urdu → Hindi (Devanagari), dictionary + WX. |
| indic_xlit | `ai4bharat-transliteration` (IndicXlit) | off | Urdu native ↔ roman, transformer (~11M). |
| indic_translit | `indic-transliteration` | off | Devanagari ↔ Roman scheme conversion / normalisation helper. |
| rule | existing `transliterate.py` | off | dictionary + heuristic fallback. |
| gpt / claude / gemini | LLM APIs | key | context-aware vowel restoration; the "tie-breaker". |

### Translation (Urdu → English / Hindi)

| id | lib / model | mode | note |
|---|---|---|---|
| indictrans2 | AI4Bharat IndicTrans2 (`IndicTransToolkit` + HF ckpt) | off | Perso-Arabic Urdu supported; best offline quality. |
| nllb | `transformers` `facebook/nllb-200-distilled-600M`, `urd_Arab` | off | strong, ~2.4 GB. |
| m2m100 | `transformers` `facebook/m2m100_418M` | off | lighter, lower quality. |
| argos | `argostranslate` ur→en package | off | small, CPU-friendly. |
| deep_translator | `deep-translator` (Google/MyMemory/Libre endpoints) | free-net | no key; needs internet. |
| gpt / claude / gemini | LLM APIs | key | highest quality. |

### In-place render (erase Urdu, draw transliteration)

| id | how | mode | note |
|---|---|---|---|
| pil_box | OCR boxes → fill rect in sampled bg colour → draw text (bundled Noto fonts, auto-shrink to fit) | off | fast, deterministic; needs an OCR provider that returns boxes (easyocr/paddle/rapidocr/surya). |
| opencv_inpaint | `cv2.inpaint` over the text mask, then draw text | off | cleaner erase on textured bg. |
| lama | `simple-lama-inpainting` / `iopaint` LaMa erase, then draw text | off | best erase; ~200 MB model, CPU ok. |
| gpt_image | existing `imgedit.py` (gpt-image-1) | key | AI re-render; layout can drift. |
| gemini_image | Gemini image edit | key | alternative AI re-render. |

## Modes / UI

Split `index.html` into `templates/index.html` + `static/app.css` + `static/app.js`.

### Common controls

- **Provider pickers** per capability: checkbox list from `GET /api/providers`.
  Unavailable ones render greyed with the `available()` reason as a tooltip and a
  badge: `offline` / `free (net)` / `needs key` / `not installed`.
- Quick buttons: **All offline**, **Recommended** (OCR: surya+easyocr+rapidocr;
  translit: aksharamukha+uroman+rule; translate: indictrans2).
- **Target script**: Devanagari / Roman / both. **Roman style**: Natural / ISO / IAST.
- **Settings panel** (gear): text fields for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GOOGLE_API_KEY`, `OCRSPACE_API_KEY`, Azure key+endpoint, GCP creds path; HF
  checkpoint overrides; per-provider timeout. Saved to `.env` via `settings.py`.
  Never logged, never returned to the client after save.

### Single image (compare)

1. Upload → pick OCR providers → **Run**.
2. `POST /api/ocr` streams **SSE**; result **columns** fill in as each provider
   finishes: label, `ms`, editable text, **Use this** button, notes.
3. **Use this** copies that text into the transliteration input (also editable).
4. Pick translit + translate providers → **Run** → more columns (Devanagari,
   Roman, English/Hindi), each with a copy button.
5. Optional **In-place render**: pick a render provider + script → shows image +
   **Download PNG**. (API render providers only listed when their key is set.)

### Batch (100–200 images)

- Pick OCR provider(s) + translit provider(s) + target script. Any combination;
  no hard cap — UI shows a time estimate ("~14 min for 200 images with easyocr")
  from a one-image calibration run.
- `POST /api/batch` → `{job_id}`. Client uploads images in chunks of ~10 to
  `POST /api/batch/<id>/upload`, then `POST /api/batch/<id>/start`.
- Background worker processes **one image at a time** (providers parallel within
  the image), appends a line to `results.jsonl`, updates `meta.json`.
- Page polls `GET /api/batch/<id>/status` every 2s: `{status, done, total,
  recent[], errors[]}`. **Refresh-safe** — rejoin by `job_id`. **Resume** on
  server restart: skip images already in `results.jsonl`.
- On done: sortable table (`thumb | file | provider | urdu | devanagari | roman |
  ms | error`) + downloads: `results.csv`, `results.xlsx`, `texts.zip`.
- One active job; a second `start` returns 409.

### Paste text

Textarea → pick translit + translate providers → compare columns. No image, no
render.

## API (Flask)

| method | path | purpose |
|---|---|---|
| GET | `/` | page |
| GET | `/api/providers` | per capability: `[{id,label,kind,badge,available,reason,note}]` |
| POST | `/api/ocr` | SSE. form: `image`, `providers=a,b,c` → one `event: result` per provider |
| POST | `/api/transliterate` | json: `text`, `providers`, `roman_style`, `targets` → `{results:[...]}` |
| POST | `/api/translate` | json: `text`, `providers`, `targets` → `{results:[...]}` |
| POST | `/api/render` | form: `image`, `provider`, `script` → `image/png` |
| POST | `/api/settings` | json: key→value; writes `.env`; returns `{saved:[keys]}` only |
| POST | `/api/batch` | json: `ocr`, `translit`, `targets` → `{job_id}` |
| POST | `/api/batch/<id>/upload` | form: `images[]` (chunk) → `{received}` |
| POST | `/api/batch/<id>/start` | begin worker → `{status}` or 409 |
| GET | `/api/batch/<id>/status` | poll payload |
| GET | `/api/batch/<id>/results.(csv\|xlsx)` , `/texts.zip` , `/results.json` | outputs |

`MAX_CONTENT_LENGTH` raised to 32 MB (chunked upload of ~10 images), not one
400 MB request.

## Dependencies

One `requirements.txt`, grouped with comments. "Install everything":

```
# web + config + io
Flask  python-dotenv  Pillow  requests  openpyxl
# API providers (optional at runtime, cheap to install)
openai  anthropic  google-generativeai
# OCR — offline
pytesseract  easyocr  paddleocr  paddlepaddle  rapidocr-onnxruntime
surya-ocr  python-doctr[torch]  transformers  torch  timm  sentencepiece
kraken            # experimental; ok if it fails to build
# transliteration — offline
aksharamukha  uroman  PyICU  indic-transliteration  ai4bharat-transliteration
# (python-hutrans installed from GitHub in setup notes)
# translation — offline / no-key
argostranslate  deep-translator  IndicTransToolkit
# render — offline
opencv-python-headless  simple-lama-inpainting  iopaint
```

**Reality:** first install + first-run model downloads total roughly **8–12 GB**
and 30–60 min on a normal connection. Windows caveats, documented in README and
handled gracefully in code:
- Tesseract needs the separate binary; `tesseract` provider auto-disables without it.
- `PyICU`, `paddlepaddle`, `kraken` may need prebuilt wheels / build tools; if any
  fails to import, its provider is simply "not installed", app still runs.
- `hf_trocr` needs a working Urdu checkpoint; ships disabled if none loads.
- GPU optional throughout; everything must run CPU-only.

`warmup.py` downloads all model weights up front and prints a per-provider
OK / SKIPPED / FAILED table.

## Error handling

- Provider failure is data, not an exception: `ok=False, error=...` in its result
  row; siblings unaffected; UI shows the row greyed with the message.
- Registry `available()` failures never raise to the request path.
- Batch worker: per-image try/except → error recorded in `results.jsonl`, loop
  continues; worker crash → `meta.status="crashed"` with last index, resumable.
- Missing API key on an API provider → `available()` false with a clear reason;
  it isn't offered in the picker until Settings saves the key.
- Upload: non-image / empty → 400 with message; oversize chunk → 413.

## Testing

- **Per provider:** a `tests/fixtures/urdu_line.png` (rendered) + a known Urdu
  string. Test: (a) when the lib imports, the provider returns `ok=True` and a
  non-empty result; (b) when the import is monkey-patched to fail, `available()`
  returns false and the provider is skipped, not crashed. Accuracy is *not*
  asserted (varies by engine); only shape + non-emptiness.
- **runner:** N providers parallel; one raising → its row `ok=False`, others
  `ok=True`; one sleeping past timeout → its row times out, others return.
- **registry:** `/api/providers` lists every module once with correct badge;
  toggling an env key flips an API provider's availability.
- **batch:** create → upload 3 imgs → start → poll to `done` → csv has 3 rows;
  one corrupt image → that row has `error`, other two ok; second `start` while
  running → 409; kill mid-run, restart worker → finishes remaining, no dupes.
- **endpoints:** `/api/ocr` SSE emits one result per provider; `/api/transliterate`
  and `/api/translate` return one row per provider; `/api/render` returns PNG;
  `/api/settings` writes `.env` and echoes only key names.
- **aksharamukha / rule:** known Urdu input → expected Devanagari (exact).
- Existing GPT paths keep their mocked tests.
- CI-friendly: heavy providers marked `@pytest.mark.heavy` and skipped unless
  `RUN_HEAVY=1`, so the default suite runs without the 10 GB of models.

## Rollout / implementation order (for the plan)

1. `providers/base.py` + `registry.py` + `runner.py` + tests (no real providers yet, use fakes).
2. Wrap what already exists: `translit/rule.py`, `ocr/gpt.py`, `translit/gpt.py`,
   `render/gpt_image.py`. App still works end to end through the new pipeline.
3. `app.py` new endpoints + `/api/providers`; split `static/`; single-image
   compare UI over the wrapped providers.
4. Offline OCR providers, one file + test each: rapidocr, easyocr, surya, paddle,
   doctr, tesseract, (hf_trocr, kraken behind graceful-disable).
5. Offline translit providers: aksharamukha, uroman, icu, hutrans, indic_xlit,
   indic_translit. Roman-style option.
6. Translate capability: providers + `/api/translate` + UI section: indictrans2,
   nllb, m2m100, argos, deep_translator, (gpt/claude/gemini).
7. Render providers: pil_box, opencv_inpaint, lama (+ gemini_image).
8. `batch.py` + batch endpoints + batch UI + resume + exports.
9. Remaining API providers: claude, gemini, ocrspace, google_vision, azure_vision.
10. `warmup.py`, README rewrite, `requirements.txt`, `.gitignore` (`jobs/`, models),
    manual full-run verification with real images.
```
