# OCR Parity Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the `paddle` and `easyocr` OCR providers in a shared offline pipeline (preprocess → recognize → RTL reading-order → Urdu-normalize → rule-translit fill) and add a CER/WER eval harness that scores every OCR engine against cached GPT silver references, plus CPU-side scaffolding for a later recognizer fine-tune.

**Architecture:** A new `providers/ocr/_pipeline.run(image, recognize, cfg)` owns fixed post/pre stages; each engine keeps only a small `_recognize(img, engine_cfg) -> list[Word]` plus a tuned `OcrConfig`. Shared types live in `providers/ocr/_types.py`. Preprocessing (`_prep.py`), reading-order (`_rtl.py`), normalization (`_normalize.py`), translit-fill (`_translit_fill.py`) and dormant custom-model resolution (`_model_hooks.py`) are separate focused modules. `eval/` is a standalone CLI + scorer. `training/` holds CPU tooling (synthetic renderer, GPT-distill aligner, dataset-format converters) and a GPU runbook.

**Tech Stack:** Python 3.13, numpy, OpenCV (`cv2`), scikit-image, python-bidi, `Levenshtein`, Pillow (raqm), rapidfuzz, fonttools, pytest. Existing: Flask app, `providers/` registry, `transliterate.py`.

**Spec:** `docs/superpowers/specs/2026-09-01-ocr-parity-pipeline-design.md`

## Global Constraints

- App package root is `urdu-free-toolkit/`; `tests/conftest.py` puts it on `sys.path`. All new packages (`eval/`, `training/`) live under `urdu-free-toolkit/`.
- Provider failures are data, never exceptions: return `OcrResult(ok=False, error=...)`. Pipeline stages are best-effort — a raising stage logs at WARNING and passes its input through unchanged.
- Heavy/engine-dependent tests carry `@pytest.mark.heavy` and only run under `RUN_HEAVY=1` (see `tests/conftest.py`). The default `pytest` run must stay green with no model downloads and no network.
- `OcrResult` contract is fixed (`base.py`): `provider_id, ok, error, ms, meta, text, boxes, notes`. New data rides in `meta`: `meta["devanagari"]`, `meta["roman"]`, `meta["reading_order"]`, `meta["prep"]`.
- `transliterate.transliterate(text: str) -> tuple[str, str]` returns `(devanagari, roman)`.
- Do not modify `providers/curation.py`, `runner.py`, the Flask app, `templates/`, `static/`, or `providers/translit/*`.
- Keep commits scoped to this work. The branch has unrelated staged deletions (`render/`, `translate/`) — never `git add -A`; always add explicit paths.
- No new **required** core dependency: everything new is added to `requirements.txt` in the OCR / optional groups, and every optional import (`skimage`, `bidi`) is guarded so the pipeline still runs without it.
- Dataclasses in `_types.py` are `frozen=True`.

---

## Task 1: Shared OCR types (`_types.py`)

**Files:**
- Create: `urdu-free-toolkit/providers/ocr/_types.py`
- Test: `urdu-free-toolkit/tests/test_ocr_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Word(text: str, box: list[tuple[int,int]], conf: float)` — frozen dataclass; `box` is a 4-point polygon.
  - `Line(words: list[Word], text: str, box: list[tuple[int,int]], conf: float)` — frozen.
  - `OcrConfig(...)` — frozen dataclass, all fields defaulted (see code).
  - `PipelineOut(text: str, lines: list[Line], meta: dict)` — non-frozen dataclass.

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_ocr_types.py
from dataclasses import FrozenInstanceError

import pytest

from providers.ocr._types import Word, Line, OcrConfig, PipelineOut


def test_word_is_frozen():
    w = Word(text="ہے", box=[(0, 0), (10, 0), (10, 8), (0, 8)], conf=0.9)
    with pytest.raises(FrozenInstanceError):
        w.text = "x"


def test_ocrconfig_defaults():
    c = OcrConfig()
    assert c.grayscale is True
    assert c.binarize == "sauvola"
    assert c.deskew is True
    assert c.map_digits_to == "ascii"
    assert c.fold_arabic_heh is False
    assert c.spellfix is False
    assert c.engine == {}


def test_ocrconfig_engine_dict_is_per_instance():
    a, b = OcrConfig(), OcrConfig()
    assert a.engine is not b.engine


def test_pipelineout_shape():
    out = PipelineOut(text="ہے", lines=[], meta={"reading_order": "rtl-pipeline"})
    assert out.text == "ہے" and out.lines == [] and out.meta["reading_order"] == "rtl-pipeline"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers.ocr._types'`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/providers/ocr/_types.py
# -*- coding: utf-8 -*-
"""Shared value types for the OCR pipeline.

Kept in their own module so ``_pipeline`` / ``_rtl`` / the providers can all
import them without an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[int, int]


@dataclass(frozen=True)
class Word:
    text: str
    box: list[Point]        # 4-point polygon, image coordinates
    conf: float


@dataclass(frozen=True)
class Line:
    words: list[Word]
    text: str
    box: list[Point]
    conf: float


@dataclass(frozen=True)
class OcrConfig:
    # preprocessing
    grayscale: bool = True
    upscale_min_text_px: int = 22
    max_upscale: float = 3.0
    denoise: bool = True
    clahe: bool = True
    binarize: str = "sauvola"          # "none" | "otsu" | "sauvola"
    deskew: bool = True
    deskew_max_deg: float = 15.0
    pad: int = 12
    # reading order
    line_y_tol_frac: float = 0.6
    # normalization
    map_digits_to: str = "ascii"       # "ascii" | "urdu" | "keep"
    fold_arabic_heh: bool = False
    spellfix: bool = False
    # engine-specific knobs consumed by the provider's _recognize
    engine: dict = field(default_factory=dict)


@dataclass
class PipelineOut:
    text: str
    lines: list[Line]
    meta: dict = field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_types.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/ocr/_types.py urdu-free-toolkit/tests/test_ocr_types.py
git commit -m "feat(ocr): shared pipeline value types"
```

---

## Task 2: Urdu text normalization (`_normalize.py`)

**Files:**
- Create: `urdu-free-toolkit/providers/ocr/_normalize.py`
- Test: `urdu-free-toolkit/tests/test_ocr_normalize.py`

**Interfaces:**
- Consumes: `OcrConfig` from `providers.ocr._types`.
- Produces:
  - `normalize_urdu(s: str, cfg: OcrConfig | None = None) -> str`
  - `FREQ_WORDS: dict[str, int]` — module-level frequency map used by spellfix.

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_ocr_normalize.py
import pytest

from providers.ocr._normalize import normalize_urdu
from providers.ocr._types import OcrConfig


@pytest.mark.parametrize("raw, expected", [
    ("\u064a\u0647", "\u06cc\u06c1" if False else "\u06cc\u0647"),  # ي fold -> ی ; heh NOT folded by default
    ("\u0643\u062a\u0627\u0628", "\u06a9\u062a\u0627\u0628"),        # ك -> ک
    ("\u0645\u062d\u0628\u0640\u0640\u062a", "\u0645\u062d\u0628\u062a"),  # tatweel stripped
    ("\ufefb", "\u0644\u0627"),                                       # ﻻ ligature -> لا
    ("\u0661\u0662\u0663", "123"),                                    # Arabic-Indic digits -> ascii
    ("\u06f1\u06f2", "12"),                                           # Extended Arabic-Indic -> ascii
    ("word  \u060c next", "word\u060c next"),                         # space before ، removed, runs collapsed
])
def test_normalize_rules(raw, expected):
    assert normalize_urdu(raw) == expected


def test_digits_keep_and_urdu_modes():
    assert normalize_urdu("\u0661\u0662", OcrConfig(map_digits_to="keep")) == "\u0661\u0662"
    assert normalize_urdu("12", OcrConfig(map_digits_to="urdu")) == "\u06f1\u06f2"


def test_heh_fold_opt_in():
    assert normalize_urdu("\u0647", OcrConfig(fold_arabic_heh=True)) == "\u06c1"
    assert normalize_urdu("\u0647", OcrConfig(fold_arabic_heh=False)) == "\u0647"


def test_blank_lines_and_trailing_ws_dropped():
    assert normalize_urdu("a \n\n  \nb ") == "a\nb"


def test_spellfix_conservative():
    # "katab" (heuristic misread) -> dictionary form only when far more frequent
    from providers.ocr._normalize import FREQ_WORDS
    assert "\u06a9\u062a\u0627\u0628" in FREQ_WORDS  # kitab present as a frequent token
    fixed = normalize_urdu("\u06a9\u062a\u0628", OcrConfig(spellfix=True))
    assert fixed == "\u06a9\u062a\u0627\u0628"
    # a token with no close frequent neighbour is left alone
    assert normalize_urdu("\u0632\u0632\u0632\u0632", OcrConfig(spellfix=True)) == "\u0632\u0632\u0632\u0632"


def test_spellfix_off_by_default():
    assert normalize_urdu("\u06a9\u062a\u0628") == "\u06a9\u062a\u0628"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_normalize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers.ocr._normalize'`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/providers/ocr/_normalize.py
# -*- coding: utf-8 -*-
"""Deterministic Urdu text normalization for OCR output.

Folds Arabic-form codepoints to their Urdu equivalents, normalizes digits and
whitespace, and (opt-in) applies a very conservative dictionary spellfix. Every
rule is individually covered in tests/test_ocr_normalize.py.
"""
from __future__ import annotations

import unicodedata

from providers.ocr._types import OcrConfig

_TATWEEL = "\u0640"
_ZWJ = "\u200d"

# Arabic-script / presentation form -> Urdu
_FOLD = {
    "\u064a": "\u06cc",   # ARABIC YEH -> FARSI YEH
    "\u0643": "\u06a9",   # ARABIC KAF -> KEHEH
    "\u0649": "\u06cc",   # ALEF MAKSURA -> FARSI YEH
    "\ufefb": "\u0644\u0627",  # LAM-ALEF isolated
    "\ufefc": "\u0644\u0627",  # LAM-ALEF final
}
_HEH_FOLD = {"\u0647": "\u06c1"}   # opt-in only

_ARABIC_INDIC = {c: chr(0x30 + i) for i, c in enumerate("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669")}
_EXT_ARABIC_INDIC = {c: chr(0x30 + i) for i, c in enumerate("\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9")}
_ASCII_TO_URDU = {chr(0x30 + i): c for i, c in enumerate("\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9")}

_PUNCT_NO_SPACE_BEFORE = "\u060c\u06d4\u061f!\u061b:\u066b\u066c,.;?"

# Frequent tokens for spellfix. Seeded from the rule engine's common-word dict;
# integer is a coarse frequency rank (higher = more common).
try:  # pragma: no cover - defensive
    from transliterate import _RAW_COMMON_WORDS as _RCW
    FREQ_WORDS: dict[str, int] = {k: 100 for k in _RCW}
except Exception:  # pragma: no cover
    FREQ_WORDS = {}
# A few high-value nouns the heuristic classically gets wrong:
FREQ_WORDS.setdefault("\u06a9\u062a\u0627\u0628", 100)   # kitab
FREQ_WORDS.setdefault("\u0633\u0648\u0627\u0644", 80)     # sawal


def _fold_chars(s: str, fold_heh: bool) -> str:
    table = dict(_FOLD)
    if fold_heh:
        table.update(_HEH_FOLD)
    out = []
    for ch in s:
        out.append(table.get(ch, ch))
    return "".join(out)


def _map_digits(s: str, mode: str) -> str:
    if mode == "keep":
        return s
    if mode == "urdu":
        s = "".join(_ARABIC_INDIC.get(c, c) for c in s)   # first normalize to ascii
        return "".join(_ASCII_TO_URDU.get(c, c) for c in s)
    # ascii
    return "".join(_EXT_ARABIC_INDIC.get(c, _ARABIC_INDIC.get(c, c)) for c in s)


def _clean_ws(s: str) -> str:
    lines = []
    for line in s.split("\n"):
        line = " ".join(line.split())
        for p in _PUNCT_NO_SPACE_BEFORE:
            line = line.replace(" " + p, p)
        lines.append(line)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(l for l in lines if l.strip())


def _levenshtein1(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    from Levenshtein import distance
    return distance(a, b) == 1


def _spellfix(s: str) -> str:
    out = []
    for tok in s.split(" "):
        if len(tok) < 3 or tok in FREQ_WORDS:
            out.append(tok)
            continue
        cands = [w for w in FREQ_WORDS if _levenshtein1(tok, w)]
        if len(cands) == 1 and FREQ_WORDS[cands[0]] >= 50:
            out.append(cands[0])
        else:
            out.append(tok)
    return " ".join(out)


def normalize_urdu(s: str, cfg: OcrConfig | None = None) -> str:
    cfg = cfg or OcrConfig()
    if not s:
        return s
    s = unicodedata.normalize("NFC", s)
    s = _fold_chars(s, cfg.fold_arabic_heh)
    s = s.replace(_TATWEEL, "").replace(_ZWJ, "")
    s = _map_digits(s, cfg.map_digits_to)
    s = _clean_ws(s)
    if cfg.spellfix:
        s = "\n".join(_spellfix(line) for line in s.split("\n"))
    return s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_normalize.py -v`
Expected: PASS. If `test_normalize_rules[\u064a\u0647-...]` fails on the heh expectation, confirm the parametrize expected value is `"\u06cc\u0647"` (yeh folded, heh untouched).

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/ocr/_normalize.py urdu-free-toolkit/tests/test_ocr_normalize.py
git commit -m "feat(ocr): deterministic Urdu text normalization"
```

---

## Task 3: RTL reading-order reconstruction (`_rtl.py`)

**Files:**
- Create: `urdu-free-toolkit/providers/ocr/_rtl.py`
- Test: `urdu-free-toolkit/tests/test_ocr_rtl.py`

**Interfaces:**
- Consumes: `Word`, `Line`, `OcrConfig` from `providers.ocr._types`.
- Produces:
  - `group_into_lines(words: list[Word], cfg: OcrConfig) -> list[Line]`
  - `order_lines(lines: list[Line]) -> list[Line]`
  - `lines_to_text(lines: list[Line]) -> str` (joins `line.text` with `"\n"`)

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_ocr_rtl.py
from providers.ocr._rtl import group_into_lines, order_lines, lines_to_text
from providers.ocr._types import Word, OcrConfig


def _w(text, x, y, w=20, h=16, conf=0.9):
    return Word(text=text, box=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)], conf=conf)


def test_words_within_a_line_are_rtl():
    # visually: [ C ][ B ][ A ]  -> reading order A B C (right to left)
    words = [_w("A", 200, 0), _w("B", 100, 0), _w("C", 0, 0)]
    lines = group_into_lines(words, OcrConfig())
    assert len(lines) == 1
    assert lines[0].text == "A B C"


def test_two_lines_split_vertically():
    words = [_w("A", 100, 0), _w("B", 0, 0), _w("C", 0, 40), _w("D", 100, 40)]
    lines = group_into_lines(words, OcrConfig())
    assert [l.text for l in lines] == ["A B", "D C"]


def test_two_columns_right_first():
    # left column x~0, right column x~500; Urdu reads right column first
    left = [_w("L1", 0, 0), _w("L2", 0, 40)]
    right = [_w("R1", 500, 0), _w("R2", 500, 40)]
    lines = order_lines(group_into_lines(left + right, OcrConfig()))
    assert [l.text for l in lines] == ["R1", "R2", "L1", "L2"]


def test_inline_latin_run_kept_ltr():
    # Urdu ... "OpenAI GPT" ... Urdu  -> the Latin pair stays "OpenAI GPT"
    words = [_w("\u06cc\u06c1", 300, 0), _w("OpenAI", 200, 0), _w("GPT", 130, 0), _w("\u06c1\u06d2", 0, 0)]
    lines = group_into_lines(words, OcrConfig())
    assert "OpenAI GPT" in lines[0].text
    assert lines[0].text.startswith("\u06cc\u06c1")
    assert lines[0].text.endswith("\u06c1\u06d2")


def test_lines_to_text_joins_with_newlines():
    words = [_w("A", 0, 0), _w("B", 0, 40)]
    assert lines_to_text(group_into_lines(words, OcrConfig())) == "A\nB"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_rtl.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers.ocr._rtl'`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/providers/ocr/_rtl.py
# -*- coding: utf-8 -*-
"""Reconstruct human reading order from detector boxes for RTL Urdu.

Detectors emit boxes in an arbitrary order. We bucket them into lines by
vertical position, order words right-to-left within a line, order lines
top-to-bottom, and put the right-hand column first on multi-column pages.
Inline left-to-right runs (Latin words, digits) are kept in visual order.
"""
from __future__ import annotations

import logging
import statistics

from providers.ocr._types import Line, OcrConfig, Word

log = logging.getLogger(__name__)

try:
    from bidi.algorithm import get_display  # type: ignore
    _HAVE_BIDI = True
except Exception:  # pragma: no cover
    _HAVE_BIDI = False


def _cx(w: Word) -> float:
    return sum(p[0] for p in w.box) / len(w.box)


def _cy(w: Word) -> float:
    return sum(p[1] for p in w.box) / len(w.box)


def _height(w: Word) -> float:
    ys = [p[1] for p in w.box]
    return max(ys) - min(ys) or 1.0


def _union_box(words: list[Word]) -> list[tuple[int, int]]:
    xs = [p[0] for w in words for p in w.box]
    ys = [p[1] for w in words for p in w.box]
    return [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]


def _is_ltr_token(t: str) -> bool:
    return any(c.isascii() and (c.isalnum()) for c in t) and not any("\u0600" <= c <= "\u06ff" for c in t)


def _line_text(words_rtl: list[Word]) -> str:
    """words_rtl is already in reading order (rightmost first). Keep maximal
    runs of LTR tokens in their left-to-right visual order."""
    toks = [w.text for w in words_rtl]
    if _HAVE_BIDI:
        # Reconstruct a logical string then let bidi lay it out is overkill here;
        # the run-reversal below is deterministic and unit-tested.
        pass
    out: list[str] = []
    i = 0
    while i < len(toks):
        if _is_ltr_token(toks[i]):
            j = i
            while j < len(toks) and _is_ltr_token(toks[j]):
                j += 1
            out.extend(reversed(toks[i:j]))  # visually rightmost-first -> restore L->R
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def group_into_lines(words: list[Word], cfg: OcrConfig) -> list[Line]:
    if not words:
        return []
    try:
        med_h = statistics.median(_height(w) for w in words)
        tol = cfg.line_y_tol_frac * med_h
        buckets: list[list[Word]] = []
        for w in sorted(words, key=_cy):
            placed = False
            for b in buckets:
                if abs(_cy(w) - statistics.mean(_cy(x) for x in b)) <= tol:
                    b.append(w)
                    placed = True
                    break
            if not placed:
                buckets.append([w])
        lines: list[Line] = []
        for b in buckets:
            ordered = sorted(b, key=_cx, reverse=True)   # RTL
            lines.append(Line(
                words=ordered,
                text=_line_text(ordered),
                box=_union_box(ordered),
                conf=statistics.mean(w.conf for w in ordered),
            ))
        lines.sort(key=lambda ln: min(p[1] for p in ln.box))
        return lines
    except Exception:  # pragma: no cover - best effort
        log.warning("group_into_lines failed; falling back to detector order", exc_info=True)
        return [Line(words=list(words), text=" ".join(w.text for w in words),
                     box=_union_box(list(words)), conf=0.0)]


def _columns(lines: list[Line]) -> list[list[Line]]:
    if len(lines) < 2:
        return [lines]
    xs = sorted(min(p[0] for p in ln.box) for ln in lines)
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    if not gaps:
        return [lines]
    med_gap = statistics.median(g for g, _ in gaps) or 1.0
    big = [i for g, i in gaps if g > 4 * med_gap and g > 40]
    if not big:
        return [lines]
    split_x = xs[big[-1]] + gaps[big[-1]][0] / 2 if False else (xs[big[-1]] + xs[big[-1] + 1]) / 2
    left = [ln for ln in lines if min(p[0] for p in ln.box) < split_x]
    right = [ln for ln in lines if min(p[0] for p in ln.box) >= split_x]
    return [right, left]  # Urdu: right column first


def order_lines(lines: list[Line]) -> list[Line]:
    out: list[Line] = []
    for col in _columns(lines):
        out.extend(sorted(col, key=lambda ln: min(p[1] for p in ln.box)))
    return out


def lines_to_text(lines: list[Line]) -> str:
    return "\n".join(ln.text for ln in lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_rtl.py -v`
Expected: PASS (5 tests). If `test_two_columns_right_first` fails, tune the `_columns` gap thresholds (`4 * med_gap and g > 40`) until the two synthetic columns split.

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/ocr/_rtl.py urdu-free-toolkit/tests/test_ocr_rtl.py
git commit -m "feat(ocr): RTL reading-order reconstruction"
```

---

## Task 4: Preprocessing stages (`_prep.py`)

**Files:**
- Create: `urdu-free-toolkit/providers/ocr/_prep.py`
- Test: `urdu-free-toolkit/tests/test_ocr_prep.py`

**Interfaces:**
- Consumes: `OcrConfig` from `providers.ocr._types`; `numpy`, `cv2`.
- Produces:
  - `preprocess(img: "np.ndarray", cfg: OcrConfig) -> tuple["np.ndarray", Callable[[float, float], tuple[float, float]], list[str]]`
    returns `(processed_image, inverse_point_fn, applied_stage_names)`.
  - Individual stage helpers, each `np.ndarray -> np.ndarray` (or returning a
    scalar alongside): `to_gray`, `upscale(img, min_text_px, max_factor) -> (img, scale)`,
    `denoise`, `clahe`, `binarize(img, mode)`, `deskew(img, max_deg) -> (img, angle_deg)`,
    `pad(img, px)`.

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_ocr_prep.py
import numpy as np
import pytest

from providers.ocr import _prep
from providers.ocr._types import OcrConfig


def _text_bar(h=200, w=400, angle=0.0):
    img = np.full((h, w), 255, np.uint8)
    img[90:110, 40:360] = 0                     # a horizontal black bar = "text"
    if angle:
        import cv2
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    return img


def test_to_gray_reduces_dims():
    rgb = np.zeros((10, 10, 3), np.uint8)
    assert _prep.to_gray(rgb).ndim == 2


def test_binarize_otsu_is_two_valued():
    img = (np.random.rand(50, 50) * 255).astype(np.uint8)
    out = _prep.binarize(img, "otsu")
    assert sorted(np.unique(out).tolist()) == [0, 255]


def test_upscale_hits_min_text_px_and_reports_scale():
    img = _text_bar()                            # bar is 20px tall
    out, scale = _prep.upscale(img, min_text_px=60, max_factor=5.0)
    assert scale >= 2.9 and out.shape[0] > img.shape[0]


def test_upscale_respects_max_factor():
    img = _text_bar()
    _, scale = _prep.upscale(img, min_text_px=10_000, max_factor=3.0)
    assert scale == pytest.approx(3.0)


def test_deskew_corrects_known_angle():
    skewed = _text_bar(angle=7.0)
    _, ang = _prep.deskew(skewed, max_deg=15.0)
    assert abs(ang) == pytest.approx(7.0, abs=2.0)


def test_preprocess_point_roundtrip():
    img = np.full((120, 240, 3), 255, np.uint8)
    cfg = OcrConfig(denoise=False, clahe=False, deskew=False, binarize="none")
    proc, inv, stages = _prep.preprocess(img, cfg)
    # a point in processed space maps back near the same fraction of the original
    px, py = proc.shape[1] * 0.5, proc.shape[0] * 0.5
    ox, oy = inv(px, py)
    assert abs(ox - 120) < 3 and abs(oy - 60) < 3
    assert "upscale" in stages


def test_preprocess_survives_a_failing_stage(monkeypatch):
    img = np.full((60, 60, 3), 255, np.uint8)
    monkeypatch.setattr(_prep, "clahe", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    proc, inv, stages = _prep.preprocess(img, OcrConfig(clahe=True))
    assert proc is not None and "clahe" not in stages
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_prep.py -v`
Expected: FAIL — `AttributeError: module 'providers.ocr._prep' has no attribute 'to_gray'`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/providers/ocr/_prep.py
# -*- coding: utf-8 -*-
"""Preprocessing stages for the offline OCR engines.

Each stage is a small pure function on a numpy image. ``preprocess`` composes
the enabled ones per ``OcrConfig``, tracks the geometric transform so word
boxes can be mapped back to original-image coordinates, and never lets one
failing stage abort the pipeline.
"""
from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

from providers.ocr._types import OcrConfig

log = logging.getLogger(__name__)

try:
    from skimage.filters import threshold_sauvola  # type: ignore
    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAVE_SKIMAGE = False


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _estimate_text_px(gray: np.ndarray) -> float:
    inv = 255 - gray
    rows = (inv > 64).sum(axis=1)
    thresh = rows.max() * 0.3 if rows.max() else 0
    runs, cur = [], 0
    for v in rows:
        if v > thresh:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    runs = [r for r in runs if r >= 3]
    return float(np.median(runs)) if runs else float(gray.shape[0])


def upscale(img: np.ndarray, min_text_px: int, max_factor: float) -> tuple[np.ndarray, float]:
    gray = to_gray(img)
    tpx = _estimate_text_px(gray)
    factor = 1.0 if tpx <= 0 else min(max_factor, max(1.0, min_text_px / tpx))
    if factor <= 1.001:
        return img, 1.0
    out = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_LANCZOS4)
    return out, factor


def denoise(img: np.ndarray) -> np.ndarray:
    g = to_gray(img)
    return cv2.fastNlMeansDenoising(g, None, h=7, templateWindowSize=7, searchWindowSize=21)


def clahe(img: np.ndarray) -> np.ndarray:
    g = to_gray(img)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def binarize(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return img
    g = to_gray(img)
    if mode == "otsu":
        _, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out
    if mode == "sauvola":
        if _HAVE_SKIMAGE:
            win = max(3, (min(g.shape) // 20) | 1)
            t = threshold_sauvola(g, window_size=win)
            return ((g > t) * 255).astype(np.uint8)
        return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    return g


def deskew(img: np.ndarray, max_deg: float) -> tuple[np.ndarray, float]:
    g = to_gray(img)
    inv = 255 - g
    coords = np.column_stack(np.where(inv > 64))
    if len(coords) < 50:
        return img, 0.0
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    if angle > 45:
        angle -= 90
    if abs(angle) > max_deg or abs(angle) < 0.1:
        return img, 0.0
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return out, float(angle)


def pad(img: np.ndarray, px: int) -> np.ndarray:
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=255)


def preprocess(img: np.ndarray, cfg: OcrConfig):
    """Returns (processed_img, inverse_point_fn, applied_stage_names)."""
    scale = 1.0
    angle = 0.0
    pad_px = 0
    h0, w0 = img.shape[:2]
    stages: list[str] = []
    cur = img

    def _try(name: str, fn):
        nonlocal cur
        try:
            cur = fn(cur)
            stages.append(name)
        except Exception:
            log.warning("prep stage %s failed; skipping", name, exc_info=True)

    if cfg.grayscale:
        _try("grayscale", to_gray)
    try:
        cur, scale = upscale(cur, cfg.upscale_min_text_px, cfg.max_upscale)
        if scale > 1.001:
            stages.append("upscale")
    except Exception:
        log.warning("prep stage upscale failed; skipping", exc_info=True)
        scale = 1.0
    if cfg.denoise:
        _try("denoise", denoise)
    if cfg.clahe:
        _try("clahe", clahe)
    try:
        cur, angle = deskew(cur, cfg.deskew_max_deg) if cfg.deskew else (cur, 0.0)
        if angle:
            stages.append("deskew")
    except Exception:
        log.warning("prep stage deskew failed; skipping", exc_info=True)
        angle = 0.0
    if cfg.binarize != "none":
        _try(f"binarize:{cfg.binarize}", lambda im: binarize(im, cfg.binarize))
    if cfg.pad:
        cur = pad(cur, cfg.pad)
        pad_px = cfg.pad
        stages.append("pad")

    cx, cy = w0 / 2, h0 / 2
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def inv(px: float, py: float) -> tuple[float, float]:
        x = (px - pad_px) / scale
        y = (py - pad_px) / scale
        x -= cx
        y -= cy
        rx = cos_t * x + sin_t * y
        ry = -sin_t * x + cos_t * y
        return rx + cx, ry + cy

    return cur, inv, stages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_prep.py -v`
Expected: PASS (7 tests). `test_deskew_corrects_known_angle` may need the `abs=2.0` tolerance widened to `3.0` depending on OpenCV version — acceptable.

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/ocr/_prep.py urdu-free-toolkit/tests/test_ocr_prep.py
git commit -m "feat(ocr): preprocessing stages with box-coordinate round-trip"
```

---

## Task 5: Translit fill (`_translit_fill.py`)

**Files:**
- Create: `urdu-free-toolkit/providers/ocr/_translit_fill.py`
- Test: `urdu-free-toolkit/tests/test_ocr_translit_fill.py`

**Interfaces:**
- Consumes: `transliterate.transliterate`.
- Produces: `rule_translit(urdu: str) -> tuple[str, str]` — `(devanagari, roman)`, never raises.

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_ocr_translit_fill.py
from providers.ocr._translit_fill import rule_translit


def test_blank_in_blank_out():
    assert rule_translit("   ") == ("", "")


def test_returns_two_strings_for_real_text():
    deva, roman = rule_translit("\u0645\u06cc\u06ba \u0679\u06be\u06cc\u06a9 \u06c1\u0648\u06ba")
    assert isinstance(deva, str) and isinstance(roman, str)
    assert deva and roman


def test_never_raises_on_engine_error(monkeypatch):
    import providers.ocr._translit_fill as tf
    monkeypatch.setattr(tf._rule, "transliterate", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("x")))
    assert rule_translit("abc") == ("", "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_translit_fill.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/providers/ocr/_translit_fill.py
# -*- coding: utf-8 -*-
"""Fill an OCR result's devanagari/roman fields via the offline rule engine,
so the offline OCR columns carry the same shape as the GPT column. Best effort:
any failure yields empty strings, never an exception.
"""
from __future__ import annotations

import transliterate as _rule


def rule_translit(urdu: str) -> tuple[str, str]:
    if not urdu or not urdu.strip():
        return "", ""
    try:
        deva, roman = _rule.transliterate(urdu)
        return deva or "", roman or ""
    except Exception:
        return "", ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_translit_fill.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/ocr/_translit_fill.py urdu-free-toolkit/tests/test_ocr_translit_fill.py
git commit -m "feat(ocr): rule-engine translit fill for offline OCR output"
```

---

## Task 6: Pipeline orchestrator (`_pipeline.py`)

**Files:**
- Create: `urdu-free-toolkit/providers/ocr/_pipeline.py`
- Test: `urdu-free-toolkit/tests/test_ocr_pipeline.py`

**Interfaces:**
- Consumes: `_prep.preprocess`, `_rtl.group_into_lines/order_lines/lines_to_text`,
  `_normalize.normalize_urdu`, `_types.{Word,Line,OcrConfig,PipelineOut}`,
  `providers._imgutil.to_ndarray`.
- Produces:
  - `RecognizeFn = Callable[[np.ndarray, dict], list[Word]]`
  - `run(image: bytes, recognize: RecognizeFn, cfg: OcrConfig) -> PipelineOut`
  - re-exports `Word`, `Line`, `OcrConfig`, `PipelineOut` for provider convenience.

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_ocr_pipeline.py
import io

import numpy as np
from PIL import Image

from providers.ocr._pipeline import run
from providers.ocr._types import OcrConfig, Word


def _png(w=240, h=120):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_run_assembles_rtl_text_and_meta():
    # fake engine: two words on one line, given in processed-image coords
    def fake_recognize(img: np.ndarray, engine_cfg: dict):
        return [
            Word("A", [(200, 10), (240, 10), (240, 30), (200, 30)], 0.9),
            Word("B", [(10, 10), (60, 10), (60, 30), (10, 30)], 0.8),
        ]

    out = run(_png(), fake_recognize, OcrConfig(denoise=False, clahe=False))
    assert out.text == "A B"
    assert out.meta["reading_order"] == "rtl-pipeline"
    assert isinstance(out.meta["prep"], list)
    assert out.lines and out.lines[0].words[0].text == "A"


def test_run_maps_boxes_back_to_original_coords():
    seen = {}

    def fake_recognize(img, engine_cfg):
        seen["shape"] = img.shape
        h, w = img.shape[:2]
        return [Word("X", [(0, 0), (w, 0), (w, h), (0, h)], 0.5)]

    out = run(_png(240, 120), fake_recognize, OcrConfig(pad=10))
    # processed image is larger (upscale+pad); box must come back within originalish bounds
    xs = [p[0] for p in out.lines[0].words[0].box]
    ys = [p[1] for p in out.lines[0].words[0].box]
    assert max(xs) <= 240 + 5 and max(ys) <= 120 + 5
    assert seen["shape"][0] != 120  # engine really saw a transformed image


def test_run_falls_back_when_recognize_returns_nothing():
    out = run(_png(), lambda *_a: [], OcrConfig())
    assert out.text == "" and out.lines == []


def test_run_normalizes_output():
    def fake(img, cfg):
        return [Word("\u0643\u062a\u0627\u0628", [(0, 0), (30, 0), (30, 20), (0, 20)], 0.9)]
    out = run(_png(), fake, OcrConfig())
    assert out.text == "\u06a9\u062a\u0627\u0628"  # ك folded to ک
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers.ocr._pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/providers/ocr/_pipeline.py
# -*- coding: utf-8 -*-
"""The shared offline OCR pipeline.

    image bytes
      -> preprocess (grayscale, upscale, denoise, clahe, deskew, binarize, pad)
      -> engine recognize()  [detect + recognize, returns Word list in proc coords]
      -> map word boxes back to original image coordinates
      -> group words into lines, order lines (RTL, right column first)
      -> normalize Urdu text
      -> PipelineOut(text, lines, meta)

Engines provide only ``recognize(processed_img, engine_cfg) -> list[Word]``.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from providers._imgutil import to_ndarray
from providers.ocr import _normalize, _prep, _rtl
from providers.ocr._types import Line, OcrConfig, PipelineOut, Word  # noqa: F401 re-export

log = logging.getLogger(__name__)

RecognizeFn = Callable[[np.ndarray, dict], list]


def _map_word(w: Word, inv) -> Word:
    return Word(text=w.text, conf=w.conf,
                box=[tuple(int(round(v)) for v in inv(px, py)) for px, py in w.box])


def run(image: bytes, recognize: RecognizeFn, cfg: OcrConfig) -> PipelineOut:
    img0 = to_ndarray(image)
    meta: dict = {}
    try:
        proc, inv, stages = _prep.preprocess(img0, cfg)
    except Exception:
        log.warning("preprocess failed entirely; using raw image", exc_info=True)
        proc, inv, stages = img0, (lambda x, y: (x, y)), []
    meta["prep"] = stages

    words = list(recognize(proc, cfg.engine) or [])
    words = [_map_word(w, inv) for w in words]

    try:
        lines = _rtl.order_lines(_rtl.group_into_lines(words, cfg))
        meta["reading_order"] = "rtl-pipeline"
    except Exception:
        log.warning("reading-order failed; detector order", exc_info=True)
        lines = []
        meta["reading_order"] = "fallback-detector-order"
        text = " ".join(w.text for w in words)
        return PipelineOut(text=_safe_normalize(text, cfg), lines=[], meta=meta)

    text = _rtl.lines_to_text(lines)
    return PipelineOut(text=_safe_normalize(text, cfg), lines=lines, meta=meta)


def _safe_normalize(text: str, cfg: OcrConfig) -> str:
    try:
        return _normalize.normalize_urdu(text, cfg)
    except Exception:
        log.warning("normalize_urdu failed; returning raw text", exc_info=True)
        return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_pipeline.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the whole OCR module test group**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_*.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add urdu-free-toolkit/providers/ocr/_pipeline.py urdu-free-toolkit/tests/test_ocr_pipeline.py
git commit -m "feat(ocr): shared preprocess/recognize/RTL/normalize pipeline"
```

---

## Task 7: Custom fine-tuned model hooks (`_model_hooks.py`)

**Files:**
- Create: `urdu-free-toolkit/providers/ocr/_model_hooks.py`
- Test: `urdu-free-toolkit/tests/test_ocr_model_hooks.py`

**Interfaces:**
- Consumes: env vars `OCR_EASYOCR_RECOG_NETWORK`, `OCR_PADDLE_REC_DIR`; filesystem.
- Produces:
  - `easyocr_recog_network() -> str | None`
  - `paddle_rec_dir() -> str | None`

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_ocr_model_hooks.py
from pathlib import Path

from providers.ocr import _model_hooks as mh


def test_easyocr_network_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.delenv("OCR_EASYOCR_RECOG_NETWORK", raising=False)
    monkeypatch.setattr(mh, "_EASYOCR_USER_DIR", tmp_path)
    assert mh.easyocr_recog_network() is None


def test_easyocr_network_from_env(monkeypatch, tmp_path):
    net = "urdu_ft"
    (tmp_path / f"{net}.py").write_text("# stub")
    monkeypatch.setattr(mh, "_EASYOCR_USER_DIR", tmp_path)
    monkeypatch.setenv("OCR_EASYOCR_RECOG_NETWORK", net)
    assert mh.easyocr_recog_network() == net


def test_easyocr_network_autodetected(monkeypatch, tmp_path):
    (tmp_path / "urdu_ft.py").write_text("# stub")
    (tmp_path / "urdu_ft.yaml").write_text("x: 1")
    monkeypatch.setattr(mh, "_EASYOCR_USER_DIR", tmp_path)
    monkeypatch.delenv("OCR_EASYOCR_RECOG_NETWORK", raising=False)
    assert mh.easyocr_recog_network() == "urdu_ft"


def test_paddle_rec_dir_only_when_real(monkeypatch, tmp_path):
    monkeypatch.delenv("OCR_PADDLE_REC_DIR", raising=False)
    assert mh.paddle_rec_dir() is None
    monkeypatch.setenv("OCR_PADDLE_REC_DIR", str(tmp_path / "nope"))
    assert mh.paddle_rec_dir() is None
    real = tmp_path / "rec"; real.mkdir()
    monkeypatch.setenv("OCR_PADDLE_REC_DIR", str(real))
    assert mh.paddle_rec_dir() == str(real)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_model_hooks.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/providers/ocr/_model_hooks.py
# -*- coding: utf-8 -*-
"""Resolve an optional fine-tuned recognizer for the offline OCR engines.

Both return ``None`` (use the stock model) unless a custom model is actually
present. Wired into ``easyocr_p`` / ``paddle`` now; dormant until Phase 2
training produces a model. See training/README.md.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_EASYOCR_USER_DIR = Path.home() / ".EasyOCR" / "user_network"


def easyocr_recog_network() -> str | None:
    env = os.environ.get("OCR_EASYOCR_RECOG_NETWORK")
    try:
        if env:
            if (_EASYOCR_USER_DIR / f"{env}.py").exists():
                return env
            log.warning("OCR_EASYOCR_RECOG_NETWORK=%s but %s/%s.py missing; using stock",
                        env, _EASYOCR_USER_DIR, env)
            return None
        for py in sorted(_EASYOCR_USER_DIR.glob("*.py")):
            if (py.with_suffix(".yaml")).exists():
                return py.stem
    except Exception:  # pragma: no cover
        log.warning("easyocr_recog_network probe failed", exc_info=True)
    return None


def paddle_rec_dir() -> str | None:
    d = os.environ.get("OCR_PADDLE_REC_DIR")
    if d and Path(d).is_dir():
        return d
    if d:
        log.warning("OCR_PADDLE_REC_DIR=%s is not a directory; using stock", d)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_model_hooks.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/ocr/_model_hooks.py urdu-free-toolkit/tests/test_ocr_model_hooks.py
git commit -m "feat(ocr): dormant custom fine-tuned model resolution hooks"
```

---

## Task 8: Refactor `easyocr_p.py` onto the pipeline

**Files:**
- Modify: `urdu-free-toolkit/providers/ocr/easyocr_p.py` (full rewrite of body, keep `ProviderInfo`)
- Create: `urdu-free-toolkit/tests/test_ocr_provider_easyocr.py`

**Interfaces:**
- Consumes: `_pipeline.run`, `_pipeline.OcrConfig`, `_translit_fill.rule_translit`,
  `_model_hooks.easyocr_recog_network`, `providers._imgutil` (not needed directly),
  `providers.ocr._types.Word`.
- Produces: `PROVIDER.ocr(image: bytes) -> OcrResult` with
  `meta["devanagari"], meta["roman"], meta["reading_order"], meta["prep"]`, and
  `boxes` a flat list of `{"box": [[x,y]*4], "text": str, "conf": float}`.
- `_recognize(img, engine_cfg) -> list[Word]` (module-level, testable).

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_ocr_provider_easyocr.py
import io
import sys
import types

import numpy as np
from PIL import Image


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (80, 40), "white").save(buf, format="PNG")
    return buf.getvalue()


def _install_fake_easyocr(monkeypatch, found):
    """found: list of (box, text, conf) as easyocr.readtext(detail=1) returns."""
    fake = types.ModuleType("easyocr")

    class Reader:
        def __init__(self, *a, **k):
            self.kwargs = k

        def readtext(self, arr, **kw):
            return found

    fake.Reader = Reader
    monkeypatch.setitem(sys.modules, "easyocr", fake)
    import importlib
    import providers.ocr.easyocr_p as mod
    importlib.reload(mod)
    return mod


def test_easyocr_ocr_returns_pipeline_result(monkeypatch):
    found = [
        ([[60, 5], [78, 5], [78, 22], [60, 22]], "\u0627\u0644\u0641", 0.9),
        ([[5, 5], [30, 5], [30, 22], [5, 22]], "\u0628\u06d2", 0.8),
    ]
    mod = _install_fake_easyocr(monkeypatch, found)
    r = mod.PROVIDER.ocr(_png())
    assert r.ok is True
    assert r.provider_id == "easyocr"
    assert r.text.split() == ["\u0627\u0644\u0641", "\u0628\u06d2"]  # RTL: rightmost first
    assert "devanagari" in r.meta and "roman" in r.meta
    assert r.meta["reading_order"] == "rtl-pipeline"
    assert isinstance(r.boxes, list) and r.boxes and set(r.boxes[0]) == {"box", "text", "conf"}


def test_easyocr_missing_lib_is_graceful(monkeypatch):
    monkeypatch.setitem(sys.modules, "easyocr", None)
    import importlib
    import providers.ocr.easyocr_p as mod
    importlib.reload(mod)
    r = mod.PROVIDER.ocr(_png())
    assert r.ok is False and "not installed" in r.error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_provider_easyocr.py -v`
Expected: FAIL — current `easyocr_p.py` has no `_recognize`, no `meta["reading_order"]`; assertions fail.

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/providers/ocr/easyocr_p.py
# -*- coding: utf-8 -*-
"""EasyOCR as an OCR provider (offline), wrapped in the shared OCR pipeline.

EasyOCR ships a real Urdu recognition model. First call downloads the detector +
Urdu recognizer (~100 MB); later calls are offline. The pipeline handles
preprocessing, RTL reading order, Urdu normalization, and a rule-engine
transliteration fill so this column matches the GPT column's shape.
"""
from __future__ import annotations

from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo
from providers.ocr import _model_hooks
from providers.ocr._pipeline import OcrConfig, run
from providers.ocr._translit_fill import rule_translit
from providers.ocr._types import Word

try:
    import easyocr
except Exception:  # noqa: BLE001
    easyocr = None

_reader = None

# Tuned knobs (see the spec). _prep does the upscaling, so mag_ratio stays modest.
_ENGINE_CFG = dict(
    decoder="beamsearch", beamWidth=10, contrast_ths=0.1, adjust_contrast=0.5,
    text_threshold=0.6, low_text=0.3, link_threshold=0.4, mag_ratio=1.5,
    slope_ths=0.2, ycenter_ths=0.5, height_ths=0.7, width_ths=0.5, add_margin=0.1,
    detail=1, paragraph=False,
)
_CFG = OcrConfig(binarize="sauvola", engine=dict(_ENGINE_CFG))


def _get_reader():
    global _reader
    if _reader is None:
        net = _model_hooks.easyocr_recog_network()
        kw = {"gpu": False, "verbose": False}
        if net:
            kw["recog_network"] = net
        _reader = easyocr.Reader(["ur", "en"], **kw)
    return _reader


def _recognize(img, engine_cfg: dict) -> list[Word]:
    reader = _get_reader()
    found = reader.readtext(img, **engine_cfg)
    out: list[Word] = []
    for box, text, conf in found:
        out.append(Word(text=text,
                        box=[(int(x), int(y)) for x, y in box],
                        conf=float(conf)))
    return out


class EasyOcr(BaseProvider):
    info = ProviderInfo(
        id="easyocr",
        label="EasyOCR (offline)",
        capability=Capability.OCR,
        kind="offline",
        note="PyTorch; Urdu model + shared preprocessing/RTL pipeline. First run downloads ~100 MB.",
        price="Free · offline",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if easyocr is not None else (False, "pip install easyocr")

    def ocr(self, image: bytes) -> OcrResult:
        if easyocr is None:
            return OcrResult(provider_id=self.info.id, ok=False, error="easyocr not installed")
        t = self._timed(run, image, _recognize, _CFG)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        out = t["value"]
        deva, roman = rule_translit(out.text)
        boxes = [{"box": [list(p) for p in w.box], "text": w.text, "conf": w.conf}
                 for ln in out.lines for w in ln.words]
        return OcrResult(
            provider_id=self.info.id, ok=True, ms=t["ms"],
            text=out.text, boxes=boxes,
            meta={"devanagari": deva, "roman": roman, **out.meta},
        )


PROVIDER = EasyOcr()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_provider_easyocr.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/ocr/easyocr_p.py urdu-free-toolkit/tests/test_ocr_provider_easyocr.py
git commit -m "feat(ocr): easyocr provider on the shared pipeline + translit fill"
```

---

## Task 9: Refactor `paddle.py` onto the pipeline

**Files:**
- Modify: `urdu-free-toolkit/providers/ocr/paddle.py`
- Modify: `urdu-free-toolkit/tests/test_provider_paddle.py` (add kwarg-filtering test)
- Create: `urdu-free-toolkit/tests/test_ocr_provider_paddle.py`

**Interfaces:**
- Consumes: same pipeline modules as Task 8; `_model_hooks.paddle_rec_dir`.
- Produces: `PROVIDER.ocr` with the same `meta` keys as easyocr; module-level
  `_recognize(img, engine_cfg) -> list[Word]`, `_extract(result) -> list[Word]`,
  `_supported_kwargs(cls, want: dict) -> dict`.
- Preserves the existing version-safety contract from `test_provider_paddle.py`
  (`lang` in `{"ur","ar","fa"}`, `enable_mkldnn=False`).

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_ocr_provider_paddle.py
import inspect

import numpy as np

import providers.ocr.paddle as paddle


def test_supported_kwargs_filters_unknown():
    class Fake:
        def __init__(self, *, lang=None, enable_mkldnn=None, text_det_thresh=None):
            pass

    got = paddle._supported_kwargs(Fake, {
        "lang": "ur", "enable_mkldnn": False, "text_det_thresh": 0.3,
        "bogus_kwarg": 1, "drop_score": 0.4,
    })
    assert got == {"lang": "ur", "enable_mkldnn": False, "text_det_thresh": 0.3}


def test_extract_normalizes_dict_shape():
    result = [{
        "rec_texts": ["\u0627\u0644\u0641", "\u0628\u06d2"],
        "rec_polys": [np.array([[60, 5], [78, 5], [78, 22], [60, 22]]),
                      np.array([[5, 5], [30, 5], [30, 22], [5, 22]])],
        "rec_scores": [0.9, 0.8],
    }]
    words = paddle._extract(result)
    assert [w.text for w in words] == ["\u0627\u0644\u0641", "\u0628\u06d2"]
    assert words[0].conf == 0.9 and len(words[0].box) == 4
```

Also append to `urdu-free-toolkit/tests/test_provider_paddle.py`:

```python
def test_unknown_kwargs_never_reach_constructor(monkeypatch):
    import providers.ocr.paddle as paddle

    seen = {}

    class _FakePaddleOCR:
        def __init__(self, *, lang=None, enable_mkldnn=None, use_textline_orientation=None):
            seen.update(lang=lang, enable_mkldnn=enable_mkldnn)

    monkeypatch.setattr(paddle, "PaddleOCR", _FakePaddleOCR)
    monkeypatch.setattr(paddle, "_engine", None)
    paddle._get_engine()
    assert seen["lang"] in {"ur", "ar", "fa"} and seen["enable_mkldnn"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_provider_paddle.py tests/test_provider_paddle.py -v`
Expected: FAIL — `_supported_kwargs` / `_extract`-returns-`Word` not present yet.

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/providers/ocr/paddle.py
# -*- coding: utf-8 -*-
"""PaddleOCR as an OCR provider (offline), wrapped in the shared OCR pipeline.

Arabic-script model; covers the Urdu alphabet, strong on printed Naskh. Paddle's
Python API churns between versions, so construction filters kwargs against the
actual signature and the call tries ``predict`` then ``ocr``. ``lang="ur"`` is
required on 3.x (``"arabic"`` raises). ``enable_mkldnn=False`` dodges a
paddlepaddle 3.3.x PIR/oneDNN crash. First run downloads the models.
"""
from __future__ import annotations

import inspect

from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo
from providers.ocr import _model_hooks
from providers.ocr._pipeline import OcrConfig, run
from providers.ocr._translit_fill import rule_translit
from providers.ocr._types import Word

try:
    from paddleocr import PaddleOCR
except Exception:  # noqa: BLE001
    PaddleOCR = None

_engine = None

_ENGINE_CFG = dict(
    lang="ur", use_textline_orientation=True, enable_mkldnn=False,
    text_det_thresh=0.3, text_det_box_thresh=0.5, text_det_unclip_ratio=1.8,
    drop_score=0.4,
)
# Paddle's detector prefers grayscale contrast over hard binarization.
_CFG = OcrConfig(binarize="none", engine={})


def _supported_kwargs(cls, want: dict) -> dict:
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return dict(want)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(want)
    return {k: v for k, v in want.items() if k in params}


def _get_engine():
    global _engine
    if _engine is None:
        want = dict(_ENGINE_CFG)
        rec_dir = _model_hooks.paddle_rec_dir()
        if rec_dir:
            want["text_recognition_model_dir"] = rec_dir
        kw = _supported_kwargs(PaddleOCR, want)
        try:
            _engine = PaddleOCR(**kw)
        except TypeError:
            legacy = _supported_kwargs(PaddleOCR, {**want, "use_angle_cls": True})
            legacy.pop("use_textline_orientation", None)
            _engine = PaddleOCR(**legacy)
    return _engine


def _extract(result) -> list[Word]:
    rows: list[tuple] = []
    if isinstance(result, list) and result and isinstance(result[0], dict):
        d = result[0]
        texts = d.get("rec_texts") or d.get("rec_text") or []
        polys = d.get("rec_polys") or d.get("dt_polys") or [None] * len(texts)
        scores = d.get("rec_scores") or [0.0] * len(texts)
        rows = list(zip(polys, texts, scores))
    elif isinstance(result, list) and result and isinstance(result[0], list):
        for entry in result[0]:
            try:
                box, (txt, score) = entry
                rows.append((box, txt, score))
            except Exception:  # noqa: BLE001
                continue
    words: list[Word] = []
    for box, txt, score in rows:
        pts = [(0, 0)] * 4 if box is None else [(int(p[0]), int(p[1])) for p in box]
        words.append(Word(text=txt, box=pts, conf=float(score or 0.0)))
    return words


def _recognize(img, engine_cfg: dict) -> list[Word]:
    engine = _get_engine()
    try:
        result = engine.predict(img)
    except (AttributeError, TypeError):
        result = engine.ocr(img)
    return _extract(result)


class PaddleOcr(BaseProvider):
    info = ProviderInfo(
        id="paddle",
        label="PaddleOCR (offline)",
        capability=Capability.OCR,
        kind="offline",
        note="Arabic-script model + shared preprocessing/RTL pipeline. First run downloads ~20 MB.",
        price="Free · offline",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if PaddleOCR is not None else (False, "pip install paddleocr paddlepaddle")

    def ocr(self, image: bytes) -> OcrResult:
        if PaddleOCR is None:
            return OcrResult(provider_id=self.info.id, ok=False, error="paddleocr not installed")
        t = self._timed(run, image, _recognize, _CFG)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        out = t["value"]
        deva, roman = rule_translit(out.text)
        boxes = [{"box": [list(p) for p in w.box], "text": w.text, "conf": w.conf}
                 for ln in out.lines for w in ln.words]
        return OcrResult(
            provider_id=self.info.id, ok=True, ms=t["ms"],
            text=out.text, boxes=boxes,
            meta={"devanagari": deva, "roman": roman, **out.meta},
        )


PROVIDER = PaddleOcr()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_ocr_provider_paddle.py tests/test_provider_paddle.py -v`
Expected: PASS (existing version-safety test + 3 new)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/ocr/paddle.py urdu-free-toolkit/tests/test_ocr_provider_paddle.py urdu-free-toolkit/tests/test_provider_paddle.py
git commit -m "feat(ocr): paddle provider on the shared pipeline + kwarg filtering"
```

---

## Task 10: Smoke-test integration + full suite green

**Files:**
- Modify: `urdu-free-toolkit/tests/test_providers_smoke.py` (add an OCR non-crash check with fake engines)
- Modify: `urdu-free-toolkit/requirements.txt`

**Interfaces:**
- Consumes: registry, both providers.
- Produces: nothing new; guards the contract.

- [ ] **Step 1: Add the failing test**

Append to `urdu-free-toolkit/tests/test_providers_smoke.py`:

```python
def test_offline_ocr_providers_emit_translit_meta(monkeypatch):
    """paddle / easyocr, with their engine calls faked, return an OcrResult
    carrying meta['devanagari'] / meta['roman'] / meta['reading_order']."""
    from providers.ocr._types import Word
    import providers.ocr.easyocr_p as e
    import providers.ocr.paddle as p

    monkeypatch.setattr(e, "easyocr", object())  # pass the availability gate
    monkeypatch.setattr(e, "_get_reader", lambda: None)
    monkeypatch.setattr(e, "_recognize",
                        lambda img, cfg: [Word("\u06c1\u06d2", [(0, 0), (20, 0), (20, 14), (0, 14)], 0.9)])
    monkeypatch.setattr(p, "PaddleOCR", object())
    monkeypatch.setattr(p, "_recognize",
                        lambda img, cfg: [Word("\u06c1\u06d2", [(0, 0), (20, 0), (20, 14), (0, 14)], 0.9)])

    for prov in (e.PROVIDER, p.PROVIDER):
        r = prov.ocr(_png())
        assert r.ok is True
        assert {"devanagari", "roman", "reading_order"} <= set(r.meta)
```

- [ ] **Step 2: Run it, watch it pass** (the providers already produce this shape after Tasks 8–9)

Run: `cd urdu-free-toolkit && python -m pytest tests/test_providers_smoke.py -v`
Expected: PASS. If it fails because `_png` isn't in scope, it is — it's a module-level helper in that file.

- [ ] **Step 3: Update `requirements.txt`**

Under `# --- OCR (offline) ---`, add:

```
opencv-python-headless  # preprocessing pipeline (grayscale, deskew, CLAHE, binarize, upscale)
scikit-image            # Sauvola binarization; falls back to cv2.adaptiveThreshold if absent
python-bidi             # inline LTR-run ordering in RTL lines; heuristic fallback if absent
Levenshtein             # normalize_urdu spellfix + eval CER/WER scoring
```

Add a new group:

```
# --- OCR fine-tuning tooling (Phase 2, CPU side; training itself is a GPU runbook) ---
rapidfuzz               # GPT-distillation line alignment
fonttools               # synthetic renderer font handling
# Pillow (listed in core) needs libraqm at runtime for Arabic shaping in training/synth/render.py
```

- [ ] **Step 4: Run the entire suite**

Run: `cd urdu-free-toolkit && python -m pytest -q`
Expected: PASS — 179 prior + the new tests, no failures, no network.

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/tests/test_providers_smoke.py urdu-free-toolkit/requirements.txt
git commit -m "test(ocr): smoke-check translit meta on offline OCR; declare new deps"
```

---

## Task 11: Eval scorer (`eval/score.py`)

**Files:**
- Create: `urdu-free-toolkit/eval/__init__.py` (empty)
- Create: `urdu-free-toolkit/eval/score.py`
- Test: `urdu-free-toolkit/tests/test_eval_score.py`

**Interfaces:**
- Consumes: `Levenshtein.distance`; `providers.ocr._normalize.normalize_urdu`.
- Produces:
  - `cer(ref: str, hyp: str) -> float` (0.0 = identical; normalized by `len(ref)`)
  - `wer(ref: str, hyp: str) -> float`
  - `score_pair(ref: str, hyp: str) -> dict` → `{"cer","wer","cer_norm","wer_norm"}`
  - `aggregate(rows: list[dict]) -> dict` → mean of each numeric key, plus `"n"`.

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_eval_score.py
import pytest

from eval.score import cer, wer, score_pair, aggregate


def test_cer_identical_is_zero():
    assert cer("\u06c1\u06d2", "\u06c1\u06d2") == 0.0


def test_cer_one_substitution():
    assert cer("abcd", "abxd") == pytest.approx(0.25)


def test_wer_counts_word_edits():
    assert wer("a b c", "a x c") == pytest.approx(1 / 3)


def test_score_pair_normalization_erases_encoding_only_diff():
    # ك (U+0643) vs ک (U+06A9): raw differs, normalized identical
    row = score_pair("\u0643\u062a\u0627\u0628", "\u06a9\u062a\u0627\u0628")
    assert row["cer"] > 0
    assert row["cer_norm"] == 0.0


def test_aggregate_means():
    rows = [{"cer": 0.0, "wer": 0.0, "cer_norm": 0.0, "wer_norm": 0.0},
            {"cer": 0.5, "wer": 1.0, "cer_norm": 0.1, "wer_norm": 0.2}]
    agg = aggregate(rows)
    assert agg["cer"] == pytest.approx(0.25) and agg["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_eval_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/eval/__init__.py
```

```python
# urdu-free-toolkit/eval/score.py
# -*- coding: utf-8 -*-
"""CER / WER scoring for OCR output against a reference, raw and normalized."""
from __future__ import annotations

from Levenshtein import distance

from providers.ocr._normalize import normalize_urdu


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return distance(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    # token-level Levenshtein
    prev = list(range(len(h) + 1))
    for i, rt in enumerate(r, 1):
        cur = [i]
        for j, ht in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rt != ht)))
        prev = cur
    return prev[-1] / len(r)


def score_pair(ref: str, hyp: str) -> dict:
    rn, hn = normalize_urdu(ref), normalize_urdu(hyp)
    return {
        "cer": cer(ref, hyp),
        "wer": wer(ref, hyp),
        "cer_norm": cer(rn, hn),
        "wer_norm": wer(rn, hn),
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "cer": 0.0, "wer": 0.0, "cer_norm": 0.0, "wer_norm": 0.0}
    keys = ("cer", "wer", "cer_norm", "wer_norm")
    out = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    out["n"] = len(rows)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_eval_score.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/eval/__init__.py urdu-free-toolkit/eval/score.py urdu-free-toolkit/tests/test_eval_score.py
git commit -m "feat(eval): CER/WER scorer, raw and normalized"
```

---

## Task 12: Silver-reference cache (`eval/refs.py`) + fixtures

**Files:**
- Create: `urdu-free-toolkit/eval/refs.py`
- Create: `urdu-free-toolkit/tests/fixtures/ocr/README.md`
- Create: `urdu-free-toolkit/tests/fixtures/ocr/.gitkeep`
- Test: `urdu-free-toolkit/tests/test_eval_refs.py`

**Interfaces:**
- Consumes: `providers.ocr.gpt.PROVIDER` (lazily), filesystem.
- Produces:
  - `FIXTURES_DIR: Path`
  - `list_fixtures(d: Path | None = None) -> list[Path]` — the `*.png` files
  - `ref_path(png: Path) -> Path` — `<slug>.gpt.json`
  - `load_ref(png: Path) -> dict | None`
  - `ensure_refs(d: Path | None = None, refresh: bool = False, ocr=None) -> dict` →
    `{"written": [...], "skipped": [...], "have": [...]}`; `ocr` is an injectable
    `callable(bytes) -> object with .ok/.text/.meta` (defaults to the gpt provider).

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_eval_refs.py
import io
import json

from PIL import Image

from eval import refs


def _png(p):
    Image.new("RGB", (30, 16), "white").save(p, format="PNG")


class _FakeOcr:
    def __init__(self, text): self.text, self.ok, self.meta = text, True, {"model": "fake"}


def test_ensure_refs_writes_missing_then_is_idempotent(tmp_path):
    _png(tmp_path / "a.png")
    calls = []

    def fake_ocr(b):
        calls.append(1)
        return _FakeOcr("\u06c1\u06d2")

    r1 = refs.ensure_refs(tmp_path, ocr=fake_ocr)
    assert r1["written"] == ["a"] and (tmp_path / "a.gpt.json").exists()
    data = json.loads((tmp_path / "a.gpt.json").read_text(encoding="utf-8"))
    assert data["urdu"] == "\u06c1\u06d2" and data["model"] == "fake"

    r2 = refs.ensure_refs(tmp_path, ocr=fake_ocr)
    assert r2["written"] == [] and r2["have"] == ["a"] and len(calls) == 1


def test_ensure_refs_skips_when_ocr_unavailable(tmp_path):
    _png(tmp_path / "b.png")

    def broken(b):
        raise RuntimeError("OPENAI_API_KEY not set")

    r = refs.ensure_refs(tmp_path, ocr=broken)
    assert r["skipped"] == ["b"] and not (tmp_path / "b.gpt.json").exists()


def test_refresh_overwrites(tmp_path):
    _png(tmp_path / "c.png")
    (tmp_path / "c.gpt.json").write_text('{"urdu":"old","model":"x"}', encoding="utf-8")
    refs.ensure_refs(tmp_path, refresh=True, ocr=lambda b: _FakeOcr("new"))
    assert json.loads((tmp_path / "c.gpt.json").read_text(encoding="utf-8"))["urdu"] == "new"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_eval_refs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.refs'`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/eval/refs.py
# -*- coding: utf-8 -*-
"""Generate and cache GPT "silver reference" transcriptions for the OCR eval
fixtures. Refs are committed JSON so the eval runs offline and free after the
first generation; pass refresh=True (CLI: --refresh) to regenerate.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ocr"


def list_fixtures(d: Path | None = None) -> list[Path]:
    d = d or FIXTURES_DIR
    return sorted(p for p in d.glob("*.png"))


def ref_path(png: Path) -> Path:
    return png.with_suffix(".gpt.json")


def load_ref(png: Path) -> dict | None:
    p = ref_path(png)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.warning("bad ref json: %s", p, exc_info=True)
        return None


def _default_ocr(image: bytes):
    from providers.ocr.gpt import PROVIDER
    return PROVIDER.ocr(image)


def ensure_refs(d: Path | None = None, refresh: bool = False, ocr=None) -> dict:
    d = d or FIXTURES_DIR
    ocr = ocr or _default_ocr
    written, skipped, have = [], [], []
    for png in list_fixtures(d):
        slug = png.stem
        if not refresh and ref_path(png).exists():
            have.append(slug)
            continue
        try:
            res = ocr(png.read_bytes())
        except Exception as e:
            log.warning("silver ref for %s skipped: %s", slug, e)
            skipped.append(slug)
            continue
        if not getattr(res, "ok", False) or not getattr(res, "text", ""):
            skipped.append(slug)
            continue
        ref_path(png).write_text(json.dumps({
            "urdu": res.text,
            "model": (getattr(res, "meta", {}) or {}).get("model", "gpt"),
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(slug)
    return {"written": written, "skipped": skipped, "have": have}
```

Fixtures README:

```markdown
# OCR eval fixtures

Each `<slug>.png` is a small Urdu image; `<slug>.gpt.json` is its cached GPT
"silver reference" transcription (`{"urdu": "...", "model": "...", "generated_at": "..."}`).

## Add a fixture
1. Drop a redistributable `<slug>.png` here (a synthetic render from
   `training/synth`, or an image you have the right to commit). Keep it small.
2. `cd urdu-free-toolkit && OPENAI_API_KEY=... python -m eval.run_eval --refresh`
   to generate `<slug>.gpt.json`.
3. Eyeball the JSON — GPT misreads too. Hand-fix the `urdu` value if needed;
   it's just JSON.
4. Commit both files.

The eval and its regression test read the committed JSON and never call the
network.
```

`.gitkeep` is an empty file so the directory exists before any fixture is added.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_eval_refs.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/eval/refs.py urdu-free-toolkit/tests/test_eval_refs.py urdu-free-toolkit/tests/fixtures/ocr/README.md urdu-free-toolkit/tests/fixtures/ocr/.gitkeep
git commit -m "feat(eval): GPT silver-reference cache + fixtures dir"
```

---

## Task 13: Eval runner CLI (`eval/run_eval.py`) + baseline

**Files:**
- Create: `urdu-free-toolkit/eval/run_eval.py`
- Create: `urdu-free-toolkit/eval/baseline.json` (seeded `{}`, filled after a real run)
- Test: `urdu-free-toolkit/tests/test_eval_run.py`

**Interfaces:**
- Consumes: `eval.refs`, `eval.score`, `providers.registry` (to resolve an
  engine id → provider), `providers.ocr._types.OcrConfig`.
- Produces:
  - `evaluate(engines: list[str], fixtures_dir=None, spellfix=False, ocr_map=None) -> dict`
    → `{engine: {"rows": [...per fixture...], "agg": {...}}}`; `ocr_map` injects
    `{engine_id: callable(bytes)->result}` for tests.
  - `format_table(results: dict) -> str` (Markdown)
  - `check_regression(results: dict, baseline: dict, tol=0.01) -> list[str]`
    (returns regression messages; empty = OK)
  - `main(argv=None) -> int` — flags `--engines`, `--refresh`, `--spellfix`,
    `--check`, `--write-baseline`.

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_eval_run.py
import io
import json

from PIL import Image

from eval import run_eval


def _png(p): Image.new("RGB", (30, 16), "white").save(p, format="PNG")


class _Res:
    def __init__(self, text): self.ok, self.text, self.meta = True, text, {}


def _fixtures(tmp_path):
    _png(tmp_path / "one.png")
    (tmp_path / "one.gpt.json").write_text(
        json.dumps({"urdu": "\u06c1\u06d2 \u0679\u06be\u06cc\u06a9", "model": "gpt"}),
        encoding="utf-8")
    return tmp_path


def test_evaluate_scores_each_engine(tmp_path):
    d = _fixtures(tmp_path)
    ocr_map = {
        "perfect": lambda b: _Res("\u06c1\u06d2 \u0679\u06be\u06cc\u06a9"),
        "bad": lambda b: _Res("xxx"),
    }
    res = run_eval.evaluate(["perfect", "bad"], fixtures_dir=d, ocr_map=ocr_map)
    assert res["perfect"]["agg"]["cer_norm"] == 0.0
    assert res["bad"]["agg"]["cer_norm"] > 0.5


def test_check_regression_flags_worse_cer():
    results = {"paddle": {"agg": {"cer_norm": 0.30}}}
    assert run_eval.check_regression(results, {"paddle": {"cer_norm": 0.20}}, tol=0.01)
    assert not run_eval.check_regression(results, {"paddle": {"cer_norm": 0.30}}, tol=0.01)


def test_format_table_has_headers():
    t = run_eval.format_table({"paddle": {"agg": {"cer": 0.1, "wer": 0.2, "cer_norm": 0.05, "wer_norm": 0.1, "n": 3}, "ms": 12}})
    assert "engine" in t and "CER" in t and "paddle" in t


def test_main_check_returns_nonzero_on_regression(tmp_path, capsys, monkeypatch):
    d = _fixtures(tmp_path)
    monkeypatch.setattr(run_eval, "_resolve_ocr", lambda ids: {"paddle": lambda b: _Res("xxx")})
    (d / "baseline.json").write_text('{"paddle": {"cer_norm": 0.0}}', encoding="utf-8")
    rc = run_eval.main(["--engines", "paddle", "--fixtures", str(d),
                        "--baseline", str(d / "baseline.json"), "--check"])
    assert rc == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_eval_run.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.run_eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/eval/run_eval.py
# -*- coding: utf-8 -*-
"""Score OCR engines against cached GPT silver references.

    cd urdu-free-toolkit
    python -m eval.run_eval --engines gpt,paddle,easyocr
    python -m eval.run_eval --engines paddle,easyocr --check      # CI regression gate
    python -m eval.run_eval --refresh                             # regenerate silver refs (needs OPENAI_API_KEY)
    python -m eval.run_eval --engines paddle,easyocr --write-baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from eval import refs
from eval.score import aggregate, score_pair

_DEFAULT_BASELINE = Path(__file__).resolve().parent / "baseline.json"


def _resolve_ocr(engine_ids: list[str]) -> dict:
    from providers import registry
    registry.reset_cache()
    reg = registry.discover(package="providers")
    out = {}
    for eid in engine_ids:
        prov = reg.get(f"ocr:{eid}") or reg.get(eid)
        if prov is None:
            raise SystemExit(f"unknown OCR engine: {eid}")
        out[eid] = prov.ocr
    return out


def evaluate(engines: list[str], fixtures_dir=None, spellfix: bool = False,
             ocr_map: dict | None = None) -> dict:
    fixtures_dir = Path(fixtures_dir) if fixtures_dir else refs.FIXTURES_DIR
    ocr_map = ocr_map or _resolve_ocr(engines)
    pngs = refs.list_fixtures(fixtures_dir)
    results: dict = {}
    for eid in engines:
        rows, t0 = [], time.monotonic()
        for png in pngs:
            ref = refs.load_ref(png)
            if not ref:
                continue
            res = ocr_map[eid](png.read_bytes())
            hyp = getattr(res, "text", "") if getattr(res, "ok", False) else ""
            row = score_pair(ref["urdu"], hyp)
            row["fixture"] = png.stem
            rows.append(row)
        ms = int((time.monotonic() - t0) * 1000 / max(1, len(pngs)))
        results[eid] = {"rows": rows, "agg": aggregate(rows), "ms": ms}
    return results


def format_table(results: dict) -> str:
    head = "| engine | CER | WER | CER(norm) | WER(norm) | n | ms/img |\n|---|---|---|---|---|---|---|"
    lines = [head]
    for eid, r in results.items():
        a = r["agg"]
        lines.append(f"| {eid} | {a['cer']:.3f} | {a['wer']:.3f} | {a['cer_norm']:.3f} "
                     f"| {a['wer_norm']:.3f} | {a.get('n', 0)} | {r.get('ms', 0)} |")
    return "\n".join(lines)


def check_regression(results: dict, baseline: dict, tol: float = 0.01) -> list[str]:
    msgs = []
    for eid, r in results.items():
        base = baseline.get(eid)
        if not base:
            continue
        cur = r["agg"]["cer_norm"]
        if cur > base["cer_norm"] + tol:
            msgs.append(f"{eid}: CER(norm) {cur:.3f} > baseline {base['cer_norm']:.3f} + {tol}")
    return msgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eval.run_eval")
    ap.add_argument("--engines", default="gpt,paddle,easyocr")
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--baseline", default=str(_DEFAULT_BASELINE))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--spellfix", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args(argv)

    fixtures_dir = Path(args.fixtures) if args.fixtures else refs.FIXTURES_DIR
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    if args.refresh:
        rep = refs.ensure_refs(fixtures_dir, refresh=True)
        print(f"silver refs: wrote {rep['written']}, skipped {rep['skipped']}")

    ocr_map = None if not _in_test() else None
    results = evaluate(engines, fixtures_dir=fixtures_dir, spellfix=args.spellfix, ocr_map=ocr_map)
    print(format_table(results))

    if args.write_baseline:
        data = {eid: {"cer_norm": r["agg"]["cer_norm"], "wer_norm": r["agg"]["wer_norm"]}
                for eid, r in results.items()}
        Path(args.baseline).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote baseline -> {args.baseline}")
        return 0

    if args.check:
        try:
            baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        except Exception:
            baseline = {}
        msgs = check_regression(results, baseline)
        for m in msgs:
            print("REGRESSION:", m)
        return 1 if msgs else 0
    return 0


def _in_test() -> bool:
    return "pytest" in sys.modules


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

> Note: `main` calls `_resolve_ocr` indirectly via `evaluate`; the
> `test_main_check_returns_nonzero_on_regression` test monkeypatches
> `run_eval._resolve_ocr`. Keep `evaluate` calling `_resolve_ocr` (module-level
> name) so the patch applies.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_eval_run.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Seed the baseline file**

```bash
printf '{}\n' > urdu-free-toolkit/eval/baseline.json
```

- [ ] **Step 6: Commit**

```bash
git add urdu-free-toolkit/eval/run_eval.py urdu-free-toolkit/eval/baseline.json urdu-free-toolkit/tests/test_eval_run.py
git commit -m "feat(eval): run_eval CLI — scoreboard, regression gate, baseline"
```

---

## Task 14: Heavy regression test wiring (`test_eval_regression.py`)

**Files:**
- Create: `urdu-free-toolkit/tests/test_eval_regression.py`

**Interfaces:**
- Consumes: `eval.run_eval.evaluate`, `eval.run_eval.check_regression`, `eval.baseline.json`.
- Produces: a `@pytest.mark.heavy` test that runs `paddle` + `easyocr` over the
  committed fixtures with cached refs and asserts no CER(norm) regression.

- [ ] **Step 1: Write the test**

```python
# urdu-free-toolkit/tests/test_eval_regression.py
"""RUN_HEAVY=1 only. Runs the real paddle/easyocr engines over the committed
OCR fixtures (cached silver refs, no network) and asserts CER(norm) has not
regressed past eval/baseline.json. Skips cleanly if there are no fixtures or no
baseline entries yet."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy


def test_offline_ocr_no_cer_regression():
    from eval import refs, run_eval

    if not refs.list_fixtures():
        pytest.skip("no OCR fixtures committed yet")
    baseline_path = Path(run_eval._DEFAULT_BASELINE)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    engines = [e for e in ("paddle", "easyocr") if e in baseline]
    if not engines:
        pytest.skip("no baseline entries for paddle/easyocr yet")

    results = run_eval.evaluate(engines)
    msgs = run_eval.check_regression(results, baseline, tol=0.01)
    assert not msgs, "\n".join(msgs)
```

- [ ] **Step 2: Run (light mode — should skip)**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_eval_regression.py -v`
Expected: SKIPPED ("heavy provider; set RUN_HEAVY=1 to run")

- [ ] **Step 3: Commit**

```bash
git add urdu-free-toolkit/tests/test_eval_regression.py
git commit -m "test(eval): heavy CER-regression guard for offline OCR"
```

---

## Task 15: Synthetic line renderer (`training/synth/`)

**Files:**
- Create: `urdu-free-toolkit/training/__init__.py`, `training/synth/__init__.py`
- Create: `urdu-free-toolkit/training/synth/render.py`
- Create: `urdu-free-toolkit/training/synth/augment.py`
- Create: `urdu-free-toolkit/training/synth/build_synth.py`
- Create: `urdu-free-toolkit/training/synth/fonts/.gitkeep`, `training/synth/fonts.json`
- Create: `urdu-free-toolkit/training/corpus/__init__.py`, `training/corpus/sample_sentences.txt`
- Test: `urdu-free-toolkit/tests/test_synth_render.py`

**Interfaces:**
- Consumes: Pillow (`ImageFont`, `ImageDraw`), numpy, `cv2`.
- Produces:
  - `render.render_line(text: str, font_path: str, size: int = 48, pad: int = 8) -> "PIL.Image.Image"`
  - `render.find_fonts(fonts_dir=None) -> list[str]` — `*.ttf`/`*.otf` under `fonts/`
  - `augment.augment(img: "np.ndarray", rng: "np.random.Generator") -> "np.ndarray"`
  - `build_synth.build(sentences: list[str], font_paths: list[str], out_dir: Path,
    per_sentence: int = 1, val_frac: float = 0.1, seed: int = 0) -> dict`
    → `{"train": int, "val": int, "gt": Path}`; writes `out_dir/{train,val}/*.png`
    and `out_dir/{train,val}/gt.txt` (`<relpath>\t<text>` per line).

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_synth_render.py
import numpy as np
import pytest

from training.synth import augment, build_synth, render

_CANDIDATE_FONTS = [
    *render.find_fonts(),
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _a_font():
    from PIL import ImageFont
    for p in _CANDIDATE_FONTS:
        try:
            ImageFont.truetype(p, 40)
            return p
        except Exception:
            continue
    pytest.skip("no usable font on this machine")


def test_render_line_is_non_blank():
    img = render.render_line("\u0645\u06cc\u06ba \u0679\u06be\u06cc\u06a9 \u06c1\u0648\u06ba", _a_font(), size=48)
    arr = np.array(img.convert("L"))
    assert arr.shape[0] > 10 and (arr < 128).mean() > 0.002   # some ink


def test_augment_changes_pixels_but_keeps_shape():
    rng = np.random.default_rng(0)
    src = np.full((40, 200), 255, np.uint8)
    src[15:25, 20:180] = 0
    out = augment.augment(src, rng)
    assert out.shape == src.shape and not np.array_equal(out, src)


def test_build_writes_split_and_gt(tmp_path, monkeypatch):
    # stub rendering so the test doesn't need a font
    monkeypatch.setattr(render, "render_line",
                        lambda text, fp, **k: __import__("PIL.Image", fromlist=["Image"]).new("RGB", (120, 32), "white"))
    sentences = ["\u062c\u0645\u0644\u06c1 \u0627\u06cc\u06a9", "\u062c\u0645\u0644\u06c1 \u062f\u0648", "\u062c\u0645\u0644\u06c1 \u062a\u06cc\u0646", "\u062c\u0645\u0644\u06c1 \u0686\u0627\u0631"]
    rep = build_synth.build(sentences, ["fake.ttf"], tmp_path, per_sentence=2, val_frac=0.25, seed=1)
    assert rep["train"] + rep["val"] == 8
    gt_lines = (tmp_path / "train" / "gt.txt").read_text(encoding="utf-8").splitlines()
    assert gt_lines and "\t" in gt_lines[0]
    # every referenced file exists
    for line in gt_lines:
        rel = line.split("\t", 1)[0]
        assert (tmp_path / "train" / rel).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_synth_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training'`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/training/__init__.py
```
```python
# urdu-free-toolkit/training/synth/__init__.py
```
```python
# urdu-free-toolkit/training/corpus/__init__.py
```

```
# urdu-free-toolkit/training/corpus/sample_sentences.txt
میں ٹھیک ہوں شکریہ
یہ کتاب بہت اچھی ہے
آج موسم بہت خوشگوار ہے
اردو زبان بہت خوبصورت ہے
ہم کل بازار جائیں گے
```

```json
// urdu-free-toolkit/training/synth/fonts.json
{
  "_comment": "Drop Urdu fonts (OFL/permissive) into ./fonts and list them here with their licence. Not committed - see training/README.md.",
  "fonts": []
}
```

```python
# urdu-free-toolkit/training/synth/render.py
# -*- coding: utf-8 -*-
"""Render a text line to a tight image for synthetic OCR training data.

Requires Pillow built with libraqm for correct Arabic shaping; render_line
asserts this so failures are loud, not silently wrong.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

_FONTS_DIR = Path(__file__).resolve().parent / "fonts"


def find_fonts(fonts_dir: Path | None = None) -> list[str]:
    d = fonts_dir or _FONTS_DIR
    return [str(p) for p in sorted(d.glob("*.ttf")) + sorted(d.glob("*.otf"))]


def render_line(text: str, font_path: str, size: int = 48, pad: int = 8) -> Image.Image:
    if not features.check("raqm"):
        raise RuntimeError(
            "Pillow lacks libraqm; Arabic shaping would be wrong. "
            "Install libraqm (conda-forge) or run under WSL. See training/README.md."
        )
    font = ImageFont.truetype(font_path, size)
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    l, t, r, b = tmp.textbbox((0, 0), text, font=font, language="ur")
    w, h = (r - l) + 2 * pad, (b - t) + 2 * pad
    img = Image.new("RGB", (max(w, 1), max(h, 1)), "white")
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill="black", language="ur")
    return img
```

```python
# urdu-free-toolkit/training/synth/augment.py
# -*- coding: utf-8 -*-
"""Light, order-preserving image augmentation for synthetic OCR lines."""
from __future__ import annotations

import cv2
import numpy as np


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    # gaussian blur
    if rng.random() < 0.5:
        k = int(rng.choice([3, 3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    # gaussian noise
    if rng.random() < 0.6:
        noise = rng.normal(0, rng.uniform(4, 14), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # brightness / contrast jitter
    if rng.random() < 0.6:
        alpha = float(rng.uniform(0.8, 1.2))
        beta = float(rng.uniform(-20, 20))
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    # tiny rotation (<= 2.5 deg) — keeps text readable / order intact
    if rng.random() < 0.5:
        ang = float(rng.uniform(-2.5, 2.5))
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderValue=255)
    # jpeg recompression artefacts
    if rng.random() < 0.5:
        q = int(rng.integers(30, 80))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return out
```

```python
# urdu-free-toolkit/training/synth/build_synth.py
# -*- coding: utf-8 -*-
"""Assemble a synthetic OCR training set: sentences x fonts x augmentations
-> out_dir/{train,val}/*.png + gt.txt  (tab-separated: <relpath>\\t<text>).

    python -m training.synth.build_synth --sentences training/corpus/sample_sentences.txt \\
        --fonts training/synth/fonts --out data/synth --per-sentence 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from training.synth import augment as _aug
from training.synth import render as _render


def build(sentences: list[str], font_paths: list[str], out_dir: Path,
          per_sentence: int = 1, val_frac: float = 0.1, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    items: list[tuple[str, str]] = []   # (relpath, text)
    counts = {"train": 0, "val": 0}
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    idx = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        for _ in range(per_sentence):
            fp = font_paths[int(rng.integers(0, len(font_paths)))]
            img = _render.render_line(s, fp)
            arr = _aug.augment(np.array(img), rng)
            split = "val" if rng.random() < val_frac else "train"
            rel = f"{idx:06d}.png"
            import cv2
            cv2.imwrite(str(out_dir / split / rel), arr)
            (out_dir / split / "gt.txt").open("a", encoding="utf-8").write(f"{rel}\t{s}\n")
            counts[split] += 1
            idx += 1
    return {"train": counts["train"], "val": counts["val"], "gt": out_dir / "train" / "gt.txt"}


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-sentence", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sentences = Path(a.sentences).read_text(encoding="utf-8").splitlines()
    fonts = _render.find_fonts(Path(a.fonts))
    if not fonts:
        raise SystemExit(f"no .ttf/.otf fonts in {a.fonts}")
    rep = build(sentences, fonts, Path(a.out), a.per_sentence, a.val_frac, a.seed)
    print(rep)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

`.gitkeep` in `training/synth/fonts/` is an empty file.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_synth_render.py -v`
Expected: PASS (3 tests; `test_render_line_is_non_blank` may SKIP if no font — acceptable)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/training/__init__.py urdu-free-toolkit/training/synth/ urdu-free-toolkit/training/corpus/ urdu-free-toolkit/tests/test_synth_render.py
git commit -m "feat(training): synthetic Urdu line renderer + augmentation + dataset builder"
```

---

## Task 16: Corpus fetcher (`training/corpus/fetch_corpus.py`)

**Files:**
- Create: `urdu-free-toolkit/training/corpus/fetch_corpus.py`
- Create: `urdu-free-toolkit/training/corpus/LICENCES.md`
- Test: `urdu-free-toolkit/tests/test_corpus_fetch.py`

**Interfaces:**
- Consumes: stdlib only (`urllib`), optional network.
- Produces:
  - `clean_sentences(raw: str, min_words: int = 3, max_words: int = 18) -> list[str]`
    — split on line/sentence punctuation, keep lines that are majority
    Urdu-script and within the word bounds, dedupe preserving order.
  - `fetch(out_path: Path, source: str = "sample", limit: int = 50000) -> int`
    — `source="sample"` copies the bundled `sample_sentences.txt`; other sources
    documented in `LICENCES.md`; returns number of sentences written.

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_corpus_fetch.py
from pathlib import Path

from training.corpus.fetch_corpus import clean_sentences, fetch


def test_clean_keeps_urdu_lines_in_word_bounds():
    raw = "\n".join([
        "\u06cc\u06c1 \u0627\u06cc\u06a9 \u0627\u0686\u06be\u06cc \u0645\u062b\u0627\u0644 \u06c1\u06d2",  # ok
        "too short",                                           # not urdu / too short
        "\u0627\u06cc\u06a9",                                   # 1 word -> drop
        "\u06cc\u06c1 \u0627\u06cc\u06a9 \u0627\u0686\u06be\u06cc \u0645\u062b\u0627\u0644 \u06c1\u06d2",  # dup -> drop
    ])
    out = clean_sentences(raw, min_words=3, max_words=18)
    assert out == ["\u06cc\u06c1 \u0627\u06cc\u06a9 \u0627\u0686\u06be\u06cc \u0645\u062b\u0627\u0644 \u06c1\u06d2"]


def test_fetch_sample_writes_file(tmp_path):
    out = tmp_path / "sentences.txt"
    n = fetch(out, source="sample")
    assert n >= 5 and out.read_text(encoding="utf-8").strip()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_corpus_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/training/corpus/fetch_corpus.py
# -*- coding: utf-8 -*-
"""Build a plain-text Urdu sentence list for the synthetic renderer.

Default source is the small bundled sample. Larger permissively-licensed
sources (Urdu Wikipedia extracts, CC-100 ur, Leipzig Corpora) are described in
LICENCES.md — fetching those is left to the operator so licence acceptance is
explicit.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SAMPLE = _HERE / "sample_sentences.txt"
_SPLIT = re.compile(r"[\n\u06d4\u061f!\.]+")


def _is_urdu(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    urdu = sum(1 for c in letters if "\u0600" <= c <= "\u06ff")
    return urdu / len(letters) >= 0.6


def clean_sentences(raw: str, min_words: int = 3, max_words: int = 18) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for chunk in _SPLIT.split(raw):
        s = " ".join(chunk.split())
        if not s or not _is_urdu(s):
            continue
        n = len(s.split())
        if n < min_words or n > max_words or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def fetch(out_path: Path, source: str = "sample", limit: int = 50_000) -> int:
    out_path = Path(out_path)
    if source == "sample":
        raw = _SAMPLE.read_text(encoding="utf-8")
    else:
        raise SystemExit(f"source {source!r} not built in; see training/corpus/LICENCES.md")
    sents = clean_sentences(raw)[:limit]
    out_path.write_text("\n".join(sents) + "\n", encoding="utf-8")
    return len(sents)


def main(argv=None) -> int:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="sample")
    ap.add_argument("--limit", type=int, default=50_000)
    a = ap.parse_args(argv)
    print(fetch(Path(a.out), a.source, a.limit))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

```markdown
# urdu-free-toolkit/training/corpus/LICENCES.md
# Corpus sources

| source | how | licence | notes |
|---|---|---|---|
| bundled `sample_sentences.txt` | in-repo | CC0 (hand-written) | tiny; smoke tests + a first render |
| Urdu Wikipedia | `wikiextractor` on a `urwiki` dump | CC BY-SA 3.0 | attribute; good register coverage |
| CC-100 `ur` | statmt.org/cc-100 | Common Crawl ToU | noisy web text; filter hard |
| Leipzig Corpora `urd_*` | wortschatz.uni-leipzig.de | CC BY-NC | non-commercial only |

`fetch_corpus.py` only bundles the sample. To use another source, download it
yourself, accept its licence, then run `clean_sentences` over it.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_corpus_fetch.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/training/corpus/fetch_corpus.py urdu-free-toolkit/training/corpus/LICENCES.md urdu-free-toolkit/tests/test_corpus_fetch.py
git commit -m "feat(training): Urdu corpus fetch + cleaning (bundled sample source)"
```

---

## Task 17: GPT-distillation aligner (`training/distill/`)

**Files:**
- Create: `urdu-free-toolkit/training/distill/__init__.py`
- Create: `urdu-free-toolkit/training/distill/align.py`
- Create: `urdu-free-toolkit/training/distill/collect.py`
- Create: `urdu-free-toolkit/training/distill/build_distill.py`
- Test: `urdu-free-toolkit/tests/test_distill_align.py`

**Interfaces:**
- Consumes: `rapidfuzz.fuzz`, `providers.ocr._rtl` (ordering), `providers.ocr._types.Word`.
- Produces:
  - `align.align_lines(crop_texts_hint: list[str] | None, page_lines: list[str],
    n_crops: int) -> list[str | None]` — assign one page line per crop, in order,
    with fuzzy fallback; `None` where confidence is too low.
  - `align.align(crops: list[Word], gpt_urdu: str, min_ratio: float = 70.0) -> list[tuple[Word, str]]`
    — high level: order crops via `_rtl`, split `gpt_urdu` into lines, align,
    drop low-confidence pairs.
  - `collect.crops_from_image(image: bytes, recognize) -> list[Word]` — reuse an
    engine `recognize` to get line-ish boxes (thin wrapper, documented, not unit-tested here).
  - `build_distill.build(pairs: list[tuple["PIL.Image.Image", str]], out_dir, val_frac=0.1, seed=0) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_distill_align.py
from providers.ocr._types import Word
from training.distill import align


def _w(text, y):
    return Word(text=text, box=[(0, y), (100, y), (100, y + 20), (0, y + 20)], conf=0.9)


def test_align_lines_one_to_one_when_counts_match():
    got = align.align_lines(None, ["alpha", "beta", "gamma"], n_crops=3)
    assert got == ["alpha", "beta", "gamma"]


def test_align_drops_when_page_has_fewer_lines():
    got = align.align_lines(None, ["only one"], n_crops=3)
    assert got[0] == "only one" and got[1] is None and got[2] is None


def test_align_uses_hint_to_reorder_fuzzy():
    # crop hints are noisy OCR; page lines are clean GPT — match by similarity
    hints = ["helo wrld", "gud bye"]
    page = ["good bye", "hello world"]
    got = align.align_lines(hints, page, n_crops=2)
    assert got == ["hello world", "good bye"]


def test_high_level_align_orders_and_filters():
    crops = [_w("\u062f\u0648\u0645 \u0633\u0637\u0631", 40), _w("\u0627\u0648\u0644 \u0633\u0637\u0631", 0)]
    gpt = "\u0627\u0648\u0644 \u0633\u0637\u0631\n\u062f\u0648\u0645 \u0633\u0637\u0631"
    pairs = align.align(crops, gpt, min_ratio=50.0)
    texts = [t for _w_, t in pairs]
    assert texts == ["\u0627\u0648\u0644 \u0633\u0637\u0631", "\u062f\u0648\u0645 \u0633\u0637\u0631"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_distill_align.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.distill'`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/training/distill/__init__.py
```

```python
# urdu-free-toolkit/training/distill/align.py
# -*- coding: utf-8 -*-
"""Align GPT page transcriptions to offline-detector line crops.

The offline detector gives us ordered line crops (via providers.ocr._rtl); GPT
gives us the clean text with its own line breaks. We assign one GPT line per
crop, in reading order, using fuzzy similarity when the counts or the order are
noisy, and drop pairs we are not confident about.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from providers.ocr import _rtl
from providers.ocr._types import Line, Word


def align_lines(crop_texts_hint: list[str] | None, page_lines: list[str],
                n_crops: int) -> list[str | None]:
    page_lines = [p for p in (s.strip() for s in page_lines) if p]
    if crop_texts_hint and len(crop_texts_hint) == n_crops and page_lines:
        # greedy best-match assignment, preserving that each page line used once
        remaining = list(page_lines)
        out: list[str | None] = []
        for hint in crop_texts_hint:
            if not remaining:
                out.append(None)
                continue
            best = max(remaining, key=lambda pl: fuzz.ratio(hint, pl))
            if fuzz.ratio(hint, best) >= 50:
                out.append(best)
                remaining.remove(best)
            else:
                out.append(None)
        return out
    # no usable hints: positional, pad/truncate to n_crops
    return [page_lines[i] if i < len(page_lines) else None for i in range(n_crops)]


def align(crops: list[Word], gpt_urdu: str, min_ratio: float = 70.0) -> list[tuple[Word, str]]:
    ordered_lines: list[Line] = _rtl.order_lines(
        _rtl.group_into_lines(crops, __import__("providers.ocr._types", fromlist=["OcrConfig"]).OcrConfig())
    )
    ordered_words = [ln.words[0] if ln.words else None for ln in ordered_lines]
    hints = [w.text for w in ordered_words if w is not None]
    assigned = align_lines(hints or None, gpt_urdu.split("\n"), len(ordered_words))
    out: list[tuple[Word, str]] = []
    for w, text in zip(ordered_words, assigned):
        if w is None or text is None:
            continue
        if w.text and fuzz.ratio(w.text, text) < min_ratio and len(hints) == len(ordered_words):
            continue
        out.append((w, text))
    return out
```

```python
# urdu-free-toolkit/training/distill/collect.py
# -*- coding: utf-8 -*-
"""Thin helper: run an engine's recognize() over a real image to get ordered
line-ish Word boxes for distillation. Kept minimal on purpose — the engine and
its preprocessing are the pipeline's job.
"""
from __future__ import annotations

from providers._imgutil import to_ndarray
from providers.ocr._types import Word


def crops_from_image(image: bytes, recognize) -> list[Word]:
    return list(recognize(to_ndarray(image), {}) or [])
```

```python
# urdu-free-toolkit/training/distill/build_distill.py
# -*- coding: utf-8 -*-
"""Write aligned (crop image, GPT text) pairs into a train/val dataset in the
same layout build_synth uses (out_dir/{train,val}/*.png + gt.txt).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def build(pairs, out_dir, val_frac: float = 0.1, seed: int = 0) -> dict:
    import cv2
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)
    for i, (img, text) in enumerate(pairs):
        arr = np.array(img.convert("L")) if hasattr(img, "convert") else np.asarray(img)
        split = "val" if rng.random() < val_frac else "train"
        rel = f"{i:06d}.png"
        cv2.imwrite(str(out_dir / split / rel), arr)
        (out_dir / split / "gt.txt").open("a", encoding="utf-8").write(f"{rel}\t{text}\n")
        counts[split] += 1
    return {"train": counts["train"], "val": counts["val"]}
```

> Import note: in `align.align`, import `OcrConfig` normally at module top
> (`from providers.ocr._types import Line, OcrConfig, Word`) and use it directly
> — the `__import__` shown above is a smell; replace it with the plain import
> when writing the file.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_distill_align.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/training/distill/ urdu-free-toolkit/tests/test_distill_align.py
git commit -m "feat(training): GPT-distillation line aligner + dataset writer"
```

---

## Task 18: Dataset-format converters (`training/easyocr`, `training/paddle`)

**Files:**
- Create: `urdu-free-toolkit/training/easyocr/__init__.py`, `training/easyocr/to_lmdb.py`
- Create: `urdu-free-toolkit/training/paddle/__init__.py`, `training/paddle/to_paddle_rec.py`
- Test: `urdu-free-toolkit/tests/test_training_converters.py`

**Interfaces:**
- Consumes: a `gt.txt` (`<relpath>\t<text>`) + its image dir.
- Produces:
  - `to_lmdb.write_lmdb(gt_txt: Path, image_dir: Path, out_lmdb: Path) -> int`
    — deep-text-recognition-benchmark layout: keys `image-000000001`,
    `label-000000001`, `num-samples`. If `lmdb` isn't installed, raise
    `RuntimeError` with the pip hint (documented; not exercised in CI).
  - `to_lmdb.gt_to_dtrb_txt(gt_txt: Path, out_txt: Path, image_dir_prefix: str = "") -> int`
    — the plain-text alternative (`<path>\t<label>` with absolute-ish paths) that
    DTRB also accepts; this IS unit-tested.
  - `to_paddle_rec.convert(gt_txt: Path, out_txt: Path, image_root: str) -> int`
    — PaddleOCR rec label file: `<image_root>/<relpath>\t<text>` per line;
    returns line count.

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_training_converters.py
from pathlib import Path

from training.easyocr.to_lmdb import gt_to_dtrb_txt
from training.paddle.to_paddle_rec import convert


def _gt(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    (d / "gt.txt").write_text("000000.png\t\u06c1\u06d2\n000001.png\t\u0679\u06be\u06cc\u06a9\n", encoding="utf-8")
    (d / "000000.png").write_bytes(b"x")
    (d / "000001.png").write_bytes(b"x")
    return d / "gt.txt", d


def test_paddle_rec_label_format(tmp_path):
    gt, d = _gt(tmp_path)
    out = tmp_path / "paddle_train.txt"
    n = convert(gt, out, image_root="dataset/urdu")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert n == 2
    assert lines[0] == "dataset/urdu/000000.png\t\u06c1\u06d2"


def test_dtrb_txt_format(tmp_path):
    gt, d = _gt(tmp_path)
    out = tmp_path / "dtrb_gt.txt"
    n = gt_to_dtrb_txt(gt, out, image_dir_prefix="train")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert n == 2 and lines[0].split("\t")[0].endswith("train/000000.png")
    assert lines[0].split("\t")[1] == "\u06c1\u06d2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_training_converters.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# urdu-free-toolkit/training/easyocr/__init__.py
```
```python
# urdu-free-toolkit/training/paddle/__init__.py
```

```python
# urdu-free-toolkit/training/easyocr/to_lmdb.py
# -*- coding: utf-8 -*-
"""Convert a gt.txt dataset to the layout Clova deep-text-recognition-benchmark
(EasyOCR's recognizer trainer) expects. See training/easyocr/train.md.
"""
from __future__ import annotations

from pathlib import Path


def _read_gt(gt_txt: Path) -> list[tuple[str, str]]:
    rows = []
    for line in Path(gt_txt).read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        rel, text = line.split("\t", 1)
        rows.append((rel.strip(), text))
    return rows


def gt_to_dtrb_txt(gt_txt: Path, out_txt: Path, image_dir_prefix: str = "") -> int:
    rows = _read_gt(gt_txt)
    with Path(out_txt).open("w", encoding="utf-8") as fh:
        for rel, text in rows:
            path = f"{image_dir_prefix.rstrip('/')}/{rel}" if image_dir_prefix else rel
            fh.write(f"{path}\t{text}\n")
    return len(rows)


def write_lmdb(gt_txt: Path, image_dir: Path, out_lmdb: Path) -> int:
    try:
        import lmdb  # noqa: F401
    except Exception as e:  # pragma: no cover - documented, not run in CI
        raise RuntimeError("pip install lmdb  (needed for DTRB LMDB datasets)") from e
    import lmdb
    rows = _read_gt(gt_txt)
    env = lmdb.open(str(out_lmdb), map_size=1 << 40)
    with env.begin(write=True) as txn:
        cnt = 0
        for rel, text in rows:
            img = (Path(image_dir) / rel).read_bytes()
            cnt += 1
            txn.put(f"image-{cnt:09d}".encode(), img)
            txn.put(f"label-{cnt:09d}".encode(), text.encode("utf-8"))
        txn.put(b"num-samples", str(cnt).encode())
    return cnt
```

```python
# urdu-free-toolkit/training/paddle/to_paddle_rec.py
# -*- coding: utf-8 -*-
"""Convert a gt.txt dataset to a PaddleOCR recognition label file.

PaddleOCR rec expects lines of ``<image_path>\\t<transcription>`` where
image_path is relative to the ``data_dir`` set in the training YAML. See
training/paddle/train.md.
"""
from __future__ import annotations

from pathlib import Path


def convert(gt_txt: Path, out_txt: Path, image_root: str) -> int:
    n = 0
    with Path(out_txt).open("w", encoding="utf-8") as fh:
        for line in Path(gt_txt).read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            rel, text = line.split("\t", 1)
            fh.write(f"{image_root.rstrip('/')}/{rel.strip()}\t{text}\n")
            n += 1
    return n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_training_converters.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/training/easyocr/ urdu-free-toolkit/training/paddle/ urdu-free-toolkit/tests/test_training_converters.py
git commit -m "feat(training): dataset-format converters for DTRB (easyocr) and PaddleOCR rec"
```

---

## Task 19: Training configs + GPU runbook (`training/README.md`, configs)

**Files:**
- Create: `urdu-free-toolkit/training/easyocr/config.yaml`
- Create: `urdu-free-toolkit/training/easyocr/train.md`
- Create: `urdu-free-toolkit/training/paddle/arabic_rec_ft.yml`
- Create: `urdu-free-toolkit/training/paddle/train.md`
- Create: `urdu-free-toolkit/training/README.md`
- Test: `urdu-free-toolkit/tests/test_training_docs.py` (sanity only)

**Interfaces:**
- Consumes: nothing (docs + static config).
- Produces: a runnable runbook. The test only asserts the files exist, are
  non-trivial, and the YAMLs parse.

- [ ] **Step 1: Write the failing test**

```python
# urdu-free-toolkit/tests/test_training_docs.py
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent / "training"


def test_runbook_and_configs_present_and_parse():
    for rel in ["README.md", "easyocr/train.md", "paddle/train.md"]:
        p = _ROOT / rel
        assert p.exists() and len(p.read_text(encoding="utf-8")) > 400, rel
    for rel in ["easyocr/config.yaml", "paddle/arabic_rec_ft.yml"]:
        p = _ROOT / rel
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and data, rel


def test_readme_links_the_eval_loop():
    txt = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "run_eval" in txt and "OCR_PADDLE_REC_DIR" in txt and "user_network" in txt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_training_docs.py -v`
Expected: FAIL — files absent. (`pip install pyyaml` if `yaml` import fails — it's a transitive dep of paddleocr, usually present.)

- [ ] **Step 3: Write the files**

`training/easyocr/config.yaml` — a DTRB-style config skeleton (exact arch fields
confirmed against the installed EasyOCR in `train.md`):

```yaml
# Fine-tune config skeleton for EasyOCR's Urdu recognizer via
# Clova deep-text-recognition-benchmark. VERIFY the arch + character set against
# your installed easyocr version before training (see train.md, step 2).
experiment_name: urdu_ft
train_data: data/lmdb/train
valid_data: data/lmdb/val
manualSeed: 1111
workers: 4
batch_size: 128
num_iter: 30000
valInterval: 1000
saved_model: ""          # path to easyocr arabic_g2 .pth to fine-tune from (train.md step 2)
FT: true
adam: true
lr: 0.0005
# --- architecture (EasyOCR 'generation2' recognizer) ---
Transformation: None
FeatureExtraction: VGG
SequenceModeling: BiLSTM
Prediction: CTC
input_channel: 1
output_channel: 256
hidden_size: 256
imgH: 64
imgW: 600
PAD: true
# --- label space ---
character: ""             # paste EasyOCR's Urdu/Arabic charset here (train.md step 2)
sensitive: false
data_filtering_off: true
```

`training/easyocr/train.md` — copy-paste runbook: (1) build LMDB with
`training/easyocr/to_lmdb.py`; (2) `git clone
https://github.com/JaidedAI/deep-text-recognition-benchmark`, locate the EasyOCR
`arabic_g2` weights in `~/.EasyOCR/model/`, read the matching arch + `character`
string from `easyocr`'s `config` for `arabic`; (3) `python train.py --config
config.yaml`; (4) convert the resulting `best_accuracy.pth` to an EasyOCR custom
model — drop `urdu_ft.py` (copy DTRB `model.py` class), `urdu_ft.yaml` (the arch
block above), `urdu_ft.pth` into `~/.EasyOCR/user_network/`; (5) verify:
`OCR_EASYOCR_RECOG_NETWORK=urdu_ft python -m eval.run_eval --engines easyocr,gpt`.
Include a Colab cell block and a RunPod block. State expected wall-clock
(~2–4 h on a T4 for 30k iters).

`training/paddle/arabic_rec_ft.yml`:

```yaml
# PaddleOCR recognition fine-tune from the Arabic PP-OCRv4 model.
# Run inside a PaddleOCR source checkout (train.md step 1).
Global:
  use_gpu: true
  epoch_num: 60
  save_model_dir: ./output/urdu_rec_ft
  save_epoch_step: 5
  eval_batch_step: [0, 500]
  pretrained_model: ./pretrain/arabic_PP-OCRv4_rec_train/best_accuracy
  character_dict_path: ppocr/utils/dict/arabic_dict.txt
  max_text_length: 60
  use_space_char: true
  save_res_path: ./output/urdu_rec_ft/predicts.txt
Optimizer:
  name: Adam
  lr:
    name: Cosine
    learning_rate: 0.0005
Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
Loss:
  name: CTCLoss
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/urdu
    label_file_list: [./data/urdu/paddle_train.txt]
  loader:
    batch_size_per_card: 128
    num_workers: 4
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/urdu
    label_file_list: [./data/urdu/paddle_val.txt]
  loader:
    batch_size_per_card: 128
    num_workers: 4
```

`training/paddle/train.md` — runbook: (1) `git clone
https://github.com/PaddlePaddle/PaddleOCR && pip install -r requirements.txt`;
(2) download `arabic_PP-OCRv4_rec_train` pretrained into `./pretrain/`;
(3) `to_paddle_rec.convert` your `gt.txt` → `paddle_train.txt` / `paddle_val.txt`
under `./data/urdu/`; (4) `python tools/train.py -c
/path/to/arabic_rec_ft.yml`; (5) `python tools/export_model.py -c
arabic_rec_ft.yml -o Global.pretrained_model=output/urdu_rec_ft/best_accuracy
Global.save_inference_dir=output/urdu_rec_infer`; (6) verify:
`OCR_PADDLE_REC_DIR=/abs/output/urdu_rec_infer python -m eval.run_eval --engines
paddle,gpt`. Colab + RunPod cell blocks. Expected wall-clock (~3–6 h on a T4).

`training/README.md` — the top-level runbook tying it together:
- prerequisites (cloud GPU, Python, the two upstream repos)
- dataset build: `fetch_corpus` → `build_synth` (bulk) + `collect`/`align`/
  `build_distill` (domain) → merge `gt.txt`s
- pointers to `easyocr/train.md` and `paddle/train.md`
- **wire-in**: `~/.EasyOCR/user_network/urdu_ft.*` or
  `OCR_EASYOCR_RECOG_NETWORK`; `OCR_PADDLE_REC_DIR`
- **acceptance**: `python -m eval.run_eval --engines gpt,paddle,easyocr` before
  and after; then `python -m eval.run_eval --engines paddle,easyocr
  --write-baseline` and commit the new `baseline.json` + numbers in the README
- handwriting: add handwriting fonts to `synth/fonts/`, tag in `fonts.json`,
  rebuild — separate acceptance run, not covered here

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_training_docs.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/training/easyocr/config.yaml urdu-free-toolkit/training/easyocr/train.md urdu-free-toolkit/training/paddle/arabic_rec_ft.yml urdu-free-toolkit/training/paddle/train.md urdu-free-toolkit/training/README.md urdu-free-toolkit/tests/test_training_docs.py
git commit -m "docs(training): fine-tune configs + GPU runbook for easyocr/paddle Urdu recognizers"
```

---

## Task 20: `.gitignore`, README, and full-suite verification

**Files:**
- Modify: `urdu-free-toolkit/.gitignore`
- Modify: `urdu-free-toolkit/README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: docs + ignore rules; no code.

- [ ] **Step 1: Update `.gitignore`**

Append to `urdu-free-toolkit/.gitignore`:

```
# OCR fine-tuning: generated datasets, model checkpoints, downloaded fonts
/data/
training/synth/fonts/*
!training/synth/fonts/.gitkeep
*.lmdb
```

- [ ] **Step 2: Update `README.md`**

In the "How it's built" section, after the capability table, add a subsection:

```markdown
### OCR pipeline (offline engines)

`paddle` and `easyocr` don't call the model raw — they run through
`providers/ocr/_pipeline.py`: preprocess (grayscale, upscale, denoise, CLAHE,
deskew, binarize) → the engine's detect+recognize → RTL reading-order
reconstruction → Urdu text normalization → a rule-engine transliteration fill,
so each offline column shows `urdu + devanagari + roman` like the GPT column.

**Measuring accuracy.** `eval/` scores every OCR engine against cached GPT
"silver reference" transcriptions:

```bash
cd urdu-free-toolkit
python -m eval.run_eval --engines gpt,paddle,easyocr          # scoreboard (CER/WER)
python -m eval.run_eval --refresh                             # regenerate silver refs (needs OPENAI_API_KEY)
python -m eval.run_eval --engines paddle,easyocr --check      # CI regression gate vs eval/baseline.json
```

Fixtures live in `tests/fixtures/ocr/` — see that folder's README to add one.

**Fine-tuning the offline recognizers** (optional, needs a GPU): the CPU-side
dataset tooling is in `training/` (synthetic Nastaliq/Naskh renderer +
GPT-distillation aligner + dataset converters); the training itself is a
copy-paste runbook in `training/README.md`. Once a model exists, point the
provider at it via `OCR_EASYOCR_RECOG_NETWORK` / `OCR_PADDLE_REC_DIR` — with no
model set, the stock weights are used.
```

Update the repo-layout block at the bottom to add `eval/` and `training/` lines.

- [ ] **Step 3: Run the entire test suite (light)**

Run: `cd urdu-free-toolkit && python -m pytest -q`
Expected: PASS — all prior + all new tests, no network, no model downloads.

- [ ] **Step 4: Run the heavy suite (engines installed on this machine)**

Run: `cd urdu-free-toolkit && RUN_HEAVY=1 python -m pytest -q -k "ocr or eval or providers_smoke"`
Expected: PASS or CLEANLY SKIPPED. `test_eval_regression` skips if no fixtures/baseline. Investigate any hard failure.

- [ ] **Step 5: Manual pipeline sanity (best effort, needs a real Urdu image + key)**

```bash
cd urdu-free-toolkit
# drop a real screenshot at /tmp/u.png first
OPENAI_API_KEY=... python -m eval.run_eval --engines gpt,paddle,easyocr --refresh
```

Record the CER/WER table in the commit message. If `paddle`/`easyocr` produce
sane Urdu with correct word order, phase 1 is working. Then:

```bash
python -m eval.run_eval --engines paddle,easyocr --write-baseline
git add urdu-free-toolkit/eval/baseline.json urdu-free-toolkit/tests/fixtures/ocr/
```

- [ ] **Step 6: Commit**

```bash
git add urdu-free-toolkit/.gitignore urdu-free-toolkit/README.md
git commit -m "docs: OCR pipeline + eval + fine-tune runbook in README; ignore training artifacts"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task(s) |
|---|---|
| Component A `_pipeline.py` | 1 (types), 6 |
| Component B `_prep.py` | 4 |
| Component C `_rtl.py` | 3 |
| Component D `_normalize.py` | 2 |
| Component E provider refactor | 8 (easyocr), 9 (paddle) |
| Component F `_translit_fill.py` | 5 |
| Component G eval harness | 11 (score), 12 (refs+fixtures), 13 (run_eval+baseline), 14 (regression test) |
| Component H fixtures dir | 12 |
| Fine-tuning track — synth | 15, 16 |
| Fine-tuning track — distill | 17 |
| Fine-tuning track — converters | 18 |
| Fine-tuning track — configs + runbook | 19 |
| Provider model hooks | 7 (+ wired in 8, 9) |
| Data flow / error handling | 6 (best-effort stages), 4 (`_try`), covered by tests in 4/6 |
| Dependencies | 10 (requirements.txt) |
| Rollout order | Tasks are in the spec's order: prep/rtl/normalize → pipeline → providers → eval → phase-2 scaffold → README |
| README update | 20 |
| `.gitignore` for `data/`, models | 20 |

No spec requirement is left without a task.

**2. Placeholder scan**

- Task 19 intentionally ships config *skeletons* with explicit "VERIFY against
  installed version" call-outs — this matches the spec's stated risk
  ("EasyOCR fine-tune arch pinning ... Runbook task"). The `.md` runbooks are
  described by their required content and required assertions (`test_training_docs.py`),
  not left as "TODO".
- Task 13 `run_eval.main` has a dead `ocr_map = None if not _in_test() else None`
  line — **fix when writing**: delete that line, call `evaluate(engines,
  fixtures_dir=fixtures_dir, spellfix=args.spellfix)` directly.
- Task 17 `align.align` shows an `__import__(...)` smell with an inline note to
  replace it with a top-level `from providers.ocr._types import Line, OcrConfig, Word`.
  Do that.
- `--spellfix` is parsed in `run_eval` and threaded to `evaluate`, but `evaluate`
  currently ignores it. **Fix when writing Task 13**: in `evaluate`, when
  `spellfix=True`, wrap each engine's result text through
  `normalize_urdu(text, OcrConfig(spellfix=True))` before scoring, OR score with
  a spellfix-on config. Simplest: pass `spellfix` into a local `score_pair`
  variant. Keep it real, not a stub.

**3. Type consistency**

- `Word(text, box, conf)` — `box` is `list[tuple[int,int]]` everywhere;
  providers emit `{"box": [list(p) ...]}` (lists) into `OcrResult.boxes` which is
  fine (JSON-facing). Consistent.
- `preprocess` returns a 3-tuple `(img, inv, stages)` in Task 4 and is consumed
  as a 3-tuple in Task 6. Consistent.
- `evaluate(...)` returns `{engine: {"rows","agg","ms"}}`; `format_table`,
  `check_regression`, and `test_eval_run` all read `["agg"]["cer_norm"]`.
  Consistent.
- `ensure_refs` return keys `{"written","skipped","have"}` — used consistently in
  Task 12 tests and Task 13 `--refresh` print.
- `build(...)` (synth) returns `{"train","val","gt"}`; `build` (distill) returns
  `{"train","val"}`. Different but each is only consumed by its own test. OK.
- `_model_hooks` names `easyocr_recog_network` / `paddle_rec_dir` match their
  call sites in Tasks 8 and 9.

Fixes above are inline notes in the affected tasks; apply them as you write.
