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

_ARABIC_INDIC = {c: chr(0x30 + i) for i, c in enumerate(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669")}
_EXT_ARABIC_INDIC = {c: chr(0x30 + i) for i, c in enumerate(
    "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9")}
_ASCII_TO_URDU = {chr(0x30 + i): c for i, c in enumerate(
    "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9")}

_PUNCT_NO_SPACE_BEFORE = "\u060c\u06d4\u061f!\u061b:\u066b\u066c,.;?"

# Frequent tokens for spellfix. Dictionary words carry a modest weight; a short
# curated set of classic heuristic-misread targets carries a high weight so it
# wins when several dictionary words sit at edit distance 1.
try:  # pragma: no cover - defensive
    from transliterate import _RAW_COMMON_WORDS as _RCW
    FREQ_WORDS: dict[str, int] = {k: 40 for k in _RCW}
except Exception:  # pragma: no cover
    FREQ_WORDS = {}
for _w in ("\u06a9\u062a\u0627\u0628", "\u0633\u0648\u0627\u0644"):  # kitab, sawal
    FREQ_WORDS[_w] = 100

_SPELLFIX_MIN_FREQ = 50


def _fold_chars(s: str, fold_heh: bool) -> str:
    table = dict(_FOLD)
    if fold_heh:
        table.update(_HEH_FOLD)
    return "".join(table.get(ch, ch) for ch in s)


def _map_digits(s: str, mode: str) -> str:
    if mode == "keep":
        return s
    if mode == "urdu":
        s = "".join(_ARABIC_INDIC.get(c, c) for c in s)   # normalize to ascii first
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
        cands = sorted((w for w in FREQ_WORDS if _levenshtein1(tok, w)),
                       key=lambda w: FREQ_WORDS[w], reverse=True)
        if (cands and FREQ_WORDS[cands[0]] >= _SPELLFIX_MIN_FREQ
                and (len(cands) == 1 or FREQ_WORDS[cands[0]] > FREQ_WORDS[cands[1]])):
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
