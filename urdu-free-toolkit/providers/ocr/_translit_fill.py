# -*- coding: utf-8 -*-
"""Fill an OCR result's script fields via the default offline neural hybrid.

The hybrid includes context rules, the exact dictionary, the full lexicon and
the trained model (with a rule fallback when model assets are unavailable).
Best effort: any failure yields empty strings, never an exception.
"""
from __future__ import annotations

import neural_translit as _neural


def rule_translit(urdu: str) -> tuple[str, str]:
    """Backward-compatible name for the default offline transliterator."""
    if not urdu or not urdu.strip():
        return "", ""
    try:
        deva, roman, _roman_diacritic = _neural.transliterate(urdu)
        return deva or "", roman or ""
    except Exception:
        return "", ""
