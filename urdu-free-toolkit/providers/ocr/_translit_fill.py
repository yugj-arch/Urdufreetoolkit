# -*- coding: utf-8 -*-
"""Fill an OCR result's devanagari/roman fields via the offline rule engine, so
the offline OCR columns carry the same shape as the GPT column. Best effort:
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
