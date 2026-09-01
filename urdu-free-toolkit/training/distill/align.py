# -*- coding: utf-8 -*-
"""Align GPT page transcriptions to offline-detector line crops.

The offline detector gives us ordered line crops (via ``providers.ocr._rtl``);
GPT gives us the clean text with its own line breaks. We assign one GPT line
per crop, in reading order, using fuzzy similarity when the counts or the order
are noisy, and drop pairs we are not confident about.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from providers.ocr import _rtl
from providers.ocr._types import Line, OcrConfig, Word


def align_lines(crop_texts_hint: list[str] | None, page_lines: list[str],
                n_crops: int) -> list[str | None]:
    page_lines = [p for p in (s.strip() for s in page_lines) if p]
    if crop_texts_hint and len(crop_texts_hint) == n_crops and page_lines:
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
    return [page_lines[i] if i < len(page_lines) else None for i in range(n_crops)]


def align(crops: list[Word], gpt_urdu: str, min_ratio: float = 70.0) -> list[tuple[Word, str]]:
    ordered_lines: list[Line] = _rtl.order_lines(_rtl.group_into_lines(crops, OcrConfig()))
    ordered_words = [ln.words[0] if ln.words else None for ln in ordered_lines]
    hints = [w.text for w in ordered_words if w is not None]
    assigned = align_lines(hints or None, gpt_urdu.split("\n"), len(ordered_words))
    out: list[tuple[Word, str]] = []
    for w, text in zip(ordered_words, assigned):
        if w is None or text is None:
            continue
        if w.text and len(hints) == len(ordered_words) and fuzz.ratio(w.text, text) < min_ratio:
            continue
        out.append((w, text))
    return out
