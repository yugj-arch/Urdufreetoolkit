# -*- coding: utf-8 -*-
"""Reconstruct human reading order from detector boxes for RTL Urdu.

Detectors emit boxes in an arbitrary order. We split words into columns on wide
vertical corridors (right column first, Urdu-style), bucket each column's words
into lines by vertical position, order words right-to-left within a line, and
order lines top-to-bottom. Inline left-to-right runs (Latin words, acronyms,
multi-digit numbers) are kept in their natural left-to-right order.
"""
from __future__ import annotations

import logging
import math
import statistics

from providers.ocr._types import Line, OcrConfig, Word

log = logging.getLogger(__name__)


def _cx(w: Word) -> float:
    return sum(p[0] for p in w.box) / len(w.box)


def _cy(w: Word) -> float:
    return sum(p[1] for p in w.box) / len(w.box)


def _height(w: Word) -> float:
    ys = [p[1] for p in w.box]
    return (max(ys) - min(ys)) or 1.0


def _width(w: Word) -> float:
    xs = [p[0] for p in w.box]
    return (max(xs) - min(xs)) or 1.0


def _union_box(words: list[Word]) -> list[tuple[int, int]]:
    xs = [p[0] for w in words for p in w.box]
    ys = [p[1] for w in words for p in w.box]
    return [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]


def _is_ltr_token(t: str) -> bool:
    """A Latin/number run worth keeping left-to-right. Single characters are
    left in RTL context (too ambiguous, and Urdu-word stand-ins in tests)."""
    if len(t) < 2:
        return False
    if any("؀" <= c <= "ۿ" for c in t):
        return False
    return any(c.isascii() and c.isalnum() for c in t)


def _line_text(words_rtl: list[Word]) -> str:
    """words_rtl is in reading order (rightmost first). Reverse each maximal run
    of LTR tokens so it reads left-to-right."""
    toks = [w.text for w in words_rtl]
    out: list[str] = []
    i = 0
    while i < len(toks):
        if _is_ltr_token(toks[i]):
            j = i
            while j < len(toks) and _is_ltr_token(toks[j]):
                j += 1
            out.extend(reversed(toks[i:j]))
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def _column_cuts(words: list[Word]) -> list[float]:
    """x positions separating columns, or [] for a single column."""
    if len(words) < 4:
        return []
    levels = {round(_cy(w) / 5) for w in words}
    if len(levels) < 2:
        return []
    mww = statistics.median(_width(w) for w in words) or 1.0
    thresh = max(60.0, 8.0 * mww)
    spans = sorted((min(p[0] for p in w.box), max(p[0] for p in w.box)) for w in words)
    cuts: list[float] = []
    cur_end = spans[0][1]
    for a, b in spans[1:]:
        if a - cur_end > thresh:
            cuts.append((cur_end + a) / 2)
        cur_end = max(cur_end, b)
    return cuts


def _lines_in_block(words: list[Word], cfg: OcrConfig) -> list[Line]:
    med_h = statistics.median(_height(w) for w in words)
    tol = cfg.line_y_tol_frac * med_h
    buckets: list[list[Word]] = []
    for w in sorted(words, key=_cy):
        for b in buckets:
            if abs(_cy(w) - statistics.mean(_cy(x) for x in b)) <= tol:
                b.append(w)
                break
        else:
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


def group_into_lines(words: list[Word], cfg: OcrConfig) -> list[Line]:
    if not words:
        return []
    try:
        cuts = _column_cuts(words)
        if not cuts:
            return _lines_in_block(list(words), cfg)
        bounds = [-math.inf, *cuts, math.inf]
        cols: list[list[Word]] = []
        for lo, hi in zip(bounds, bounds[1:]):
            grp = [w for w in words if lo <= _cx(w) < hi]
            if grp:
                cols.append(grp)
        cols.sort(key=lambda g: -statistics.mean(_cx(w) for w in g))   # right column first
        out: list[Line] = []
        for g in cols:
            out.extend(_lines_in_block(g, cfg))
        return out
    except Exception:  # pragma: no cover - best effort
        log.warning("group_into_lines failed; detector order", exc_info=True)
        return [Line(words=list(words), text=" ".join(w.text for w in words),
                     box=_union_box(list(words)), conf=0.0)]


def order_lines(lines: list[Line]) -> list[Line]:
    """group_into_lines already returns reading order (columns + top-to-bottom).
    Kept for API symmetry and as the place to hang future page-level logic."""
    return list(lines)


def lines_to_text(lines: list[Line]) -> str:
    return "\n".join(ln.text for ln in lines)
