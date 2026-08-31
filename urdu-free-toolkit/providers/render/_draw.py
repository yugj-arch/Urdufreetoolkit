# -*- coding: utf-8 -*-
"""Shared helpers for the offline render providers: find a usable TrueType font
for a given script, and box detection via EasyOCR."""
from __future__ import annotations

import re

from PIL import ImageFont

_DEVA_RE = re.compile(r"[\u0900-\u097F]")

_FONT_CANDIDATES = {
    "devanagari": [
        "assets/fonts/NotoSansDevanagari-Regular.ttf",
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/mangal.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    ],
    "roman": [
        "assets/fonts/NotoSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
}


def script_of(line: str) -> str:
    return "devanagari" if _DEVA_RE.search(line or "") else "roman"


def load_font(script: str, size: int):
    import os

    for path in _FONT_CANDIDATES.get(script, []):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def detect_line_boxes(pil):
    """Return [(x0, y0, x1, y1), ...] top-to-bottom via EasyOCR. Empty if EasyOCR
    is unavailable or finds nothing."""
    try:
        import numpy as np

        from providers.ocr.easyocr_p import _get_reader, easyocr
    except Exception:  # noqa: BLE001
        return []
    if easyocr is None:
        return []
    found = _get_reader().readtext(np.array(pil), detail=1, paragraph=False)
    rects = []
    for box, _txt, _conf in found:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        rects.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    rects.sort(key=lambda r: r[1])
    return rects
