# -*- coding: utf-8 -*-
"""Render a text line to a tight image for synthetic OCR training data.

Requires Pillow built with libraqm for correct Arabic shaping; ``render_line``
asserts this so failures are loud, not silently wrong.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

_FONTS_DIR = Path(__file__).resolve().parent / "fonts"


def find_fonts(fonts_dir: Path | None = None) -> list[str]:
    d = Path(fonts_dir) if fonts_dir else _FONTS_DIR
    return [str(p) for p in sorted(d.glob("*.ttf")) + sorted(d.glob("*.otf"))]


def render_line(text: str, font_path: str, size: int = 48, pad: int = 8) -> Image.Image:
    if not features.check("raqm"):
        raise RuntimeError(
            "Pillow lacks libraqm; Arabic shaping would be wrong. Install libraqm "
            "(conda-forge) or run under WSL. See training/README.md."
        )
    font = ImageFont.truetype(font_path, size)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    l, t, r, b = probe.textbbox((0, 0), text, font=font, language="ur")
    w, h = (r - l) + 2 * pad, (b - t) + 2 * pad
    img = Image.new("RGB", (max(w, 1), max(h, 1)), "white")
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill="black", language="ur")
    return img
