# -*- coding: utf-8 -*-
"""Offline in-place redraw with Pillow: detect the Urdu line boxes (via EasyOCR),
cover each with a rectangle sampled from its surroundings, and draw the matching
transliteration line on top, shrunk to fit. Deterministic and fast; the erase is
a flat fill, so it looks best on plain backgrounds.
"""
from __future__ import annotations

import io

from PIL import ImageDraw

from providers._imgutil import to_pil
from providers.base import BaseProvider, Capability, ProviderInfo, RenderResult
from providers.render._draw import detect_line_boxes, load_font, script_of


def _bg_sample(img, rect):
    x0, y0, x1, y1 = rect
    w, h = img.size
    pts = [(min(x1 + 3, w - 1), (y0 + y1) // 2), (max(x0 - 3, 0), (y0 + y1) // 2),
           ((x0 + x1) // 2, max(y0 - 3, 0)), ((x0 + x1) // 2, min(y1 + 3, h - 1))]
    px = img.load()
    cols = [px[p] for p in pts]
    return tuple(sum(c[i] for c in cols) // len(cols) for i in range(3))


def _fit_font(draw, text, script, max_w, max_h):
    size = max(10, int(max_h * 0.9))
    while size > 8:
        font = load_font(script, size)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        if (r - l) <= max_w and (b - t) <= max_h:
            return font
        size -= 2
    return load_font(script, 8)


class PilBoxRender(BaseProvider):
    info = ProviderInfo(
        id="pil_box",
        label="Pillow box-cover (offline)",
        capability=Capability.RENDER,
        kind="offline",
        note="Detect boxes (EasyOCR), flat-fill, draw text. Best on plain backgrounds.",
    )

    def available(self) -> tuple[bool, str]:
        try:
            from providers.ocr.easyocr_p import easyocr
        except Exception:  # noqa: BLE001
            return (False, "needs EasyOCR for box detection")
        return (True, "") if easyocr is not None else (False, "pip install easyocr")

    def render(self, image: bytes, lines: list[str]) -> RenderResult:
        def _call():
            img = to_pil(image)
            draw = ImageDraw.Draw(img)
            rects = detect_line_boxes(img)
            if not rects:
                raise RuntimeError("no text lines detected to replace")
            for rect, line in zip(rects, lines):
                x0, y0, x1, y1 = rect
                draw.rectangle(rect, fill=_bg_sample(img, rect))
                script = script_of(line)
                font = _fit_font(draw, line, script, x1 - x0, y1 - y0)
                lb, tb, rb, bb = draw.textbbox((0, 0), line, font=font)
                tx = x0 + max(0, ((x1 - x0) - (rb - lb)) // 2)
                ty = y0 + max(0, ((y1 - y0) - (bb - tb)) // 2) - tb
                draw.text((tx, ty), line, fill=(20, 20, 20), font=font)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        t = self._timed(_call)
        if not t["ok"]:
            return RenderResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        return RenderResult(provider_id=self.info.id, ok=True, ms=t["ms"], png=t["value"])


PROVIDER = PilBoxRender()
