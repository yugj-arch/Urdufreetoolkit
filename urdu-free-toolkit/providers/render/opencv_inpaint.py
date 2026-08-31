# -*- coding: utf-8 -*-
"""Offline in-place redraw with OpenCV inpainting: mask the Urdu line boxes,
``cv2.inpaint`` to erase them into the background, then draw the transliteration
on top with Pillow. Cleaner erase than a flat fill on textured backgrounds.
"""
from __future__ import annotations

import io

from PIL import Image, ImageDraw

from providers._imgutil import to_pil
from providers.base import BaseProvider, Capability, ProviderInfo, RenderResult
from providers.render._draw import detect_line_boxes, load_font, script_of


class OpencvInpaintRender(BaseProvider):
    info = ProviderInfo(
        id="opencv_inpaint",
        label="OpenCV inpaint (offline)",
        capability=Capability.RENDER,
        kind="offline",
        note="Mask boxes, cv2.inpaint erase, draw text. Better on textured backgrounds.",
    )

    def available(self) -> tuple[bool, str]:
        try:
            import cv2  # noqa: F401
        except Exception:  # noqa: BLE001
            return (False, "pip install opencv-python-headless")
        try:
            from providers.ocr.easyocr_p import easyocr
        except Exception:  # noqa: BLE001
            return (False, "needs EasyOCR for box detection")
        return (True, "") if easyocr is not None else (False, "pip install easyocr")

    def render(self, image: bytes, lines: list[str]) -> RenderResult:
        def _call():
            import cv2
            import numpy as np

            pil = to_pil(image)
            rects = detect_line_boxes(pil)
            if not rects:
                raise RuntimeError("no text lines detected to replace")
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            mask = np.zeros(bgr.shape[:2], np.uint8)
            for x0, y0, x1, y1 in rects:
                cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
            erased = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
            out = Image.fromarray(cv2.cvtColor(erased, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(out)
            for (x0, y0, x1, y1), line in zip(rects, lines):
                script = script_of(line)
                size = max(10, int((y1 - y0) * 0.9))
                font = load_font(script, size)
                lb, tb, rb, bb = draw.textbbox((0, 0), line, font=font)
                while (rb - lb) > (x1 - x0) and size > 8:
                    size -= 2
                    font = load_font(script, size)
                    lb, tb, rb, bb = draw.textbbox((0, 0), line, font=font)
                draw.text((x0, y0 - tb), line, fill=(20, 20, 20), font=font)
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            return buf.getvalue()

        t = self._timed(_call)
        if not t["ok"]:
            return RenderResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        return RenderResult(provider_id=self.info.id, ok=True, ms=t["ms"], png=t["value"])


PROVIDER = OpencvInpaintRender()
