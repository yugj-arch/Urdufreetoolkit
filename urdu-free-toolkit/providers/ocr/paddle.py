# -*- coding: utf-8 -*-
"""PaddleOCR as an OCR provider (offline).

Uses the Arabic-script model, which covers the Urdu alphabet and does well on
printed Naskh. PaddleOCR's Python API has changed a lot between versions, so the
call is written defensively (``predict`` then ``ocr``) and any mismatch surfaces
as a failed result row rather than a crash. First run downloads ~20 MB of models.
"""
from __future__ import annotations

from providers._imgutil import to_pil
from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo

try:
    from paddleocr import PaddleOCR
except Exception:  # noqa: BLE001
    PaddleOCR = None

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        try:
            _engine = PaddleOCR(lang="arabic", use_textline_orientation=True)
        except TypeError:
            _engine = PaddleOCR(lang="arabic", use_angle_cls=True)
    return _engine


def _extract(result) -> tuple[str, list]:
    """Normalize the several shapes PaddleOCR .ocr()/.predict() can return."""
    lines: list[tuple] = []
    if isinstance(result, list) and result and isinstance(result[0], dict):
        d = result[0]
        texts = d.get("rec_texts") or d.get("rec_text") or []
        polys = d.get("rec_polys") or d.get("dt_polys") or [None] * len(texts)
        scores = d.get("rec_scores") or [0.0] * len(texts)
        lines = list(zip(polys, texts, scores))
    elif isinstance(result, list) and result and isinstance(result[0], list):
        for entry in result[0]:
            try:
                box, (txt, score) = entry
                lines.append((box, txt, score))
            except Exception:  # noqa: BLE001
                continue
    boxes = []
    texts = []
    for box, txt, score in lines:
        texts.append(txt)
        if box is not None:
            pts = [[int(p[0]), int(p[1])] for p in box]
            boxes.append({"box": pts, "text": txt, "conf": float(score or 0.0)})
    return "\n".join(texts), boxes


class PaddleOcr(BaseProvider):
    info = ProviderInfo(
        id="paddle",
        label="PaddleOCR (offline)",
        capability=Capability.OCR,
        kind="offline",
        note="Arabic-script model; good on printed Naskh. First run downloads ~20 MB.",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if PaddleOCR is not None else (False, "pip install paddleocr paddlepaddle")

    def ocr(self, image: bytes) -> OcrResult:
        if PaddleOCR is None:
            return OcrResult(provider_id=self.info.id, ok=False, error="paddleocr not installed")

        def _call():
            import numpy as np

            arr = np.array(to_pil(image))
            engine = _get_engine()
            try:
                result = engine.predict(arr)
            except (AttributeError, TypeError):
                result = engine.ocr(arr)
            return _extract(result)

        t = self._timed(_call)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        text, boxes = t["value"]
        return OcrResult(provider_id=self.info.id, ok=True, ms=t["ms"], text=text, boxes=boxes)


PROVIDER = PaddleOcr()
