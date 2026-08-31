# -*- coding: utf-8 -*-
"""RapidOCR (ONNX) as an OCR provider (offline).

Fast and torch-free. The bundled models are Latin/CJK, so out of the box Urdu
accuracy is low — point ``RAPIDOCR_ARABIC=1`` at Arabic-script ONNX models to
improve it (a later-phase enhancement). Kept here so it shows in the compare
view and works immediately for any Latin text mixed into the image.
"""
from __future__ import annotations

from providers._imgutil import to_ndarray
from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # noqa: BLE001
    RapidOCR = None

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


class RapidOcr(BaseProvider):
    info = ProviderInfo(
        id="rapidocr",
        label="RapidOCR (offline)",
        capability=Capability.OCR,
        kind="offline",
        note="ONNX, no PyTorch. Bundled models are Latin/CJK — low Urdu accuracy by default.",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if RapidOCR is not None else (False, "pip install rapidocr-onnxruntime")

    def ocr(self, image: bytes) -> OcrResult:
        if RapidOCR is None:
            return OcrResult(provider_id=self.info.id, ok=False,
                             error="rapidocr-onnxruntime not installed")

        def _call():
            arr = to_ndarray(image)
            result, _elapse = _get_engine()(arr)
            if not result:
                return "", []
            boxes = [
                {"box": [[int(x), int(y)] for x, y in box], "text": txt, "conf": float(score)}
                for box, txt, score in result
            ]
            return "\n".join(r[1] for r in result), boxes

        t = self._timed(_call)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        text, boxes = t["value"]
        return OcrResult(provider_id=self.info.id, ok=True, ms=t["ms"], text=text, boxes=boxes)


PROVIDER = RapidOcr()
