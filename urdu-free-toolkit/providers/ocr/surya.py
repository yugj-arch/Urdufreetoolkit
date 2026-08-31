# -*- coding: utf-8 -*-
"""Surya as an OCR provider (offline).

Transformer detection + recognition with native Urdu support — the strongest
free engine on Nastaliq. The first call downloads the Surya models (~1 GB) from
the Hugging Face hub into the local cache; after that it runs offline on CPU.
"""
from __future__ import annotations

from providers._imgutil import to_pil
from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo

try:
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor
except Exception:  # noqa: BLE001
    RecognitionPredictor = None
    DetectionPredictor = None

_rec = None
_det = None


def _predictors():
    global _rec, _det
    if _rec is None:
        _det = DetectionPredictor()
        _rec = RecognitionPredictor()
    return _rec, _det


class SuryaOcr(BaseProvider):
    info = ProviderInfo(
        id="surya",
        label="Surya (offline)",
        capability=Capability.OCR,
        kind="offline",
        note="Transformer OCR, native Urdu. Best free on Nastaliq. First run downloads ~1 GB.",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if RecognitionPredictor is not None else (False, "pip install surya-ocr")

    def ocr(self, image: bytes) -> OcrResult:
        if RecognitionPredictor is None:
            return OcrResult(provider_id=self.info.id, ok=False, error="surya-ocr not installed")

        def _call():
            pil = to_pil(image)
            rec, det = _predictors()
            preds = rec([pil], det_predictor=det)
            page = preds[0]
            lines = getattr(page, "text_lines", []) or []
            boxes = []
            texts = []
            for ln in lines:
                txt = getattr(ln, "text", "") or ""
                texts.append(txt)
                bbox = getattr(ln, "bbox", None)
                if bbox and len(bbox) == 4:
                    x0, y0, x1, y1 = (int(v) for v in bbox)
                    boxes.append({
                        "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        "text": txt,
                        "conf": float(getattr(ln, "confidence", 0.0) or 0.0),
                    })
            return "\n".join(texts), boxes

        t = self._timed(_call)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        text, boxes = t["value"]
        return OcrResult(provider_id=self.info.id, ok=True, ms=t["ms"], text=text, boxes=boxes)


PROVIDER = SuryaOcr()
