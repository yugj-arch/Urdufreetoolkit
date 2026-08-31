# -*- coding: utf-8 -*-
"""EasyOCR as an OCR provider (offline).

EasyOCR ships a real Urdu recognition model. The first call downloads the
detector + Urdu recognizer (~100 MB) into the EasyOCR cache; subsequent calls
are offline. Reading order for RTL scripts follows the detected boxes, so word
order can differ from the original on multi-column layouts.
"""
from __future__ import annotations

from providers._imgutil import to_ndarray
from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo

try:
    import easyocr
except Exception:  # noqa: BLE001
    easyocr = None

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["ur", "en"], gpu=False, verbose=False)
    return _reader


class EasyOcr(BaseProvider):
    info = ProviderInfo(
        id="easyocr",
        label="EasyOCR (offline)",
        capability=Capability.OCR,
        kind="offline",
        note="PyTorch; has an Urdu model. First run downloads ~100 MB.",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if easyocr is not None else (False, "pip install easyocr")

    def ocr(self, image: bytes) -> OcrResult:
        if easyocr is None:
            return OcrResult(provider_id=self.info.id, ok=False, error="easyocr not installed")

        def _call():
            arr = to_ndarray(image)
            reader = _get_reader()
            found = reader.readtext(arr, detail=1, paragraph=False)
            boxes = [
                {"box": [[int(x), int(y)] for x, y in box], "text": txt, "conf": float(conf)}
                for box, txt, conf in found
            ]
            text = "\n".join(txt for _, txt, _ in found)
            return text, boxes

        t = self._timed(_call)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        text, boxes = t["value"]
        return OcrResult(provider_id=self.info.id, ok=True, ms=t["ms"], text=text, boxes=boxes)


PROVIDER = EasyOcr()
