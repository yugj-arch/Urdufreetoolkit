# -*- coding: utf-8 -*-
"""Tesseract as an OCR provider (offline).

Needs the Tesseract *binary* installed separately (Windows: the UB Mannheim
build) plus the ``urd`` language data. Lightweight and fully offline, but the
weakest of the offline engines on Nastaliq. Auto-disables with a clear reason
when the binary or the Urdu data is missing.
"""
from __future__ import annotations

from providers._imgutil import to_pil
from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo

try:
    import pytesseract
except Exception:  # noqa: BLE001
    pytesseract = None


def _check() -> tuple[bool, str]:
    if pytesseract is None:
        return (False, "pip install pytesseract")
    try:
        langs = set(pytesseract.get_languages(config=""))
    except Exception as e:  # noqa: BLE001 - binary not on PATH
        return (False, f"Tesseract binary not found ({type(e).__name__})")
    if "urd" not in langs:
        return (False, "Tesseract has no 'urd' language data")
    return (True, "")


class TesseractOcr(BaseProvider):
    info = ProviderInfo(
        id="tesseract",
        label="Tesseract (offline)",
        capability=Capability.OCR,
        kind="offline",
        note="Needs the Tesseract binary + urd data. Lightest engine; weak on Nastaliq.",
    )

    def available(self) -> tuple[bool, str]:
        return _check()

    def ocr(self, image: bytes) -> OcrResult:
        ok, reason = _check()
        if not ok:
            return OcrResult(provider_id=self.info.id, ok=False, error=reason)

        def _call():
            return pytesseract.image_to_string(to_pil(image), lang="urd").strip()

        t = self._timed(_call)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        return OcrResult(provider_id=self.info.id, ok=True, ms=t["ms"], text=t["value"])


PROVIDER = TesseractOcr()
