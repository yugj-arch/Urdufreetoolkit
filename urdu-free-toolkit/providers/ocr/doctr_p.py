# -*- coding: utf-8 -*-
"""docTR as an OCR provider (offline).

Detection + recognition with a PyTorch backend. docTR's recognizers are trained
on Latin script, so Urdu accuracy is low — it is here for completeness and for
any Latin text embedded in the image. First call downloads the models (~150 MB).
"""
from __future__ import annotations

from providers._imgutil import to_pil
from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo

try:
    from doctr.models import ocr_predictor
except Exception:  # noqa: BLE001
    ocr_predictor = None

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = ocr_predictor(pretrained=True)
    return _model


class DoctrOcr(BaseProvider):
    info = ProviderInfo(
        id="doctr",
        label="docTR (offline)",
        capability=Capability.OCR,
        kind="offline",
        note="PyTorch; Latin-trained recognizers — low Urdu accuracy. First run ~150 MB.",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if ocr_predictor is not None else (False, "pip install python-doctr[torch]")

    def ocr(self, image: bytes) -> OcrResult:
        if ocr_predictor is None:
            return OcrResult(provider_id=self.info.id, ok=False, error="python-doctr not installed")

        def _call():
            import numpy as np

            arr = np.array(to_pil(image))
            result = _get_model()([arr])
            return result.render()

        t = self._timed(_call)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        return OcrResult(provider_id=self.info.id, ok=True, ms=t["ms"], text=(t["value"] or "").strip())


PROVIDER = DoctrOcr()
