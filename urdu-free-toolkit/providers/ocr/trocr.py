# -*- coding: utf-8 -*-
"""TrOCR (Hugging Face) as an OCR provider (offline).

TrOCR is a single-line recognizer with no layout model, so it works best on a
tight crop of one text line. The default checkpoint is English-printed — set
``HF_TROCR_CKPT`` in Settings to an Urdu-finetuned checkpoint for real Urdu use.
First call downloads the chosen checkpoint.
"""
from __future__ import annotations

import os

from providers._imgutil import to_pil
from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except Exception:  # noqa: BLE001
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None

_DEFAULT = "microsoft/trocr-base-printed"
_cache: dict = {}


def _load(ckpt: str):
    if ckpt not in _cache:
        _cache[ckpt] = (
            TrOCRProcessor.from_pretrained(ckpt),
            VisionEncoderDecoderModel.from_pretrained(ckpt),
        )
    return _cache[ckpt]


class TrocrOcr(BaseProvider):
    info = ProviderInfo(
        id="hf_trocr",
        label="TrOCR / HF checkpoint (offline)",
        capability=Capability.OCR,
        kind="offline",
        note="Single-line recognizer. Set HF_TROCR_CKPT to an Urdu checkpoint; default is English.",
    )

    def available(self) -> tuple[bool, str]:
        if TrOCRProcessor is None:
            return (False, "pip install transformers torch")
        return (True, "")

    def ocr(self, image: bytes) -> OcrResult:
        if TrOCRProcessor is None:
            return OcrResult(provider_id=self.info.id, ok=False, error="transformers not installed")

        ckpt = os.environ.get("HF_TROCR_CKPT", _DEFAULT)

        def _call():
            processor, model = _load(ckpt)
            pil = to_pil(image)
            pixel_values = processor(images=pil, return_tensors="pt").pixel_values
            ids = model.generate(pixel_values, max_new_tokens=256)
            return processor.batch_decode(ids, skip_special_tokens=True)[0]

        t = self._timed(_call)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        return OcrResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                         text=(t["value"] or "").strip(), meta={"checkpoint": ckpt})


PROVIDER = TrocrOcr()
