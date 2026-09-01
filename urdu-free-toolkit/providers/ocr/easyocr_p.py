# -*- coding: utf-8 -*-
"""EasyOCR as an OCR provider (offline), wrapped in the shared OCR pipeline.

EasyOCR ships a real Urdu recognition model. The first call downloads the
detector + Urdu recognizer (~100 MB); later calls are offline. The pipeline
(providers/ocr/_pipeline.py) handles preprocessing, RTL reading order, Urdu
normalization, and a rule-engine transliteration fill so this column carries the
same urdu + devanagari + roman shape as the GPT column.
"""
from __future__ import annotations

from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo
from providers.ocr import _model_hooks
from providers.ocr._pipeline import OcrConfig, run
from providers.ocr._translit_fill import rule_translit
from providers.ocr._types import Word

try:
    import easyocr
except Exception:  # noqa: BLE001
    easyocr = None

_reader = None

# Tuned knobs. _prep already upscales, so mag_ratio stays modest.
_ENGINE_CFG = dict(
    decoder="beamsearch", beamWidth=10, contrast_ths=0.1, adjust_contrast=0.5,
    text_threshold=0.6, low_text=0.3, link_threshold=0.4, mag_ratio=1.5,
    slope_ths=0.2, ycenter_ths=0.5, height_ths=0.7, width_ths=0.5, add_margin=0.1,
    detail=1, paragraph=False,
)
_CFG = OcrConfig(binarize="sauvola", engine=dict(_ENGINE_CFG))


def _get_reader():
    global _reader
    if _reader is None:
        net = _model_hooks.easyocr_recog_network()
        kw = {"gpu": False, "verbose": False}
        if net:
            kw["recog_network"] = net
        _reader = easyocr.Reader(["ur", "en"], **kw)
    return _reader


def _recognize(img, engine_cfg: dict) -> list[Word]:
    found = _get_reader().readtext(img, **engine_cfg)
    return [
        Word(text=text, box=[(int(x), int(y)) for x, y in box], conf=float(conf))
        for box, text, conf in found
    ]


class EasyOcr(BaseProvider):
    info = ProviderInfo(
        id="easyocr",
        label="EasyOCR (offline)",
        capability=Capability.OCR,
        kind="offline",
        note="PyTorch Urdu model + shared preprocessing/RTL pipeline. First run downloads ~100 MB.",
        price="Free · offline",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if easyocr is not None else (False, "pip install easyocr")

    def ocr(self, image: bytes) -> OcrResult:
        if easyocr is None:
            return OcrResult(provider_id=self.info.id, ok=False, error="easyocr not installed")
        t = self._timed(run, image, _recognize, _CFG)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        out = t["value"]
        deva, roman = rule_translit(out.text)
        boxes = [{"box": [list(p) for p in w.box], "text": w.text, "conf": w.conf}
                 for ln in out.lines for w in ln.words]
        return OcrResult(
            provider_id=self.info.id, ok=True, ms=t["ms"],
            text=out.text, boxes=boxes,
            meta={"devanagari": deva, "roman": roman, **out.meta},
        )


PROVIDER = EasyOcr()
