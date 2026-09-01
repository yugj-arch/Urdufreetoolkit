# -*- coding: utf-8 -*-
"""PaddleOCR as an OCR provider (offline), wrapped in the shared OCR pipeline.

Arabic-script model; covers the Urdu alphabet, strong on printed Naskh. Paddle's
Python API churns between versions, so construction filters kwargs against the
actual ``__init__`` signature and the call tries ``predict`` then ``ocr``.
``lang="ur"`` is required on 3.x (``"arabic"`` raises ``ValueError: No models
are available``). ``enable_mkldnn=False`` dodges a paddlepaddle 3.3.x PIR/oneDNN
crash (``ConvertPirAttribute2RuntimeAttribute not support``). First run
downloads the models.
"""
from __future__ import annotations

import inspect

from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo
from providers.ocr import _model_hooks
from providers.ocr._pipeline import OcrConfig, run
from providers.ocr._translit_fill import rule_translit
from providers.ocr._types import Word

try:
    from paddleocr import PaddleOCR
except Exception:  # noqa: BLE001
    PaddleOCR = None

_engine = None

_ENGINE_CFG = dict(
    lang="ur", use_textline_orientation=True, enable_mkldnn=False,
    text_det_thresh=0.3, text_det_box_thresh=0.5, text_det_unclip_ratio=1.8,
    drop_score=0.4,
)
# Paddle's detector prefers grayscale contrast over hard binarization.
_CFG = OcrConfig(binarize="none", engine={})


def _supported_kwargs(cls, want: dict) -> dict:
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return dict(want)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(want)
    return {k: v for k, v in want.items() if k in params}


def _get_engine():
    global _engine
    if _engine is None:
        want = dict(_ENGINE_CFG)
        rec_dir = _model_hooks.paddle_rec_dir()
        if rec_dir:
            want["text_recognition_model_dir"] = rec_dir
        kw = _supported_kwargs(PaddleOCR, want)
        try:
            _engine = PaddleOCR(**kw)
        except TypeError:
            legacy = _supported_kwargs(PaddleOCR, {**want, "use_angle_cls": True})
            legacy.pop("use_textline_orientation", None)
            _engine = PaddleOCR(**legacy)
    return _engine


def _extract(result) -> list[Word]:
    rows: list[tuple] = []
    if isinstance(result, list) and result and isinstance(result[0], dict):
        d = result[0]
        texts = d.get("rec_texts") or d.get("rec_text") or []
        polys = d.get("rec_polys") or d.get("dt_polys") or [None] * len(texts)
        scores = d.get("rec_scores") or [0.0] * len(texts)
        rows = list(zip(polys, texts, scores))
    elif isinstance(result, list) and result and isinstance(result[0], list):
        for entry in result[0]:
            try:
                box, (txt, score) = entry
                rows.append((box, txt, score))
            except Exception:  # noqa: BLE001
                continue
    words: list[Word] = []
    for box, txt, score in rows:
        pts = [(0, 0)] * 4 if box is None else [(int(p[0]), int(p[1])) for p in box]
        words.append(Word(text=txt, box=pts, conf=float(score or 0.0)))
    return words


def _recognize(img, engine_cfg: dict) -> list[Word]:
    engine = _get_engine()
    try:
        result = engine.predict(img)
    except (AttributeError, TypeError):
        result = engine.ocr(img)
    return _extract(result)


class PaddleOcr(BaseProvider):
    info = ProviderInfo(
        id="paddle",
        label="PaddleOCR (offline)",
        capability=Capability.OCR,
        kind="offline",
        note="Arabic-script model + shared preprocessing/RTL pipeline. First run downloads ~20 MB.",
        price="Free · offline",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if PaddleOCR is not None else (False, "pip install paddleocr paddlepaddle")

    def ocr(self, image: bytes) -> OcrResult:
        if PaddleOCR is None:
            return OcrResult(provider_id=self.info.id, ok=False, error="paddleocr not installed")
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


PROVIDER = PaddleOcr()
