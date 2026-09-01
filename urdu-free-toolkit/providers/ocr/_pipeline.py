# -*- coding: utf-8 -*-
"""The shared offline OCR pipeline.

    image bytes
      -> preprocess (grayscale, upscale, denoise, clahe, deskew, binarize, pad)
      -> engine recognize()  [detect + recognize, returns Word list in proc coords]
      -> map word boxes back to original image coordinates
      -> group words into lines, order lines (RTL, right column first)
      -> normalize Urdu text
      -> PipelineOut(text, lines, meta)

Engines provide only ``recognize(processed_img, engine_cfg) -> list[Word]``.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from providers._imgutil import to_ndarray
from providers.ocr import _normalize, _prep, _rtl
from providers.ocr._types import Line, OcrConfig, PipelineOut, Word  # noqa: F401  re-export

log = logging.getLogger(__name__)

RecognizeFn = Callable[[np.ndarray, dict], list]


def _map_word(w: Word, inv) -> Word:
    return Word(text=w.text, conf=w.conf,
                box=[tuple(int(round(v)) for v in inv(px, py)) for px, py in w.box])


def _safe_normalize(text: str, cfg: OcrConfig) -> str:
    try:
        return _normalize.normalize_urdu(text, cfg)
    except Exception:
        log.warning("normalize_urdu failed; returning raw text", exc_info=True)
        return text


def run(image: bytes, recognize: RecognizeFn, cfg: OcrConfig) -> PipelineOut:
    img0 = to_ndarray(image)
    meta: dict = {}
    try:
        proc, inv, stages = _prep.preprocess(img0, cfg)
    except Exception:
        log.warning("preprocess failed entirely; using raw image", exc_info=True)
        proc, inv, stages = img0, (lambda x, y: (x, y)), []
    meta["prep"] = stages

    words = list(recognize(proc, cfg.engine) or [])
    words = [_map_word(w, inv) for w in words]

    try:
        lines = _rtl.order_lines(_rtl.group_into_lines(words, cfg))
        meta["reading_order"] = "rtl-pipeline"
    except Exception:
        log.warning("reading-order failed; detector order", exc_info=True)
        meta["reading_order"] = "fallback-detector-order"
        text = " ".join(w.text for w in words)
        return PipelineOut(text=_safe_normalize(text, cfg), lines=[], meta=meta)

    text = _rtl.lines_to_text(lines)
    return PipelineOut(text=_safe_normalize(text, cfg), lines=lines, meta=meta)
