# -*- coding: utf-8 -*-
"""Shared value types for the OCR pipeline.

Kept in their own module so ``_pipeline`` / ``_rtl`` / the providers can all
import them without an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[int, int]


@dataclass(frozen=True)
class Word:
    text: str
    box: list[Point]        # 4-point polygon, image coordinates
    conf: float


@dataclass(frozen=True)
class Line:
    words: list[Word]
    text: str
    box: list[Point]
    conf: float


@dataclass(frozen=True)
class OcrConfig:
    # preprocessing
    grayscale: bool = True
    upscale_min_text_px: int = 22
    max_upscale: float = 3.0
    denoise: bool = True
    clahe: bool = True
    binarize: str = "sauvola"          # "none" | "otsu" | "sauvola"
    deskew: bool = True
    deskew_max_deg: float = 15.0
    pad: int = 12
    # reading order
    line_y_tol_frac: float = 0.6
    # normalization
    map_digits_to: str = "ascii"       # "ascii" | "urdu" | "keep"
    fold_arabic_heh: bool = False
    spellfix: bool = False
    # engine-specific knobs consumed by the provider's _recognize
    engine: dict = field(default_factory=dict)


@dataclass
class PipelineOut:
    text: str
    lines: list[Line]
    meta: dict = field(default_factory=dict)
