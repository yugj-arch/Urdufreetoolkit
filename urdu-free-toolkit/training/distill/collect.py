# -*- coding: utf-8 -*-
"""Thin helper: run an engine's recognize() over a real image to get ordered
line-ish Word boxes for distillation. Kept minimal on purpose — the engine and
its preprocessing (providers/ocr/_pipeline.py) already do the real work.
"""
from __future__ import annotations

from providers._imgutil import to_ndarray
from providers.ocr._types import Word


def crops_from_image(image: bytes, recognize) -> list[Word]:
    return list(recognize(to_ndarray(image), {}) or [])
