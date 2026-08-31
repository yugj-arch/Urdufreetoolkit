# -*- coding: utf-8 -*-
"""Small image helpers shared by the offline OCR / render providers."""
from __future__ import annotations

import io

from PIL import Image


def to_pil(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    return img.convert("RGB")


def to_ndarray(image_bytes: bytes):
    """RGB uint8 H x W x 3 numpy array."""
    import numpy as np

    return np.array(to_pil(image_bytes))
