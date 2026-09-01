# -*- coding: utf-8 -*-
"""Light, order-preserving image augmentation for synthetic OCR lines."""
from __future__ import annotations

import cv2
import numpy as np


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    if rng.random() < 0.5:
        k = int(rng.choice([3, 3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.6:
        noise = rng.normal(0, rng.uniform(4, 14), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        alpha = float(rng.uniform(0.8, 1.2))
        beta = float(rng.uniform(-20, 20))
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ang = float(rng.uniform(-2.5, 2.5))
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderValue=255)
    if rng.random() < 0.5:
        q = int(rng.integers(30, 80))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return out
