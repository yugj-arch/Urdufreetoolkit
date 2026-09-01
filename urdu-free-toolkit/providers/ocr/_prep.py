# -*- coding: utf-8 -*-
"""Preprocessing stages for the offline OCR engines.

Each stage is a small function on a numpy image. ``preprocess`` composes the
enabled ones per ``OcrConfig``, tracks the geometric transform so word boxes can
be mapped back to original-image coordinates, and never lets one failing stage
abort the pipeline.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from providers.ocr._types import OcrConfig

log = logging.getLogger(__name__)

try:
    from skimage.filters import threshold_sauvola  # type: ignore
    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAVE_SKIMAGE = False


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _estimate_text_px(gray: np.ndarray) -> float:
    inv = 255 - gray
    rows = (inv > 64).sum(axis=1)
    if not rows.max():
        return float(gray.shape[0])
    thresh = rows.max() * 0.3
    runs, cur = [], 0
    for v in rows:
        if v > thresh:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    runs = [r for r in runs if r >= 3]
    return float(np.median(runs)) if runs else float(gray.shape[0])


def upscale(img: np.ndarray, min_text_px: int, max_factor: float) -> tuple[np.ndarray, float]:
    tpx = _estimate_text_px(to_gray(img))
    factor = 1.0 if tpx <= 0 else min(max_factor, max(1.0, min_text_px / tpx))
    if factor <= 1.001:
        return img, 1.0
    out = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_LANCZOS4)
    return out, factor


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(to_gray(img), None, h=7,
                                    templateWindowSize=7, searchWindowSize=21)


def clahe(img: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(to_gray(img))


def binarize(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return img
    g = to_gray(img)
    if mode == "otsu":
        _, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out
    if mode == "sauvola":
        if _HAVE_SKIMAGE:
            win = max(3, (min(g.shape) // 20) | 1)
            t = threshold_sauvola(g, window_size=win)
            return ((g > t) * 255).astype(np.uint8)
        return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    return g


def deskew(img: np.ndarray, max_deg: float) -> tuple[np.ndarray, float]:
    g = to_gray(img)
    inv = 255 - g
    coords = np.column_stack(np.where(inv > 64))
    if len(coords) < 50:
        return img, 0.0
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) > max_deg or abs(angle) < 0.1:
        return img, 0.0
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return out, float(angle)


def pad(img: np.ndarray, px: int) -> np.ndarray:
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=255)


def preprocess(img: np.ndarray, cfg: OcrConfig):
    """Returns ``(processed_img, inverse_point_fn, applied_stage_names)``.

    ``inverse_point_fn(px, py)`` maps a point in processed-image space back to
    original-image coordinates (undo pad, scale, rotation).
    """
    scale = 1.0
    angle = 0.0
    pad_px = 0
    h0, w0 = img.shape[:2]
    stages: list[str] = []
    cur = img

    def _try(name, fn):
        nonlocal cur
        try:
            cur = fn(cur)
            stages.append(name)
        except Exception:
            log.warning("prep stage %s failed; skipping", name, exc_info=True)

    if cfg.grayscale:
        _try("grayscale", to_gray)
    try:
        cur, scale = upscale(cur, cfg.upscale_min_text_px, cfg.max_upscale)
        if scale > 1.001:
            stages.append("upscale")
    except Exception:
        log.warning("prep stage upscale failed; skipping", exc_info=True)
        scale = 1.0
    if cfg.denoise:
        _try("denoise", denoise)
    if cfg.clahe:
        _try("clahe", clahe)
    if cfg.deskew:
        try:
            cur, angle = deskew(cur, cfg.deskew_max_deg)
            if angle:
                stages.append("deskew")
        except Exception:
            log.warning("prep stage deskew failed; skipping", exc_info=True)
            angle = 0.0
    if cfg.binarize != "none":
        _try(f"binarize:{cfg.binarize}", lambda im: binarize(im, cfg.binarize))
    if cfg.pad:
        try:
            cur = pad(cur, cfg.pad)
            pad_px = cfg.pad
            stages.append("pad")
        except Exception:
            log.warning("prep stage pad failed; skipping", exc_info=True)

    cx, cy = w0 / 2, h0 / 2
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def inv(px: float, py: float) -> tuple[float, float]:
        x = (px - pad_px) / scale - cx
        y = (py - pad_px) / scale - cy
        rx = cos_t * x + sin_t * y
        ry = -sin_t * x + cos_t * y
        return rx + cx, ry + cy

    return cur, inv, stages
