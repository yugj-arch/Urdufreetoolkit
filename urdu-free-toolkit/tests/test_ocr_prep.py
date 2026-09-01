import numpy as np
import pytest

from providers.ocr import _prep
from providers.ocr._types import OcrConfig


def _text_bar(h=200, w=400, angle=0.0):
    img = np.full((h, w), 255, np.uint8)
    img[90:110, 40:360] = 0                     # a horizontal black bar = "text"
    if angle:
        import cv2
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    return img


def _page_with_bar(h=120, w=240):
    img = np.full((h, w, 3), 255, np.uint8)
    img[55:65, 20:220] = 0                       # ~10px "text" -> triggers upscale
    return img


def test_to_gray_reduces_dims():
    rgb = np.zeros((10, 10, 3), np.uint8)
    assert _prep.to_gray(rgb).ndim == 2


def test_binarize_otsu_is_two_valued():
    img = (np.random.rand(50, 50) * 255).astype(np.uint8)
    out = _prep.binarize(img, "otsu")
    assert sorted(np.unique(out).tolist()) == [0, 255]


def test_upscale_hits_min_text_px_and_reports_scale():
    img = _text_bar()                            # bar is 20px tall
    out, scale = _prep.upscale(img, min_text_px=60, max_factor=5.0)
    assert scale >= 2.9 and out.shape[0] > img.shape[0]


def test_upscale_respects_max_factor():
    img = _text_bar()
    _, scale = _prep.upscale(img, min_text_px=10_000, max_factor=3.0)
    assert scale == pytest.approx(3.0)


def test_deskew_corrects_known_angle():
    skewed = _text_bar(angle=7.0)
    _, ang = _prep.deskew(skewed, max_deg=15.0)
    assert abs(ang) == pytest.approx(7.0, abs=2.5)


def test_preprocess_point_roundtrip():
    img = _page_with_bar()
    cfg = OcrConfig(denoise=False, clahe=False, deskew=False, binarize="none")
    proc, inv, stages = _prep.preprocess(img, cfg)
    px, py = proc.shape[1] * 0.5, proc.shape[0] * 0.5
    ox, oy = inv(px, py)
    assert abs(ox - 120) < 3 and abs(oy - 60) < 3
    assert "upscale" in stages


def test_preprocess_survives_a_failing_stage(monkeypatch):
    img = np.full((60, 60, 3), 255, np.uint8)
    monkeypatch.setattr(_prep, "clahe",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    proc, inv, stages = _prep.preprocess(img, OcrConfig(clahe=True))
    assert proc is not None and "clahe" not in stages
