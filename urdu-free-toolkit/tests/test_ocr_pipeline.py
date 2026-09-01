import io

import numpy as np
from PIL import Image

from providers.ocr._pipeline import run
from providers.ocr._types import OcrConfig, Word


def _png(w=240, h=120):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_run_assembles_rtl_text_and_meta():
    def fake_recognize(img, engine_cfg):
        return [
            Word("A", [(200, 10), (240, 10), (240, 30), (200, 30)], 0.9),
            Word("B", [(10, 10), (60, 10), (60, 30), (10, 30)], 0.8),
        ]

    out = run(_png(), fake_recognize, OcrConfig(denoise=False, clahe=False))
    assert out.text == "A B"
    assert out.meta["reading_order"] == "rtl-pipeline"
    assert isinstance(out.meta["prep"], list)
    assert out.lines and out.lines[0].words[0].text == "A"


def test_run_maps_boxes_back_to_original_coords():
    seen = {}

    def fake_recognize(img, engine_cfg):
        seen["shape"] = img.shape
        h, w = img.shape[:2]
        return [Word("X", [(0, 0), (w, 0), (w, h), (0, h)], 0.5)]

    out = run(_png(240, 120), fake_recognize, OcrConfig(pad=10))
    xs = [p[0] for p in out.lines[0].words[0].box]
    ys = [p[1] for p in out.lines[0].words[0].box]
    # box came back near original bounds (240x120) within pad slop, not in
    # processed-image space
    assert max(xs) <= 240 + 12 and max(ys) <= 120 + 12
    assert max(xs) < seen["shape"][1]          # engine saw a larger, transformed image
    assert seen["shape"][0] != 120


def test_run_falls_back_when_recognize_returns_nothing():
    out = run(_png(), lambda *_a: [], OcrConfig())
    assert out.text == "" and out.lines == []


def test_run_normalizes_output():
    def fake(img, cfg):
        return [Word("كتاب", [(0, 0), (30, 0), (30, 20), (0, 20)], 0.9)]

    out = run(_png(), fake, OcrConfig())
    assert out.text == "کتاب"          # Arabic kaf U+0643 folded to keheh U+06A9
