import numpy as np
import pytest
from PIL import Image

from training.synth import augment, build_synth, render

_CANDIDATE_FONTS = [
    *render.find_fonts(),
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _a_font():
    from PIL import ImageFont
    for p in _CANDIDATE_FONTS:
        try:
            ImageFont.truetype(p, 40)
            return p
        except Exception:
            continue
    pytest.skip("no usable font on this machine")


def test_render_line_is_non_blank():
    img = render.render_line("میں ٹھیک ہوں", _a_font(), size=48)
    arr = np.array(img.convert("L"))
    assert arr.shape[0] > 10 and (arr < 128).mean() > 0.002   # some ink


def test_augment_changes_pixels_but_keeps_shape():
    rng = np.random.default_rng(0)
    src = np.full((40, 200), 255, np.uint8)
    src[15:25, 20:180] = 0
    out = augment.augment(src, rng)
    assert out.shape == src.shape and not np.array_equal(out, src)


def test_build_writes_split_and_gt(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "render_line",
                        lambda text, fp, **k: Image.new("RGB", (120, 32), "white"))
    sentences = ["جملہ ایک", "جملہ دو", "جملہ تین", "جملہ چار"]
    rep = build_synth.build(sentences, ["fake.ttf"], tmp_path, per_sentence=2, val_frac=0.25, seed=1)
    assert rep["train"] + rep["val"] == 8
    gt_lines = (tmp_path / "train" / "gt.txt").read_text(encoding="utf-8").splitlines()
    assert gt_lines and "\t" in gt_lines[0]
    for line in gt_lines:
        rel = line.split("\t", 1)[0]
        assert (tmp_path / "train" / rel).exists()
