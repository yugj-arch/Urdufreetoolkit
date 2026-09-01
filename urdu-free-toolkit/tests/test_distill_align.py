from providers.ocr._types import Word
from training.distill import align


def _w(text, y):
    return Word(text=text, box=[(0, y), (100, y), (100, y + 20), (0, y + 20)], conf=0.9)


def test_align_lines_one_to_one_when_counts_match():
    got = align.align_lines(None, ["alpha", "beta", "gamma"], n_crops=3)
    assert got == ["alpha", "beta", "gamma"]


def test_align_drops_when_page_has_fewer_lines():
    got = align.align_lines(None, ["only one"], n_crops=3)
    assert got[0] == "only one" and got[1] is None and got[2] is None


def test_align_uses_hint_to_reorder_fuzzy():
    # crop hints are noisy OCR; page lines are clean GPT — match by similarity
    hints = ["helo wrld", "gud bye"]
    page = ["good bye", "hello world"]
    got = align.align_lines(hints, page, n_crops=2)
    assert got == ["hello world", "good bye"]


def test_high_level_align_orders_and_filters():
    crops = [_w("دوم سطر", 40), _w("اول سطر", 0)]
    gpt = "اول سطر\nدوم سطر"
    pairs = align.align(crops, gpt, min_ratio=50.0)
    texts = [t for _w_, t in pairs]
    assert texts == ["اول سطر", "دوم سطر"]
