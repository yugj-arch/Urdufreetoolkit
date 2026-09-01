import pytest

from providers.ocr._normalize import normalize_urdu
from providers.ocr._types import OcrConfig


@pytest.mark.parametrize("raw, expected", [
    ("يه", "یه"),                                 # ي -> ی ; heh untouched by default
    ("كتاب", "کتاب"),         # ك -> ک
    ("محبــت", "محبت"),  # tatweel stripped
    ("ﻻ", "لا"),                                       # ﻻ ligature -> لا
    ("١٢٣", "123"),                                    # Arabic-Indic digits -> ascii
    ("۱۲", "12"),                                           # Extended Arabic-Indic -> ascii
    ("word  ، next", "word، next"),                         # space before ، removed, runs collapsed
])
def test_normalize_rules(raw, expected):
    assert normalize_urdu(raw) == expected


def test_digits_keep_and_urdu_modes():
    assert normalize_urdu("١٢", OcrConfig(map_digits_to="keep")) == "١٢"
    assert normalize_urdu("12", OcrConfig(map_digits_to="urdu")) == "۱۲"


def test_heh_fold_opt_in():
    assert normalize_urdu("ه", OcrConfig(fold_arabic_heh=True)) == "ہ"
    assert normalize_urdu("ه", OcrConfig(fold_arabic_heh=False)) == "ه"


def test_blank_lines_and_trailing_ws_dropped():
    assert normalize_urdu("a \n\n  \nb ") == "a\nb"


def test_spellfix_conservative():
    from providers.ocr._normalize import FREQ_WORDS
    assert "کتاب" in FREQ_WORDS  # kitab present as a frequent token
    fixed = normalize_urdu("کتب", OcrConfig(spellfix=True))
    assert fixed == "کتاب"
    # a token with no close frequent neighbour is left alone
    assert normalize_urdu("زززز", OcrConfig(spellfix=True)) == "زززز"


def test_spellfix_off_by_default():
    assert normalize_urdu("کتب") == "کتب"
