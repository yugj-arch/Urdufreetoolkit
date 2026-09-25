# -*- coding: utf-8 -*-
"""GPT-distilled tier of the neural engine (training/translit/gpt_distill.py):
line table verbatim, word / context readings, joins, and the aligner that
builds them from GPT's replies."""
from __future__ import annotations

import gzip
import json

import pytest

import neural_translit
from training.translit.gpt_distill import align, urdu_tokens

GHALIB_1 = "آ کہ مری جان کو قرار نہیں ہے"
GHALIB_1_GPT = ["आ कि मेरी जाँ को क़रार नहीं है", "aa ki meri jaan ko qaraar nahin hai",
                "ā ki merī jāñ ko qarār nahīñ hai"]


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    if not neural_translit.LEXICON_PATH.exists():
        pytest.skip("neural model files not built")
    table = {
        "lines": {neural_translit._line_key(GHALIB_1): GHALIB_1_GPT},
        "words": {"طاقت": ["ताक़त", "taaqat", "ṭāqat"], "و": ["ओ", "o", "o"],
                  "کیوں": ["क्यूँ", "kyun", "kyūñ"], "دل": [None, "dill", None]},
        "ctx": {"R\tکیا\tہے": [None, "kiya", None]},
        "joins": {"طاقت\tبیداد": ["-e-", "-e-", None], "سنگ\tو": [" ", " ", " "],
                  "و\tخشت": [" ", " ", " "], "اندازۂ\tخمار": [" ", " ", " "]},
        "o_join": ["-", "-", "-"],
    }
    p = tmp_path_factory.mktemp("g") / "gpt.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(table, fh, ensure_ascii=False)
    missing = tmp_path_factory.mktemp("m") / "missing.pt"
    return (neural_translit.NeuralTransliterator(model_path=missing, gpt_path=None),
            neural_translit.NeuralTransliterator(model_path=missing, gpt_path=p))


def test_line_gpt_has_read_comes_back_verbatim(engines):
    _, gpt = engines
    assert gpt.transliterate(GHALIB_1) == tuple(GHALIB_1_GPT)
    # harakat / spacing differences still hit the same line
    assert gpt.transliterate("  آ کہ مِری جان کو قرار نہیں ہے ") == tuple(GHALIB_1_GPT)


def test_known_and_unknown_lines_mix_in_order(engines):
    base, gpt = engines
    other = "دل ہی تو ہے"
    out = gpt.transliterate(f"{other}\n{GHALIB_1}\n\n{other}")
    lines = [o.split("\n") for o in out]
    assert [l[1] for l in lines] == GHALIB_1_GPT
    assert [l[2] for l in lines] == ["", "", ""]
    assert lines[1][0] == lines[1][3]


def test_word_reading_per_script_falls_through(engines):
    base, gpt = engines
    d, p, r = gpt.transliterate("کیوں")
    assert (d, p, r) == ("क्यूँ", "kyun", "kyūñ")
    bd, bp, br = base.transliterate("دل")
    d, p, r = gpt.transliterate("دل")
    assert p == "dill" and d == bd and r == br     # only plain Roman known to GPT's table


def test_context_reading(engines):
    _, gpt = engines
    assert gpt.transliterate("قتل کیا ہے")[1].split()[1] == "kiya"


def test_joins_izafat_and_spaced_o(engines):
    base, gpt = engines
    d, p, r = gpt.transliterate("طاقت بیداد")
    assert p == "taaqat-e-" + p.split("-e-")[1] and "-ए-" in d and "-e-" in r
    assert "-o-" in base.transliterate("سنگ و خشت")[1]
    assert gpt.transliterate("سنگ و خشت")[1].split()[1] == "o"
    assert "-o-" in gpt.transliterate("شعر و شاعری")[1]          # unseen pair: o_join default
    # GPT's join overrides a written izafat
    assert "-e-" not in gpt.transliterate("اندازۂ خمار")[1]
    assert "-e-" in base.transliterate("اندازۂ خمار")[1]


@pytest.mark.parametrize("urdu,out,want", [
    ("طاقت بیداد انتظار نہیں ہے", "taaqat-e-bedaad-e-intizaar nahin hai",
     [("taaqat", "-e-"), ("bedaad", "-e-"), ("intizaar", " "), ("nahin", " "), ("hai", " ")]),
    ("نشہ بہ اندازۂ خمار", "nasha ba-andaaza-e-khumaar",
     [("nasha", " "), ("ba", "-"), ("andaaza", "-e-"), ("khumaar", " ")]),
    ("لطف جلوہ ہائے معانی", "लुत्फ़-ए-जल्वा-हा-ए-मआनी",
     [("लुत्फ़", "-e-"), ("जल्वा", "-"), ("हा", "-e-"), ("मआनी", " ")]),
    ("سنگ و خشت۔", "संग ओ ख़िश्त।", [("संग", " "), ("ओ", " "), ("ख़िश्त", " ")]),
    ("سنگ و خشت", "sang o", None),
    # GPT writes two Urdu words as one / one as two / drops the year sign
    ("نیپال اس کا پڑوسی ملک", "Nepal iska padosi mulk",
     [("Nepal", " "), ("iska", "+"), (None, " "), ("padosi", " "), ("mulk", " ")]),
    ("نکھر کرسامنے آیا", "nikhar kar saamne aaya",
     [("nikhar", " "), ("kar saamne", " "), ("aaya", " ")]),
    ("سنہ 1828ء میں", "sanah 1828 mein", [("sanah", " "), (None, " "), ("mein", " ")]),
    # a split that doesn't fit -> reject the line rather than misalign
    ("پنجاب محکمہ صحت میں برسرخدمت رہتے ہوئے",
     "Panjab mahkama-e-sehat mein bar-sar-e-khidmat rehte hue", None),
])
def test_align(urdu, out, want):
    script = "deva" if "ऀ" <= out[0] <= "ॿ" else "roman"
    assert align(urdu_tokens(urdu), out, script) == want


def test_merged_pair_spelled_as_one_word(engines, tmp_path):
    _, gpt = engines
    gpt.gpt_joins["روئیں	گے"] = ["+", "+", None]
    gpt.gpt_merge["روئیں	گے"] = ["रोएँगे", "royenge", None]
    try:
        d, p, r = gpt.transliterate("وہ روئیں گے")
    finally:
        del gpt.gpt_joins["روئیں	گے"], gpt.gpt_merge["روئیں	گے"]
    assert d.endswith(" रोएँगे") and p.endswith(" royenge")
    assert len(r.split()) == 3                           # no merged Rekhta spelling: unchanged
