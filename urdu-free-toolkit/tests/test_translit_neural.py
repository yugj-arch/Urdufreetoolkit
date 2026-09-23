# -*- coding: utf-8 -*-
"""Neural transliterator: spelling schemes, the runtime engine's structure
guarantees, context rules, and the provider contract."""
from __future__ import annotations

import pytest

from urdu_nn.scheme import (
    deva_to_urdu_candidates, fold_roman, norm_urdu, to_plain, to_rekhta, wik_to_rich,
)


# -- scheme: Wiktionary -> R -------------------------------------------------

@pytest.mark.parametrize("roman,urdu,source,rich", [
    ("kyā", "کیا", "ur", "kyā"),
    ("h͠ai", "ہیں", "ur", "hai~"),          # double tilde on a consonant
    ("nahī̃", "نہیں", "ur", "nahī~"),
    ("ḵẖayāl", "خیال", "ur", "xayāl"),        # ALA ḵẖ
    ("khayaal", "خیال", "ur", "xayaal"),      # hand-typed kh for خ
    ("cirāġ", "چراغ", "ur", "cirāġ"),
    ("tağyīr", "تغییر", "ur", "taġyīr"),      # breve g
    ("manz̤ar", "منظر", "ur", "manzar"),
    ("ʻilm", "علم", "ur", "'ilm"),
    ("kŕṣṇa", "", "hi", "kriṣṇa"),
])
def test_wik_to_rich(roman, urdu, source, rich):
    assert wik_to_rich(roman, urdu, source) == rich


@pytest.mark.parametrize("bad", ["karẽ, kījie", "tyauhār~tyohār", "a-", "candr(a)bind(u)", ""])
def test_wik_to_rich_rejects_non_single_readings(bad):
    assert wik_to_rich(bad) is None


# -- scheme: renders ---------------------------------------------------------

@pytest.mark.parametrize("rich,plain,rekhta", [
    ("kyā", "kya", "kyā"),
    ("hai~", "hain", "haiñ"),
    ("me~", "mein", "meñ"),
    ("hā~", "haan", "hāñ"),
    ("kahā~", "kahan", "kahāñ"),
    ("nahī~", "nahin", "nahīñ"),
    ("kahnā", "kehna", "kahnā"),            # ah+C -> eh, plain only
    ("xayāl", "khayaal", "ḳhayāl"),
    ("kuch", "kuchh", "kuchh"),              # c = چ, ch = چھ
    ("ma'lūm", "maloom", "ma'lūm"),
    ("baṛā", "bada", "baṛā"),
    ("vaqt", "waqt", "vaqt"),
    ("jāe", "jaaye", "jāe"),                 # hiatus glide, plain only
    ("gae", "gaye", "gae"),
    ("jāe~", "jaayein", "jāeñ"),
])
def test_renders(rich, plain, rekhta):
    assert to_plain(rich) == plain
    assert to_rekhta(rich) == rekhta


def test_fold_roman_ignores_style_only_differences():
    assert fold_roman("kitaab") == fold_roman("kitāb") == fold_roman("kitab")
    assert fold_roman("waqt") == fold_roman("vaqt")
    assert fold_roman("mein") != fold_roman("main")


def test_norm_urdu_folds_variants_and_keeps_hamza():
    assert norm_urdu("كتاب") == "کتاب"
    assert norm_urdu("گئے") == "گئے"                 # hamza seat kept
    assert norm_urdu("کِتاب") == "کتاب"
    assert norm_urdu("کِتاب", keep_harakat=True) == "کِتاب"


@pytest.mark.parametrize("deva,urdu", [
    ("है", "ہے"), ("हैं", "ہیں"), ("में", "میں"), ("गए", "گئے"), ("आई", "آئی"),
    ("अच्छा", "اچھا"), ("किताब", "کتاب"), ("ज्ञान", "گیان"), ("औरत", "عورت"),
    ("ज़्यादा", "زیادہ"), ("इल्म", "علم"),
])
def test_deva_to_urdu_candidates_contains_real_spelling(deva, urdu):
    assert urdu in deva_to_urdu_candidates(deva)


# -- runtime engine ----------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    import neural_translit
    if not neural_translit.LEXICON_PATH.exists():
        pytest.skip("neural model files not built")
    return neural_translit.get_engine()


def test_keeps_lines_punctuation_digits_and_latin(engine):
    text = "دل ہی تو ہے\nنہ سنگ و خشت۔\n\nABC 123"
    deva, plain, dia = engine.transliterate(text)
    for out in (deva, plain, dia):
        assert out.count("\n") == text.count("\n")
        assert out.endswith("ABC 123")
    assert "।" in deva and "." in plain


def test_three_outputs_agree_on_the_reading(engine):
    deva, plain, dia = engine.transliterate("زندگی")
    assert deva == "ज़िंदगी" or deva == "ज़िन्दगी"
    assert plain == "zindagi"
    assert dia == "zindagī"


def test_mein_main_context(engine):
    assert engine.transliterate("میں نے کہا")[1].startswith("main ")
    assert engine.transliterate("وہ گھر میں ہے")[1] == "wo ghar mein hai"


def test_kya_kiya_context(engine):
    assert engine.transliterate("یہ کیا ہے")[1] == "ye kya hai"
    assert engine.transliterate("اس نے کام کیا۔")[1].endswith("kiya.")
    assert "kiya gaya" in engine.transliterate("فیصلہ کیا گیا")[1]


def test_conjunctive_waw_and_izafat(engine):
    _, plain, dia = engine.transliterate("شعر و شاعری")
    assert "-o-" in plain and "-o-" in dia
    _, _, dia = engine.transliterate("دلِ ناداں")
    assert dia.startswith("dil-e")


def test_english_loans_use_english_spelling_in_plain_only(engine):
    deva, plain, dia = engine.transliterate("ڈاکٹر اسکول گئے")
    assert plain.split()[:2] == ["doctor", "school"]
    assert "ḍ" in dia or "d" in dia.split()[0]          # Rekhta stays phonetic
    assert deva.startswith("डॉक्टर") or deva.startswith("डाक्टर")


def test_native_word_beats_english_homograph(engine):
    # کار is kaar (work) in Urdu, not "car"
    assert engine.transliterate("کار")[1] != "car"


def test_silent_he_is_short_a(engine):
    assert engine.transliterate("سبزہ")[2] == "sabza"


def test_arabic_article(engine):
    assert engine.transliterate("الرحمن")[2] == "ar-rahmān"        # sun letter assimilates
    assert engine.transliterate("الدین")[2] == "ad-dīn"
    assert engine.transliterate("العلوم")[2].startswith("al-")    # moon letter keeps l
    assert "ul-hukūmat" in engine.transliterate("دار الحکومت")[2]   # after a word: ul
    assert engine.transliterate("الماری")[1] == "almaari"          # a real al- word stays whole


def test_extra_curated_names(engine):
    deva, plain, dia = engine.transliterate("سید عباس مرزا")
    assert plain == "syed abbaas mirza"
    assert dia == "sayyid abbās mirzā"


def test_unknown_word_never_leaks_urdu_letters(engine):
    deva, plain, dia = engine.transliterate("قراقرمستان")
    for out in (deva, plain, dia):
        assert not any("؀" <= ch <= "ۿ" for ch in out)


def test_works_without_model_file(tmp_path):
    import neural_translit
    if not neural_translit.LEXICON_PATH.exists():
        pytest.skip("neural model files not built")
    eng = neural_translit.NeuralTransliterator(model_path=tmp_path / "missing.pt")
    assert not eng.has_model
    deva, plain, dia = eng.transliterate("محبت اور زندگی")
    assert plain.split()[1] == "aur"
    assert not any("؀" <= ch <= "ۿ" for ch in deva + plain + dia)


# -- provider ----------------------------------------------------------------

def test_provider_contract(engine):
    from providers.base import TranslitOpts
    from providers.translit.neural import PROVIDER
    assert PROVIDER.info.id == "neural" and PROVIDER.info.kind == "offline"
    assert PROVIDER.available()[0]
    res = PROVIDER.translit("یہ کتاب ہے", TranslitOpts())
    assert res.ok, res.error
    assert res.devanagari and res.roman and res.roman_diacritic
    assert res.roman.startswith("ye ")
