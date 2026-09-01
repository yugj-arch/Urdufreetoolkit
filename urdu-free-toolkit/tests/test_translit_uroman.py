# -*- coding: utf-8 -*-
"""uroman transliteration provider.

uroman is a universal romanizer: it maps every Urdu letter faithfully but
invents no short vowels, so ``محبت`` comes out ``mhbt``. This provider keeps
that skeleton reading as the out-of-vocabulary behaviour -- the honest
baseline the compare view wants -- but routes every token through the same
shared curated dictionary + bundled lexicon + per-script punctuation layer the
rule engine uses, so on clean text the output matches the GPT-4o compare
column byte-for-byte (both scripts). Devanagari for OOV words comes from the
rule engine's character fallback, since uroman produces none.
"""
import pytest

import transliterate as rule_engine
from providers.translit import _uroman_engine as ur_engine
from providers.translit.uroman_p import PROVIDER
from providers.base import TranslitOpts, Capability

pytestmark = pytest.mark.skipif(
    ur_engine._UR is None, reason="uroman not installed"
)

PERSO_ARABIC = range(0x0600, 0x0700)


def _leaks(*strings):
    return {c for s in strings for c in s if ord(c) in PERSO_ARABIC}


# ---------------------------------------------------------------------------
# provider wiring
# ---------------------------------------------------------------------------

def test_info():
    assert PROVIDER.info.id == "uroman"
    assert PROVIDER.info.capability == Capability.TRANSLIT
    assert PROVIDER.info.kind == "offline"
    assert PROVIDER.available() == (True, "")


def test_translit_known_words():
    r = PROVIDER.translit("میں ٹھیک ہوں", TranslitOpts())
    assert r.ok is True
    assert r.roman.startswith(("mein", "main"))
    assert r.ms >= 0


# ---------------------------------------------------------------------------
# uroman still drives out-of-vocabulary words -- the raw consonant skeleton,
# distinct from the rule engine's schwa-filled guess. Devanagari for those
# words falls back to the rule engine (uroman produces none).
# ---------------------------------------------------------------------------

def test_oov_words_keep_uromans_skeleton_romanization():
    _, roman = ur_engine.transliterate("محبت")        # not in the dictionary
    assert roman == "mhbt"
    assert roman != rule_engine.transliterate("محبت")[1]  # rule guesses "mahabat"


def test_oov_devanagari_comes_from_the_rule_fallback():
    deva, _ = ur_engine.transliterate("محبت")
    assert deva == rule_engine.transliterate("محبت")[0]
    assert not _leaks(deva)


# ---------------------------------------------------------------------------
# no Perso-Arabic character may survive into the output
# ---------------------------------------------------------------------------

TRICKY_WORDS = [
    "دیا", "آیا", "لڑکیاں",
    "لائے", "گئے", "کوئی", "آؤ", "رئیس",
    "کاـم", "کتـاب",
    "سال ۲۰۲۴", "صفحہ ١٢٣",
    "جملہ،",
    "پڑوسی", "کھڑکی", "قلمکار", "محبت",
]


@pytest.mark.parametrize("src", TRICKY_WORDS)
def test_no_perso_arabic_leaks_into_output(src):
    d, r = ur_engine.transliterate(src)
    assert not _leaks(d, r), f"leaked {sorted(_leaks(d, r))}: {d!r} / {r!r}"


def test_tatweel_is_transparent():
    assert ur_engine.transliterate("کاـم") == ur_engine.transliterate("کام")


def test_arabic_indic_digits_become_ascii():
    d, r = ur_engine.transliterate("سال ۲۰۲۴")
    assert "2024" in d and "2024" in r


# ---------------------------------------------------------------------------
# shared curated dictionary is exact and identical to the rule engine
# ---------------------------------------------------------------------------

POEM_WORDS = {
    "عید": ("ईद", "eid"),
    "مبارک": ("मुबारक", "mubaarak"),
    "نظام": ("निज़ाम", "nizaam"),
    "سرکار": ("सरकार", "sarkaar"),
    "فلاحی": ("फ़लाही", "falaahi"),
    "آزادی": ("आज़ादी", "aazaadi"),
    "بزم": ("बज़्म", "bazm"),
}


@pytest.mark.parametrize("src,expected", list(POEM_WORDS.items()))
def test_curated_vocabulary_exact(src, expected):
    assert ur_engine.transliterate(src) == expected


def test_curated_dictionary_is_shared_with_rule_engine():
    for src in ["راستے", "فاصلے", "منزلوں", "ذریعہ", "کہنا", "کتاب", "ہوتا"]:
        assert ur_engine.transliterate(src) == rule_engine.transliterate(src)


# ---------------------------------------------------------------------------
# per-script punctuation (۔ -> danda / period), never leaked
# ---------------------------------------------------------------------------

def test_urdu_full_stop_is_danda_in_devanagari_and_period_in_roman():
    deva, roman = ur_engine.transliterate("یہ گھر ہے۔")
    assert deva == "ये घर है।"
    assert roman == "ye ghar hai."


def test_arabic_comma_and_question_mark_stay_ascii_in_both_scripts():
    deva, roman = ur_engine.transliterate("کیا؟ ہاں،")
    assert deva == "क्या? हाँ,"
    assert roman == "kya? haan,"


def test_urdu_full_stop_glued_to_a_word_still_never_leaks():
    deva, roman = ur_engine.transliterate("ہے۔اور")
    assert not _leaks(deva, roman)
    assert "।" in deva and "." in roman


def test_newlines_are_preserved():
    deva, roman = ur_engine.transliterate("گھر\nہے")
    assert deva == "घर\nहै"
    assert roman == "ghar\nhai"


# ---------------------------------------------------------------------------
# whole-sentence parity with the GPT-4o reference column, both scripts. Same
# reference strings the rule engine is pinned to.
# ---------------------------------------------------------------------------

_REF_SENTENCE = (
    "راستے اور فاصلے میں بڑا فرق ہوتا ہے۔ "
    "راستے طے کرنے پڑتے ہیں اور فاصلے سمیٹنے پڑتے ہیں۔ "
    "راستے وہ جو منزلوں تک لے جانے کا سبب بنتے ہیں اور "
    "فاصلے وہ جو اپنوں سے دور لے جانے کا ذریعہ بنتے ہیں۔"
)

_REF_ROMAN = (
    "raaste aur faasle mein badaa farq hota hai. "
    "raaste tay karne padte hain aur faasle sametne padte hain. "
    "raaste wo jo manzilon tak le jaane ka sabab bante hain aur "
    "faasle wo jo apnon se door le jaane ka zariya bante hain."
)

_REF_DEVA = (
    "रास्ते और फ़ासले में बड़ा फ़र्क होता है। "
    "रास्ते तय करने पड़ते हैं और फ़ासले समेटने पड़ते हैं। "
    "रास्ते वो जो मंज़िलों तक ले जाने का सबब बनते हैं और "
    "फ़ासले वो जो अपनों से दूर ले जाने का ज़रिया बनते हैं।"
)


def test_reference_sentence_roman_matches_gpt():
    _, roman = ur_engine.transliterate(_REF_SENTENCE)
    assert roman == _REF_ROMAN


def test_reference_sentence_devanagari_matches_gpt():
    deva, _ = ur_engine.transliterate(_REF_SENTENCE)
    assert deva == _REF_DEVA


def test_reference_sentence_has_no_perso_arabic_leaks():
    deva, roman = ur_engine.transliterate(_REF_SENTENCE)
    assert not _leaks(deva, roman)


_GHAZAL_SRC = "\n".join([
    "عید مبارک",
    "وہ عید بھیج کہ لائے نوید آبادی",
    "ہلال عید کو کر دے کلید آزادی",
    "علی آج نہ سائل پھرے ترا محروم",
    "ترا غلام ہو تیرے در کا فرمادی",
])

_GHAZAL_DEVA = "\n".join([
    "ईद मुबारक",
    "वो ईद भेज कि लाए नवेद आबादी",
    "हिलाल ईद को कर दे कलीद आज़ादी",
    "अली आज न साइल फिरे तेरा महरूम",
    "तेरा गुलाम हो तेरे दर का फरमादी",
])

_GHAZAL_ROMAN = "\n".join([
    "eid mubaarak",
    "wo eid bhej ki laaye naveed aabaadi",
    "hilaal eid ko kar de kaleed aazaadi",
    "ali aaj na saail phire tera mahroom",
    "tera ghulaam ho tere dar ka farmaadi",
])


def test_ghazal_devanagari_matches_gpt():
    deva, _ = ur_engine.transliterate(_GHAZAL_SRC)
    assert deva == _GHAZAL_DEVA


def test_ghazal_roman_matches_gpt():
    _, roman = ur_engine.transliterate(_GHAZAL_SRC)
    assert roman == _GHAZAL_ROMAN


def test_ghazal_has_no_perso_arabic_leaks():
    deva, roman = ur_engine.transliterate(_GHAZAL_SRC)
    assert not _leaks(deva, roman)
