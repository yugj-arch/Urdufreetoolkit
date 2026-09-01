# -*- coding: utf-8 -*-
"""Aksharamukha transliteration provider.

Raw Aksharamukha "Urdu" leaks common letters (پ ک ڑ), mishandles ع/ح and
restores no short vowels. This provider wraps it in the same machinery the
rule engine uses -- shared curated dictionary + bundled lexicon + per-script
punctuation -- and repairs whatever Aksharamukha leaks on the residual
out-of-vocabulary words. On clean text it matches the GPT-4o compare column
byte-for-byte, exactly like the rule engine.
"""
import pytest

import transliterate as rule_engine
from providers.translit import _aksharamukha_engine as ak_engine
from providers.translit.aksharamukha_p import PROVIDER
from providers.base import TranslitOpts, Capability

pytestmark = pytest.mark.skipif(
    ak_engine._ak is None, reason="aksharamukha not installed"
)

PERSO_ARABIC = range(0x0600, 0x0700)


def _leaks(*strings):
    """Perso-Arabic codepoints that leaked through into a transliteration."""
    return {c for s in strings for c in s if ord(c) in PERSO_ARABIC}


# ---------------------------------------------------------------------------
# provider wiring
# ---------------------------------------------------------------------------

def test_info():
    assert PROVIDER.info.id == "aksharamukha"
    assert PROVIDER.info.capability == Capability.TRANSLIT
    assert PROVIDER.info.kind == "offline"
    assert PROVIDER.available() == (True, "")


def test_translit_known_words():
    r = PROVIDER.translit("میں ٹھیک ہوں", TranslitOpts())
    assert r.ok is True
    assert "में" in r.devanagari or "मैं" in r.devanagari
    assert r.roman.startswith(("mein", "main"))
    assert r.ms >= 0


def test_translit_returns_both_scripts():
    r = PROVIDER.translit("عید مبارک", TranslitOpts())
    assert r.ok is True
    assert r.devanagari == "ईद मुबारक"
    assert r.roman == "eid mubaarak"


# ---------------------------------------------------------------------------
# no Perso-Arabic character may ever survive into the output -- this is the
# whole point of the repair pass, since raw Aksharamukha leaks پ ک ڑ ع ح …
# ---------------------------------------------------------------------------

TRICKY_WORDS = [
    "دیا", "آیا", "لڑکیاں",          # orphan long-vowel letters after a vowel
    "لائے", "گئے", "کوئی", "آؤ", "رئیس",  # hamza carriers (ء ئ ؤ)
    "کاـم", "کتـاب",                  # tatweel / kashida filler
    "سال ۲۰۲۴", "صفحہ ١٢٣",           # Arabic-Indic + Extended digits
    "جملہ،",                          # Arabic comma glued to a token
    "پڑوسی", "کھڑکی", "پکوان",        # پ ک ڑ clusters raw Aksharamukha drops
    "محبت", "تعلیم", "معلم",          # ع / ح mid-word
]


@pytest.mark.parametrize("src", TRICKY_WORDS)
def test_no_perso_arabic_leaks_into_output(src):
    d, r = ak_engine.transliterate(src)
    assert not _leaks(d, r), f"leaked {sorted(_leaks(d, r))}: {d!r} / {r!r}"


def test_ain_is_never_left_in_output():
    for src in ["ذریعہ", "بعد", "شعر", "شروع", "تعلیم"]:
        d, r = ak_engine.transliterate(src)
        assert not _leaks(d, r), f"ain leaked: {d!r} / {r!r}"


def test_tatweel_is_transparent():
    assert ak_engine.transliterate("کاـم") == ak_engine.transliterate("کام")


def test_arabic_indic_digits_become_ascii():
    d, r = ak_engine.transliterate("سال ۲۰۲۴")
    assert "2024" in d and "2024" in r


# ---------------------------------------------------------------------------
# shared curated dictionary is exact (dictionary tier, identical to rule)
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
    assert ak_engine.transliterate(src) == expected


def test_curated_dictionary_is_shared_with_rule_engine():
    # every curated/lexicon word must transliterate identically in both engines
    for src in ["راستے", "فاصلے", "منزلوں", "ذریعہ", "کہنا", "کتاب", "ہوتا"]:
        assert ak_engine.transliterate(src) == rule_engine.transliterate(src)


# ---------------------------------------------------------------------------
# the divergence from the rule engine: a fully-vowelled word (harakat) is read
# by Aksharamukha, not the heuristic -- this is the one case it is better at.
# Curated keys are un-vowelled, so کِتاب / مُحَبَّت miss the dictionary and go
# down the Aksharamukha path.
# ---------------------------------------------------------------------------

def test_aksharamukha_drives_fully_vowelled_words():
    d1, r1 = ak_engine.transliterate("کِتاب")
    assert (d1, r1) == ("किताब", "kitaab")
    d2, r2 = ak_engine.transliterate("مُحَبَّت")   # heuristic would say "mahabbat"
    assert (d2, r2) == ("मुहब्बत", "muhabbat")
    assert not _leaks(d1, r1, d2, r2)


def test_vowelled_word_reading_beats_the_default_a_heuristic():
    ak_d, ak_r = ak_engine.transliterate("مُحَبَّت")
    rule_d, rule_r = rule_engine.transliterate("مُحَبَّت")
    assert (ak_d, ak_r) != (rule_d, rule_r)
    assert "u" in ak_r  # Aksharamukha restored the short u; the heuristic did not


def test_plain_unvowelled_oov_words_match_the_rule_engine():
    # no harakat -> rule engine's character fallback, byte-for-byte. This is
    # what keeps whole-sentence parity with GPT identical to the rule engine.
    for src in ["پڑوسی", "کھڑکی", "قلمکار", "کمپیوٹر", "ہے۔اور"]:
        assert ak_engine.transliterate(src) == rule_engine.transliterate(src)


# ---------------------------------------------------------------------------
# Urdu sentence punctuation renders per script (۔ -> danda / period), and the
# mark must never leak -- even glued to a word
# ---------------------------------------------------------------------------

def test_urdu_full_stop_is_danda_in_devanagari_and_period_in_roman():
    deva, roman = ak_engine.transliterate("یہ گھر ہے۔")
    assert deva == "ये घर है।"
    assert roman == "ye ghar hai."


def test_arabic_comma_and_question_mark_stay_ascii_in_both_scripts():
    deva, roman = ak_engine.transliterate("کیا؟ ہاں،")
    assert deva == "क्या? हाँ,"
    assert roman == "kya? haan,"


def test_urdu_full_stop_glued_to_a_word_still_never_leaks():
    deva, roman = ak_engine.transliterate("ہے۔اور")
    assert not _leaks(deva, roman)
    assert "।" in deva and "." in roman


# ---------------------------------------------------------------------------
# line breaks are structural (poetry / OCR line output) -- kept
# ---------------------------------------------------------------------------

def test_newlines_are_preserved():
    deva, roman = ak_engine.transliterate("گھر\nہے")
    assert deva == "घर\nहै"
    assert roman == "ghar\nhai"


# ---------------------------------------------------------------------------
# whole-sentence parity with the GPT-4o reference column (README compare
# view). Same reference strings the rule engine is pinned to.
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
    _, roman = ak_engine.transliterate(_REF_SENTENCE)
    assert roman == _REF_ROMAN


def test_reference_sentence_devanagari_matches_gpt():
    deva, _ = ak_engine.transliterate(_REF_SENTENCE)
    assert deva == _REF_DEVA


def test_reference_sentence_has_no_perso_arabic_leaks():
    deva, roman = ak_engine.transliterate(_REF_SENTENCE)
    assert not _leaks(deva, roman)


# ---------------------------------------------------------------------------
# whole-ghazal parity with the GPT-4o reference column. Line breaks kept.
# ---------------------------------------------------------------------------

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
    deva, _ = ak_engine.transliterate(_GHAZAL_SRC)
    assert deva == _GHAZAL_DEVA


def test_ghazal_roman_matches_gpt():
    _, roman = ak_engine.transliterate(_GHAZAL_SRC)
    assert roman == _GHAZAL_ROMAN


def test_ghazal_has_no_perso_arabic_leaks():
    deva, roman = ak_engine.transliterate(_GHAZAL_SRC)
    assert not _leaks(deva, roman)
