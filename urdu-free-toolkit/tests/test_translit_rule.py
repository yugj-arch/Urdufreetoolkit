import gzip
import json

import pytest

import transliterate as rule_engine
from providers.translit.rule import PROVIDER
from providers.base import TranslitOpts, Capability

PERSO_ARABIC = range(0x0600, 0x0700)


def _leaks(*strings):
    """Perso-Arabic codepoints that leaked through into a transliteration."""
    return {c for s in strings for c in s if ord(c) in PERSO_ARABIC}


# ---------------------------------------------------------------------------
# provider wiring (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_info():
    assert PROVIDER.info.id == "rule"
    assert PROVIDER.info.capability == Capability.TRANSLIT
    assert PROVIDER.available() == (True, "")


def test_translit_known_words():
    r = PROVIDER.translit("میں ٹھیک ہوں", TranslitOpts())
    assert r.ok is True
    assert "में" in r.devanagari or "मैं" in r.devanagari  # rule engine defaults to "mein"
    assert r.roman.startswith(("mein", "main"))
    assert r.ms >= 0


# ---------------------------------------------------------------------------
# no Perso-Arabic character may ever survive into the output
# ---------------------------------------------------------------------------

TRICKY_WORDS = [
    "دیا", "آیا", "لڑکیاں",          # orphan long-vowel letters after a vowel
    "لائے", "گئے", "کوئی", "آؤ", "رئیس",  # hamza carriers (ء ئ ؤ)
    "کاـم", "کتـاب",                  # tatweel / kashida filler
    "سال ۲۰۲۴", "صفحہ ١٢٣",           # Arabic-Indic + Extended digits
    "جملہ،",                          # Arabic comma glued to a token
]


@pytest.mark.parametrize("src", TRICKY_WORDS)
def test_no_perso_arabic_leaks_into_output(src):
    d, r = rule_engine.transliterate(src)
    assert not _leaks(d, r), f"leaked {sorted(_leaks(d, r))}: {d!r} / {r!r}"


def test_tatweel_is_transparent():
    assert rule_engine.transliterate("کاـم") == rule_engine.transliterate("کام")


def test_arabic_indic_digits_become_ascii():
    d, r = rule_engine.transliterate("سال ۲۰۲۴")
    assert "2024" in d and "2024" in r


def test_shadda_geminates_consonant():
    d, r = rule_engine.transliterate("مکّہ")
    assert "kk" in r
    assert not _leaks(d, r)


# ---------------------------------------------------------------------------
# curated vocabulary is exact (dictionary tier)
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
    assert rule_engine.transliterate(src) == expected


def test_eid_mubarak_headline():
    assert rule_engine.transliterate("عید مبارک") == ("ईद मुबारक", "eid mubaarak")


# ---------------------------------------------------------------------------
# bundled lexicon tier
# ---------------------------------------------------------------------------

def test_load_lexicon_missing_file_is_not_fatal(tmp_path):
    assert rule_engine._load_lexicon(tmp_path / "does-not-exist.json.gz") == {}


def test_load_lexicon_reads_gzipped_json(tmp_path):
    p = tmp_path / "lex.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump({"فلانی": ["फ़लानी", "falaani"]}, fh, ensure_ascii=False)
    lex = rule_engine._load_lexicon(p)
    assert lex["فلانی"] == ("फ़लानी", "falaani")


def test_lexicon_resolves_words_the_heuristic_would_miss(monkeypatch):
    monkeypatch.setitem(rule_engine._LEXICON, "زقندہ", ("ज़क़ंदा", "zaqanda"))
    assert rule_engine.transliterate("زقندہ") == ("ज़क़ंदा", "zaqanda")


def test_curated_dictionary_wins_over_lexicon(monkeypatch):
    monkeypatch.setitem(rule_engine._LEXICON, "اور", ("ग़लत", "ghalat"))
    assert rule_engine.transliterate("اور") == ("और", "aur")


# ---------------------------------------------------------------------------
# ع (ain) must be voiced or dropped -- never left raw in the output
# (regression: mid-word ع was absent from every table and leaked verbatim,
#  e.g. ذریعہ -> "ज़रीعह" / "zariعh")
# ---------------------------------------------------------------------------

AIN_WORDS = ["ذریعہ", "بعد", "شعر", "جمع", "موقع", "شروع", "بعض"]


@pytest.mark.parametrize("src", AIN_WORDS)
def test_ain_is_never_left_in_output(src):
    d, r = rule_engine.transliterate(src)
    assert not _leaks(d, r), f"ain leaked: {d!r} / {r!r}"


# ---------------------------------------------------------------------------
# Roman schwa-deletion in the character fallback (applies to *new* words,
# not dictionary entries). Pattern  nucleus C ə C V  ->  the medial ə is
# dropped in Roman. The first syllable's schwa is always kept. Devanagari
# is left exactly as before -- readers apply the deletion themselves.
# ---------------------------------------------------------------------------

def test_roman_medial_schwa_is_deleted():
    # چمکتے: was "chamakate" -- the ک–ت schwa should drop -> "chamakte"
    _, roman = rule_engine.transliterate("چمکتے")
    assert roman == "chamakte"


def test_roman_first_syllable_schwa_is_kept():
    # نکلتے: the ن–ک schwa stays, only the ل–ت one drops -> "nakalte"
    _, roman = rule_engine.transliterate("نکلتے")
    assert roman == "nakalte"


def test_schwa_deletion_does_not_touch_devanagari():
    deva, _ = rule_engine.transliterate("چمکتے")
    assert deva == "चमकते"
    assert not _leaks(deva)


def test_curated_word_is_not_reprocessed_by_schwa_deletion():
    # کہنا is curated as "kehna"; the bare fallback would say "kahna"
    assert rule_engine.transliterate("کہنا") == ("कहना", "kehna")


# ---------------------------------------------------------------------------
# curated vocabulary for the GPT-4o reference comparison sentence
# ---------------------------------------------------------------------------

SENTENCE_VOCAB = {
    "راستے": ("रास्ते", "raaste"),
    "فاصلے": ("फ़ासले", "faasle"),
    "فرق": ("फ़र्क", "farq"),
    "طے": ("तय", "tay"),
    "کرنے": ("करने", "karne"),
    "سمیٹنے": ("समेटने", "sametne"),
    "پڑتے": ("पड़ते", "padte"),
    "منزلوں": ("मंज़िलों", "manzilon"),
    "سبب": ("सबब", "sabab"),
    "بنتے": ("बनते", "bante"),
    "دور": ("दूर", "door"),
    "ذریعہ": ("ज़रिया", "zariya"),
    "اپنوں": ("अपनों", "apnon"),
    "ہوتا": ("होता", "hota"),
}


@pytest.mark.parametrize("src,expected", list(SENTENCE_VOCAB.items()))
def test_sentence_vocab_is_exact(src, expected):
    assert rule_engine.transliterate(src) == expected


def test_colloquial_pronouns_match_reference():
    assert rule_engine.transliterate("وہ") == ("वो", "wo")
    assert rule_engine.transliterate("یہ") == ("ये", "ye")


# ---------------------------------------------------------------------------
# Urdu sentence punctuation renders per script: ۔ (Urdu full stop) becomes a
# Devanagari danda in Hindi text but a plain period in Roman; ، ؛ ؟ stay
# ASCII in both. The mark must never leak, even glued to a word.
# ---------------------------------------------------------------------------

def test_urdu_full_stop_is_danda_in_devanagari_and_period_in_roman():
    deva, roman = rule_engine.transliterate("یہ گھر ہے۔")
    assert deva == "ये घर है।"
    assert roman == "ye ghar hai."


def test_danda_never_leaks_into_roman_and_period_never_into_devanagari():
    deva, roman = rule_engine.transliterate("چلو۔ اب سو جاؤ۔")
    assert "।" not in roman and "." not in deva
    assert deva.count("।") == 2 and roman.count(".") == 2


def test_arabic_comma_and_question_mark_stay_ascii_in_both_scripts():
    deva, roman = rule_engine.transliterate("کیا؟ ہاں،")
    assert deva == "क्या? हाँ,"
    assert roman == "kya? haan,"


def test_urdu_full_stop_glued_to_a_word_still_never_leaks():
    deva, roman = rule_engine.transliterate("ہے۔اور")
    assert not _leaks(deva, roman)
    assert "।" in deva and "." in roman


# ---------------------------------------------------------------------------
# whole-sentence parity with the GPT-4o reference column (README compare
# view). Both scripts are expected to match the reference exactly.
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
    _, roman = rule_engine.transliterate(_REF_SENTENCE)
    assert roman == _REF_ROMAN


def test_reference_sentence_devanagari_matches_gpt():
    deva, _ = rule_engine.transliterate(_REF_SENTENCE)
    assert deva == _REF_DEVA


def test_reference_sentence_has_no_perso_arabic_leaks():
    deva, roman = rule_engine.transliterate(_REF_SENTENCE)
    assert not _leaks(deva, roman)
