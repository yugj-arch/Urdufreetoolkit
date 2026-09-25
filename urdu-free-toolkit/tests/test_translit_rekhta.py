# -*- coding: utf-8 -*-
"""Rekhta-style engine: the line cutter, Rekhta's two Roman schemes, the
Devanagari -> R reader, the teacher runtime and the engine end to end."""
from __future__ import annotations

import pytest

from urdu_nn import rekhta_roman as rr
from urdu_nn.rekhta_text import MAX_RUN, segments, urdu_runs


# -- cutting lines into model runs --------------------------------------------

def test_segments_keep_punctuation_digits_and_latin_out_of_runs():
    assert segments("دل ہی تو ہے، نہ سنگ و خشت؟") == [
        ("u", "دل ہی تو ہے"), ("x", "، "), ("u", "نہ سنگ و خشت"), ("x", "؟")]
    assert segments("وہ 1947ء میں آئے۔") == [
        ("u", "وہ"), ("x", " 1947ء "), ("u", "میں آئے"), ("x", "۔")]
    assert segments("hello دنیا") == [("x", "hello "), ("u", "دنیا")]


def test_segments_fold_letters_and_keep_izafat_zer():
    # Arabic yeh/kaf fold; harakat drop except a word-final zer (izafat)
    assert urdu_runs("دلِ ناداں تجھے ہوا كيا ہے") == ["دلِ ناداں تجھے ہوا کیا ہے"]
    assert urdu_runs("مُحَبَّت") == ["محبت"]


def test_long_runs_are_cut_into_balanced_misra_sized_pieces():
    words = ("یہ ایک بہت لمبا جملہ ہے جس میں بہت سارے الفاظ ہیں اور "
             "یہ ختم ہونے کا نام ہی نہیں لیتا بلکہ چلتا رہتا ہے").split()
    runs = urdu_runs(" ".join(words))
    assert len(runs) > 1 and all(len(r) <= MAX_RUN for r in runs)
    assert " ".join(runs).split() == words
    assert min(len(r) for r in runs) > MAX_RUN // 3       # no one-word scrap at the end


# -- Rekhta's ASCII table (aa ii uu, KH G, T D, .D, ñ) ------------------------

@pytest.mark.parametrize("rich,ascii_", [
    # every row of Rekhta's table, as R -> ascii
    ("ab", "ab"), ("āg", "aag"), ("ibādat", "ibaadat"), ("jīt", "jiit"), ("pānī", "paanii"),
    ("uns", "uns"), ("dūr", "duur"), ("ek", "ek"), ("sitāre", "sitaare"), ("aiś", "aish"),
    ("maidān", "maidaan"), ("dost", "dost"), ("aur", "aur"), ("nau", "nau"),
    ("ā~c", "aañch"), ("samā~", "samaañ"), ("bharosā", "bharosaa"), ("kabhī", "kabhii"),
    ("jībh", "jiibh"), ("cain", "chain"), ("soc", "soch"), ("chat", "chhat"),
    ("bichaunā", "bichhaunaa"), ("kuch", "kuchh"), ("dāman", "daaman"), ("dhokā", "dhokaa"),
    ("dūdh", "duudh"), ("ḍor", "Dor"), ("suḍaul", "suDaul"), ("ghamaṇḍ", "ghamanD"),
    ("ḍhol", "Dhol"), ("niḍhāl", "niDhaal"), ("peṛ", "pe.D"), ("beṛā", "be.Daa"),
    ("gaṛh", "ga.Dh"), ("fanā", "fanaa"), ("ghar", "ghar"), ("ġam", "Gam"),
    ("kāġaz", "kaaGaz"), ("dāġ", "daaG"), ("xat", "KHat"), ("zaxm", "zaKHm"),
    ("surx", "surKH"), ("ṭolī", "Tolii"), ("maṭak", "maTak"), ("āhaṭ", "aahaT"),
    ("ṭhokar", "Thokar"), ("uṭhān", "uThaan"), ("pīṭh", "piiTh"), ("thakan", "thakan"),
    ("vādā", "vaadaa"), ("uzv", "uzv"), ("qayāmat", "qayaamat"), ("žālā", "zhaalaa"),
    ("aždahā", "azhdahaa"), ("śak", "shak"), ("iśq", "ishq"), ("hai~", "haiñ"),
])
def test_ascii_render_matches_rekhtas_table(rich, ascii_):
    assert rr.render(rich, "ascii") == ascii_


def test_ascii_render_marks_hiatus_izafat_and_ain():
    assert rr.render("paṛhāī", "ascii") == "pa.Dhaa.ii"       # the table's own example
    assert rr.render("koī", "ascii") == "ko.ii"
    assert rr.render("āe kyū~", "ascii") == "aa.e kyuuñ"
    assert rr.render("xauf-e-rasan", "ascii") == "KHauf-e-rasan"
    assert rr.render("saṅg-o-xiśt", "ascii") == "sang-o-KHisht"
    assert rr.render("e'lān", "ascii") == "e'laan"


def test_diacritic_render_is_rekhtas_marked_roman():
    assert rr.render("xauf-e-rasan", "rekhta") == "ḳhauf-e-rasan"
    assert rr.render("āe kyū~", "rekhta") == "ā.e kyūñ"
    assert rr.render("paṛhāī kāġaz", "rekhta") == "paṛhā.ī kāġaz"


# -- Devanagari -> R ------------------------------------------------------------

def test_deva_units_izafat_is_a_joiner_and_conjunctive_o_a_word():
    assert rr.deva_units("ख़ौफ़-ए-रसन ले") == [["ख़ौफ़", "-e-"], ["रसन", " "], ["ले", " "]]
    assert rr.deva_units("संग-ओ-ख़िश्त") == [["संग", "-"], ["ओ", "-"], ["ख़िश्त", " "]]


@pytest.mark.parametrize("deva,urdu,rich", [
    ("ख़ौफ़", "خوف", "xauf"),
    ("आँख", "آنکھ", "ā~kh"),            # candrabindu: a nasal vowel, not āṅkh
    ("ए'लान", "اعلان", "e'lān"),         # ain apostrophe splits the syllables
    ("अफ़्साना", "افسانہ", "afsāna"),      # silent he: short final a
    ("पढ़ाई", "پڑھائی", "paṛhāī"),
])
def test_reader_reads_rekhta_devanagari(deva, urdu, rich):
    assert rr.Reader().read(deva, urdu) == rich


def test_reader_lets_human_evidence_pick_the_schwa():
    ev = {"وسوسے": {"waswase": 3.0}}
    assert rr.Reader().read("वसवसे", "وسوسے") == "vasavse"      # bare Hindi schwa rule
    assert rr.Reader(evidence=ev).read("वसवसे", "وسوسے") == "vasvase"


# -- distillation data ---------------------------------------------------------

def test_dataset_drops_looping_or_misread_teacher_labels():
    from training.rekhta.dataset import check
    good = {"ur": "دل ہی تو ہے", "hi": "दिल ही तो है", "back": "دل ہی تو ہے"}
    assert check(good, 0.85) == ""
    looped = {"ur": "جو کو", "hi": "जो को को को को", "back": "جو کو کو کو کو"}
    assert check(looped, 0.85) == "words"
    misread = {"ur": "مکر", "hi": "मुकर्रर", "back": "مقرر"}
    assert check(misread, 0.85) == "roundtrip"
    izafat = {"ur": "خوف رسن", "hi": "ख़ौफ़-ए-रसन", "back": "خوف رسن"}
    assert check(izafat, 0.85) == ""                     # -ए- is a joiner, not a word


@pytest.mark.parametrize("urdu,gold,rekhta", [
    ("یہاں", "यहां", "यहाँ"),        # nasal vowel with no top matra -> candrabindu
    ("ہوں", "हूं", "हूँ"),
    ("میں", "में", "में"),           # top matra: anusvara stays
    ("قانون", "कानून", "क़ानून"),     # nukta where the Urdu letter calls for it
])
def test_gold_words_are_rewritten_in_rekhtas_conventions(urdu, gold, rekhta):
    from training.rekhta.dataset import rekhta_conventions
    assert rekhta_conventions(urdu, gold) == rekhta


# -- the teacher and the engine (need the downloaded models) -------------------

@pytest.fixture(scope="module")
def teacher():
    from urdu_nn import rekhta_teacher as rt
    if not (rt.MODELS / "ur-2-hi" / rt.FILES["ur2hi"][2]).exists():
        pytest.skip("Rekhta models not downloaded (data/rekhta_models/)")
    return rt.Teacher("ur2hi", device="cpu")


def test_batched_teacher_reads_like_the_model_card(teacher):
    lines = ["وسوسے دل میں نہ رکھ خوف رسن لے کے نہ چل", "محبت", "کتاب پڑھ رہا ہوں"]
    # the model card's own output for its example line
    assert teacher(lines)[0] == "वसवसे दिल में न रख ख़ौफ़-ए-रसन ले के न चल"
    # padding a batch must not change a line's reading
    assert teacher(lines) == [teacher([l])[0] for l in lines]


def test_engine_end_to_end(teacher):
    import rekhta_translit
    eng = rekhta_translit.RekhtaTransliterator(use_teacher=True)
    r = eng.transliterate_full("دل ہی تو ہے نہ سنگ و خشت، درد سے بھر نہ آئے کیوں؟\nوہ 1947ء میں آئے۔")
    deva, ascii_, dia = r["devanagari"].split("\n"), r["roman"].split("\n"), r["roman_diacritic"].split("\n")
    assert deva[0] == "दिल ही तो है न संग-ओ-ख़िश्त, दर्द से भर न आए क्यूँ?"
    assert ascii_[0] == "dil hii to hai na sang-o-KHisht, dard se bhar na aa.e kyuuñ?"
    assert dia[0] == "dil hī to hai na sang-o-ḳhisht, dard se bhar na ā.e kyūñ?"
    assert deva[1].startswith("वो 1947 में") and deva[1].endswith("।")     # में, not मैं
    assert ascii_[1].endswith(".")


def test_provider_is_registered():
    from providers import registry
    from providers.base import Capability
    assert registry.get(Capability.TRANSLIT, "rekhta").info.kind == "offline"
