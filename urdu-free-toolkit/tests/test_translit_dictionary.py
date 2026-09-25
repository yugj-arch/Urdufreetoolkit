# -*- coding: utf-8 -*-
"""Exact transliteration dictionary (data/translit_dictionary.tsv): the file
itself, its builder, and how the rule and neural engines use it."""
from __future__ import annotations

import pytest

import transliterate
from training.translit import build_dictionary as bd

URDU = range(0x0600, 0x0700)


def _has_urdu(s: str) -> bool:
    return any(ord(ch) in URDU for ch in s)


# -- the shipped dictionary --------------------------------------------------

@pytest.fixture(scope="module")
def shipped():
    d = transliterate.load_exact_dictionary()
    if not d:
        pytest.skip("data/translit_dictionary.tsv not built")
    return d


def test_shipped_dictionary_is_complete_and_clean(shipped):
    assert len(shipped) >= 9000
    for urdu, (deva, plain, rekhta) in shipped.items():
        assert deva and plain and rekhta, urdu
        assert not _has_urdu(deva + plain + rekhta), urdu
        assert plain == plain.strip() and "\t" not in plain


@pytest.mark.parametrize("urdu,deva,plain,rekhta", [
    ("ایک", "एक", "ek", "ek"),                 # not the Pakistani "aik"
    ("عام", "आम", "aam", "ām"),                # ain-initial: no stray व
    ("کرتا", "करता", "karta", "kartā"),        # not कर्ता (doer)
    ("کافی", "काफ़ी", "kaafi", "kāfī"),        # enough -- not "coffee"
    ("جماعت", "जमात", "jamaat", "jamā'at"),    # dropped ain never triples a vowel
    ("شائع", "शाया", "shaya", "shā'e"),
    ("ہند", "हिंद", "hind", "hind"),            # not "india"
])
def test_shipped_dictionary_spot_checks(shipped, urdu, deva, plain, rekhta):
    assert shipped[urdu] == (deva, plain, rekhta)


@pytest.mark.parametrize("rich,plain", [
    ("ā gayā", "aa gaya"),          # آ on its own is "aa", not "a"
    ("dev", "dev"),                 # word-final و after a vowel stays v ...
    ("dabāv", "dabaav"),
    ("vaqt", "waqt"),               # ... elsewhere the casual w
    ("jamā'at", "jamaat"),          # a dropped ain never leaves "aaa"
])
def test_plain_render_rules(rich, plain):
    from urdu_nn.scheme import to_plain
    assert to_plain(rich) == plain


def test_nasal_normalisation():
    assert bd.norm_nasals("हिन्दी") == "हिंदी"
    assert bd.norm_nasals("नवम्बर") == "नवंबर"
    assert bd.norm_nasals("उन्हें") == "उन्हें"          # nh is not a class nasal
    assert bd.norm_nasals("जन्नत") == "जन्नत"


# -- builder -----------------------------------------------------------------

def test_build_applies_fixes(tmp_path, monkeypatch):
    draft = tmp_path / "draft.tsv"
    draft.write_text(
        "# header\n"
        "ایک\tऐक\taik\taik\taik\t9\tlexicon\t\n"
        "عام\tवाम\tām\taam\tām\t8\tlexicon\t\n"
        "ھ\tह\th\th\th\t7\tlexicon\t\n"
        "کتاب\tकिताब\tkitāb\tkitaab\tkitāb\t6\tcurated\t\n",
        encoding="utf-8")
    fixes = tmp_path / "fixes.txt"
    fixes.write_text(
        "# comment\n"
        "ایک | एक | ek\n"                 # new reading: renders plain + rekhta
        "عام | आम | .\n"                  # devanagari only
        "ھ | -\n"                          # dropped
        "کمپنی | कंपनी | kampanī | company\n",   # extra word with a plain override
        encoding="utf-8")
    out = tmp_path / "dict.tsv"
    monkeypatch.setattr(bd, "DRAFT", draft)
    monkeypatch.setattr(bd, "FIXES", fixes)
    monkeypatch.setattr(bd, "OUT", out)
    bd.build()
    got = transliterate.load_exact_dictionary(out)
    assert got == {
        "ایک": ("एक", "ek", "ek"),
        "عام": ("आम", "aam", "ām"),
        "کتاب": ("किताब", "kitaab", "kitāb"),
        "کمپنی": ("कंपनी", "company", "kampanī"),
    }
    assert list(got) == ["ایک", "عام", "کتاب", "کمپنی"]      # frequency order kept


def test_build_rejects_duplicate_fixes(tmp_path, monkeypatch):
    draft = tmp_path / "draft.tsv"
    draft.write_text("ایک\tऐक\taik\taik\taik\t9\tlexicon\t\n", encoding="utf-8")
    fixes = tmp_path / "fixes.txt"
    fixes.write_text("ایک | एक | ek\nایک | एक | ek\n", encoding="utf-8")
    monkeypatch.setattr(bd, "DRAFT", draft)
    monkeypatch.setattr(bd, "FIXES", fixes)
    monkeypatch.setattr(bd, "OUT", tmp_path / "dict.tsv")
    with pytest.raises(SystemExit):
        bd.build()


def test_loader_skips_comments_and_short_rows(tmp_path):
    p = tmp_path / "d.tsv"
    p.write_text("# c\nایک\tएक\tek\tek\nبری\tबुरी\n\nعام\tआम\taam\tām\n", encoding="utf-8")
    assert transliterate.load_exact_dictionary(p) == {
        "ایک": ("एक", "ek", "ek"), "عام": ("आम", "aam", "ām")}
    assert transliterate.load_exact_dictionary(tmp_path / "missing.tsv") == {}
    assert transliterate.load_exact_dictionary(None) == {}


# -- rule engine -------------------------------------------------------------

@pytest.fixture
def rule_dict(monkeypatch):
    exact = {transliterate._fold_key("ایک"): ("एक", "ek", "ek"),
             transliterate._fold_key("جماعت"): ("जमात", "jamaat", "jamā'at"),
             transliterate._fold_key("کتاب"): ("XX", "xx", "xx")}
    monkeypatch.setattr(transliterate, "_EXACT", exact)


def test_rule_engine_uses_dictionary_in_both_styles(rule_dict):
    assert transliterate.transliterate("ایک جماعت") == ("एक जमात", "ek jamaat")
    assert transliterate.transliterate("ایک جماعت", style="diacritic")[1] == "ek jamā'at"


def test_rule_engine_curated_words_still_win(rule_dict):
    assert transliterate.transliterate("کتاب")[1] != "xx"


def test_rule_engine_written_harakat_bypass_the_dictionary(monkeypatch):
    # the writer spelled the vowels out (حَسَن, not the dictionary's حسن = husn)
    monkeypatch.setattr(transliterate, "_EXACT",
                        {transliterate._fold_key("حسن"): ("हुस्न", "husn", "husn")})
    assert transliterate.transliterate("حسن")[1] == "husn"
    assert transliterate.transliterate("حَسَن")[1] != "husn"


# -- neural engine -----------------------------------------------------------

@pytest.fixture(scope="module")
def neural(tmp_path_factory):
    import neural_translit
    if not neural_translit.LEXICON_PATH.exists():
        pytest.skip("neural model files not built")
    p = tmp_path_factory.mktemp("d") / "dict.tsv"
    p.write_text("ایک\tएक\tek\tek\n"
                 "الدین\tअद-दीन\tad-deen\tad-dīn\n"
                 "میں\tमें\tmein\tmeñ\n"
                 "کافی\tकाफ़ी\tkaafi\tkāfī\n", encoding="utf-8")
    return neural_translit.NeuralTransliterator(
        model_path=tmp_path_factory.mktemp("m") / "missing.pt", dictionary_path=p,
        gpt_path=None)


def test_neural_uses_dictionary_verbatim(neural):
    assert neural.transliterate("ایک") == ("एक", "ek", "ek")
    assert neural.transliterate("کافی")[1] == "kaafi"       # English layer never overrides


def test_neural_context_rules_beat_dictionary(neural):
    assert neural.transliterate("میں نے کہا")[1].startswith("main ")
    assert neural.transliterate("وہ گھر میں ہے")[1].split()[2] == "mein"


def test_neural_dictionary_article_takes_u_after_a_word(neural):
    assert neural.transliterate("الدین")[1] == "ad-deen"
    deva, plain, dia = neural.transliterate("نصیر الدین")
    assert plain.endswith(" ud-deen") and dia.endswith(" ud-dīn") and deva.endswith(" उद-दीन")


def test_neural_ban_so_jald_context(neural):
    assert neural.transliterate("وہ بن گیا")[1].split()[1] == "ban"
    assert neural.transliterate("وہ سو گیا")[1].split()[1] == "so"
    assert neural.transliterate("جلد ہی")[1].startswith("jald ")
    assert neural.transliterate("بچہ جلد سو گیا")[1].split()[1] == "jald"


def test_neural_kal_kul_context(neural):
    assert neural.transliterate("وہ کل پاکستان جائے گا")[1].split()[1] == "kal"
    assert neural.transliterate("کل آبادی دس لاکھ تھی")[1].startswith("kul ")
