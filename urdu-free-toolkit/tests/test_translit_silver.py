# -*- coding: utf-8 -*-
"""Silver training pairs for the neural transliterator: Devanagari -> R
readings (urdu_nn.scheme.deva_to_rich_candidates) and the Urdu x Hindi join
helpers in training/translit/silver.py."""
from __future__ import annotations

from collections import Counter

import pytest

from training.translit.silver import pick_reading, spell_with_nuktas
from urdu_nn.scheme import deva_to_rich_candidates


@pytest.mark.parametrize("deva, rich", [
    ("किताब", "kitāb"),
    ("करती", "kartī"),          # medial schwa dropped: V C _ C V
    ("हिचकता", "hicaktā"),      # ... right to left, so only the second one
    ("लहंगे", "lahaṅge"),       # a nasalised schwa is heard; homorganic ṅ
    ("संभालिये", "sambhāliye"),
    ("संयोग", "sa~yog"),        # anusvara before a semivowel: nasal vowel
    ("वंश", "vañś"),
    ("फाँसी", "phā~sī"),        # candrabindu: nasal vowel ...
    ("तरूँगी", "tarūṅgī"),      # ... but a velar nasal before g
    ("हैं", "hai~"),
    ("कई", "kaī"),              # schwa heard before a vowel letter
    ("दुःख", "duhkh"),
    ("क़ानून", "qānūn"),
])
def test_default_reading(deva, rich):
    assert deva_to_rich_candidates(deva)[0] == rich


def test_other_schwa_readings_are_offered():
    # the default rule says adālto~; people say adaalaton
    assert "adālato~" in deva_to_rich_candidates("अदालतों")


def test_unmodelled_text_gives_nothing():
    assert deva_to_rich_candidates("abc") == []
    assert deva_to_rich_candidates("दो शब्द") == []


def test_nuktas_come_from_the_urdu_letters():
    assert spell_with_nuktas("قانون", "कानून") == "क़ानून"
    assert spell_with_nuktas("کام", "काम") == "काम"             # nothing to restore
    assert spell_with_nuktas("کتاب", "मकान") is None


def test_reading_follows_human_romanisations():
    assert pick_reading("عدالتوں", "अदालतों", Counter({"adaalaton": 1.0})) == ("adālato~", 1.0)


def test_no_match_keeps_the_default_reading():
    # "olsen" is textually nearer "olsn" than "olsan"; neither matches exactly
    assert pick_reading("اولسن", "ओलसन", Counter({"olsen": 1.0}))[0] == "olsan"


def test_final_silent_he_is_a_short_a():
    assert pick_reading("افسانہ", "अफ़साना", Counter())[0] == "afsāna"
