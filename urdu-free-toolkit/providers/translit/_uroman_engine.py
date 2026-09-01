# -*- coding: utf-8 -*-
"""uroman-backed transliteration engine.

uroman romanizes every Urdu letter but restores no short vowels
(``محبت`` -> ``mhbt``). This module keeps that skeleton reading as the
out-of-vocabulary behaviour -- the honest baseline the compare view is for --
while routing every token through the shared curated dictionary + bundled
lexicon + per-script punctuation layer (``transliterate.transliterate_with``),
so on clean text the output matches the GPT-4o compare column byte-for-byte in
both scripts. uroman produces no Devanagari, so the Devanagari side of an OOV
word comes from the rule engine's character fallback.
"""
from __future__ import annotations

import re
import unicodedata

import transliterate as _rule

try:
    import uroman as _uroman_mod

    _UR = _uroman_mod.Uroman()
except Exception:  # noqa: BLE001 - any import failure means "not installed"
    _UR = None

_PERSO = re.compile(r"[؀-ۿ]")


def _clean_roman(s: str) -> str:
    """uroman's romanization -> the rule engine's plain-ASCII house style: drop
    the ع/ء apostrophes and connector underscores, strip any combining marks,
    lower-case, collapse whitespace."""
    s = "".join(ch for ch in unicodedata.normalize("NFD", s)
                if not unicodedata.combining(ch))
    s = s.replace("ʼ", "").replace("'", "").replace("_", "")
    return re.sub(r"\s+", " ", s).strip().lower()


def _uroman_word(word: str):
    """OOV fallback: Devanagari from the rule engine's character rules (uroman
    has none), Roman from uroman's skeleton reading. Falls back to the rule
    engine's Roman too if uroman errors or returns nothing usable."""
    deva, rule_roman = _rule._transliterate_word_rule_based(word)
    if _UR is None:
        return deva, rule_roman
    try:
        roman = _clean_roman(_UR.romanize_string(word))
    except Exception:  # noqa: BLE001 - uroman failure is not our crash
        return deva, rule_roman
    if not roman or _PERSO.search(roman):
        roman = rule_roman
    return deva, roman


def transliterate(text: str):
    """Main entry point. Returns ``(devanagari_text, roman_text)``."""
    return _rule.transliterate_with(_uroman_word, text)
