# -*- coding: utf-8 -*-
"""Aksharamukha-backed transliteration engine.

Aksharamukha is a script-conversion library. Its ``Urdu`` reader is excellent
on fully-vowelled text but on ordinary un-vowelled Urdu it is rough: it leaks
the leading consonant of most words, restores no short vowels, voices ``ع``/``ء``
as a spurious ``अ``/``a`` and turns ``ح``/``ہ`` into ``ः``. Shipping that next to
the GPT column is the embarrassment this module removes.

The fix is the same one the rule engine uses: run every token through the
*shared* curated dictionary + bundled lexicon + per-script punctuation layer
(``transliterate.transliterate_with``), and only hand the residual
out-of-vocabulary words to Aksharamukha -- then repair whatever it leaks so no
Perso-Arabic codepoint ever survives into the output. On clean text every
content word is curated, so the result matches the GPT-4o compare column
byte-for-byte, exactly like the rule engine; the two engines diverge only on
rare OOV words, which is what the compare view is there to show.
"""
from __future__ import annotations

import re
import unicodedata

import transliterate as _rule

try:
    from aksharamukha import transliterate as _ak
except Exception:  # noqa: BLE001 - any import failure means "not installed"
    _ak = None

_PERSO = re.compile(r"[؀-ۿ]")

# Short-vowel marks (tanwin, zabar/zer/pesh, superscript alef -- not the
# vowel-less shadda/sukun). Their presence is the signal that Aksharamukha's
# reading is worth taking: fully-vowelled Urdu is the one case it is genuinely
# better at. Everything else goes to the rule engine's character fallback,
# which restores schwas, treats ع/ء as a silent seat and renders interior
# Perso-Arabic punctuation per script -- none of which Aksharamukha does.
_HARAKAT = re.compile("[ً-ِٰ]")

# Aksharamukha voices these as a bare vowel mid-word; the rule engine treats
# them as a silent seat. Strip before the call so Aksharamukha can't add the
# spurious अ / a. Word-initially they *are* a vowel seat -> map to alif.
_DROP_BEFORE_AK = str.maketrans("", "", "عء")  # ع ء
_INITIAL_SEAT = str.maketrans("عء", "اا")

# Leaked Perso-Arabic -> Devanagari / Roman, reusing the rule engine's own
# per-letter tables so the two engines agree on the letter values. A leaked
# consonant is voiced with its inherent "a" (Devanagari) / bare (Roman);
# anything Perso-Arabic with no mapping is dropped rather than left to leak.
_EXTRA_DEVA = {
    "ا": "ा", "آ": "आ", "و": "ो",
    "ی": "ी", "ے": "े", "ں": "ं",
    "ھ": "", "ة": "ह", "ۃ": "ह",
    "ك": "क", "ي": "ी",
}
_EXTRA_ROMAN = {
    "ا": "a", "آ": "aa", "و": "o", "ی": "i", "ے": "e",
    "ں": "n", "ھ": "h", "ة": "h", "ۃ": "h",
    "ك": "k", "ي": "i",
}
_P2D = {u: d for u, (d, _r) in _rule.CONSONANTS.items()} | _EXTRA_DEVA
_P2R = {u: r for u, (_d, r) in _rule.CONSONANTS.items()} | _EXTRA_ROMAN

# IAST diacritics Aksharamukha emits -> the rule engine's plain-ASCII house
# style. Applied to NFC text; residual combining marks are then stripped.
_IAST_MAP = {
    "ā": "aa", "ī": "i", "ū": "u", "ē": "e", "ō": "o",
    "ṃ": "n", "ṅ": "n", "ñ": "n", "ṇ": "n",
    "ṛ": "r", "ṭ": "t", "ḍ": "d",
    "ś": "sh", "ṣ": "sh", "ḥ": "h", "ġ": "gh",
}

# IAST -> the Rekhta-style diacritic house style: ā ī ū ṛ ṭ ḍ ġ are already
# right, so only the nasals, sibilants and mid vowels are remapped; macrons
# and retroflex underdots are kept (not NFD-stripped like _deaccent_roman).
_IAST_REKHTA_MAP = {
    "ē": "e", "ō": "o",
    "ṃ": "ñ", "ṅ": "ñ", "ñ": "ñ", "ṇ": "ñ",
    "ś": "sh", "ṣ": "sh", "ḥ": "h",
}


def _strip_perso(s: str, table: dict) -> str:
    """Replace every leaked Perso-Arabic codepoint via ``table``; drop any that
    has no mapping. Guarantees the result is Perso-Arabic-free."""
    return "".join(
        table[ch] if ch in table else ("" if _PERSO.match(ch) else ch)
        for ch in s
    )


def _deaccent_roman(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = "".join(_IAST_MAP.get(ch, ch) for ch in s)
    s = s.replace("ch", "\x00").replace("c", "ch").replace("\x00", "chh")
    s = "".join(ch for ch in unicodedata.normalize("NFD", s)
                if not unicodedata.combining(ch))
    s = s.replace("_", "").replace("ʼ", "").replace("'", "")
    return re.sub(r"\s+", " ", s).strip().lower()


def _reaccent_roman(s: str) -> str:
    """Like ``_deaccent_roman`` but keeps the Rekhta-style marks (ā ī ū ṛ ṭ ḍ
    ġ ñ): remap only the nasals / sibilants / mid vowels, and drop stray
    *combining* marks without decomposing the precomposed letters we want."""
    s = unicodedata.normalize("NFC", s)
    s = "".join(_IAST_REKHTA_MAP.get(ch, ch) for ch in s)
    s = s.replace("ch", "\x00").replace("c", "ch").replace("\x00", "chh")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("_", "").replace("ʼ", "").replace("'", "")
    return re.sub(r"\s+", " ", s).strip().lower()


def _ak_word(word: str, style: str = "plain"):
    """OOV fallback. A fully-vowelled word is read by Aksharamukha (and the
    result repaired so it never leaks); every other word -- the common case --
    goes to the rule engine's character rules, which Aksharamukha cannot match
    on schwa restoration, silent ع/ء and interior punctuation. Any Aksharamukha
    error or empty/garbled result also falls back to the rule engine.

    ``style`` is forwarded to the rule fallback; the Aksharamukha IAST reading
    is mapped to the Rekhta-style marks instead of being stripped."""
    if _ak is None or not _HARAKAT.search(word):
        return _rule._transliterate_word_rule_based(word, style)
    # mid-word ع/ء is a silent seat (strip it); word-initial ع/ء is a vowel
    # seat -- hand it to Aksharamukha as an alif so the vowel still renders.
    seed = (word[:1].translate(_INITIAL_SEAT)
            + word[1:].translate(_DROP_BEFORE_AK))
    if not seed:
        return _rule._transliterate_word_rule_based(word, style)
    clean = _reaccent_roman if style == "diacritic" else _deaccent_roman
    try:
        deva = _strip_perso(_ak.process("Urdu", "Devanagari", seed), _P2D)
        roman = _strip_perso(clean(_ak.process("Urdu", "IAST", seed)), _P2R)
    except Exception:  # noqa: BLE001 - Aksharamukha failure is not our crash
        return _rule._transliterate_word_rule_based(word, style)
    if not deva.strip() or not roman.strip():
        return _rule._transliterate_word_rule_based(word, style)
    return deva, roman


def transliterate(text: str, style: str = "plain"):
    """Main entry point. Returns ``(devanagari_text, roman_text)``."""
    return _rule.transliterate_with(
        lambda w: _ak_word(w, style), text, style=style)
