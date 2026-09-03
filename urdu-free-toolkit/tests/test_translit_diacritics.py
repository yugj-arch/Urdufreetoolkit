# -*- coding: utf-8 -*-
"""Diacritic ("Rekhta-style") Roman output.

Two layers:

* ``transliterate(text, style="diacritic")`` — the offline engine marks the
  Roman side the way the Rekhta poetry site does (long vowels ``ā ī ū``, ``ñ``
  for nūn-ġunna, ``ḳh`` for خ, ``ġ`` for غ, ``ṭ ḍ ṛ`` for the retroflexes).
  Devanagari is untouched, and the default ``style="plain"`` stays byte-for-byte
  what it always was (guarded by the reference-sentence tests in
  ``test_translit_rule.py``).

* every transliteration provider fills BOTH ``roman`` (plain) and
  ``roman_diacritic`` on one call, so the UI toggle can swap between them with
  no re-run.
"""
import pytest

import transliterate as rule_engine
from providers.base import TranslitOpts
from providers.translit.rule import PROVIDER


def dia(text: str) -> str:
    _, roman = rule_engine.transliterate(text, style="diacritic")
    return roman


# ---------------------------------------------------------------------------
# character fallback: marks come from letter identity, so they are exact
# ---------------------------------------------------------------------------

def test_long_alif_becomes_macron_a():
    assert dia("ناز") == "nāz"


def test_khe_is_k_underdot_h_distinct_from_aspirated_kh():
    assert dia("خراب") == "ḳharāb"


def test_ghain_is_g_with_dot():
    assert dia("غزل") == "ġazal"


def test_retroflex_aspirate_keeps_its_underdot():
    # ٹھیک: ٹھ -> ṭh (retroflex + aspiration), ی -> ī
    assert dia("ٹھیک") == "ṭhīk"


def test_noon_ghunna_in_fallback_is_n_tilde():
    assert dia("مہرباں") == "maharbāñ"


def test_retroflex_long_vowels_and_nasal_together():
    assert dia("لڑکیاں") == "laṛkīāñ"


# ---------------------------------------------------------------------------
# curated / lexicon tier: conservative transform on the pre-baked ASCII value
# ---------------------------------------------------------------------------

def test_curated_word_nasalises_when_source_ends_in_noon_ghunna():
    assert dia("ہیں") == "haiñ"
    assert dia("اپنوں") == "apnoñ"


def test_word_ending_in_plain_noon_is_not_nasalised():
    d, r = rule_engine.transliterate("اذان", style="diacritic")
    assert not r.endswith("ñ")


def test_curated_long_vowels_are_lengthened():
    assert dia("ہاں") == "hāñ"          # "haan" -> "hān" -> nasal "hāñ"
    assert dia("دور") == "dūr"          # curated "door"


# ---------------------------------------------------------------------------
# Devanagari is never touched by the Roman style
# ---------------------------------------------------------------------------

def test_devanagari_identical_across_styles():
    src = "خراب آنکھ لڑکیاں ہیں"
    plain_d, _ = rule_engine.transliterate(src, style="plain")
    dia_d, _ = rule_engine.transliterate(src, style="diacritic")
    assert plain_d == dia_d


# ---------------------------------------------------------------------------
# default is plain and completely unchanged
# ---------------------------------------------------------------------------

def test_default_style_is_plain():
    assert rule_engine.transliterate("خراب") == ("ख़राब", "kharaab")


def test_explicit_plain_matches_default():
    assert (rule_engine.transliterate("لڑکیاں ہیں")
            == rule_engine.transliterate("لڑکیاں ہیں", style="plain"))


def test_no_perso_arabic_leaks_in_diacritic_mode():
    perso = range(0x0600, 0x0700)
    for src in ["خراب", "لڑکیاں", "مہرباں", "ٹھیک", "دیا", "گئے", "کوئی"]:
        d, r = rule_engine.transliterate(src, style="diacritic")
        assert not [c for c in d + r if ord(c) in perso], f"leak in {src!r}: {d!r}/{r!r}"


# ---------------------------------------------------------------------------
# every provider fills BOTH roman spellings on one call
# ---------------------------------------------------------------------------

def test_rule_provider_returns_both_roman_spellings():
    r = PROVIDER.translit("خراب", TranslitOpts())
    assert r.ok
    assert r.roman == "kharaab"
    assert r.roman_diacritic == "ḳharāb"
    assert r.devanagari == "ख़राब"


from providers.translit import _uroman_engine as _ur      # noqa: E402
from providers.translit import _aksharamukha_engine as _ak  # noqa: E402
from providers.translit.uroman_p import PROVIDER as UROMAN   # noqa: E402
from providers.translit.aksharamukha_p import PROVIDER as AKSHARA  # noqa: E402

_uroman_missing = pytest.mark.skipif(_ur._UR is None, reason="uroman not installed")
_akshara_missing = pytest.mark.skipif(_ak._ak is None, reason="aksharamukha not installed")


@_uroman_missing
def test_uroman_engine_style_kwarg():
    assert _ur.transliterate("ہیں", style="diacritic") == ("हैं", "haiñ")
    assert _ur.transliterate("ہیں") == ("हैं", "hain")


@_uroman_missing
def test_uroman_provider_returns_both_roman_spellings():
    r = UROMAN.translit("خراب", TranslitOpts())
    assert r.ok
    assert r.roman == "khrab"              # uroman's skeleton reading
    assert r.roman_diacritic == "ḳharāb"   # rule engine's diacritic fallback


@_akshara_missing
def test_aksharamukha_engine_style_kwarg():
    assert _ak.transliterate("ہیں", style="diacritic") == ("हैं", "haiñ")
    assert _ak.transliterate("ہیں") == ("हैं", "hain")


@_akshara_missing
def test_aksharamukha_provider_returns_both_roman_spellings():
    r = AKSHARA.translit("لڑکیاں", TranslitOpts())
    assert r.ok
    assert r.roman == "larkiaan"
    assert r.roman_diacritic == "laṛkīāñ"


# ---------------------------------------------------------------------------
# the LLM providers ask for both spellings in one prompt and parse both
# ---------------------------------------------------------------------------

import json as _json      # noqa: E402
import types as _types    # noqa: E402

_PAYLOAD = {"devanagari": "मैं ठीक हूँ", "roman": "main theek hoon",
            "roman_diacritic": "maiñ ṭhīk hūñ", "notes": ""}


def test_openai_text_prompt_asks_for_both_spellings():
    import providers._openai_common as common
    assert "roman_diacritic" in common.TEXT_SYSTEM
    assert "ā" in common.TEXT_SYSTEM and "ñ" in common.TEXT_SYSTEM


def test_gpt_translit_parses_both_spellings(monkeypatch):
    import providers._openai_common as common

    class _Msg:
        content = _json.dumps(_PAYLOAD)

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Chat:
        class completions:
            @staticmethod
            def create(**kw):
                return _Resp()

    monkeypatch.setattr(common, "get_client",
                        lambda: _types.SimpleNamespace(chat=_Chat()))
    monkeypatch.setattr(common, "have_key", lambda: True)
    from providers.translit.gpt import PROVIDER as GPT
    r = GPT.translit("میں ٹھیک ہوں", TranslitOpts())
    assert r.roman == "main theek hoon"
    assert r.roman_diacritic == "maiñ ṭhīk hūñ"


def test_claude_translit_parses_both_spellings(monkeypatch):
    import providers._anthropic_common as common

    class _Block:
        type = "text"
        text = _json.dumps(_PAYLOAD)

    class _Resp:
        content = [_Block()]

    class _Messages:
        @staticmethod
        def create(**kw):
            return _Resp()

    monkeypatch.setattr(common, "get_client",
                        lambda: _types.SimpleNamespace(messages=_Messages()))
    monkeypatch.setattr(common, "have_key", lambda: True)
    from providers.translit.claude import PROVIDER as CLAUDE
    r = CLAUDE.translit("میں ٹھیک ہوں", TranslitOpts())
    assert r.roman == "main theek hoon"
    assert r.roman_diacritic == "maiñ ṭhīk hūñ"
