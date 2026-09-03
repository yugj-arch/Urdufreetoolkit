# -*- coding: utf-8 -*-
"""Aksharamukha as a transliteration provider (offline).

Aksharamukha's ``Urdu`` script reader alone is rough on un-vowelled Urdu -- it
leaks the leading consonant of most words and restores no short vowels. This
provider runs it under the same shared machinery the rule engine uses (curated
dictionary + bundled lexicon + per-script punctuation) and repairs whatever
Aksharamukha leaks, so on clean text the output matches the GPT compare column
byte-for-byte. See ``_aksharamukha_engine`` for the details.
"""
from __future__ import annotations

from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslitOpts,
    TranslitResult,
)
from providers.translit import _aksharamukha_engine as engine


class AksharamukhaTranslit(BaseProvider):
    info = ProviderInfo(
        id="aksharamukha",
        label="Aksharamukha (offline)",
        capability=Capability.TRANSLIT,
        kind="offline",
        note="Aksharamukha script model + shared curated dictionary. "
             "Matches GPT on clean text; diverges only on rare words.",
        price="Free · offline",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if engine._ak is not None else (False, "pip install aksharamukha")

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        if engine._ak is None:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error="aksharamukha not installed")
        def _both():
            deva, roman = engine.transliterate(text, style="plain")
            _, roman_dia = engine.transliterate(text, style="diacritic")
            return deva, roman, roman_dia

        t = self._timed(_both)
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error=t["error"], ms=t["ms"])
        deva, roman, roman_dia = t["value"]
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari=deva, roman=roman, roman_diacritic=roman_dia)


PROVIDER = AksharamukhaTranslit()
