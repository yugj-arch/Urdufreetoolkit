# -*- coding: utf-8 -*-
"""uroman as a transliteration provider (offline).

uroman is a universal romanizer with no short-vowel restoration (``mhbt``, not
``mohabbat``). This provider keeps that skeleton reading for out-of-vocabulary
words but runs every token through the shared curated dictionary + per-script
punctuation layer, so on clean text the output matches the GPT compare column
in both scripts. Devanagari for OOV words comes from the rule engine, since
uroman produces none. See ``_uroman_engine``.
"""
from __future__ import annotations

from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslitOpts,
    TranslitResult,
)
from providers.translit import _uroman_engine as engine


class UromanTranslit(BaseProvider):
    info = ProviderInfo(
        id="uroman",
        label="uroman (offline)",
        capability=Capability.TRANSLIT,
        kind="offline",
        note="Universal romanizer + shared curated dictionary. Matches GPT on "
             "clean text; raw consonant skeleton (mhbt) on rare words.",
        price="Free · offline",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if engine._UR is not None else (False, "pip install uroman")

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        if engine._UR is None:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error="uroman not installed")
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


PROVIDER = UromanTranslit()
