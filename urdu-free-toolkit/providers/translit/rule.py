# -*- coding: utf-8 -*-
"""Rule-based transliterator as a provider.

Wraps the hand-built engine in ``transliterate.py`` (dictionary of high-frequency
Hindustani words + a character-level heuristic fallback). Free, offline,
deterministic. It does not restore Urdu's unwritten short vowels beyond what the
dictionary covers — see the notes in ``transliterate.py``.
"""
from __future__ import annotations

from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslitOpts,
    TranslitResult,
)
import transliterate as _rule


class RuleTranslit(BaseProvider):
    info = ProviderInfo(
        id="rule",
        label="Rule engine (offline)",
        capability=Capability.TRANSLIT,
        kind="offline",
        note="Dictionary + heuristic. Free and deterministic; limited vowel restoration.",
        price="Free · offline",
    )

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        def _both():
            deva, roman = _rule.transliterate(text, style="plain")
            _, roman_dia = _rule.transliterate(text, style="diacritic")
            return deva, roman, roman_dia

        t = self._timed(_both)
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error=t["error"], ms=t["ms"])
        deva, roman, roman_dia = t["value"]
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari=deva, roman=roman, roman_diacritic=roman_dia)


PROVIDER = RuleTranslit()
