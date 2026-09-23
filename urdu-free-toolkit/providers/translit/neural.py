# -*- coding: utf-8 -*-
"""Trained neural transliterator as a provider (offline, free).

Wraps ``neural_translit.py``: curated words -> 23k-word Wiktionary gold
lexicon -> a character-level Transformer trained on ~700k human-authored
Urdu romanisations (``training/translit/``). Returns Devanagari plus BOTH
Roman spellings (plain + Rekhta-style diacritic) from one reading, so the
UI's ā toggle never disagrees with itself.
"""
from __future__ import annotations

from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslitOpts,
    TranslitResult,
)


class NeuralTranslit(BaseProvider):
    info = ProviderInfo(
        id="neural",
        label="Urdu Neural (offline)",
        capability=Capability.TRANSLIT,
        kind="offline",
        note="Trained model + 23k-word gold lexicon. Free, offline, never drops a line.",
        price="Free · offline",
    )

    def available(self) -> tuple[bool, str]:
        import neural_translit
        if not neural_translit.LEXICON_PATH.exists():
            return False, "model files missing (data/translit_model/)"
        return True, ""

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        def _run():
            import neural_translit
            return neural_translit.transliterate(text)

        t = self._timed(_run)
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error=t["error"], ms=t["ms"])
        deva, roman, roman_dia = t["value"]
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari=deva, roman=roman, roman_diacritic=roman_dia)


PROVIDER = NeuralTranslit()
