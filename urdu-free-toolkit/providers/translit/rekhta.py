# -*- coding: utf-8 -*-
"""Rekhta-style transliteration as a provider (offline, free).

Wraps ``rekhta_translit.py``: Rekhta Labs' published Urdu->Hindi poetry model
and our line model read each misra-sized run of words in context (izafat,
conjunctive و, poetic forms), a poem's lines are checked against its meter,
and words are spelt as rekhta.org spells them. Roman comes in rekhta.org's
two spellings, as its own Roman toggle has them -- the simple one
(``dil-e-nadan tujhe hua kya hai``) as the plain spelling and the marked one
(``dil-e-nādāñ tujhe huā kyā hai``) behind the ā toggle.
"""
from __future__ import annotations

from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslitOpts,
    TranslitResult,
)


class RekhtaTranslit(BaseProvider):
    info = ProviderInfo(
        id="rekhta",
        label="DotSyndicate Fine-tune (offline)",
        capability=Capability.TRANSLIT,
        kind="offline",
        note="Rekhta Labs' poetry transliterator, checked by meter and spelt as rekhta.org spells. "
             "Devanagari as Rekhta writes it; Roman as rekhta.org prints it (simple, or marked: ā ī ū, ḳh ġh, ñ).",
        price="Free · offline",
    )

    def available(self) -> tuple[bool, str]:
        # runs on torch when installed, else on numpy (the Vercel build)
        try:
            import numpy  # noqa: F401
        except ImportError:
            return False, "needs numpy"
        import rekhta_translit
        from urdu_nn.rekhta_models import model_dir
        if not (rekhta_translit.MODEL_PATH.exists() or model_dir("ur2hi")):
            return False, "model files missing (data/rekhta_model/)"
        return True, ""

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        def _run():
            import rekhta_translit
            return rekhta_translit.transliterate(text)

        t = self._timed(_run)
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error=t["error"], ms=t["ms"])
        deva, roman, roman_dia = t["value"]
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari=deva, roman=roman, roman_diacritic=roman_dia)


PROVIDER = RekhtaTranslit()
