# -*- coding: utf-8 -*-
"""Rekhta-style transliteration as a provider (offline, free).

Wraps ``rekhta_translit.py``: our line model, distilled from Rekhta Labs'
published Urdu->Hindi poetry model, reads each misra-sized run of words in
context (izafat, conjunctive و, poetic forms); Roman comes from the same
reading in Rekhta's two schemes -- the ASCII table (``KHauf-e-rasan``,
``aañkh``, ``pa.Dhaa.ii``) as the plain spelling and the diacritic one
(``ḳhauf-e-rasan``, ``āñkh``) behind the ā toggle.
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
        label="Rekhta-style (offline)",
        capability=Capability.TRANSLIT,
        kind="offline",
        note="Line model distilled from Rekhta Labs' poetry transliterator. "
             "Devanagari as Rekhta writes it; Roman in Rekhta's scheme (aa ii uu, KH G, T D, .D, ñ).",
        price="Free · offline",
    )

    def available(self) -> tuple[bool, str]:
        import rekhta_translit
        from urdu_nn import rekhta_teacher
        teacher = rekhta_teacher.MODELS / "ur-2-hi" / rekhta_teacher.FILES["ur2hi"][2]
        if not (rekhta_translit.MODEL_PATH.exists() or teacher.exists()):
            return False, "model files missing (data/rekhta_model/)"
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "needs torch"
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
