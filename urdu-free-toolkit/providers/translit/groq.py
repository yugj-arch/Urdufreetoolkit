# -*- coding: utf-8 -*-
"""Groq (Llama 3.3 70B) as a transliteration provider — free, context-aware.

Same context-aware vowel restoration as the GPT / Gemini providers, served on
Groq's free tier. Get a key at console.groq.com.
"""
from __future__ import annotations

from providers import _groq_common as gq
from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslitOpts,
    TranslitResult,
)


class GroqTranslit(BaseProvider):
    info = ProviderInfo(
        id="groq",
        label="Groq Llama 3.3 70B (free tier)",
        capability=Capability.TRANSLIT,
        kind="api",
        needs=["GROQ_API_KEY"],
        note="Free key from console.groq.com. Context-aware vowel restoration.",
        price="Free tier",
    )

    def available(self) -> tuple[bool, str]:
        try:
            import groq  # noqa: F401  (the SDK, not this module)
        except Exception:  # noqa: BLE001
            return (False, "pip install groq")
        return (True, "") if gq.have_key() else (False, "GROQ_API_KEY not set")

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        t = self._timed(gq.generate_json, gq.TEXT_SYSTEM, text)
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error=t["error"], ms=t["ms"])
        d = t["value"]
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari=(d.get("devanagari") or "").strip(),
                              roman=(d.get("roman") or "").strip())


PROVIDER = GroqTranslit()
