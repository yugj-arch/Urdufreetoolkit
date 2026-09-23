# -*- coding: utf-8 -*-
"""Google Gemini as a transliteration provider (free API tier).

Context-aware, like the GPT provider — resolves Urdu's unwritten short vowels
from sentence context.
"""
from __future__ import annotations

from providers import _gemini_common as gem
from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslitOpts,
    TranslitResult,
)


class GeminiTranslit(BaseProvider):
    info = ProviderInfo(
        id="gemini",
        label="Google Gemini (free tier)",
        capability=Capability.TRANSLIT,
        kind="api",
        needs=["GOOGLE_API_KEY"],
        note="Free key from aistudio.google.com. Context-aware vowel restoration.",
        price="Free tier",
    )

    def available(self) -> tuple[bool, str]:
        try:
            import google.genai  # noqa: F401
        except Exception:  # noqa: BLE001
            return (False, "pip install google-genai")
        return (True, "") if gem.have_key() else (False, "GOOGLE_API_KEY not set")

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        t = self._timed(gem.generate_json, gem.TEXT_SYSTEM, [text])
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error=t["error"], ms=t["ms"])
        d = t["value"]
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari=(d.get("devanagari") or "").strip(),
                              roman=(d.get("roman") or "").strip(),
                              roman_diacritic=(d.get("roman_diacritic") or "").strip())


PROVIDER = GeminiTranslit()
