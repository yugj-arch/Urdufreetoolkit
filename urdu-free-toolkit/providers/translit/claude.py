# -*- coding: utf-8 -*-
"""Anthropic Claude as a transliteration provider (text-only path).

Context-aware vowel restoration, like the GPT / Gemini providers — resolves
Urdu's unwritten short vowels from sentence context instead of a dictionary.
"""
from __future__ import annotations

from providers import _anthropic_common as claude
from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslitOpts,
    TranslitResult,
)


class ClaudeTranslit(BaseProvider):
    info = ProviderInfo(
        id="claude",
        label="Anthropic Claude",
        capability=Capability.TRANSLIT,
        kind="api",
        needs=["ANTHROPIC_API_KEY"],
        note="Context-aware vowel restoration.",
        price="≈ $0.01 / run",
    )

    def available(self) -> tuple[bool, str]:
        try:
            import anthropic  # noqa: F401
        except Exception:  # noqa: BLE001
            return (False, "pip install anthropic")
        return (True, "") if claude.have_key() else (False, "ANTHROPIC_API_KEY not set")

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        t = self._timed(claude.generate_json, claude.TEXT_SYSTEM,
                        [claude.text_block(text)])
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error=t["error"], ms=t["ms"])
        d = t["value"]
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari=(d.get("devanagari") or "").strip(),
                              roman=(d.get("roman") or "").strip(),
                              roman_diacritic=(d.get("roman_diacritic") or "").strip())


PROVIDER = ClaudeTranslit()
