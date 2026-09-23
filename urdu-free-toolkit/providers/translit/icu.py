# -*- coding: utf-8 -*-
"""ICU transliteration as a provider (offline, Roman only).

Uses ICU's built-in ``Arabic-Latin`` transform. It maps letters without
inventing short vowels. No Devanagari output.
"""
from __future__ import annotations

from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslitOpts,
    TranslitResult,
)

try:
    import icu

    _TR = icu.Transliterator.createInstance("Arabic-Latin")
except Exception:  # noqa: BLE001
    _TR = None


class IcuTranslit(BaseProvider):
    info = ProviderInfo(
        id="icu",
        label="ICU Arabic-Latin (offline)",
        capability=Capability.TRANSLIT,
        kind="offline",
        note="ICU transform. Roman only, letter-level, no vowel restoration.",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if _TR is not None else (False, "pip install PyICU")

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        if _TR is None:
            return TranslitResult(provider_id=self.info.id, ok=False, error="PyICU not installed")
        t = self._timed(_TR.transliterate, text)
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error=t["error"], ms=t["ms"])
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari="", roman=(t["value"] or "").strip())


PROVIDER = IcuTranslit()
