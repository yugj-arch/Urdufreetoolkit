# -*- coding: utf-8 -*-
"""uroman as a transliteration provider (offline, Roman only).

uroman is a universal romanizer. It maps the Urdu letters faithfully but does
not invent the short vowels Urdu omits, so ``محبت`` comes out ``mhbt`` rather
than ``mohabbat``. Useful as a deterministic consonant-skeleton reference in the
compare view. It produces no Devanagari.
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
    import uroman as _uroman_mod

    _UR = _uroman_mod.Uroman()
except Exception:  # noqa: BLE001
    _UR = None


class UromanTranslit(BaseProvider):
    info = ProviderInfo(
        id="uroman",
        label="uroman (offline)",
        capability=Capability.TRANSLIT,
        kind="offline",
        note="Universal romanizer. Roman only, no vowel restoration (mhbt, not mohabbat).",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if _UR is not None else (False, "pip install uroman")

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        if _UR is None:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error="uroman not installed")
        t = self._timed(_UR.romanize_string, text)
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error=t["error"], ms=t["ms"])
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari="", roman=(t["value"] or "").strip())


PROVIDER = UromanTranslit()
