# -*- coding: utf-8 -*-
"""Aksharamukha as a transliteration provider (offline).

Aksharamukha is a script-conversion library. On fully-vowelled Urdu (with
harakat) it is excellent; on ordinary un-vowelled Urdu it leaves gaps and is
noticeably weaker than the rule engine — which is exactly the kind of thing the
compare view is meant to surface.
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
    from aksharamukha import transliterate as _ak
except Exception:  # noqa: BLE001 - any import failure means "not installed"
    _ak = None

_ROMAN = {"natural": "HK", "iso": "ISO", "iast": "IAST"}


class AksharamukhaTranslit(BaseProvider):
    info = ProviderInfo(
        id="aksharamukha",
        label="Aksharamukha (offline)",
        capability=Capability.TRANSLIT,
        kind="offline",
        note="Script-mapping library. Best with harakat; weaker on plain Urdu.",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if _ak is not None else (False, "pip install aksharamukha")

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        if _ak is None:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error="aksharamukha not installed")

        def _call():
            deva = _ak.process("Urdu", "Devanagari", text)
            roman = _ak.process("Urdu", _ROMAN.get(opts.roman_style, "IAST"), text)
            return deva, roman

        t = self._timed(_call)
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error=t["error"], ms=t["ms"])
        deva, roman = t["value"]
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari=deva, roman=roman)


PROVIDER = AksharamukhaTranslit()
