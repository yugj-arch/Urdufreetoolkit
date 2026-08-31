# -*- coding: utf-8 -*-
"""deep-translator as a translation provider (free, needs internet, no key).

Wraps the free Google Translate web endpoint. No API key, but it does call out
to the network, so it is marked as an API-kind provider with the "free (net)"
badge.
"""
from __future__ import annotations

from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslateOpts,
    TranslateResult,
)

try:
    from deep_translator import GoogleTranslator
except Exception:  # noqa: BLE001
    GoogleTranslator = None


class DeepTranslate(BaseProvider):
    info = ProviderInfo(
        id="deep_translator",
        label="deep-translator (Google, free)",
        capability=Capability.TRANSLATE,
        kind="api",
        needs=[],
        note="Free Google endpoint. No key, but needs internet.",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if GoogleTranslator is not None else (False, "pip install deep-translator")

    def translate(self, text: str, opts: TranslateOpts) -> TranslateResult:
        if GoogleTranslator is None:
            return TranslateResult(provider_id=self.info.id, ok=False,
                                   error="deep-translator not installed")

        def _call():
            out = {}
            if "english" in opts.targets:
                out["english"] = GoogleTranslator(source="ur", target="en").translate(text)
            if "hindi" in opts.targets:
                out["hindi"] = GoogleTranslator(source="ur", target="hi").translate(text)
            return out

        t = self._timed(_call)
        if not t["ok"]:
            return TranslateResult(provider_id=self.info.id, ok=False,
                                   error=t["error"], ms=t["ms"])
        v = t["value"]
        return TranslateResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                               english=v.get("english", ""), hindi=v.get("hindi", ""))


PROVIDER = DeepTranslate()
