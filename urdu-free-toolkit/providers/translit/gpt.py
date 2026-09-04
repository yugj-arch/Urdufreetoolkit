# -*- coding: utf-8 -*-
"""OpenAI GPT-4o as a transliteration provider (text-only path).

This is the context-aware "tie-breaker": unlike the offline engines it can
resolve Urdu's unwritten short vowels from sentence context.
"""
from __future__ import annotations

from providers import _openai_common as common
from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslitOpts,
    TranslitResult,
)


class GptTranslit(BaseProvider):
    info = ProviderInfo(
        id="gpt",
        label="OpenAI GPT",
        capability=Capability.TRANSLIT,
        kind="api",
        needs=["OPENAI_API_KEY"],
        note="Context-aware vowel restoration -- the tie-breaker.",
        price="≈ $0.005 / image",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if common.have_key() else (False, "OPENAI_API_KEY not set")

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        def _call():
            client = common.get_client()
            resp = client.chat.completions.create(
                model=common.MODEL,
                response_format={"type": "json_object"},
                **common.sampling_kwargs(),
                messages=[
                    {"role": "system", "content": common.TEXT_SYSTEM},
                    {"role": "user", "content": text},
                ],
            )
            return common.parse_json(resp.choices[0].message.content)

        t = self._timed(_call)
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False,
                                  error=t["error"], ms=t["ms"])
        d = t["value"]
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari=(d.get("devanagari") or "").strip(),
                              roman=(d.get("roman") or "").strip(),
                              roman_diacritic=(d.get("roman_diacritic") or "").strip())


PROVIDER = GptTranslit()
