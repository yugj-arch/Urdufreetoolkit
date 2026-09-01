# -*- coding: utf-8 -*-
"""Anthropic Claude vision as an OCR provider.

One vision call transcribes the Urdu *and* returns Devanagari + Roman in the
same response (the model restores short vowels from context). The extra
transliteration is stashed in ``meta`` so the compare UI can show it without a
second call — same contract as the GPT / Gemini OCR providers.
"""
from __future__ import annotations

from providers import _anthropic_common as claude
from providers._openai_common import prepare_image
from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo


class ClaudeOcr(BaseProvider):
    info = ProviderInfo(
        id="claude",
        label="Anthropic Claude vision",
        capability=Capability.OCR,
        kind="api",
        needs=["ANTHROPIC_API_KEY"],
        note="Strong on Nastaliq; restores short vowels from context.",
        price="≈ $0.02 / image",
    )

    def available(self) -> tuple[bool, str]:
        try:
            import anthropic  # noqa: F401
        except Exception:  # noqa: BLE001
            return (False, "pip install anthropic")
        return (True, "") if claude.have_key() else (False, "ANTHROPIC_API_KEY not set")

    def ocr(self, image: bytes) -> OcrResult:
        def _call():
            b64, mime = prepare_image(image)
            return claude.generate_json(claude.VISION_SYSTEM, [
                claude.image_block(b64, mime),
                claude.text_block("Transcribe and transliterate the Urdu in this image."),
            ])

        t = self._timed(_call)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        d = t["value"]
        return OcrResult(
            provider_id=self.info.id, ok=True, ms=t["ms"],
            text=(d.get("urdu") or "").strip(),
            notes=(d.get("notes") or "").strip(),
            meta={
                "model": claude.MODEL,
                "devanagari": (d.get("devanagari") or "").strip(),
                "roman": (d.get("roman") or "").strip(),
            },
        )


PROVIDER = ClaudeOcr()
