# -*- coding: utf-8 -*-
"""Google Gemini vision as an OCR provider (free API tier).

Currently the most accurate reader of Urdu Nastaliq available at no cost — get a
key free at aistudio.google.com and paste it in Settings. One call transcribes
the Urdu and returns Devanagari + Roman (short vowels restored from context),
stashed in ``meta`` like the GPT provider.
"""
from __future__ import annotations

from providers import _gemini_common as gem
from providers._openai_common import prepare_image
from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo


class GeminiOcr(BaseProvider):
    info = ProviderInfo(
        id="gemini",
        label="Google Gemini vision (free tier)",
        capability=Capability.OCR,
        kind="api",
        needs=["GOOGLE_API_KEY"],
        note="Free key from aistudio.google.com. Top accuracy on Nastaliq; restores vowels.",
        price="Free tier · ≈ $0.003 / image",
    )

    def available(self) -> tuple[bool, str]:
        try:
            import google.genai  # noqa: F401
        except Exception:  # noqa: BLE001
            return (False, "pip install google-genai")
        return (True, "") if gem.have_key() else (False, "GOOGLE_API_KEY not set")

    def ocr(self, image: bytes) -> OcrResult:
        def _call():
            import base64

            b64, mime = prepare_image(image)
            part = gem.image_part(base64.b64decode(b64), mime)
            return gem.generate_json(
                gem.VISION_SYSTEM,
                ["Transcribe and transliterate the Urdu in this image.", part],
            )

        t = self._timed(_call)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        d = t["value"]
        return OcrResult(
            provider_id=self.info.id, ok=True, ms=t["ms"],
            text=(d.get("urdu") or "").strip(),
            notes=(d.get("notes") or "").strip(),
            meta={"model": gem.MODEL,
                  "devanagari": (d.get("devanagari") or "").strip(),
                  "roman": (d.get("roman") or "").strip()},
        )


PROVIDER = GeminiOcr()
