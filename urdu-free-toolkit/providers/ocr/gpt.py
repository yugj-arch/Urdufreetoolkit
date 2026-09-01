# -*- coding: utf-8 -*-
"""OpenAI GPT-4o vision as an OCR provider.

One vision call transcribes the Urdu *and* returns Devanagari + Roman in the
same response (the model restores short vowels from context). The extra
transliteration is stashed in ``meta`` so the compare UI can show it without a
second call.
"""
from __future__ import annotations

from providers import _openai_common as common
from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo


class GptOcr(BaseProvider):
    info = ProviderInfo(
        id="gpt",
        label="OpenAI GPT vision",
        capability=Capability.OCR,
        kind="api",
        needs=["OPENAI_API_KEY"],
        note="Best overall; restores short vowels from context.",
        price="≈ $0.012 / image",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if common.have_key() else (False, "OPENAI_API_KEY not set")

    def ocr(self, image: bytes) -> OcrResult:
        def _call():
            b64, mime = common.prepare_image(image)
            client = common.get_client()
            resp = client.chat.completions.create(
                model=common.MODEL,
                response_format={"type": "json_object"},
                **common.sampling_kwargs(),
                messages=[
                    {"role": "system", "content": common.VISION_SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text",
                         "text": "Transcribe and transliterate the Urdu in this image."},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ]},
                ],
            )
            return common.parse_json(resp.choices[0].message.content)

        t = self._timed(_call)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        d = t["value"]
        return OcrResult(
            provider_id=self.info.id, ok=True, ms=t["ms"],
            text=(d.get("urdu") or "").strip(),
            notes=(d.get("notes") or "").strip(),
            meta={
                "model": common.MODEL,
                "devanagari": (d.get("devanagari") or "").strip(),
                "roman": (d.get("roman") or "").strip(),
            },
        )


PROVIDER = GptOcr()
