# -*- coding: utf-8 -*-
"""OpenAI gpt-image-1 as an in-place render provider: erase the Urdu in the
uploaded image and paint the given (already transliterated) lines back where it
was. Ported from the old top-level ``imgedit.py``.

This is an AI re-render, not a pixel-level patch: layout and colour match well,
but fine details can shift and the typeface is a generic sans. Needs
``gpt-image-1`` access on the API account (OpenAI may require org verification).
"""
from __future__ import annotations

import base64
import io
import os

from PIL import Image

from providers import _openai_common as common
from providers.base import BaseProvider, Capability, ProviderInfo, RenderResult

IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
IMAGE_QUALITY = os.environ.get("OPENAI_IMAGE_QUALITY", "high")  # low | medium | high
_MAX_DIM = 2048

_EDIT_PROMPT = """In this image, replace ONLY the Urdu text with the text below. Change nothing else.

New text, one entry per line of Urdu, top to bottom:
---
{lines}
---

Placement rules - follow exactly:
- Completely erase the original Urdu script first - no leftover strokes, dots, or shadows.
- Each entry above replaces exactly one line of Urdu and must stay on ONE line. Do NOT wrap it, do NOT split it, do NOT add lines.
- Keep the SAME bounding area as the Urdu it replaces: same top edge, same baseline, same horizontal span, same alignment (if the Urdu was centred, keep it centred).
- The transliteration is usually LONGER than the Urdu. If it does not fit, shrink the font size just enough that the whole line fits inside the original horizontal span. Never let any text touch or cross the image edge. Never clip a word.
- Match the original text colour exactly. Use a clean, legible sans-serif of similar weight.
- Keep the background, photos, graphics, colours, borders, lighting and layout pixel-for-pixel as they are outside the text.
- Write the text VERBATIM. Do NOT translate. Keep the spelling exactly as given.
"""


def _png_for_upload(image_bytes: bytes) -> bytes:
    """Validate, flatten to RGB, downscale if huge, return PNG bytes."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Not a readable image: {e}")
    img = img.convert("RGB")
    if max(img.size) > _MAX_DIM:
        img.thumbnail((_MAX_DIM, _MAX_DIM), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class GptImageRender(BaseProvider):
    info = ProviderInfo(
        id="gpt_image",
        label="OpenAI gpt-image-1",
        capability=Capability.RENDER,
        kind="api",
        needs=["OPENAI_API_KEY"],
        note="AI re-render; layout can drift. ~4-8 cents/image, 30-90s.",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if common.have_key() else (False, "OPENAI_API_KEY not set")

    def render(self, image: bytes, lines: list[str]) -> RenderResult:
        def _call():
            orig_size = Image.open(io.BytesIO(image)).size
            png_in = _png_for_upload(image)
            client = common.get_client()
            resp = client.images.edit(
                model=IMAGE_MODEL,
                image=("image.png", png_in, "image/png"),
                prompt=_EDIT_PROMPT.format(lines="\n".join(lines)),
                input_fidelity="high",
                size="auto",
                quality=IMAGE_QUALITY,
                output_format="png",
                timeout=240,
            )
            out_bytes = base64.b64decode(resp.data[0].b64_json)
            edited = Image.open(io.BytesIO(out_bytes)).convert("RGB")
            if edited.size != orig_size:
                edited = edited.resize(orig_size, Image.LANCZOS)
            buf = io.BytesIO()
            edited.save(buf, format="PNG")
            return buf.getvalue()

        t = self._timed(_call)
        if not t["ok"]:
            err = t["error"]
            if "403" in err or "not available" in err.lower():
                err += (f". This account likely needs organisation verification to "
                        f"use {IMAGE_MODEL}.")
            return RenderResult(provider_id=self.info.id, ok=False, error=err, ms=t["ms"])
        return RenderResult(provider_id=self.info.id, ok=True, ms=t["ms"], png=t["value"])


PROVIDER = GptImageRender()
