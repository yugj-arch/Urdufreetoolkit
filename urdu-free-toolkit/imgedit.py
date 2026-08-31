# -*- coding: utf-8 -*-
"""
imgedit.py -- an edited copy of the uploaded image with the Urdu text
replaced *in place* by its transliteration, using OpenAI's gpt-image-1
image-editing model.

Flow:
  1. ocr.extract_and_transliterate() reads the Urdu and produces the
     Roman / Devanagari lines (transliteration, NOT translation).
  2. gpt-image-1 is asked to erase the Urdu and paint that text back into
     the same spot, keeping everything else identical
     (input_fidelity="high").

This is an AI re-rendering of the image, not a pixel-level patch. It
handles textured backgrounds and matches layout / colour well, but fine
details can shift and it is not a forgery-grade copy of the original
typeface. Needs gpt-image-1 access on the API account (OpenAI may require
organisation verification to enable it).
"""

import base64
import io
import os

from PIL import Image

from ocr import _get_client, extract_and_transliterate

IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
IMAGE_QUALITY = os.environ.get("OPENAI_IMAGE_QUALITY", "high")  # low | medium | high
_MAX_DIM = 2048  # re-encode the upload no larger than this before sending

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
    except Exception as e:
        raise ValueError(f"Not a readable image: {e}")
    img = img.convert("RGB")
    if max(img.size) > _MAX_DIM:
        img.thumbnail((_MAX_DIM, _MAX_DIM), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_transliterated_image(image_bytes: bytes, script: str = "roman") -> bytes:
    """image_bytes -> PNG bytes with the Urdu replaced in place by the
    `script` ('roman' or 'devanagari') transliteration, via gpt-image-1."""
    if script not in ("roman", "devanagari"):
        raise ValueError("script must be 'roman' or 'devanagari'")

    try:
        orig_size = Image.open(io.BytesIO(image_bytes)).size
    except Exception as e:
        raise ValueError(f"Not a readable image: {e}")

    tr = extract_and_transliterate(image_bytes)
    lines = (tr.get(script) or "").strip()
    if not lines:
        raise RuntimeError("No Urdu text was found in the image to replace.")

    png_in = _png_for_upload(image_bytes)
    client = _get_client()

    try:
        resp = client.images.edit(
            model=IMAGE_MODEL,
            image=("image.png", png_in, "image/png"),
            prompt=_EDIT_PROMPT.format(lines=lines),
            input_fidelity="high",
            size="auto",
            quality=IMAGE_QUALITY,
            output_format="png",
            timeout=240,
        )
    except Exception as e:
        raise RuntimeError(
            f"{IMAGE_MODEL} image edit failed: {e}. If this is a 403 / "
            f"'model not available' error, the API account likely needs "
            f"organisation verification to use {IMAGE_MODEL}."
        )

    out_bytes = base64.b64decode(resp.data[0].b64_json)
    edited = Image.open(io.BytesIO(out_bytes)).convert("RGB")
    if edited.size != orig_size:
        edited = edited.resize(orig_size, Image.LANCZOS)
    buf = io.BytesIO()
    edited.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python imgedit.py <image_path> [roman|devanagari]")
        sys.exit(1)
    script = sys.argv[2] if len(sys.argv) > 2 else "roman"
    with open(sys.argv[1], "rb") as f:
        png = render_transliterated_image(f.read(), script)
    out_path = os.path.splitext(sys.argv[1])[0] + f".{script}.png"
    with open(out_path, "wb") as f:
        f.write(png)
    print("wrote", out_path)
