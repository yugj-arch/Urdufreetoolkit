# -*- coding: utf-8 -*-
"""
ocr.py -- image -> Urdu text -> Devanagari + Roman, in a single OpenAI
GPT-4o vision call.

This replaces the earlier free/offline Tesseract pipeline. Tesseract's
accuracy on Urdu Nastaliq (the cursive style Urdu is normally printed in)
was too low to be useful. A vision LLM reads Nastaliq far better and, in
the same call, restores the short vowels that Urdu script omits but that
Devanagari and Roman both require -- using sentence context rather than a
fixed heuristic.

Requires an OpenAI API key in the environment (OPENAI_API_KEY), loaded
from a local .env file. Costs a small amount per image (roughly 1-2 US
cents with gpt-4o).

Switching providers: change OPENAI_MODEL, or swap the two `_client` calls
for an Anthropic / Gemini client -- the JSON contract these functions
return is provider-agnostic.

Related: imgedit.py reuses OPENAI_MODEL / _get_client() from here to
produce an edited copy of the image with the transliteration burned in.
"""

import base64
import io
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
_MAX_IMAGE_DIM = 2048  # downscale the longer side before upload, to cap cost/latency

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Put it in a .env file next to app.py "
                "(OPENAI_API_KEY=sk-...) or export it in your environment."
            )
        _client = OpenAI()
    return _client


_VISION_SYSTEM = """You are an expert transcriber and transliterator of Urdu text.

You will be shown an image containing Urdu text. It may be in Nastaliq
(cursive calligraphic) style, may include English words or digits, and may
span multiple lines.

Do ALL of the following:
1. Transcribe the Urdu EXACTLY as written -- same words, same order, same
   line breaks. Do NOT translate, paraphrase, or "fix" spelling. Keep any
   English words and digits inline where they appear.
2. Transliterate that text into Devanagari (Hindi script). Restore the
   short vowels Urdu omits, using context and standard Hindustani
   pronunciation. Keep proper nouns recognizable.
3. Transliterate into Roman (natural Roman Urdu / Hindustani the way people
   actually type it -- e.g. "mohabbat", "kya haal hai"), short vowels
   restored. Plain ASCII letters only: use "aa", "ee", "oo", "n" -- never
   macrons or diacritics like a-with-bar, i-with-bar, or n-with-dot.
4. Briefly note anything blurry, cut off, or genuinely ambiguous.

Return ONLY a JSON object with exactly these keys:
{"urdu": "...", "devanagari": "...", "roman": "...", "notes": "..."}
Use \\n for line breaks inside the strings. "notes" is "" if nothing is unclear."""

_TEXT_SYSTEM = """You are an expert transliterator of Urdu text.

Given Urdu text, produce:
1. A Devanagari (Hindi-script) transliteration with the short vowels Urdu
   omits restored from context and standard Hindustani pronunciation.
2. A Roman transliteration (natural Roman Urdu / Hindustani the way people
   actually type it -- e.g. "mohabbat", "kya haal hai") with short vowels
   restored. Plain ASCII letters only: "aa", "ee", "oo", "n" -- never
   macrons or diacritics.
Do NOT translate. Preserve line breaks, English words, and digits.

Return ONLY a JSON object with exactly these keys:
{"devanagari": "...", "roman": "...", "notes": "..."}
"notes" is "" if nothing is unclear."""


def _prepare_image(image_bytes: bytes):
    """Validate, normalize, and (if large) downscale the image.
    Returns (base64_str, mime_type). Raises ValueError if it isn't a
    readable image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        raise ValueError(f"Not a readable image: {e}")

    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        fmt, mime = "PNG", "image/png"
    else:
        img = img.convert("RGB")
        fmt, mime = "JPEG", "image/jpeg"

    longest = max(img.size)
    if longest > _MAX_IMAGE_DIM:
        scale = _MAX_IMAGE_DIM / longest
        img = img.resize((round(img.width * scale), round(img.height * scale)))

    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=90)
    else:
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), mime


def _parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:
        raise RuntimeError(
            f"Model did not return valid JSON ({e}). Raw response: {content!r:.500}"
        )


def extract_and_transliterate(image_bytes: bytes) -> dict:
    """Image bytes -> {urdu, devanagari, roman, notes, model}.
    One GPT-4o vision call. Raises RuntimeError / ValueError on failure --
    there is no offline fallback, by design."""
    b64, mime = _prepare_image(image_bytes)
    client = _get_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _VISION_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Transcribe and transliterate the Urdu in this image."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]},
        ],
    )
    data = _parse_json(resp.choices[0].message.content)
    return {
        "urdu": (data.get("urdu") or "").strip(),
        "devanagari": (data.get("devanagari") or "").strip(),
        "roman": (data.get("roman") or "").strip(),
        "notes": (data.get("notes") or "").strip(),
        "model": OPENAI_MODEL,
    }


def transliterate_text(urdu_text: str) -> dict:
    """Urdu text -> {devanagari, roman, notes, model}. Text-only GPT call,
    used by the 'paste text' path."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _TEXT_SYSTEM},
            {"role": "user", "content": urdu_text},
        ],
    )
    data = _parse_json(resp.choices[0].message.content)
    return {
        "devanagari": (data.get("devanagari") or "").strip(),
        "roman": (data.get("roman") or "").strip(),
        "notes": (data.get("notes") or "").strip(),
        "model": OPENAI_MODEL,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_path>")
        sys.exit(1)
    with open(sys.argv[1], "rb") as f:
        result = extract_and_transliterate(f.read())
    print("Urdu:      ", result["urdu"])
    print("Devanagari:", result["devanagari"])
    print("Roman:     ", result["roman"])
    if result["notes"]:
        print("Notes:     ", result["notes"])
