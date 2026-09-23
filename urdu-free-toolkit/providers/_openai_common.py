# -*- coding: utf-8 -*-
"""Shared OpenAI plumbing for the ``gpt`` providers (OCR, transliteration,
image edit). Ported from the old top-level ``ocr.py`` / ``imgedit.py``.

Nothing here is imported unless an OpenAI-backed provider is actually used, and
every entry point degrades to a clear error when ``OPENAI_API_KEY`` is unset.
"""
from __future__ import annotations

import base64
import io
import json
import os

from PIL import Image

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")  # vision; override with OPENAI_MODEL (e.g. gpt-4o)
_MAX_IMAGE_DIM = 2048  # downscale the longer side before upload, to cap cost/latency


def sampling_kwargs() -> dict:
    """``temperature=0`` for models that allow it, ``{}`` for the GPT-5 /
    reasoning family (which reject any non-default temperature)."""
    if MODEL.lower().startswith(("gpt-5", "o1", "o3", "o4")):
        return {}
    return {"temperature": 0}

_client = None


def have_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def get_client():
    """Lazily build (and cache) an OpenAI client. Raises if no key is set."""
    global _client
    if _client is None:
        if not have_key():
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it in Settings, or put it in a "
                ".env file next to app.py (OPENAI_API_KEY=sk-...)."
            )
        from openai import OpenAI

        _client = OpenAI()
    return _client


VISION_SYSTEM = """You are an expert transcriber and transliterator of Urdu text.

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

TEXT_SYSTEM = """You are an expert transliterator of Urdu text.

Given Urdu text, restore the short vowels Urdu omits from context and standard
Hindustani pronunciation, and produce ALL THREE of:
1. "devanagari" -- a Devanagari (Hindi-script) transliteration.
2. "roman" -- a natural Roman Urdu / Hindustani transliteration the way people
   actually type it (e.g. "mohabbat", "kya haal hai"). Plain ASCII only:
   "aa", "ee", "oo", "n" -- no macrons or diacritics.
3. "roman_diacritic" -- the SAME transliteration in the scholarly "Rekhta"
   style with diacritics:
   - long vowels: ā, ī, ū  (e.g. "ḳharāb", "nāz", "dūr")
   - ے -> e, و -> o; diphthongs ai, au
   - ñ for nūn-ġunna / nasalization  ("haiñ", "hāloñ", "kahāñ")
   - ḳh for خ, ġ for غ, q for ق; ṭ ḍ ṛ for ٹ ڈ ڑ; ḥ for ح
   - keep aspirates as plain digraphs: kh gh th dh ph bh chh
   - join iẓāfat / compounds with a hyphen: "chashm-e-nāz", "ḳharāb-hāloñ"
Do NOT translate. Preserve line breaks, English words, and digits.

Return ONLY a JSON object with exactly these keys:
{"devanagari": "...", "roman": "...", "roman_diacritic": "...", "notes": "..."}
"notes" is "" if nothing is unclear."""


def prepare_image(image_bytes: bytes) -> tuple[str, str]:
    """Validate, normalize, and (if large) downscale the image.
    Returns ``(base64_str, mime_type)``. Raises ``ValueError`` if it isn't a
    readable image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:  # noqa: BLE001
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


def parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:
        raise RuntimeError(
            f"Model did not return valid JSON ({e}). Raw response: {content!r:.500}"
        )
