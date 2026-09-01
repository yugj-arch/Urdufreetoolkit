# -*- coding: utf-8 -*-
"""Shared Google Gemini plumbing for the ``gemini`` providers.

Gemini's free API tier (key from Google AI Studio) is currently among the most
accurate readers of Urdu Nastaliq, and like the GPT path it restores the short
vowels Urdu omits from context. Reuses the OCR / text prompts from
``_openai_common`` so the JSON contract is identical across providers.
"""
from __future__ import annotations

import json
import os

from providers._openai_common import TEXT_SYSTEM, VISION_SYSTEM  # noqa: F401 (re-exported)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")  # override with GEMINI_MODEL (e.g. gemini-2.5-flash, gemini-2.5-pro)
_client = None


def have_key() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY"))


def get_client():
    global _client
    if _client is None:
        if not have_key():
            raise RuntimeError("GOOGLE_API_KEY is not set (get one free at aistudio.google.com).")
        from google import genai

        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _client


def generate_json(system: str, parts: list) -> dict:
    from google.genai import types

    client = get_client()
    resp = client.models.generate_content(
        model=MODEL,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    text = (resp.text or "").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        raise RuntimeError(f"Gemini did not return valid JSON ({e}). Raw: {text!r:.400}")


def image_part(image_bytes: bytes, mime: str = "image/png"):
    from google.genai import types

    return types.Part.from_bytes(data=image_bytes, mime_type=mime)
