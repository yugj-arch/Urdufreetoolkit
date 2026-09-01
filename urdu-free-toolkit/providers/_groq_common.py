# -*- coding: utf-8 -*-
"""Shared Groq plumbing for the ``groq`` transliteration provider.

Groq serves open models (Llama) on its own inference hardware with a generous
free tier — a context-aware transliterator like the GPT / Gemini paths, at no
cost. The chat API is OpenAI-compatible. Reuses ``TEXT_SYSTEM`` from
``_openai_common`` so the JSON contract is identical across providers.

Nothing here imports ``groq`` until the provider is actually used, and every
entry point degrades to a clear error when ``GROQ_API_KEY`` is unset.
"""
from __future__ import annotations

import json
import os

from providers._openai_common import TEXT_SYSTEM  # noqa: F401 (re-exported)

MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")  # override with GROQ_MODEL
_client = None


def have_key() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def get_client():
    """Lazily build (and cache) a Groq client. Raises if no key is set."""
    global _client
    if _client is None:
        if not have_key():
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get one free at console.groq.com, then "
                "add it in Settings (or a .env file next to app.py)."
            )
        from groq import Groq

        _client = Groq()
    return _client


def _loads(raw: str) -> dict:
    s = (raw or "").strip()
    if s.startswith("```"):  # strip a ```json ... ``` fence if the model added one
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.rsplit("```", 1)[0].strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b > a:
            return json.loads(s[a:b + 1])
        raise RuntimeError(f"Groq did not return valid JSON. Raw response: {raw!r:.500}")


def generate_json(system: str, text: str) -> dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    )
    return _loads(resp.choices[0].message.content)
