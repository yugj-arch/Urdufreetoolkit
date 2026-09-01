# -*- coding: utf-8 -*-
"""Shared Anthropic (Claude) plumbing for the ``claude`` providers.

Claude's vision models read Urdu Nastaliq well and, like the GPT / Gemini
paths, restore the short vowels Urdu omits from context. Reuses the OCR / text
prompts from ``_openai_common`` so the JSON contract is identical across every
API provider.

Nothing here imports ``anthropic`` until a Claude-backed provider is actually
used, and every entry point degrades to a clear error when ``ANTHROPIC_API_KEY``
is unset.
"""
from __future__ import annotations

import json
import os

from providers._openai_common import TEXT_SYSTEM, VISION_SYSTEM  # noqa: F401 (re-exported)

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
_MAX_TOKENS = 4096
_client = None


def have_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def get_client():
    """Lazily build (and cache) an Anthropic client. Raises if no key is set."""
    global _client
    if _client is None:
        if not have_key():
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it in Settings, or put it in "
                "a .env file next to app.py (ANTHROPIC_API_KEY=sk-ant-...)."
            )
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def _text_of(resp) -> str:
    """First text block of a Messages response (skips any thinking block)."""
    return next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")


def _parse_json(raw: str) -> dict:
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
        raise RuntimeError(f"Claude did not return valid JSON. Raw response: {raw!r:.500}")


def generate_json(system: str, content: list) -> dict:
    """One Messages call whose user turn is ``content`` (a list of blocks).
    Returns the parsed JSON object the prompt asks for."""
    client = get_client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return _parse_json(_text_of(resp))


def image_block(b64: str, mime: str) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64}}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}
