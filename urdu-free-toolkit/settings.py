# -*- coding: utf-8 -*-
"""Read/write API keys and provider config to the ``.env`` file next to
``app.py``. The Settings panel in the UI POSTs here so keys never have to be
hand-edited into a file. Values are applied to ``os.environ`` immediately so a
provider lights up without restarting the server.

Only key *names* are ever echoed back to the client — never the values.
"""
from __future__ import annotations

import os
from pathlib import Path

import config

KNOWN_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_VISION_KEY",
    "OCRSPACE_API_KEY",
    "AZURE_VISION_KEY",
    "AZURE_VISION_ENDPOINT",
    "GROQ_API_KEY",
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
    "GEMINI_MODEL",
    "GROQ_MODEL",
    "HF_TROCR_CKPT",
)


def env_path() -> Path:
    return Path(__file__).with_name(".env")


def _read_lines() -> list[str]:
    p = env_path()
    return p.read_text(encoding="utf-8").splitlines() if p.exists() else []


def save(values: dict[str, str]) -> list[str]:
    """Upsert each known key present in ``values`` into ``.env`` (other lines and
    ordering preserved). Empty string removes the key. Returns key names written.

    A no-op on Vercel, whose filesystem is read-only — keys there come from the
    project's environment variables."""
    if config.settings_readonly():
        return []
    lines = _read_lines()
    written: list[str] = []
    for key in KNOWN_KEYS:
        if key not in values:
            continue
        val = (values[key] or "").strip()
        idx = next((i for i, ln in enumerate(lines)
                    if ln.strip().startswith(f"{key}=")), None)
        if val == "":
            if idx is not None:
                lines.pop(idx)
            os.environ.pop(key, None)
        else:
            new_line = f"{key}={val}"
            if idx is None:
                lines.append(new_line)
            else:
                lines[idx] = new_line
            os.environ[key] = val
        written.append(key)
    env_path().write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return written


def status() -> dict[str, bool]:
    """``{key: is_set}`` for every known key. No values."""
    return {k: bool(os.environ.get(k)) for k in KNOWN_KEYS}
