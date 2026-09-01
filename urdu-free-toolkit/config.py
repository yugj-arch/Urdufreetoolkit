# -*- coding: utf-8 -*-
"""Where-am-I-running switches.

Locally the app writes API keys to ``.env`` and lets the Batch tab queue 30
images. On Vercel the filesystem is read-only and the request body is capped at
~4.5 MB, so a few behaviours change. Every such decision is made here, keyed off
the ``VERCEL`` environment variable the platform injects, and read at call time
so tests can flip it with ``monkeypatch.setenv``.
"""
from __future__ import annotations

import os

BATCH_MAX_FILES_LOCAL = 30
BATCH_MAX_FILES_VERCEL = 4  # keeps a multi-image POST under Vercel's ~4.5 MB body cap

_MB = 1024 * 1024


def on_vercel() -> bool:
    """True inside a Vercel deployment (``VERCEL=1`` is set by the platform)."""
    return bool(os.environ.get("VERCEL"))


def batch_max_files() -> int:
    return BATCH_MAX_FILES_VERCEL if on_vercel() else BATCH_MAX_FILES_LOCAL


def settings_readonly() -> bool:
    """On Vercel the filesystem is read-only, so keys can't be written to
    ``.env`` — they come from the project's environment variables instead."""
    return on_vercel()


def max_content_length() -> int:
    """Flask upload ceiling. Tight on Vercel (its edge rejects >4.5 MB anyway),
    generous locally for the Batch tab."""
    return 4 * _MB if on_vercel() else 64 * _MB
