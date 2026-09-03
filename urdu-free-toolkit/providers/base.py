# -*- coding: utf-8 -*-
"""Shared types for every provider.

A *provider* is a small module under providers/<capability>/ that exposes a
module-level ``PROVIDER`` instance (a ``BaseProvider`` subclass). It declares
what it is via ``info`` and whether it can run right now via ``available()``,
and implements one capability method: ``ocr`` / ``translit``. Failures are
returned as data (``ok=False``, ``error=...``), never raised into a request
path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Capability(str, Enum):
    OCR = "ocr"
    TRANSLIT = "translit"


@dataclass
class ProviderInfo:
    id: str
    label: str
    capability: Capability
    kind: str  # "offline" | "api"
    needs: list[str] = field(default_factory=list)
    note: str = ""
    price: str = ""  # short human string for the UI price badge, e.g. "Free · offline"


@dataclass
class Result:
    provider_id: str
    ok: bool = True
    error: str = ""
    ms: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class OcrResult(Result):
    text: str = ""
    boxes: list | None = None
    notes: str = ""


@dataclass
class TranslitResult(Result):
    devanagari: str = ""
    roman: str = ""              # plain ASCII ("khraab")
    roman_diacritic: str = ""    # Rekhta-style marks ("ḳharāb"); "" if unavailable


@dataclass
class TranslitOpts:
    roman_style: str = "natural"
    targets: tuple[str, ...] = ("devanagari", "roman")


class BaseProvider:
    """Common base. Subclasses set ``info`` and implement a capability method."""

    info: ProviderInfo

    def available(self) -> tuple[bool, str]:
        """(usable_now?, reason-if-not). Default: always usable."""
        return (True, "")

    def _timed(self, fn, *args, **kwargs) -> dict:
        """Run ``fn`` catching everything. Returns
        ``{"ok", "error", "ms", "value"}`` for the subclass to fold into its
        concrete Result dataclass."""
        start = time.monotonic()
        try:
            value = fn(*args, **kwargs)
            return {
                "ok": True,
                "error": "",
                "ms": int((time.monotonic() - start) * 1000),
                "value": value,
            }
        except Exception as e:  # noqa: BLE001 - provider failures are data, not crashes
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "ms": int((time.monotonic() - start) * 1000),
                "value": None,
            }
