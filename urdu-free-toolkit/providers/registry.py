# -*- coding: utf-8 -*-
"""Discover provider modules and report which can run right now.

Every provider module under providers/<capability>/ exposes a module-level
``PROVIDER`` (a ``BaseProvider`` subclass). ``discover()`` walks the package,
imports each module, and collects those. A module whose third-party library is
not installed raises ``ImportError`` on import and is skipped silently.

The registry is keyed by ``"<capability>:<id>"`` so the same ``id`` (e.g.
``"gpt"``) can exist for more than one capability without colliding.
"""
from __future__ import annotations

import importlib
import pkgutil

from providers.base import BaseProvider, Capability

_cache: dict[str, BaseProvider] | None = None


def reset_cache() -> None:
    """Drop the discovery cache. Call after an API key changes."""
    global _cache
    _cache = None


def _key(capability: Capability, provider_id: str) -> str:
    cap = capability.value if isinstance(capability, Capability) else str(capability)
    return f"{cap}:{provider_id}"


def _iter_module_names(package: str):
    try:
        pkg = importlib.import_module(package)
    except ModuleNotFoundError:
        return
    if not hasattr(pkg, "__path__"):
        return
    for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        if m.ispkg:
            continue
        yield m.name


def discover(package: str = "providers") -> dict[str, BaseProvider]:
    """Return ``{"<capability>:<id>": provider}``. Cached for the default package
    (and for the first non-default package, so tests can pin a fake set)."""
    global _cache
    if _cache is not None and package == "providers":
        return _cache

    found: dict[str, BaseProvider] = {}
    for mod_name in _iter_module_names(package):
        short = mod_name.rsplit(".", 1)[-1]
        if short in ("base", "registry") or short.startswith("_"):
            continue
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue  # this provider's library isn't installed — skip
        provider = getattr(mod, "PROVIDER", None)
        if isinstance(provider, BaseProvider):
            found[_key(provider.info.capability, provider.info.id)] = provider

    if package == "providers" or _cache is None:
        _cache = found
    return found


def get(capability: Capability, provider_id: str) -> BaseProvider:
    """Resolve one provider. Raises ``KeyError`` if unknown."""
    return discover()[_key(capability, provider_id)]


def _badge(p: BaseProvider, ok: bool) -> str:
    if p.info.kind == "offline":
        return "offline" if ok else "not installed"
    # api
    if not p.info.needs:
        return "free (net)"
    return "api" if ok else "needs key"


def for_ui(capability: Capability | None = None) -> list[dict]:
    """Rows for the frontend picker, sorted available-first then by label."""
    reg = _cache if _cache is not None else discover()
    rows: list[dict] = []
    for p in reg.values():
        if capability is not None and p.info.capability != capability:
            continue
        ok, reason = p.available()
        rows.append({
            "id": p.info.id,
            "label": p.info.label,
            "capability": p.info.capability.value,
            "kind": p.info.kind,
            "note": p.info.note,
            "available": ok,
            "reason": reason,
            "badge": _badge(p, ok),
        })
    rows.sort(key=lambda r: (not r["available"], r["label"].lower()))
    return rows
