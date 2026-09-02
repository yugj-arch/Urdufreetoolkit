# -*- coding: utf-8 -*-
"""The short, curated engine list the compare UI actually shows.

Every provider module under ``providers/`` still loads and still works — this
only controls which engines the pickers surface, and in what order (first entry
= top pick, shown first). Re-enable a hidden engine by adding its id back to the
list here. ``registry.for_ui(cap, include_hidden=True)`` ignores this filter
entirely.

Safety net: if *none* of a capability's curated ids are installed, the registry
falls back to showing every engine it discovered for that step rather than an
empty picker.
"""
from __future__ import annotations

# Keyed by Capability.value. Order matters: it's the display / ranking order,
# and the first entry is the engine the UI pre-selects on load (the one default).
# At most four per step.
FEATURED: dict[str, list[str]] = {
    "ocr":       ["gcv", "gpt", "paddle", "easyocr"],
    "translit":  ["rule", "gpt", "aksharamukha", "uroman"],
}
