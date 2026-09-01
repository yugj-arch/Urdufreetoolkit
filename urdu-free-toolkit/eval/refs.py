# -*- coding: utf-8 -*-
"""Generate and cache GPT "silver reference" transcriptions for the OCR eval
fixtures. Refs are committed JSON so the eval runs offline and free after the
first generation; pass ``refresh=True`` (CLI: ``--refresh``) to regenerate.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ocr"


def list_fixtures(d: Path | None = None) -> list[Path]:
    d = d or FIXTURES_DIR
    return sorted(p for p in d.glob("*.png"))


def ref_path(png: Path) -> Path:
    return png.with_suffix(".gpt.json")


def load_ref(png: Path) -> dict | None:
    p = ref_path(png)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.warning("bad ref json: %s", p, exc_info=True)
        return None


def _default_ocr(image: bytes):
    from providers.ocr.gpt import PROVIDER
    return PROVIDER.ocr(image)


def ensure_refs(d: Path | None = None, refresh: bool = False, ocr=None) -> dict:
    d = d or FIXTURES_DIR
    ocr = ocr or _default_ocr
    written, skipped, have = [], [], []
    for png in list_fixtures(d):
        slug = png.stem
        if not refresh and ref_path(png).exists():
            have.append(slug)
            continue
        try:
            res = ocr(png.read_bytes())
        except Exception as e:
            log.warning("silver ref for %s skipped: %s", slug, e)
            skipped.append(slug)
            continue
        if not getattr(res, "ok", False) or not getattr(res, "text", ""):
            skipped.append(slug)
            continue
        ref_path(png).write_text(json.dumps({
            "urdu": res.text,
            "model": (getattr(res, "meta", {}) or {}).get("model", "gpt"),
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(slug)
    return {"written": written, "skipped": skipped, "have": have}
