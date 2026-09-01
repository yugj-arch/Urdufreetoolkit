# -*- coding: utf-8 -*-
"""Resolve an optional fine-tuned recognizer for the offline OCR engines.

Both return ``None`` (use the stock model) unless a custom model is actually
present. Wired into ``easyocr_p`` / ``paddle`` now; dormant until Phase 2
training produces a model. See training/README.md.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_EASYOCR_USER_DIR = Path.home() / ".EasyOCR" / "user_network"


def easyocr_recog_network() -> str | None:
    env = os.environ.get("OCR_EASYOCR_RECOG_NETWORK")
    try:
        if env:
            if (_EASYOCR_USER_DIR / f"{env}.py").exists():
                return env
            log.warning("OCR_EASYOCR_RECOG_NETWORK=%s but %s/%s.py missing; using stock",
                        env, _EASYOCR_USER_DIR, env)
            return None
        for py in sorted(_EASYOCR_USER_DIR.glob("*.py")):
            if py.with_suffix(".yaml").exists():
                return py.stem
    except Exception:  # pragma: no cover
        log.warning("easyocr_recog_network probe failed", exc_info=True)
    return None


def paddle_rec_dir() -> str | None:
    d = os.environ.get("OCR_PADDLE_REC_DIR")
    if d and Path(d).is_dir():
        return d
    if d:
        log.warning("OCR_PADDLE_REC_DIR=%s is not a directory; using stock", d)
    return None
