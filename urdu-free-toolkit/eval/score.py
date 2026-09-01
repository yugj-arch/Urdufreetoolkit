# -*- coding: utf-8 -*-
"""CER / WER scoring for OCR output against a reference, raw and normalized."""
from __future__ import annotations

from Levenshtein import distance

from providers.ocr._normalize import normalize_urdu


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return distance(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rt in enumerate(r, 1):
        cur = [i]
        for j, ht in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rt != ht)))
        prev = cur
    return prev[-1] / len(r)


def score_pair(ref: str, hyp: str) -> dict:
    rn, hn = normalize_urdu(ref), normalize_urdu(hyp)
    return {
        "cer": cer(ref, hyp),
        "wer": wer(ref, hyp),
        "cer_norm": cer(rn, hn),
        "wer_norm": wer(rn, hn),
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "cer": 0.0, "wer": 0.0, "cer_norm": 0.0, "wer_norm": 0.0}
    keys = ("cer", "wer", "cer_norm", "wer_norm")
    out = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    out["n"] = len(rows)
    return out
