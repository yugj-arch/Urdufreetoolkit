# -*- coding: utf-8 -*-
"""Build a plain-text Urdu sentence list for the synthetic renderer.

Default source is the small bundled sample. Larger permissively-licensed
sources (Urdu Wikipedia extracts, CC-100 ur, Leipzig Corpora) are described in
LICENCES.md — fetching those is left to the operator so licence acceptance is
explicit.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SAMPLE = _HERE / "sample_sentences.txt"
_SPLIT = re.compile(r"[\n۔؟!.]+")


def _is_urdu(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    urdu = sum(1 for c in letters if "؀" <= c <= "ۿ")
    return urdu / len(letters) >= 0.6


def clean_sentences(raw: str, min_words: int = 3, max_words: int = 18) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for chunk in _SPLIT.split(raw):
        s = " ".join(chunk.split())
        if not s or not _is_urdu(s):
            continue
        n = len(s.split())
        if n < min_words or n > max_words or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def fetch(out_path, source: str = "sample", limit: int = 50_000) -> int:
    out_path = Path(out_path)
    if source == "sample":
        raw = _SAMPLE.read_text(encoding="utf-8")
    else:
        raise SystemExit(f"source {source!r} not built in; see training/corpus/LICENCES.md")
    sents = clean_sentences(raw)[:limit]
    out_path.write_text("\n".join(sents) + "\n", encoding="utf-8")
    return len(sents)


def main(argv=None) -> int:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="sample")
    ap.add_argument("--limit", type=int, default=50_000)
    a = ap.parse_args(argv)
    print(fetch(Path(a.out), a.source, a.limit))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
