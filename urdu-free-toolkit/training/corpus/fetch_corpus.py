# -*- coding: utf-8 -*-
"""Build a plain-text Urdu sentence list for the synthetic renderer.

Default source is the small bundled sample. ``--source wikipedia`` pulls real
running-text extracts straight from ur.wikipedia.org's own API (CC BY-SA 3.0,
see LICENCES.md) -- a random sample of articles via ``generator=random``, the
same polite-client pattern ``scripts/build_lexicon.py`` uses for Wiktionary.
Other larger sources (CC-100 ur, Leipzig Corpora) are described in
LICENCES.md -- fetching those is left to the operator so licence acceptance is
explicit.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_SAMPLE = _HERE / "sample_sentences.txt"
_SPLIT = re.compile(r"[\n۔؟!.]+")

_WIKI_API = "https://ur.wikipedia.org/w/api.php"
_WIKI_USER_AGENT = "UrduFreeToolkitCorpusBot/1.0 (offline corpus builder; github.com repo)"


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


def fetch_wikipedia_extracts(limit: int, delay: float = 2.0, batch: int = 20) -> str:
    """Pull plain-text extracts from random ur.wikipedia.org articles via
    ``generator=random`` until ``clean_sentences`` would yield roughly
    ``limit`` sentences worth of raw text (we overfetch a bit since not every
    extract line survives the Urdu-ratio/length filter). CC BY-SA 3.0, see
    LICENCES.md. Polite: single-threaded, paced with ``delay`` between calls,
    exponential backoff + Retry-After handling on 429/5xx (same pattern as
    ``scripts/build_lexicon.py``'s ``api_get``)."""
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": _WIKI_USER_AGENT})
    chunks: list[str] = []
    sentence_estimate = 0
    seen_pageids: set[int] = set()
    attempts = 0
    max_attempts = max(80, (limit // 3) + 40)
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": batch,
        "prop": "extracts",
        "explaintext": 1,
        "exlimit": batch,
    }
    backoff = delay
    while sentence_estimate < limit and attempts < max_attempts:
        attempts += 1
        try:
            resp = session.get(_WIKI_API, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"  wikipedia network error ({exc}); retrying in {backoff:.0f}s", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else backoff
            print(f"  wikipedia {resp.status_code}; backing off {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            backoff = min(backoff * 2, 60.0)
            continue
        try:
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 -- transient network/API hiccup, keep going
            print(f"  wikipedia fetch error ({exc}); retrying in {backoff:.0f}s", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        backoff = delay  # reset after a clean response
        pages = data.get("query", {}).get("pages", {})
        new = 0
        for page in pages.values():
            pid = page.get("pageid")
            extract = page.get("extract", "")
            if pid in seen_pageids or not extract:
                continue
            seen_pageids.add(pid)
            chunks.append(extract)
            new += 1
        sentence_estimate = len(clean_sentences("\n".join(chunks)))
        print(f"  ...{len(seen_pageids)} articles, ~{sentence_estimate} usable sentences so far",
              file=sys.stderr)
        if new:
            time.sleep(delay)
    return "\n".join(chunks)


def fetch(out_path, source: str = "sample", limit: int = 50_000) -> int:
    out_path = Path(out_path)
    if source == "sample":
        raw = _SAMPLE.read_text(encoding="utf-8")
    elif source == "wikipedia":
        raw = fetch_wikipedia_extracts(limit)
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
