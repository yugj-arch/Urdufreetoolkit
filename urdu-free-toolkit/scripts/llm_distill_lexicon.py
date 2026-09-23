#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build ``data/urdu_lexicon_llm.json.gz`` -- the GPT-distilled "breadth layer"
lexicon that fills the gap the Wiktionary lexicon (``scripts/build_lexicon.py``)
leaves: Wiktionary's "Category:Urdu lemmas" is dictionary *citation forms*
(headwords), so common running-text words that are inflected, or simply
outside the lemma category, are missing from it. This script instead ranks
words BY REAL FREQUENCY in an open Urdu corpus and asks GPT for each one's
Devanagari + plain-Roman reading, in this project's exact convention.

WHAT IT DOES
------------
1. Reads ``--corpus`` (default: ``training/corpus/sentences_wiki.txt``, built
   by ``python -m training.corpus.fetch_corpus --source wikipedia`` -- CC
   BY-SA 3.0 Urdu Wikipedia extracts, see ``training/corpus/LICENCES.md``).
   Rekhta or any other copyrighted-poetry site is never used as a source --
   see the module docstring in ``transliterate.py`` and ``data/SOURCES.md``.
2. Tokenizes into Perso-Arabic word runs and counts frequency.
3. Drops words already covered by ``transliterate.COMMON_WORDS`` or the
   existing bundled lexicons (Wiktionary + any prior LLM run) -- no point
   spending an API call on a word we already have an exact answer for.
4. Sends the remaining words, ranked by frequency, to the OpenAI API
   (``providers._openai_common``, same client/model/JSON-mode config the
   app's own GPT provider uses) in numbered batches, asking for
   ``{devanagari, roman}`` per word in the app's plain-ASCII Roman
   convention (aa/ii/oo doubling, kh/gh/sh/ch digraphs, trailing "n"
   nasalisation -- matching ``normalize_roman`` in build_lexicon.py).
5. Writes ``{urdu_spelling: [devanagari, roman]}`` to a gzip JSON file, same
   shape ``transliterate._load_lexicon`` already expects.

RESUMABILITY
------------
Checkpointed to ``--checkpoint`` after every batch, like build_lexicon.py --
an interrupted run just re-launches with the same arguments.

USAGE
-----
    python -m training.corpus.fetch_corpus \\
        --out training/corpus/sentences_wiki.txt --source wikipedia --limit 4000
    python scripts/llm_distill_lexicon.py --limit 3000

Needs OPENAI_API_KEY (loaded from .env next to app.py, same as the app).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import transliterate as _rule  # noqa: E402
from providers import _openai_common as _common  # noqa: E402

DEFAULT_CORPUS = ROOT / "training" / "corpus" / "sentences_wiki.txt"
DEFAULT_OUTPUT = ROOT / "data" / "urdu_lexicon_llm.json.gz"
DEFAULT_CHECKPOINT = ROOT / "data" / ".lexicon_llm_checkpoint.json"

_URDU_WORD_RE = re.compile(r"[؀-ۿ]+")
_MIN_LEN = 2  # drop single-letter tokens (isolated conjunctions/prefixes, noise)

_DISTILL_SYSTEM = """You are an expert lexicographer transliterating Urdu (Perso-Arabic script) into Hindi (Devanagari) and Roman.

You will be given a numbered list of individual Urdu WORDS, no sentence context -- give each word its single most common/citation-form reading, restoring the short vowels Urdu's script omits.

For EACH numbered word produce:
- "d": Devanagari (Hindi-script) spelling.
- "r": plain-ASCII Roman spelling -- long vowels doubled ("aa","ii","oo"), digraphs "kh"/"gh"/"sh"/"ch"/"th"/"dh"/"ph"/"bh", nasalisation folded to a trailing "n". Lowercase only. NEVER macrons, diacritics, or non-ASCII letters in "r".

Return ONLY a JSON object {"<number>": ["<devanagari>", "<roman>"], ...} -- one entry per input number, same count as the input list. If a word is obscure or ambiguous, still give your single best reading; never omit a number."""


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    tmp.replace(path)


def write_lexicon(lexicon: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as fh:
        json.dump(lexicon, fh, ensure_ascii=False, sort_keys=True)


def _checkpoint_to_lexicon(checkpoint: dict) -> dict:
    return {
        unicodedata.normalize("NFC", w): entry
        for w, entry in checkpoint.items()
        if entry
    }


def rank_oov_words(corpus_text: str, limit: int) -> list[str]:
    """Tokenize, count frequency, drop anything already covered by the
    curated dict or either bundled lexicon, return the top ``limit`` by
    frequency (ties broken by first-seen order)."""
    counts: collections.Counter = collections.Counter()
    for tok in _URDU_WORD_RE.findall(corpus_text):
        tok = unicodedata.normalize("NFC", tok)
        if len(tok) >= _MIN_LEN:
            counts[tok] += 1
    known = set(_rule.COMMON_WORDS) | set(_rule._LEXICON)
    ranked = [w for w, _ in counts.most_common() if _rule._fold_key(w) not in known]
    return ranked[:limit]


def _call_batch(words: list[str], max_retries: int = 5) -> dict[int, tuple[str, str]]:
    """One OpenAI call for a batch of words. Returns {index: (deva, roman)}
    for whatever the model returned; missing/malformed indices are simply
    absent (caller leaves those words unchecked-pointed for a later retry)."""
    numbered = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(words))
    client = _common.get_client()
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=_common.MODEL,
                response_format={"type": "json_object"},
                **_common.sampling_kwargs(),
                messages=[
                    {"role": "system", "content": _DISTILL_SYSTEM},
                    {"role": "user", "content": numbered},
                ],
            )
            data = _common.parse_json(resp.choices[0].message.content)
            out: dict[int, tuple[str, str]] = {}
            for k, v in data.items():
                try:
                    idx = int(k) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(words) and isinstance(v, (list, tuple)) and len(v) == 2 and all(v):
                    out[idx] = (str(v[0]).strip(), str(v[1]).strip())
            return out
        except Exception as exc:  # noqa: BLE001 -- rate limits, timeouts, bad JSON, all just retry
            last_exc = exc
            wait = 4.0 * (attempt + 1)
            print(f"  batch error ({exc}); retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    print(f"  giving up on this batch after {max_retries} retries ({last_exc})", file=sys.stderr)
    return {}


def build(corpus_path: Path, limit: int, batch_size: int, checkpoint_path: Path,
          resume: bool, output_path: Path) -> dict:
    corpus_text = corpus_path.read_text(encoding="utf-8")
    ranked = rank_oov_words(corpus_text, limit)
    print(f"{len(ranked)} OOV words ranked by frequency (target limit {limit})", file=sys.stderr)

    checkpoint: dict = load_json(checkpoint_path, {}) if resume else {}
    print(f"resuming with {len(checkpoint)} already-processed words" if checkpoint
          else "starting fresh", file=sys.stderr)

    remaining = [w for w in ranked if w not in checkpoint]
    batches_done = 0
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        result = _call_batch(batch)
        for i, word in enumerate(batch):
            checkpoint[word] = list(result[i]) if i in result else None
        batches_done += 1
        save_json(checkpoint_path, checkpoint)
        accepted = sum(1 for v in checkpoint.values() if v)
        print(f"  ...batch {batches_done} ({start + len(batch)}/{len(remaining)} this run) "
              f"-- {accepted} accepted total", file=sys.stderr)
        write_lexicon(_checkpoint_to_lexicon(checkpoint), output_path)

    return _checkpoint_to_lexicon(checkpoint)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                     help=f"plain-text Urdu sentences, one per line (default: {DEFAULT_CORPUS})")
    ap.add_argument("--limit", type=int, default=3000,
                     help="target number of OOV words to distill (default: 3000)")
    ap.add_argument("--batch-size", type=int, default=40,
                     help="words per API call (default: 40)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                     help=f"output gzip JSON path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                     help="resumable progress file (default: data/.lexicon_llm_checkpoint.json)")
    ap.add_argument("--no-resume", action="store_true",
                     help="ignore any existing checkpoint and start over")
    args = ap.parse_args()

    if not _common.have_key():
        raise SystemExit("OPENAI_API_KEY not set (checked environment + .env)")
    if not args.corpus.exists():
        raise SystemExit(
            f"corpus file {args.corpus} not found -- build it first:\n"
            f"  python -m training.corpus.fetch_corpus --out {args.corpus} "
            f"--source wikipedia --limit 4000")

    lexicon = build(
        corpus_path=args.corpus,
        limit=args.limit,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        resume=not args.no_resume,
        output_path=args.output,
    )
    write_lexicon(lexicon, args.output)
    size = args.output.stat().st_size
    print(f"wrote {len(lexicon)} entries to {args.output} ({size:,} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
