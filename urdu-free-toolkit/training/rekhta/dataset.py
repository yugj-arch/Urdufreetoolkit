# -*- coding: utf-8 -*-
"""Teacher labels -> the student's train / dev / test lines.

A label is kept only when it is structurally sound and both of Rekhta's
models agree on it:

* one Devanagari word per Urdu word (izafat -ए- is a joiner, not a word;
  the conjunctive -ओ- *is* the Urdu و), so nothing was dropped or looped;
* the reverse model (Devanagari -> Urdu) gives back nearly the same Urdu
  (``--min-back`` similarity), so the teacher read the words that are there.

Splits are by source group: whole poems (by title), Dakshina's own
dev/test sentences, a hash of the Wikipedia sentence.

    python -m training.rekhta.dataset          # -> data/rekhta_ds/{train,dev,test}.tsv
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys

from urdu_nn.rekhta_roman import deva_units
from urdu_nn.rekhta_text import ZER, norm_word
from training.rekhta.label import DS, LABELS

TEST_SHARE, DEV_SHARE = 0.03, 0.01


def _bucket(key: str) -> float:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_of(row: dict) -> str:
    if row["src"] == "dak":
        return "test" if row["grp"] == "test" else "dev"
    if row["src"] == "word":
        return "train"
    h = _bucket(row["src"] + ":" + row["grp"])
    return "test" if h < TEST_SHARE else "dev" if h < TEST_SHARE + DEV_SHARE else "train"


def _nb(s: str) -> str:
    return " ".join(norm_word(w).rstrip(ZER) for w in s.split())


def check(row: dict, min_back: float) -> str:
    """"" if the label is kept, else the reason it isn't."""
    from rapidfuzz.distance import Levenshtein
    ur, hi = row["ur"], row["hi"]
    if not hi or len(hi) > 2 * len(ur) + 12:
        return "length"
    if any(c.isascii() and c.isalpha() for c in hi):
        return "latin"
    if len(deva_units(hi)) != len(ur.split()):
        return "words"
    back, want = _nb(row["back"]), _nb(ur)
    if back != want and Levenshtein.normalized_similarity(back, want) < min_back:
        return "roundtrip"
    return ""


def build(min_back: float = 0.85) -> dict:
    seen, stats = set(), collections.Counter()
    out = {s: open(DS / f"{s}.tsv", "w", encoding="utf-8", newline="\n")
           for s in ("train", "dev", "test")}
    for line in LABELS.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        why = check(row, min_back)
        stats[(row["src"], why or "kept")] += 1
        if why and row["src"] != "dak":
            continue
        s = split_of(row)
        if why and s != "test":
            continue
        exact = int(_nb(row["back"]) == _nb(row["ur"]))
        out[s].write(f"{row['src']}\t{row['ur']}\t{row['hi']}\t{exact}\t{int(not why)}\n")
        stats[s] += 1
    for fh in out.values():
        fh.close()
    rep = {f"{k[0]}:{k[1]}" if isinstance(k, tuple) else k: v for k, v in sorted(stats.items(), key=str)}
    (DS / "dataset_stats.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-back", type=float, default=0.85)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.min_back), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
