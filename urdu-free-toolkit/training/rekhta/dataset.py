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


def load_split(split: str) -> list[tuple[str, str, str, bool, bool]]:
    """(src, urdu run, teacher devanagari, round trip exact, label kept)."""
    path = DS / f"{split}.tsv"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        src, ur, hi, exact, kept = line.split("\t")
        rows.append((src, ur, hi, exact == "1", kept == "1"))
    return rows


# ---------------------------------------------------------------------------
# gold words: correct spellings from us, conventions from Rekhta
# ---------------------------------------------------------------------------

_TOP = "िीेैोौ"
ROOT = DS.parents[1]


def _conv_fold(deva: str) -> str:
    """Devanagari with every Rekhta-vs-dictionary *convention* folded away:
    ain apostrophe, nukta, ँ vs ं, half-nasal vs anusvara."""
    d = deva.replace("'", "").replace("़", "").replace("ँ", "ं")
    return re.sub(r"[नम]्(?=[क-ह])", "ं", d)


def rekhta_conventions(urdu: str, deva: str) -> str:
    """A gold spelling written the way Rekhta writes: nuktas where the Urdu
    letters call for them (कानून + قانون -> क़ानून), a nasal vowel with no
    top matra as ँ (यहां -> यहाँ, हूं -> हूँ)."""
    from training.translit.silver import spell_with_nuktas
    d = unicodedata.normalize("NFC", deva)
    d = spell_with_nuktas(urdu, d) or d
    return re.sub("(?<=[^" + _TOP + "])ं(?=$|-)", "ँ", d)


def gold_words() -> dict[str, str]:
    """urdu word -> gold Devanagari: the Wiktionary lexicon (training half)
    and the reviewed dictionary, minus every word the benchmark holds out."""
    import transliterate as rule
    test = {json.loads(l)["urdu"] for l in (ROOT / "data" / "translit_ds" / "test.jsonl")
            .open(encoding="utf-8")}
    out = {}
    lex = json.loads((ROOT / "data" / "translit_ds" / "lexicon_train.json").read_text(encoding="utf-8"))
    for u, v in lex.items():
        if v[0] and u not in test:
            out[norm_word(u)] = v[0]
    for u, v in rule.load_exact_dictionary(rule.EXACT_DICTIONARY_PATH).items():
        k = norm_word(u)
        if v[0] and k not in test and u not in test:
            out[k] = v[0]
    return {k: d for k, d in out.items() if k and " " not in k and len(deva_units(d)) == 1}


def build(min_back: float = 0.85, word_repeat: int = 3) -> dict:
    seen, stats = set(), collections.Counter()
    out = {s: open(DS / f"{s}.tsv", "w", encoding="utf-8", newline="\n")
           for s in ("train", "dev", "test")}
    gold = gold_words()
    teacher_word: dict[str, str] = {}
    for line in LABELS.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        if row["src"] == "word":
            teacher_word[row["ur"]] = row["hi"]
            continue
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
    # the gold words: Rekhta's own reading where it only differs from the
    # gold one by convention, else the gold spelling in Rekhta's conventions
    for u, d in gold.items():
        t = teacher_word.get(u)
        if t and _conv_fold(t) == _conv_fold(d):
            target, how = t, "gold:teacher"
        else:
            target, how = rekhta_conventions(u, d), "gold:converted"
        stats[how] += 1
        for _ in range(word_repeat):
            out["train"].write(f"gold\t{u}\t{target}\t1\t1\n")
        stats["train"] += word_repeat
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
