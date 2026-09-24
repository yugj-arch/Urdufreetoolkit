# -*- coding: utf-8 -*-
"""Build the exact transliteration dictionary: data/translit_dictionary.tsv

The N most frequent Urdu word forms (default 10,000 -- about 92% of running
Urdu text) each get a reviewed Devanagari, plain Roman and Rekhta-style
Roman spelling. Every offline engine looks these words up before guessing, so
most of any real text comes out exactly as reviewed.

    python -m training.translit.build_dictionary draft   # engine readings -> draft TSV for review
    python -m training.translit.build_dictionary build   # draft + reviewed fixes -> shipped TSV

How the reviewed spellings are kept reproducible:

  * ``draft`` counts word frequency over Dakshina's Urdu Wikipedia text and
    reads every word through the neural engine *without* the dictionary
    (curated words, Wiktionary gold lexicon, then the model). It writes
    ``data/translit_ds/dictionary_draft.tsv``.
  * ``dictionary_fixes.txt`` (next to this file) holds every hand correction:

        urdu | devanagari | R [| plain [| rekhta]]

    "." keeps the draft's value for that column; R is the canonical reading
    (``urdu_nn/scheme.py``) the plain and Rekhta spellings are rendered from;
    a plain / rekhta column overrides the render (English loans: school).
    A line ``urdu | -`` drops the word (stray letters, broken tokens).
  * ``build`` = draft + fixes -> ``data/translit_dictionary.tsv``
    (urdu, devanagari, roman, rekhta), frequency order.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from urdu_nn.scheme import URDU_LETTERS, norm_urdu, to_plain, to_rekhta  # noqa: E402

WIKI = (ROOT / "data" / "translit_raw" / "dakshina_dataset_v1.0" / "ur"
        / "native_script_wikipedia" / "ur.wiki-filt.train.text.shuf.txt.gz")
DRAFT = ROOT / "data" / "translit_ds" / "dictionary_draft.tsv"
FIXES = Path(__file__).with_name("dictionary_fixes.txt")
OUT = ROOT / "data" / "translit_dictionary.tsv"

HEADER = """\
# Exact Urdu transliteration dictionary -- the {n} most frequent Urdu word
# forms (Dakshina Urdu Wikipedia counts), reviewed (data/SOURCES.md). Columns:
#   urdu <TAB> devanagari <TAB> roman (plain) <TAB> rekhta (diacritic roman)
# Built by training/translit/build_dictionary.py; edit dictionary_fixes.txt
# and rebuild rather than editing this file by hand.
"""


def top_words(n: int) -> list[tuple[str, int]]:
    import neural_translit as nt
    import transliterate as rule
    cnt: collections.Counter = collections.Counter()
    with gzip.open(WIKI, "rt", encoding="utf-8") as fh:
        for line in fh:
            for run in nt._URDU_RUN.findall(line):
                _, core, _ = rule._split_leading_trailing_punct(run)
                key = norm_urdu(core)
                # a word = one run of Urdu script with at least two real letters
                # (drops ٪ ٫ ٭, lone ں / ھ and runs glued by a mid-word ، )
                if (key and nt._URDU_RUN.fullmatch(key) and "،" not in key
                        and sum(c in URDU_LETTERS for c in key) >= 2 or key in ("ء", "و", "ع", "آ")):
                    cnt[key] += 1
    return cnt.most_common(n)


def draft(n: int) -> None:
    import neural_translit as nt
    import transliterate as rule
    eng = nt.NeuralTransliterator(dictionary_path=None)
    words = top_words(n)
    rows = eng.analyze([w for w, _ in words])
    if len(rows) != len(words):
        raise SystemExit(f"analyze returned {len(rows)} rows for {len(words)} words")
    # second opinion on the Devanagari: decode it from the chosen reading
    alt = eng._deva_for([r[2] for r in rows if r[2]])
    DRAFT.parent.mkdir(parents=True, exist_ok=True)
    with DRAFT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# urdu\tdevanagari\tR\tplain\trekhta\tcount\ttier\tdeva_from_R\n")
        for (w, c), (deva, plain, rich, rekhta) in zip(words, rows):
            tier = ("curated" if rule._fold_key(w) in rule.COMMON_WORDS
                    else "lexicon" if w in eng.lexicon else "model")
            a = alt.get(rich, "") if rich else ""
            fh.write("\t".join([w, deva or "", rich or "", plain or "", rekhta or "", str(c),
                                tier, a if a != deva else ""]) + "\n")
    print(f"draft: {len(rows)} words -> {DRAFT}")


# Panchamakshar rule (modern standard Hindi): a nasal consonant + virama before
# a consonant of its own class is written as anusvara -- हिन्दी -> हिंदी,
# नवम्बर -> नवंबर, इन्सान -> इंसान; nh / nn / ny / nv / nm / mm / mh stay.
_HALF_NASAL = re.compile("ङ्(?=[क-घ])|ञ्(?=[च-झ])|ण्(?=[ट-ढ])|न्(?=[क-धसशष])|म्(?=[प-भ])")


def norm_nasals(deva: str) -> str:
    return _HALF_NASAL.sub("ं", deva)


def read_tsv(path: Path, sep: str = "\t") -> list[list[str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            out.append([unicodedata.normalize("NFC", c.strip()) for c in line.split(sep)])
    return out


def build() -> None:
    rows = read_tsv(DRAFT)
    fixes: dict[str, list[str]] = {}
    for f in read_tsv(FIXES, "|") if FIXES.exists() else []:
        key = norm_urdu(f[0])
        if key in fixes:
            raise SystemExit(f"duplicate fix for {key}")
        fixes[key] = f[1:]
    seen, out, used = set(), [], set()
    for r in rows:
        w, deva, rich, plain, rekhta = r[0], r[1], r[2], r[3], r[4]
        if w in seen:
            continue
        seen.add(w)
        fx = fixes.get(w)
        if fx is not None:
            used.add(w)
            if fx[0] == "-":
                continue
            if fx[0] != ".":
                deva = fx[0]
            if len(fx) > 1 and fx[1] != ".":
                rich = fx[1]
                plain, rekhta = to_plain(rich), to_rekhta(rich)
            if len(fx) > 2 and fx[2] not in (".", ""):
                plain = fx[2]
            if len(fx) > 3 and fx[3] not in (".", ""):
                rekhta = fx[3]
        if not (deva and plain and rekhta):
            raise SystemExit(f"incomplete entry for {w}: {deva!r} {plain!r} {rekhta!r}")
        out.append((w, norm_nasals(deva), plain, rekhta))
    # fixes for words outside the draft are extra entries (rare forms worth pinning)
    for w, fx in fixes.items():
        if w in used or fx[0] == "-":
            continue
        if len(fx) < 2 or "." in (fx[0], fx[1]):
            raise SystemExit(f"new word {w} needs devanagari and R")
        plain = fx[2] if len(fx) > 2 and fx[2] not in (".", "") else to_plain(fx[1])
        rekhta = fx[3] if len(fx) > 3 and fx[3] not in (".", "") else to_rekhta(fx[1])
        out.append((w, norm_nasals(fx[0]), plain, rekhta))
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(HEADER.format(n=f"{len(out):,}"))
        for row in out:
            fh.write("\t".join(row) + "\n")
    print(f"build: {len(out)} entries ({len(used)} hand-fixed) -> {OUT}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["draft", "build"])
    ap.add_argument("--words", type=int, default=10000)
    args = ap.parse_args(argv)
    draft(args.words) if args.step == "draft" else build()


if __name__ == "__main__":
    main()
