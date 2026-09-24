# -*- coding: utf-8 -*-
"""Label our Urdu corpus with the Rekhta teacher, round-trip every label.

Every Urdu run (see ``urdu_nn.rekhta_text``) from

* ``poem``  -- Urdu Wikisource ghazals/nazms (212k lines),
* ``wiki``  -- Dakshina's Urdu Wikipedia sentences (prose; capped),
* ``dak``   -- Dakshina's romanised dev/test sentences (human Roman: eval),
* ``word``  -- single words from the gold lexicon, the exact dictionary,
  Dakshina and Aksharantar (vocabulary the line corpora rarely repeat),

goes through ``ur2hi`` (Urdu -> Devanagari, exactly the model card's greedy
decoding) and the Devanagari straight back through ``hi2ur``. A label whose
round trip returns the same Urdu is one both of Rekhta's models agree on;
``build`` keeps those. Output is appended to ``data/rekhta_ds/labels.jsonl``
and cached by run, so the command can be stopped and resumed at will.

    python -m training.rekhta.label                     # everything, CPU
    python -m training.rekhta.label --only poem --device cuda
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

from urdu_nn.rekhta_text import urdu_runs

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "translit_raw"
DS = ROOT / "data" / "rekhta_ds"
LABELS = DS / "labels.jsonl"
POETRY = RAW / "wikisource_ur_poetry.jsonl"
DAK = RAW / "dakshina_dataset_v1.0" / "ur"
WIKI = DAK / "native_script_wikipedia" / "ur.wiki-filt.train.text.shuf.txt.gz"
_ARABIC_ONLY = re.compile("[يىكة]")


def rid(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def poem_items():
    for raw in POETRY.open(encoding="utf-8"):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for line in row["lines"]:
            for run in urdu_runs(line):
                yield "poem", row["title"], run


def dak_items():
    for split in ("dev", "test"):
        for line in (DAK / "romanized" / f"ur.romanized.rejoined.{split}.native.txt").open(encoding="utf-8"):
            for run in urdu_runs(line.strip()):
                yield "dak", split, run


def wiki_items(limit: int):
    n = 0
    with gzip.open(WIKI, "rt", encoding="utf-8") as fh:
        for line in fh:
            if len(_ARABIC_ONLY.findall(line)) > 2:     # Qur'an / Arabic quotations
                continue
            for run in urdu_runs(line.strip()):
                if " " not in run:
                    continue
                yield "wiki", rid(line)[:4], run
                n += 1
            if n >= limit:
                return


def word_items(limit: int):
    seen = set()
    sources = []
    lex = ROOT / "data" / "translit_ds" / "lexicon_full.json"
    if lex.exists():
        sources.append(list(json.loads(lex.read_text(encoding="utf-8"))))
    dic = ROOT / "data" / "translit_dictionary.tsv"
    if dic.exists():
        sources.append([l.split("\t")[0] for l in dic.read_text(encoding="utf-8").splitlines()[1:]])
    lexd = DAK / "lexicons" / "ur.translit.sampled.train.tsv"
    if lexd.exists():
        sources.append([l.split("\t")[0] for l in lexd.read_text(encoding="utf-8").splitlines()])
    ak = RAW / "aksharantar_urd.zip"
    if ak.exists():
        z = zipfile.ZipFile(ak)
        words = []
        for name in z.namelist():
            if name.endswith(".json"):
                for line in z.read(name).decode("utf-8").splitlines():
                    words.append(json.loads(line)["native word"])
        sources.append(words)
    n = 0
    for src in sources:
        for w in src:
            for run in urdu_runs(w):
                if " " in run or run in seen:
                    continue
                seen.add(run)
                yield "word", "", run
                n += 1
                if n >= limit:
                    return


def done_ids() -> set[str]:
    ids = set()
    if LABELS.exists():
        for line in LABELS.open(encoding="utf-8"):
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def label(only: list[str], device: str, wiki_limit: int, word_limit: int,
          block: int = 4096, batch: int = 128) -> None:
    from urdu_nn.rekhta_teacher import Teacher
    fwd, back = Teacher("ur2hi", device), Teacher("hi2ur", device)
    done = done_ids()
    print(f"{len(done)} runs already labelled; device {fwd.device}", flush=True)
    gens = {"dak": dak_items, "poem": poem_items,
            "wiki": lambda: wiki_items(wiki_limit), "word": lambda: word_items(word_limit)}
    DS.mkdir(parents=True, exist_ok=True)
    for name in only:
        t0, n, todo = time.time(), 0, []

        def run_block():
            nonlocal n
            if not todo:
                return
            hi = fwd([t[2] for t in todo], batch_size=batch, with_score=True)
            ur = back([h for h, _ in hi], batch_size=batch)
            with LABELS.open("a", encoding="utf-8") as fh:
                for (src, grp, run), (h, lp), u in zip(todo, hi, ur):
                    fh.write(json.dumps({"id": rid(run), "src": src, "grp": grp, "ur": run,
                                         "hi": h, "lp": round(lp, 4), "back": u},
                                        ensure_ascii=False) + "\n")
            n += len(todo)
            todo.clear()
            dt = time.time() - t0
            print(f"  {name}: {n} labelled  {dt:.0f}s  ({n / max(dt, 1e-9):.1f}/s)", flush=True)

        for src, grp, run in gens[name]():
            k = rid(run)
            if k in done:
                continue
            done.add(k)
            todo.append((src, grp, run))
            if len(todo) >= block:
                run_block()
        run_block()
        print(f"{name}: done, {n} new runs", flush=True)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="dak,poem,wiki,word")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wiki-limit", type=int, default=300_000)
    ap.add_argument("--word-limit", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args(argv)
    label(a.only.split(","), a.device, a.wiki_limit, a.word_limit, batch=a.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
