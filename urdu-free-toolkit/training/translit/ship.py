# -*- coding: utf-8 -*-
"""Copy the runtime tables built by build_data.py into data/translit_model/
(what neural_translit.py loads and the repo ships):

    lexicon.json.gz   gold lexicon (urdu -> [devanagari, R])
    evidence.json.gz  urdu -> {folded casual roman: weight}  (beam reranking)
    english.json      urdu -> English spelling for loanwords / names

    python -m training.translit.ship
"""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "translit_ds"
OUT = ROOT / "data" / "translit_model"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lex = json.loads((DS / "lexicon_full.json").read_text(encoding="utf-8"))
    with gzip.open(OUT / "lexicon.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(lex, fh, ensure_ascii=False, separators=(",", ":"))
    shutil.copy(DS / "evidence_full.json.gz", OUT / "evidence.json.gz")
    shutil.copy(DS / "english_full.json", OUT / "english.json")
    for name in ("lexicon.json.gz", "evidence.json.gz", "english.json"):
        print(f"{name:18s} {(OUT / name).stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
