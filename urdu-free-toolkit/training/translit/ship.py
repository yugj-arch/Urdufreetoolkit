# -*- coding: utf-8 -*-
"""Copy the runtime tables built by build_data.py into data/translit_model/
(what neural_translit.py loads and the repo ships):

    lexicon.json.gz   gold lexicon (urdu -> [devanagari, R])
    evidence.json.gz  urdu -> {folded casual roman: weight}  (beam reranking)
    english.json      urdu -> English spelling for loanwords / names
    hindi_forms.json.gz  urdu -> Hindi words that spell it (hindi_fold keys);
                      the runtime prefers beam readings that are real Hindi words

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


def hindi_forms() -> dict[str, list[str]]:
    """Urdu spelling -> the Hindi words (Aksharantar Hindi, ~1M) whose letters
    spell it. Devanagari writes the short vowels Urdu leaves out, so a beam
    reading that is a real Hindi word spelling this Urdu word (चक्कियों for
    چکیوں) beats one that is no word at all (चुकियों). Keys: every Urdu word
    the corpora / Aksharantar / Wiktionary know."""
    from training.translit.build_data import aksharantar, attested_counts
    from training.translit.silver import hindi_casual
    from urdu_nn.scheme import deva_to_urdu_candidates, hindi_fold
    known = set(attested_counts()) | {u for u, _, _ in aksharantar()}
    known |= set(json.loads((DS / "lexicon_full.json").read_text(encoding="utf-8")))
    forms: dict[str, set[str]] = {}
    for d in dict.fromkeys(d for d, _, _ in hindi_casual()):
        for u in deva_to_urdu_candidates(d, limit=48):
            if u in known:
                forms.setdefault(u, set()).add(hindi_fold(d))
    return {u: sorted(v) for u, v in forms.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / "hindi_forms.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(hindi_forms(), fh, ensure_ascii=False, separators=(",", ":"))
    lex = json.loads((DS / "lexicon_full.json").read_text(encoding="utf-8"))
    with gzip.open(OUT / "lexicon.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(lex, fh, ensure_ascii=False, separators=(",", ":"))
    shutil.copy(DS / "evidence_full.json.gz", OUT / "evidence.json.gz")
    shutil.copy(DS / "english_full.json", OUT / "english.json")
    for name in ("lexicon.json.gz", "evidence.json.gz", "english.json", "hindi_forms.json.gz"):
        print(f"{name:18s} {(OUT / name).stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
