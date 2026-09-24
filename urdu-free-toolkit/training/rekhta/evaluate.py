# -*- coding: utf-8 -*-
"""Benchmark the Rekhta-style engine: our student vs Rekhta's teacher vs the
word-level neural engine.

* ``poem``   -- held-out ghazal lines (whole poems never trained on): each
  system's Devanagari against the teacher's label, where both of Rekhta's
  models agreed on it. Measures how faithfully the student learnt Rekhta.
* ``human``  -- Dakshina test sentences with the Roman a person typed:
  word accuracy of each system's casual Roman, compared style-folded.
* ``words``  -- held-out Wiktionary words (``data/translit_ds/test.jsonl``):
  Devanagari exact (nasal spelling folded).

    python -m training.rekhta.evaluate --systems student,teacher,neural
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time

from training.rekhta.dataset import DS, load_split
from urdu_nn.scheme import fold_roman


def _nasal(s: str) -> str:
    s = s.replace("ँ", "ं")
    return re.sub(r"[नम]्(?=[क-ह])", "ं", s)


def _systems(names):
    out = {}
    for n in names:
        if n in ("student", "teacher"):
            import rekhta_translit
            eng = rekhta_translit.RekhtaTransliterator(use_teacher=(n == "teacher"))
            out[n] = lambda text, e=eng: (lambda r: (r["devanagari"], r["plain"]))(e.transliterate_full(text))
        elif n == "neural":
            import neural_translit
            out[n] = lambda text: neural_translit.transliterate(text)[:2]
    return out


def _word_acc(got: str, want: str, fold) -> tuple[int, int]:
    g, w = got.split(), want.split()
    if len(g) != len(w):
        return 0, len(w)
    return sum(fold(a) == fold(b) for a, b in zip(g, w)), len(w)


def _fold_r(s: str) -> str:
    return fold_roman(re.sub(r"[^\w\s'-]", "", s))


def run(names, n_poem=1500, n_human=800, n_words=1500, seed=7) -> dict:
    rng = random.Random(seed)
    systems = _systems(names)
    poem = [r for r in load_split("test") if r[0] == "poem" and r[4]]
    poem = rng.sample(poem, min(n_poem, len(poem)))
    native = (DS.parents[0] / "translit_raw" / "dakshina_dataset_v1.0" / "ur" / "romanized"
              / "ur.romanized.rejoined.test.native.txt").read_text(encoding="utf-8").splitlines()
    roman = (DS.parents[0] / "translit_raw" / "dakshina_dataset_v1.0" / "ur" / "romanized"
             / "ur.romanized.rejoined.test.roman.txt").read_text(encoding="utf-8").splitlines()
    human = rng.sample(list(zip(native, roman)), min(n_human, len(native)))
    words = [json.loads(l) for l in (DS.parents[0] / "translit_ds" / "test.jsonl").open(encoding="utf-8")]
    words = [w for w in words if w["deva"]][:n_words]
    res = {}
    for name, fn in systems.items():
        t0 = time.time()
        r = {}
        # poem lines, Devanagari vs the teacher's agreed label
        out = fn("\n".join(p[1] for p in poem)).__getitem__(0).split("\n")
        ok_l = sum(_nasal(o) == _nasal(p[2]) for o, p in zip(out, poem))
        wa = [_word_acc(o, p[2], _nasal) for o, p in zip(out, poem)]
        r["poem_line"] = round(ok_l / len(poem), 4)
        r["poem_word"] = round(sum(a for a, _ in wa) / sum(b for _, b in wa), 4)
        # human casual Roman
        out = fn("\n".join(u for u, _ in human))[1].split("\n")
        wa = [_word_acc(_fold_r(o), _fold_r(h), lambda x: x) for o, (_, h) in zip(out, human)]
        r["human_roman_word"] = round(sum(a for a, _ in wa) / sum(b for _, b in wa), 4)
        # held-out words
        out = fn("\n".join(w["urdu"] for w in words))[0].split("\n")
        r["words_deva"] = round(sum(_nasal(o.strip()) == _nasal(w["deva"]) for o, w in zip(out, words))
                                / len(words), 4)
        r["sec"] = round(time.time() - t0, 1)
        res[name] = r
        print(name, r, flush=True)
    return res


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="student,teacher,neural")
    a = ap.parse_args(argv)
    res = run(a.systems.split(","))
    (DS / "benchmark.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
