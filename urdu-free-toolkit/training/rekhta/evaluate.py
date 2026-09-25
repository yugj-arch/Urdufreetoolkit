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
from pathlib import Path

from training.rekhta.dataset import DS, _conv_fold, load_split
from urdu_nn.scheme import fold_roman


def _nasal(s: str) -> str:
    s = s.replace("ँ", "ं")
    return re.sub(r"[नम]्(?=[क-ह])", "ं", s)


def _held_out() -> set[str]:
    from urdu_nn.scheme import norm_urdu
    return {norm_urdu(json.loads(l)["urdu"])
            for l in (DS.parents[0] / "translit_ds" / "test.jsonl").open(encoding="utf-8")}


def _systems(names, model_path=None, device="cpu"):
    out = {}
    for n in names:
        if n in ("student", "teacher", "ensemble"):
            import rekhta_translit
            kw = {"model_path": Path(model_path)} if model_path else {}
            eng = rekhta_translit.RekhtaTransliterator(mode=n, corrections_path=Path("-none-"),
                                                       device=device, **kw)
            out[n] = lambda text, e=eng: (lambda r: (r["devanagari"], r["plain"]))(e.transliterate_full(text))
        elif n == "neural":
            import neural_translit
            eng = neural_translit.NeuralTransliterator()
            for w in _held_out():       # the shipped tables contain the held-out words
                for table in (eng.lexicon, eng.exact, eng.gpt_words):
                    table.pop(w, None)
            out[n] = lambda text, e=eng: e.transliterate(text)[:2]
    return out


def _tokens(s: str, fold) -> list[str]:
    """Words of a line, hyphens split and bare izafat / conjunctive parts
    dropped (khauf-e-rasan, khaufe rasan and khauf rasan all -> khauf rasan)."""
    toks = [fold(t) for t in re.split(r"[\s\-]+", s) if t]
    return [t for t in toks if t and t not in ("e", "ए", "o", "ओ")]


def _word_acc(got: str, want: str, fold) -> tuple[int, int]:
    """Reference words matched in order (LCS), so one merged or split word
    doesn't zero the rest of the line."""
    g, w = _tokens(got, fold), _tokens(want, fold)
    prev = [0] * (len(g) + 1)
    for b in w:
        cur = [0]
        for j, a in enumerate(g):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1], len(w)


def _fold_r(s: str) -> str:
    return fold_roman(re.sub(r"[^\w\s'-]", "", s))


def run(names, n_poem=1500, n_human=800, n_words=1500, seed=7, model_path=None,
        device="cpu") -> dict:
    rng = random.Random(seed)
    systems = _systems(names, model_path, device)
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
        wa = [_word_acc(o, h, _fold_r) for o, (_, h) in zip(out, human)]
        r["human_roman_word"] = round(sum(a for a, _ in wa) / sum(b for _, b in wa), 4)
        # held-out words
        out = fn("\n".join(w["urdu"] for w in words))[0].split("\n")
        # Wiktionary's spellings drop most nuktas and ain marks: compare the word, not the convention
        r["words_deva"] = round(sum(_conv_fold(o.strip()) == _conv_fold(w["deva"])
                                    for o, w in zip(out, words)) / len(words), 4)
        r["sec"] = round(time.time() - t0, 1)
        res[name] = r
        print(name, r, flush=True)
    return res


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="student,teacher,neural")
    ap.add_argument("--out", default=str(DS / "benchmark.json"))
    ap.add_argument("--model", default=None, help="student checkpoint (default: data/rekhta_model/model.pt)")
    ap.add_argument("--n", type=float, default=1.0, help="fraction of each test set, for quick runs")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    res = run(a.systems.split(","), n_poem=int(1500 * a.n), n_human=int(800 * a.n),
              n_words=int(1500 * a.n), model_path=a.model, device=a.device)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
