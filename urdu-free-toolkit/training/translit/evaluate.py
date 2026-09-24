# -*- coding: utf-8 -*-
"""Benchmark transliteration systems on held-out, human-authored gold data.

  A. words      -- Wiktionary words/inflections held out of training AND out of
                   the lexicon (data/translit_ds/test.jsonl): Devanagari exact
                   match + Roman match (style-folded, so casual vs scholarly
                   spelling conventions don't count as errors)
  B. sentences  -- Dakshina test sentences with human-typed Roman (never
                   trained on): word accuracy + character error rate, folded

    python -m training.translit.evaluate --systems rule,neural,gpt --words 500 --sents 120
    (any of gpt / claude / gemini / groq, whichever has a key in .env)

    python -m training.translit.evaluate --systems neural,neural+dict --words 0 --sents 0
    ("+dict" = with the exact dictionary data/translit_dictionary.tsv; 0 = the
    whole held-out set. Held-out Wiktionary words are removed from the
    dictionary for the word benchmark, so no system is scored on a word whose
    reading it was handed.)

Writes data/translit_model/benchmark.json and prints a table.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from urdu_nn.scheme import fold_roman, to_plain  # noqa: E402

DS = ROOT / "data" / "translit_ds"
DAK = ROOT / "data" / "translit_raw" / "dakshina_dataset_v1.0" / "ur" / "romanized"


def norm_deva(s: str) -> str:
    s = unicodedata.normalize("NFC", s).replace("़", "").replace("ँ", "ं")
    s = s.replace("‍", "").replace("‌", "")
    s = re.sub("[ङञणनम]्(?=[क-ह])", "ं", s)   # सम्बन्ध = संबंध
    return s.strip(" ।.")


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# systems: each maps list[str] (lines of Urdu) -> list[(deva, plain)] or None
# ---------------------------------------------------------------------------

def sys_rule(exact: dict | None):
    """The rule engine, with the exact dictionary swapped for ``exact``
    ({} = without it)."""
    import transliterate

    def run(lines):
        saved = transliterate._EXACT
        transliterate._EXACT = {transliterate._fold_key(k): v for k, v in (exact or {}).items()}
        try:
            return [transliterate.transliterate(l) for l in lines]
        finally:
            transliterate._EXACT = saved
    return run


def sys_neural(lexicon: Path, evidence: Path | None, ev_weight: float, exact: dict | None):
    from neural_translit import NeuralTransliterator
    eng = NeuralTransliterator(lexicon_path=lexicon, evidence_path=evidence,
                               english_path=None if evidence is None else DS / "english_train.json",
                               dictionary_path=None, evidence_weight=ev_weight)
    eng.exact = dict(exact or {})
    print(f"  neural: model={'yes' if eng.has_model else 'NO'} lexicon={len(eng.lexicon)} "
          f"evidence={len(eng.evidence)} weight={ev_weight} dictionary={len(eng.exact)}",
          flush=True)
    return lambda lines: [eng.transliterate(l)[:2] for l in lines]


def with_dictionary(fn_for, exact_all: dict, held: set[str]):
    """A system factory run with the dictionary minus held-out gold words:
    words are scored without them, sentences with the whole dictionary."""
    exact_words = {k: v for k, v in exact_all.items() if k not in held}
    fw, fs = fn_for(exact_words), fn_for(exact_all)
    return fw, fs


def sys_llm(provider_id: str):
    """Any of the app's API transliteration providers (gpt / claude / gemini /
    groq), with their own production prompts. Lines go in blocks of 40 (split
    smaller when a reply's line count doesn't match); a line the API never
    answered comes back as None and is excluded from scoring, not counted wrong.
    Replies are cached per provider+model so reruns cost nothing."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from providers import registry
    from providers.base import Capability, TranslitOpts
    prov = registry.get(Capability.TRANSLIT, provider_id)
    ok, why = prov.available()
    if not ok:
        raise SystemExit(f"{provider_id}: {why}")
    common = __import__(f"providers.{_COMMON[provider_id]}", fromlist=["MODEL"])
    model = getattr(common, "MODEL", provider_id)
    cache_path = ROOT / "data" / "translit_model" / f"llm_cache_{provider_id}_{model}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    dead = {"n": 0}

    def call(block: str):
        if dead["n"] >= 3:                 # quota gone / key revoked: stop hammering
            return None
        for attempt in range(3):
            res = prov.translit(block, TranslitOpts())
            if res.ok:
                dead["n"] = 0
                return res
            print(f"   {provider_id} error: {res.error[:160]}", flush=True)
            if "quota" in res.error or "credit" in res.error or "401" in res.error:
                dead["n"] += 1
                return None
            time.sleep(2 + attempt * 3)
        return None

    def _run(lines, chunk=40):
        out = []
        for i in range(0, len(lines), chunk):
            part = lines[i:i + chunk]
            res = call("\n".join(part))
            if res is None:
                out.extend([None] * len(part))
                continue
            dv, ro = res.devanagari.strip().split("\n"), res.roman.strip().split("\n")
            if len(dv) == len(part) and len(ro) == len(part):
                out.extend(zip(dv, ro))
            elif chunk > 1:
                out.extend(_run(part, max(1, chunk // 5)))
            else:
                out.append((dv[0], ro[0]))
        return out

    def run(lines):
        todo = [l for l in dict.fromkeys(lines) if l not in cache]
        for l, r in zip(todo, _run(todo)):
            if r is not None:
                cache[l] = list(r)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        return [tuple(cache[l]) if l in cache else None for l in lines]
    return run, f"{provider_id}: {model}"


_COMMON = {"gpt": "_openai_common", "claude": "_anthropic_common",
           "gemini": "_gemini_common", "groq": "_groq_common"}


# ---------------------------------------------------------------------------

def eval_words(fn, words):
    hyps = fn([w["urdu"] for w in words])
    answered = [(w, h) for w, h in zip(words, hyps) if h is not None]
    dv_ok = ro_ok = dv_n = 0
    errors = []
    for w, (d, p) in answered:
        g_plain = to_plain(w["rich"])
        r_ok = fold_roman(p.strip()) == fold_roman(g_plain)
        ro_ok += r_ok
        if w["deva"]:
            dv_n += 1
            d_ok = norm_deva(d) == norm_deva(w["deva"])
            dv_ok += d_ok
        else:
            d_ok = True
        if not (r_ok and d_ok) and len(errors) < 25:
            errors.append(f"{w['urdu']}  gold {w['deva']}|{g_plain}  got {d}|{p}")
    return {"deva_acc": dv_ok / max(1, dv_n), "roman_acc": ro_ok / max(1, len(answered)),
            "n": len(answered), "coverage": len(answered) / len(words), "errors": errors}


def _words(s: str) -> list[str]:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return s.split()


def eval_sents(fn, sents):
    hyps = fn([u for u, _ in sents])
    answered = [(s_, h) for s_, h in zip(sents, hyps) if h is not None]
    w_ok = w_n = 0
    ed = tot = 0
    for (u, gold), (_, p) in answered:
        g, h = [fold_roman(x) for x in _words(gold)], [fold_roman(x) for x in _words(p)]
        gs, hs = " ".join(g), " ".join(h)
        ed += levenshtein(gs, hs)
        tot += len(gs)
        if len(g) == len(h):
            w_n += len(g)
            w_ok += sum(a == b for a, b in zip(g, h))
    return {"word_acc": w_ok / max(1, w_n), "cer": ed / max(1, tot), "n": len(answered),
            "coverage": len(answered) / len(sents)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="rule,neural")
    ap.add_argument("--words", type=int, default=500, help="0 = all held-out words")
    ap.add_argument("--sents", type=int, default=120, help="0 = all test sentences")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sent-split", default="test", choices=["dev", "test"],
                    help="Dakshina split for sentences (tune on dev, report on test)")
    ap.add_argument("--ev-weight", type=float, default=2.0)
    ap.add_argument("--no-evidence", action="store_true")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    words = [json.loads(l) for l in (DS / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    held = {w["urdu"] for w in words}
    rng.shuffle(words)
    words = words[: args.words or None]
    nat = (DAK / f"ur.romanized.rejoined.{args.sent_split}.native.txt").read_text(encoding="utf-8").splitlines()
    rom = (DAK / f"ur.romanized.rejoined.{args.sent_split}.roman.txt").read_text(encoding="utf-8").splitlines()
    pairs = [(n, r) for n, r in zip(nat, rom) if 4 <= len(n.split()) <= 30]
    rng.shuffle(pairs)
    sents = pairs[: args.sents or None]

    lex_train = DS / "lexicon_train.json.gz"
    with gzip.open(lex_train, "wt", encoding="utf-8") as fh:
        json.dump(json.loads((DS / "lexicon_train.json").read_text(encoding="utf-8")), fh,
                  ensure_ascii=False)

    from urdu_nn.scheme import norm_urdu
    from transliterate import load_exact_dictionary
    exact_all = {norm_urdu(k): v for k, v in load_exact_dictionary().items()}
    ev = None if args.no_evidence else DS / "evidence_train.json.gz"

    results = {}
    for name in args.systems.split(","):
        t0 = time.time()
        print(f"== {name}", flush=True)
        base, dict_on = name.removesuffix("+dict"), name.endswith("+dict")
        if base == "rule":
            fn_for, label = sys_rule, "rule engine (offline)"
        elif base == "neural":
            fn_for = lambda ex: sys_neural(lex_train, ev, args.ev_weight, ex)  # noqa: E731
            label = "neural (offline)"
        elif name in _COMMON:
            fn, label = sys_llm(name)
            fn_for = None
        else:
            raise SystemExit(f"unknown system {name}")
        if fn_for is None:
            fw = fs = fn
        elif dict_on:
            if not exact_all:
                raise SystemExit("no data/translit_dictionary.tsv -- build it first")
            fw, fs = with_dictionary(fn_for, exact_all, held)
            label += f" + exact dictionary ({len(exact_all):,} words)"
        else:
            fw = fs = fn_for({})
        res = {"label": label, "words": eval_words(fw, words), "sents": eval_sents(fs, sents)}
        res["seconds"] = round(time.time() - t0, 1)
        results[name] = res
        w, s = res["words"], res["sents"]
        print(f"   words: deva {w['deva_acc']:.1%}  roman {w['roman_acc']:.1%}   "
              f"sents: word {s['word_acc']:.1%}  CER {s['cer']:.3f}   "
              f"coverage {w['coverage']:.0%}/{s['coverage']:.0%}  ({res['seconds']}s)", flush=True)
    out = ROOT / "data" / "translit_model" / "benchmark.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n| system | words: Devanagari | words: Roman | sentences: word acc | sentences: CER |")
    print("|---|---|---|---|---|")
    for r in results.values():
        w, s = r["words"], r["sents"]
        print(f"| {r['label']} | {w['deva_acc']:.1%} | {w['roman_acc']:.1%} | "
              f"{s['word_acc']:.1%} | {s['cer']:.3f} |")


if __name__ == "__main__":
    main()
