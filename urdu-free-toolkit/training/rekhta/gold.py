# -*- coding: utf-8 -*-
"""A gold set from rekhta.org itself: ghazals as Rekhta publishes them in
Urdu, Devanagari and both of its Roman spellings (the marked one,
``dil-e-nādāñ``, and the simple one, ``dil-e-nadan``).

It measures the engine against Rekhta (dev ghazals to look at, test ghazals
only ever scored) and gives Rekhta's word spellings to the engine's lexicon
(``urdu_nn.rekhta_lexicon``). The pages stay local in ``data/rekhta_gold/``
(gitignored); only the lexicon built from them ships. Fetches politely
(``--delay`` seconds between a worker's pages, robots.txt allows /ghazals/)
and resumes where it stopped.

    python -m training.rekhta.gold fetch --per-poet 25 --workers 2
    python -m training.rekhta.gold score --split test --lex dev
    python -m training.rekhta.gold lexicon --split all --ship   # -> data/rekhta_lexicon.json.gz
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "rekhta_gold"
GHAZALS = GOLD / "ghazals.jsonl"
SITE = "https://www.rekhta.org"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
POETS = [
    # classical
    "mirza-ghalib", "meer-taqi-meer", "momin-khan-momin", "dagh-dehlvi", "allama-iqbal",
    "bahadur-shah-zafar", "sheikh-ibrahim-zauq", "altaf-hussain-hali", "akbar-allahabadi",
    "haidar-ali-aatish", "ameer-minai", "hasrat-mohani", "fani-badayuni",
    # modern
    "faiz-ahmad-faiz", "ahmad-faraz", "jaun-eliya", "parveen-shakir", "nasir-kazmi",
    "bashir-badr", "nida-fazli", "firaq-gorakhpuri", "jigar-moradabadi",
    # more of both (a slug that isn't Rekhta's is skipped)
    "khwaja-mir-dard", "nazeer-akbarabadi", "mushafi-ghulam-hamdani", "insha-allah-khan-insha",
    "imam-bakhsh-nasikh", "mirza-mohammad-rafi-sauda", "wali-mohammad-wali", "qaim-chandpuri",
    "shad-azimabadi", "yagana-changezi", "asghar-gondvi", "josh-malihabadi", "asrar-ul-haq-majaz",
    "makhdoom-mohiuddin", "majrooh-sultanpuri", "sahir-ludhianvi", "kaifi-azmi", "ali-sardar-jafri",
    "jan-nisar-akhtar", "akhtar-ul-iman", "ibn-e-insha", "munir-niyazi", "zafar-iqbal",
    "shakeb-jalali", "mohsin-naqvi", "obaidullah-aleem", "jamal-ehsani", "irfan-siddiqi",
    "ahmad-mushtaq", "khalilur-rahman-azmi", "shahryar", "javed-akhtar", "gulzar", "rahat-indori",
    "munawwar-rana", "waseem-barelvi", "ada-jafri", "zehra-nigah", "kishwar-naheed",
    "aziz-lakhnavi", "arzoo-lakhnavi", "seemab-akbarabadi", "jaleel-manikpuri", "riyaz-khairabadi",
    "saqib-lakhnavi", "anwar-shuoor", "abbas-tabish", "tehzeeb-hafi", "ahmad-nadeem-qasmi",
    "qateel-shifai", "habib-jalib", "ehsan-danish", "hafeez-jalandhari", "akhtar-shirani",
    "krishn-bihari-noor", "bekhud-dehlvi", "jurat-qalandar-bakhsh", "mustafa-zaidi",
    "ahmad-faraz", "athar-nafees", "khumar-barabankavi", "shakeel-badayuni", "hasrat-jaipuri",
    "kaleem-aajiz", "nazir-banarasi", "nushur-wahidi", "aal-e-ahmad-suroor", "rais-amrohvi",
    "jigar-moradabadi", "akbar-allahabadi", "mohammad-alvi", "nasir-kazmi", "shuja-khawar",
    "ghulam-mohammad-qasir", "saleem-kausar", "ameer-qazalbash", "iftikhar-arif", "pirzada-qasim",
]
POETS = list(dict.fromkeys(POETS))


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


_SPAN = re.compile(r"<span(?: data-m='([^']*)')?[^>]*>(.*?)</span>", re.S)


def _text(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s))).strip()


def poem_lines(page: str) -> dict[str, list]:
    """The poem on a ghazal page -> {"off": lines, "on": lines, "units": ...}.
    ``off`` is the script the page is in (Rekhta's marked Roman on the
    English page), ``on`` the simple Roman (English page only). ``units``
    holds, per line of ``off``, its words as Rekhta's dictionary groups them:
    [id, text] -- every page tags a word with the same id, and an izafat
    compound written as two Urdu words (دل ناداں) is one id, one
    Devanagari word (दिल-ए-नादाँ) and one Roman word (dil-e-nādāñ)."""
    out: dict[str, list] = {}
    for m in re.finditer(r"<div class='pMC' data-roman='(on|off)'[^>]*>", page):
        flag = m.group(1)
        if flag in out:
            continue
        end = page.find("<div class='pMC'", m.end())
        seg = page[m.end(): end if end > 0 else m.end() + 200000]
        lines, units = [], []
        for c in re.findall(r"<div class='c'>(.*?)</div>", seg, re.S):
            for p in re.findall(r"<p[^>]*>(.*?)</p>", c, re.S):
                lines.append(_text(p))
                us: list[list[str]] = []
                for k, t in _SPAN.findall(p):
                    t = _text(t)
                    if not t:
                        continue
                    if k and us and us[-1][0] == k:
                        us[-1][1] += " " + t
                    else:
                        us.append([k, t])
                units.append(us)
        out[flag] = lines
        if flag == "off":
            out["units"] = units
    return out


def split_of(slug: str) -> str:
    """Whole ghazals go to "dev" (look at it, fix the engine) or "test"
    (only ever scored)."""
    return "test" if int(hashlib.md5(slug.encode()).hexdigest()[:8], 16) % 10 < 3 else "dev"


def load(split: str | None = None) -> list[dict]:
    if not GHAZALS.exists():
        return []
    rows = [json.loads(l) for l in GHAZALS.open(encoding="utf-8") if l.strip()]
    return [r for r in rows if split in (None, r["split"])]


def _ghazal(slug: str, poet: str, delay: float) -> dict | None:
    pages = {}
    try:
        for lang in ("ur", "hi", "en"):
            pages[lang] = poem_lines(_get(f"{SITE}/ghazals/{slug}?lang={lang}"))
            time.sleep(delay)
    except Exception as e:
        print(f"  {slug}: {e}", flush=True)
        return None
    ur, hi = pages["ur"].get("off", []), pages["hi"].get("off", [])
    ro, simple = pages["en"].get("off", []), pages["en"].get("on", [])
    if not ur or not (len(ur) == len(hi) == len(ro)):
        print(f"  {slug}: lines don't line up {len(ur)}/{len(hi)}/{len(ro)}", flush=True)
        return None
    return {"slug": slug, "poet": poet, "split": split_of(slug), "ur": ur, "hi": hi,
            "roman": ro, "simple": simple if len(simple) == len(ur) else [],
            "units": {k: pages[lang].get("units", []) for k, lang in
                      (("ur", "ur"), ("hi", "hi"), ("roman", "en"))}}


def fetch(per_poet: int, delay: float, workers: int = 1) -> None:
    """Up to ``per_poet`` ghazals of each poet (the first page of their
    list), ``workers`` pages in flight at a time, each worker pausing
    ``delay`` seconds between pages."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    GOLD.mkdir(parents=True, exist_ok=True)
    done = collections.Counter()
    seen = set()
    for r in load():
        seen.add(r["slug"])
        done[r["poet"]] += 1
    todo = []
    for poet in POETS:
        if done[poet] >= per_poet:
            continue
        try:
            listing = _get(f"{SITE}/poets/{poet}/ghazals")
        except Exception as e:
            print(f"{poet}: listing failed ({e})", flush=True)
            continue
        time.sleep(delay)
        slugs = [s for s in dict.fromkeys(re.findall(
            r'href="https://www\.rekhta\.org/ghazals/([a-z0-9-]+)"', listing)) if s not in seen]
        seen.update(slugs)
        todo += [(s, poet) for s in slugs[:per_poet - done[poet]]]
        print(f"{poet}: {min(len(slugs), per_poet - done[poet])} to fetch", flush=True)
    print(f"{len(todo)} ghazals to fetch", flush=True)
    lock = threading.Lock()
    with GHAZALS.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as pool:
        for n, row in enumerate(pool.map(lambda t: _ghazal(t[0], t[1], delay), todo), 1):
            if row:
                with lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
            if n % 50 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

_EDGE = re.compile(r"^[^\wऀ-ॿ]+|[^\wऀ-ॿ]+$")


def tokens(line: str, roman: bool = False) -> list[str]:
    """Words of a line, punctuation and the takhallus quotes ('ġhālib')
    stripped from their edges; Roman compared case-blind."""
    line = unicodedata.normalize("NFC", line.replace("’", "'").replace("‘", "'"))
    out = []
    for w in line.split():
        w = _EDGE.sub("", w)
        if w:
            out.append(w.lower() if roman else w)
    return out


def _pairs(got: list[str], want: list[str]):
    """(ours, Rekhta's) for every word that differs, lined up by difflib."""
    import difflib
    sm = difflib.SequenceMatcher(a=got, b=want, autojunk=False)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            continue
        if a1 - a0 == b1 - b0:
            yield from zip(got[a0:a1], want[b0:b1])
        else:
            yield " ".join(got[a0:a1]), " ".join(want[b0:b1])


def score(rows: list[dict], engine, label: str = "", dump: Path | None = None) -> dict:
    """Run ``engine`` (text -> {devanagari, roman_diacritic, roman, plain})
    on every ghazal and compare with what Rekhta published."""
    import collections
    keys = (("hi", "devanagari", False), ("roman", "roman_diacritic", True), ("simple", "roman", True))
    tot = {k: [0, 0, 0, 0] for k, _, _ in keys}      # lines ok, lines, words ok, words
    diffs = {k: collections.Counter() for k, _, _ in keys}
    examples = []
    for r in rows:
        out = engine("\n".join(r["ur"]))
        for k, ok_key, roman in keys:
            if not r.get(k) or ok_key not in out:
                continue
            ours = out[ok_key].split("\n")
            for i, want_line in enumerate(r[k]):
                got = tokens(ours[i] if i < len(ours) else "", roman)
                want = tokens(want_line, roman)
                t = tot[k]
                t[1] += 1
                t[3] += len(want)
                bad = list(_pairs(got, want))
                t[2] += len(want) - sum(len(w.split()) for _, w in bad)
                if got == want:
                    t[0] += 1
                else:
                    for p in bad:
                        diffs[k][p] += 1
                    examples.append((r["slug"], k, r["ur"][i], ours[i] if i < len(ours) else "", want_line))
    res = {k: {"line": round(t[0] / max(t[1], 1), 4), "word": round(t[2] / max(t[3], 1), 4),
               "lines": t[1], "words": t[3]} for k, t in tot.items()}
    print(label, json.dumps(res), flush=True)
    if dump:
        with dump.open("w", encoding="utf-8") as fh:
            for k in diffs:
                fh.write(f"## {k}: ours -> rekhta (count)\n")
                for (a, b), n in diffs[k].most_common():
                    fh.write(f"{n}\t{a}\t{b}\n")
            fh.write("\n## lines\n")
            for e in examples:
                fh.write("\t".join(e) + "\n")
    return res


def engine_fn(mode: str = None, lexicon_path=None, device: str = "cpu"):
    import rekhta_translit
    eng = rekhta_translit.RekhtaTransliterator(lexicon_path=lexicon_path, device=device,
                                               **({"mode": mode} if mode else {}))
    return eng.transliterate_full


def build_lexicon(split: str, path: Path) -> Path:
    from urdu_nn.rekhta_lexicon import RekhtaLexicon
    lex = RekhtaLexicon.build(load(None if split == "all" else split))
    lex.save(path)
    print(f"lexicon from {split}: {len(lex)} entries -> {path}", flush=True)
    return path


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "stats", "score", "lexicon"])
    ap.add_argument("--ship", action="store_true", help="lexicon: write data/rekhta_lexicon.json.gz")
    ap.add_argument("--per-poet", type=int, default=8)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--split", default="dev", help="ghazals to score: dev, test or all")
    ap.add_argument("--lex", default="none",
                    help="lexicon for scoring: none, or the split to build it from (dev/all)")
    ap.add_argument("--mode", default=None, help="engine mode (ensemble/student/teacher)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.per_poet, a.delay, a.workers)
    if a.cmd == "lexicon":
        build_lexicon(a.split, ROOT / "data" / "rekhta_lexicon.json.gz" if a.ship else GOLD / "lexicon.json.gz")
        return 0
    if a.cmd == "score":
        lex = Path("-none-") if a.lex == "none" else build_lexicon(a.lex, GOLD / f"lexicon_{a.lex}.json.gz")
        rows = load(None if a.split == "all" else a.split)
        score(rows, engine_fn(a.mode, lex, a.device), f"{a.split} lex={a.lex}",
              GOLD / f"diff_{a.split}_lex-{a.lex}.txt")
        return 0
    rows = load()
    for sp in ("dev", "test"):
        rs = [r for r in rows if r["split"] == sp]
        print(sp, len(rs), "ghazals", sum(len(r["ur"]) for r in rs), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
