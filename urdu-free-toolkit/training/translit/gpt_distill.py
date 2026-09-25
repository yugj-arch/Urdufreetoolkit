# -*- coding: utf-8 -*-
"""Distil the GPT transliteration column into the offline ``neural`` engine.

The app's GPT provider is the reference the user compares every offline engine
against. This script runs real Urdu text through *exactly* that call (same
model, same ``TEXT_SYSTEM`` prompt, same sampling) and turns the replies into
lookup tables the neural engine consults before anything else
(``data/translit_model/gpt.json.gz``):

  * lines   -- every labelled line, verbatim: a line GPT has read comes back
               byte-for-byte as GPT wrote it
  * words   -- per Urdu word, GPT's majority spelling in each script
               (Devanagari / plain Roman / Rekhta Roman, voted separately)
  * ctx     -- (previous word, word) and (word, next word) readings where GPT
               consistently reads a homograph differently in that context
  * joins   -- how GPT joins a word pair: space, hyphen, or an izafat -e-
               (GPT restores unwritten izafat: taaqat-e-bedaad)

    python -m training.translit.gpt_distill fetch            # Wikisource poetry -> data/translit_raw/
    python -m training.translit.gpt_distill label --calls 300
    python -m training.translit.gpt_distill build            # -> data/translit_model/gpt.json.gz
    python -m training.translit.gpt_distill eval             # neural vs GPT on held-out chunks

Text sources: Urdu Wikisource poetry (public domain / CC BY-SA -- pages carry
``<poem>`` blocks, mostly {{PD-old}} classical ghazals) and Dakshina's Urdu
Wikipedia sentences. Labels are cached in ``data/translit_ds/gpt_labels.jsonl``
so reruns and rebuilds cost nothing; ``label`` only pays for new chunks.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POETRY = ROOT / "data" / "translit_raw" / "wikisource_ur_poetry.jsonl"
WIKI = (ROOT / "data" / "translit_raw" / "dakshina_dataset_v1.0" / "ur"
        / "native_script_wikipedia" / "ur.wiki-filt.train.text.shuf.txt.gz")
LABELS = ROOT / "data" / "translit_ds" / "gpt_labels.jsonl"
OUT = ROOT / "data" / "translit_model" / "gpt.json.gz"

_WS_API = "https://ur.wikisource.org/w/api.php"
_URDU = re.compile(r"[؀-ۿݐ-ݿ]")
TEST_SHARE = 0.08          # chunks held out (by poem / sentence group) for `eval`
POEM_CHUNK = 16            # max lines per GPT call -- a whole ghazal, as a user would paste it
PROSE_CHUNK = 8


# ---------------------------------------------------------------------------
# fetch: Urdu Wikisource <poem> blocks
# ---------------------------------------------------------------------------

def _clean_wikitext_line(line: str) -> str:
    line = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", line)
    line = re.sub(r"\{\{[^{}]*\}\}", "", line)
    line = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", line)
    line = re.sub(r"<[^>]+>|'{2,}|&nbsp;", " ", line)
    line = line.lstrip(":*#; ").strip()
    return re.sub(r"\s+", " ", line)


def poem_lines(wikitext: str) -> list[str]:
    out = []
    for block in re.findall(r"<poem[^>]*>(.*?)</poem>", wikitext, flags=re.S):
        for raw in block.split("\n"):
            line = _clean_wikitext_line(raw)
            letters = [c for c in line if c.isalpha()]
            if not letters:
                out.append("")          # stanza / couplet break
                continue
            if sum(1 for c in letters if _URDU.match(c)) / len(letters) < 0.9:
                continue
            out.append(line)
    return out


def fetch(out: Path = POETRY, delay: float = 0.5) -> int:
    import requests
    sess = requests.Session()
    sess.headers["User-Agent"] = "urdu-free-toolkit/1.0 (transliteration research; offline tables)"
    params = {"action": "query", "generator": "allpages", "gapnamespace": 0, "gaplimit": 50,
              "prop": "revisions", "rvprop": "content", "rvslots": "main",
              "format": "json", "formatversion": 2}
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        while True:
            for attempt in range(5):
                try:
                    r = sess.get(_WS_API, params=params, timeout=60)
                    r.raise_for_status()
                    d = r.json()
                    break
                except Exception as e:  # noqa: BLE001
                    print("  retry:", e)
                    time.sleep(5 * (attempt + 1))
            else:
                raise SystemExit("Wikisource fetch failed")
            for page in d.get("query", {}).get("pages", []):
                revs = page.get("revisions") or []
                if not revs:
                    continue
                text = revs[0]["slots"]["main"].get("content", "")
                if "<poem" not in text:
                    continue
                lines = poem_lines(text)
                if sum(1 for l in lines if l) < 2:
                    continue
                m = re.search(r"\|\s*author\s*=\s*([^\n|]*)", text)
                fh.write(json.dumps({"title": page["title"],
                                     "author": (m.group(1).strip() if m else ""),
                                     "lines": lines}, ensure_ascii=False) + "\n")
                n += 1
            if "continue" not in d:
                break
            params.update(d["continue"])
            print(f"  {n} poems ...", flush=True)
            time.sleep(delay)
    print(f"wrote {n} poems -> {out}")
    return n


# ---------------------------------------------------------------------------
# label: the app's own GPT call over chunks of real text
# ---------------------------------------------------------------------------

def _split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    return "test" if h / 0xFFFFFFFF < TEST_SHARE else "train"


def _chunk_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def poem_chunks() -> list[dict]:
    """Whole ghazals (<= POEM_CHUNK lines), longer poems cut at couplet breaks."""
    chunks, seen = [], set()
    if not POETRY.exists():
        return chunks
    for raw in POETRY.open(encoding="utf-8"):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:            # a line `fetch` is still writing
            continue
        split = _split_of("poem:" + row["title"])
        cur: list[str] = []
        for line in row["lines"] + [""]:
            if line:
                if line in seen:
                    continue
                seen.add(line)
                cur.append(line)
                continue
            if len(cur) >= POEM_CHUNK - 1:
                chunks.append({"kind": "poem", "split": split, "src": row["title"], "urdu": cur})
                cur = []
        for i in range(0, len(cur), POEM_CHUNK):
            part = cur[i:i + POEM_CHUNK]
            if part:
                chunks.append({"kind": "poem", "split": split, "src": row["title"], "urdu": part})
    return chunks


def prose_chunks(limit: int = 4000) -> list[dict]:
    chunks, cur = [], []
    if not WIKI.exists():
        return chunks
    with gzip.open(WIKI, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = re.sub(r"\s+", " ", line).strip()
            if not (5 <= len(line.split()) <= 22) or not _URDU.search(line):
                continue
            cur.append(line)
            if len(cur) == PROSE_CHUNK:
                chunks.append({"kind": "prose", "split": _split_of("prose:" + cur[0]),
                               "src": "dakshina", "urdu": cur})
                cur = []
                if len(chunks) >= limit:
                    break
    return chunks


def _gpt(text: str) -> tuple[dict, dict]:
    """The exact call ``providers/translit/gpt.py`` makes -> (reply, usage)."""
    from providers import _openai_common as common
    resp = common.get_client().chat.completions.create(
        model=common.MODEL,
        response_format={"type": "json_object"},
        **common.sampling_kwargs(),
        messages=[{"role": "system", "content": common.TEXT_SYSTEM},
                  {"role": "user", "content": text}],
    )
    u = resp.usage
    usage = {"in": u.prompt_tokens, "out": u.completion_tokens,
             "reason": getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0}
    return common.parse_json(resp.choices[0].message.content), usage


def load_labels() -> dict[str, dict]:
    rows = {}
    if LABELS.exists():
        for line in LABELS.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def label(calls: int, workers: int = 16, prose_share: float = 0.2, seed: int = 7) -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    done = load_labels()
    rng = random.Random(seed)
    poems, prose = poem_chunks(), prose_chunks()
    rng.shuffle(poems)
    rng.shuffle(prose)
    n_prose = int(calls * prose_share)
    todo = []
    for pool, want in ((prose, n_prose), (poems, calls - n_prose)):
        for c in pool:
            if want <= 0:
                break
            c["id"] = _chunk_id("\n".join(c["urdu"]))
            if c["id"] in done:
                continue
            todo.append(c)
            want -= 1
    print(f"{len(done)} chunks cached; labelling {len(todo)} new "
          f"({sum(c['kind'] == 'poem' for c in todo)} poem, "
          f"{sum(c['kind'] == 'prose' for c in todo)} prose) with {workers} workers", flush=True)
    lock = threading.Lock()
    stop = threading.Event()
    tok = collections.Counter()
    LABELS.parent.mkdir(parents=True, exist_ok=True)

    def work(c):
        if stop.is_set():
            return None
        for attempt in range(6):
            try:
                reply, usage = _gpt("\n".join(c["urdu"]))
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "insufficient_quota" in msg or "credit_balance" in msg:
                    stop.set()
                    print("  OpenAI credits exhausted -- stopping", flush=True)
                    return None
                time.sleep(min(60, 4 * 2 ** attempt))
        else:
            return None
        row = dict(c, deva=_lines(reply.get("devanagari")), roman=_lines(reply.get("roman")),
                   dia=_lines(reply.get("roman_diacritic")), usage=usage, at=time.time())
        with lock:
            with LABELS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            tok.update(usage)
        return row

    t0, n = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(work, c) for c in todo]):
            if fut.result():
                n += 1
                if n % 10 == 0:
                    print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s  "
                          f"tokens in={tok['in']} out={tok['out']} (reasoning {tok['reason']})",
                          flush=True)
    print(f"done: {n} new chunks, tokens in={tok['in']} out={tok['out']}")


def _lines(s) -> list[str]:
    s = (s or "").replace("\r", "")
    return [l.strip() for l in s.split("\n") if l.strip()]


# ---------------------------------------------------------------------------
# build: align GPT's lines to the Urdu words
# ---------------------------------------------------------------------------

_IZ = {"e", "ए", "-e", "ye"}                     # izafat marker parts
_PUNCT_EDGE = re.compile(r"^(?:[^\wऀ-ॿ]|[।॥])+|(?:[^\wऀ-ॿ]|[।॥])+$")


def urdu_tokens(line: str) -> list[str]:
    """The Urdu words of a line as the neural engine keys them."""
    import neural_translit as nt
    import transliterate as rule
    from urdu_nn.scheme import norm_urdu
    toks = []
    for part in nt._split(unicodedata.normalize("NFC", line)):
        if nt._URDU_RUN.fullmatch(part):
            _, core, _ = rule._split_leading_trailing_punct(part)
            k = norm_urdu(core)
            if k:
                toks.append(k)
    return toks


def line_key(line: str) -> str:
    """Normalised line for the verbatim-line table (the engine's own key)."""
    import neural_translit as nt
    return nt._line_key(line)


def _units(out_line: str) -> list[list[str]]:
    """GPT output line -> [[word, joiner-after]]: split on spaces and hyphens;
    a bare izafat part (e / ए) becomes the joiner "-e-" of the word before
    it; digits (which the Urdu tokens never include) are dropped."""
    units = []
    for w in out_line.split():
        w = _PUNCT_EDGE.sub("", w)
        if not w:
            continue
        parts = w.split("-")
        for j, p in enumerate(parts):
            p = _PUNCT_EDGE.sub("", p)
            if not p:
                continue
            if all(ch.isdigit() or ch in ".,:/" for ch in p):
                continue
            if p.lower() in _IZ and units and j > 0:
                units[-1][1] = "-e-"
                continue
            if units and not units[-1][1]:
                units[-1][1] = "-" if j > 0 else " "
            units.append([p, ""])
    for u in units:
        u[1] = u[1] or " "
    return units


def _fold(s: str) -> str:
    """Loose form for aligning GPT's words with a rough reading."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch) and ch not in "़्ंँः'’.-")
    s = s.replace("w", "v").replace("ee", "i").replace("oo", "u")
    return re.sub(r"(.)\1+", r"\1", s)


def _rough(tok: str, script: str) -> str:
    import transliterate as rule
    deva, plain = rule.transliterate_word(rule._fold_key(tok))
    return _fold(deva if script == "deva" else plain)


def _sim(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.normalized_similarity(a, b)


_MIN_SIM, _MIN_SIM_JOINED, _OP_COST = 0.3, 0.55, 0.15


def align(utoks: list[str], out_line: str, script: str = "roman"):
    """GPT output line -> per Urdu word (reading, joiner-to-next), or None
    when the line can't be aligned confidently.

    Mostly one Urdu word per output word; GPT may also write two Urdu words
    as one (اس کا -> iska, جمشید پور -> Jamshedpur: the first gets the whole
    reading and joiner "+", the second (None, joiner)) or one as two
    (کرسامنے -> kar saamne). Choices are scored against a rough rule-engine
    reading of each word."""
    units = _units(out_line)
    n, m = len(utoks), len(units)
    if not n or not m:
        return None
    rough = [_rough(t, script) for t in utoks]
    iz = ("ए", "ये") if script == "deva" else ("e", "ye")

    def one(i, j):
        """Word i read as unit j; an izafat unit (bū + -e-) is also tried
        with the e the Urdu spells out (بوئے)."""
        u = units[j][0]
        forms = [u] + ([u + x for x in iz] if units[j][1] == "-e-" else [])
        sc = max(_sim(rough[i], _fold(f)) for f in forms)
        return max(sc, _MIN_SIM) if len(utoks[i]) < 3 else sc

    if n == m and all(one(i, i) >= _MIN_SIM for i in range(n)):
        return [(u[0], u[1]) for u in units[:-1]] + [(units[-1][0], " ")]
    NEG = -1e9
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if best[i][j] == NEG:
                continue
            cands = []
            if i < n and j < m:
                cands.append((1, 1, one(i, j), _MIN_SIM))
            if i + 1 < n and j < m:
                cands.append((2, 1, _sim(rough[i] + rough[i + 1], _fold(units[j][0])),
                              _MIN_SIM_JOINED))
            if i < n and j + 1 < m and units[j][1] != "-e-":
                # a hyphen inside one GPT word (मा-सिवा) is itself evidence
                cands.append((1, 2, _sim(rough[i], _fold(units[j][0] + units[j + 1][0])),
                              0.4 if units[j][1] == "-" else _MIN_SIM_JOINED))
            if i < n and utoks[i] == "ء":                      # 1828ء: GPT keeps just the year
                cands.append((1, 0, 0.5, 0))
            for di, dj, sc, floor in cands:
                if sc < floor:
                    continue
                if (di, dj) != (1, 1):
                    sc -= _OP_COST
                if best[i][j] + sc > best[i + di][j + dj]:
                    best[i + di][j + dj] = best[i][j] + sc
                    back[i + di][j + dj] = (di, dj)
    if best[n][m] == NEG:
        return None
    out, i, j = [], n, m
    while i or j:
        di, dj = back[i][j]
        i, j = i - di, j - dj
        if (di, dj) == (1, 1):
            out.append([(units[j][0], units[j][1])])
        elif (di, dj) == (2, 1):
            out.append([(units[j][0], "+"), (None, units[j][1])])
        elif (di, dj) == (1, 2):
            out.append([(units[j][0] + units[j][1] + units[j + 1][0], units[j + 1][1])])
        else:
            out.append([(None, " ")])
    res = [x for step in reversed(out) for x in step]
    if res[-1][1] not in ("+",):
        res[-1] = (res[-1][0], " ")
    return res


SCRIPTS = ("deva", "roman", "dia")
FUTURE = ("گا", "گے", "گی")


def build(out: Path = OUT, include_test: bool = True) -> dict:
    rows = [r for r in load_labels().values() if include_test or r["split"] == "train"]
    lines: dict[str, list[str]] = {}
    word = {s: collections.defaultdict(collections.Counter) for s in SCRIPTS}
    left = {s: collections.defaultdict(collections.Counter) for s in SCRIPTS}   # (prev, w)
    right = {s: collections.defaultdict(collections.Counter) for s in SCRIPTS}  # (w, next)
    join = {s: collections.defaultdict(collections.Counter) for s in SCRIPTS}   # (w, next)
    merge = {s: collections.defaultdict(collections.Counter) for s in SCRIPTS}  # (w, next) as one word
    absorbed = {s: collections.Counter() for s in SCRIPTS}     # w written into the word before it
    caps = {s: collections.defaultdict(collections.Counter) for s in SCRIPTS}  # mid-line capital?
    stats = collections.Counter()
    for r in rows:
        if not all(len(r[s]) == len(r["urdu"]) for s in SCRIPTS):
            stats["chunk_line_mismatch"] += 1
            continue
        for i, u in enumerate(r["urdu"]):
            lines[line_key(u)] = [r["deva"][i], r["roman"][i], r["dia"][i]]
            toks = urdu_tokens(u)
            if not toks:
                continue
            stats["lines"] += 1
            for s in SCRIPTS:
                al = align(toks, r[s][i], s)
                if al is None:
                    stats[f"unaligned_{s}"] += 1
                    continue
                stats[f"aligned_{s}"] += 1
                for k, (reading, j) in enumerate(al):
                    w = toks[k]
                    nxt = toks[k + 1] if k + 1 < len(toks) else None
                    if nxt is not None:
                        join[s][(w, nxt)][j] += 1
                    if j == "+":
                        merge[s][(w, nxt)][reading] += 1
                        continue
                    if reading is None:
                        absorbed[s][w] += 1
                        continue
                    # spelling votes are case-blind; a capital (a name) is kept
                    # only where GPT writes it mid-line (line starts may be a
                    # prose sentence capital, and ghazal lines have none)
                    word[s][w][reading.lower()] += 1
                    if k:
                        caps[s][w][reading[:1].isupper()] += 1
                    if k and al[k - 1][1] != "+":
                        left[s][(toks[k - 1], w)][reading.lower()] += 1
                    if nxt is not None and j != "+":
                        right[s][(w, nxt)][reading.lower()] += 1
    table = {"lines": lines, "words": {}, "ctx": {}, "joins": {}, "merge": {},
             "join_default": {}, "stats": dict(stats)}
    majority = {}
    for s in SCRIPTS:
        for w, c in word[s].items():
            reading = c.most_common(1)[0][0]
            if caps[s][w][True] > caps[s][w][False]:
                reading = reading[:1].upper() + reading[1:]
            majority[(s, w)] = reading
    for w in set(w for s in SCRIPTS for w in word[s]):
        # a word GPT mostly writes into the word before it (گے in karenge)
        # has no trustworthy reading of its own
        table["words"][w] = [majority.get((s, w)) if absorbed[s][w] <= sum(word[s][w].values())
                             else None for s in SCRIPTS]
        if not any(table["words"][w]):
            del table["words"][w]
    # context readings: a (prev, w) / (w, next) pair GPT reads differently
    # from the word's overall majority, consistently (>= 2 votes, >= 75%)
    for side, src in (("L", left), ("R", right)):
        for s_i, s in enumerate(SCRIPTS):
            for pair, c in src[s].items():
                w = pair[1] if side == "L" else pair[0]
                reading, n = c.most_common(1)[0]
                tot = sum(c.values())
                maj = majority[(s, w)]
                if reading == maj.lower() or n < 2 or n / tot < 0.75:
                    continue
                if maj[:1].isupper():
                    reading = reading[:1].upper() + reading[1:]
                key = f"{side}\t{pair[0]}\t{pair[1]}"
                table["ctx"].setdefault(key, [None, None, None])[s_i] = reading
    # joiners: a pair is kept when GPT doesn't just space it, or when the
    # engine's own default differs (around و: -o-; after a written izafat: -e-)
    o_join = [collections.Counter() for _ in SCRIPTS]
    for s_i, s in enumerate(SCRIPTS):
        allj = collections.Counter()
        for pair, c in join[s].items():
            allj.update(c)
            j, n = c.most_common(1)[0]
            if "و" in pair:
                o_join[s_i].update(c)
            key = f"{pair[0]}\t{pair[1]}"
            if j == "+":                          # GPT writes the pair as one word
                table["merge"].setdefault(key, [None, None, None])[s_i] = \
                    merge[s][pair].most_common(1)[0][0]
            if j != " " or "و" in pair or pair[0].endswith(("ۂ", "ٔ")):
                table["joins"].setdefault(key, [None, None, None])[s_i] = j
        table["join_default"][s] = dict(allj)
    table["o_join"] = [(c.most_common(1)[0][0] if c else "-") for c in o_join]
    # future suffix: how often GPT writes verb + گا / گے / گی as one word
    # (karega, jaaunga, denge) -> the engine merges unseen verbs the same way
    table["future_merge"] = {}
    for suf in FUTURE:
        rates = []
        for s in SCRIPTS:
            tot = sum(sum(c.values()) for (a, b), c in join[s].items() if b == suf)
            glued = sum(c["+"] for (a, b), c in join[s].items() if b == suf)
            rates.append(round(glued / tot, 3) if tot else 0.0)
        table["future_merge"][suf] = rates
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(table, fh, ensure_ascii=False)
    print(f"{len(rows)} chunks -> {len(lines)} lines, {len(table['words'])} words, "
          f"{len(table['ctx'])} context readings, {len(table['joins'])} joins -> {out}")
    print("  ", dict(stats))
    return table


# ---------------------------------------------------------------------------
# eval: offline neural vs the GPT column on held-out chunks
# ---------------------------------------------------------------------------

def evaluate(tables: bool = True, show: int = 40) -> dict:
    """Word + exact-line agreement with GPT on test chunks, neural built from
    train chunks only (test lines never reach the verbatim-line table). Each
    chunk is transliterated whole, as the app does with a pasted ghazal."""
    import difflib
    import neural_translit as nt
    tmp = OUT.with_name("gpt_eval_train.json.gz")
    if tables:
        build(tmp, include_test=False)
    eng = nt.NeuralTransliterator(gpt_path=tmp if tables else None)
    rows = [r for r in load_labels().values() if r["split"] == "test"
            and all(len(r[s]) == len(r["urdu"]) for s in SCRIPTS)]
    outs = {r["id"]: [o.split("\n") for o in eng.transliterate("\n".join(r["urdu"]))]
            for r in rows}
    res = {}
    diffs = {s: collections.Counter() for s in SCRIPTS}
    for kind in ("poem", "prose", "all"):
        sel = [r for r in rows if kind == "all" or r["kind"] == kind]
        if not sel:
            continue
        exact = collections.Counter()
        wok, wtot = collections.Counter(), collections.Counter()
        n = 0
        for r in sel:
            for i in range(len(r["urdu"])):
                n += 1
                for s_i, s in enumerate(SCRIPTS):
                    mine, ref = _canon(outs[r["id"]][s_i][i]), _canon(r[s][i])
                    exact[s] += mine == ref
                    a, b = mine.split(), ref.split()
                    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
                    wok[s] += sum(bl.size for bl in sm.get_matching_blocks())
                    wtot[s] += len(b)
                    if kind != "all":
                        continue
                    for op, i1, i2, j1, j2 in sm.get_opcodes():
                        if op != "equal":
                            diffs[s][(" ".join(a[i1:i2]), " ".join(b[j1:j2]))] += 1
        res[kind] = {s: {"line_exact": round(exact[s] / n, 4),
                         "word_acc": round(wok[s] / max(1, wtot[s]), 4)} for s in SCRIPTS}
        res[kind]["lines"] = n
    print(json.dumps(res, indent=1))
    if show:
        for s in SCRIPTS:
            print(f"-- top {s} differences (ours -> GPT)")
            for (a, b), c in diffs[s].most_common(show):
                print(f"  {c:3d}  {a!r} -> {b!r}")
    return res


def _canon(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    p = sub.add_parser("label")
    p.add_argument("--calls", type=int, default=300)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--prose-share", type=float, default=0.2)
    sub.add_parser("build")
    p = sub.add_parser("eval")
    p.add_argument("--no-tables", action="store_true", help="baseline: neural without GPT tables")
    p.add_argument("--show", type=int, default=40, help="top differences to print per script")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch()
    elif a.cmd == "label":
        label(a.calls, a.workers, a.prose_share)
    elif a.cmd == "build":
        build()
    else:
        evaluate(tables=not a.no_tables, show=a.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
