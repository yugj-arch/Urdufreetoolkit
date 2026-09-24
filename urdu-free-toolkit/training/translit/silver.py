# -*- coding: utf-8 -*-
"""Silver ``j`` pairs (urdu -> R + Devanagari) from two human romanisation sets.

Urdu script leaves short vowels out; Devanagari writes them. Aksharantar has
~700k Urdu words and ~1.3M Hindi words, each with the casual Roman a person
typed for it. Where an Urdu word and a Hindi word were romanised the same way
(fold_roman), and the Hindi word's letters really do spell that Urdu word
(deva_to_urdu_candidates), the Hindi spelling is that Urdu word's vowelled
reading: بلی (billi) meets बिल्ली (billi), never बली (bali).

The R reading then comes from the Devanagari (deva_to_rich_candidates), with
the schwa variant chosen by those same human romanisations (adaalaton, not
adaalton), nuktas restored from the Urdu letters (कानून + قانون -> क़ानून, qānūn)
and word-final silent he read as a short a (afsāna).

    python -m training.translit.silver     # precision report against the gold lexicon
"""
from __future__ import annotations

import collections
import itertools
import json
import re
import unicodedata
import zipfile
from pathlib import Path

from urdu_nn.scheme import (
    deva_to_rich_candidates, deva_to_urdu_candidates, fold_roman, to_plain,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "translit_raw"

_NUKTA = "़"
_NUKTABLE = set("कखगजफ")
_NUKTA_URDU = set("قخغزذضظژف")
_DEVA = re.compile(r"^[ऀ-ॿ]+$")
_CASUAL = re.compile(r"^[a-z]+$")


def hindi_casual():
    """Aksharantar Hindi: (devanagari, casual roman, weight)."""
    z = zipfile.ZipFile(RAW / "aksharantar_hin.zip")
    for name in ("hin_train.json", "hin_valid.json", "hin_test.json"):
        for line in z.read(name).decode("utf-8").splitlines():
            x = json.loads(line)
            d = unicodedata.normalize("NFC", x["native word"].strip())
            r = x["english word"].strip().lower()
            if x.get("source") == "IndicCorp" and (x.get("score") or 0) < -0.3:
                continue
            if _DEVA.match(d) and _CASUAL.match(r):
                yield d, r, 1.0 if x.get("source") != "IndicCorp" else 0.5


def spell_with_nuktas(urdu: str, deva: str) -> str | None:
    """``deva`` with the nuktas the Urdu letters call for (कानून + قانون ->
    क़ानून), or None if ``deva`` can't spell ``urdu`` at all. Among spellings
    that work, the one whose best Urdu candidate is ``urdu`` wins."""
    deva = unicodedata.normalize("NFC", deva)
    spots = [i for i, ch in enumerate(deva) if ch in _NUKTABLE
             and not (i + 1 < len(deva) and deva[i + 1] == _NUKTA)]
    if not any(c in _NUKTA_URDU for c in urdu) or len(spots) > 5:
        spots = []
    best = None
    for mask in itertools.product((0, 1), repeat=len(spots)):
        d = deva
        for pos in sorted((p for p, m in zip(spots, mask) if m), reverse=True):
            d = d[:pos + 1] + _NUKTA + d[pos + 1:]
        cands = deva_to_urdu_candidates(d)
        if urdu in cands:
            key = (cands.index(urdu), sum(mask))
            if best is None or key < best[0]:
                best = (key, d)
    return best[1] if best else None


def pick_reading(urdu: str, deva: str, romans: collections.Counter) -> tuple[str | None, float]:
    """(R, confidence) for ``deva`` read as ``urdu``: the schwa variant the
    human romanisations back. Confidence is the share of romanisation weight
    that folds exactly to it (0 = none do)."""
    cands = deva_to_rich_candidates(deva)
    if not cands:
        return None, 0.0
    if urdu.endswith("ہ"):          # word-final silent he: sabza, afsāna
        cands = [c[:-1] + "a" if c.endswith("ā") else c for c in cands]
    total = sum(romans.values()) or 1.0
    folded = collections.Counter()
    for r, w in romans.items():
        folded[fold_roman(r)] += w

    def score(i_c):
        # only an exact (folded) match counts: "near" similarity favours the
        # shorter string (olsen is nearer olsn than olsan), so with no match
        # the default schwa rule's reading stands
        i, c = i_c
        return (folded.get(fold_roman(to_plain(c)), 0.0), -i)

    i, best = max(enumerate(cands), key=score)
    return best, folded.get(fold_roman(to_plain(best)), 0.0) / total


def silver_pairs(urdu_casual, skip: set[str] = frozenset(), min_conf: float = 0.3,
                 log=print) -> list[dict]:
    """urdu_casual: iterable of (urdu, casual roman, weight).
    Returns [{urdu, rich, deva, conf}] for Urdu words not in ``skip``."""
    u_rom: dict[str, collections.Counter] = {}
    for u, r, w in urdu_casual:
        if u not in skip:
            u_rom.setdefault(u, collections.Counter())[r] += w
    log(f"  silver: {len(u_rom)} Urdu words with human romanisations")
    by_fold: dict[str, dict[str, collections.Counter]] = {}
    for d, r, w in hindi_casual():
        by_fold.setdefault(fold_roman(r), {}).setdefault(d, collections.Counter())[r] += w
    log(f"  silver: {len(by_fold)} Hindi romanisation keys")
    out = []
    for n, (u, roms) in enumerate(u_rom.items()):
        if n and n % 100_000 == 0:
            log(f"    {n} / {len(u_rom)} ({len(out)} pairs)")
        hits: dict[str, collections.Counter] = {}
        for r in roms:
            for d, dr in by_fold.get(fold_roman(r), {}).items():
                hits.setdefault(d, collections.Counter()).update(dr)
        best = None
        for d, dr in hits.items():
            d2 = spell_with_nuktas(u, d)
            if not d2:
                continue
            rich, conf = pick_reading(u, d2, roms + dr)
            if not rich or conf < min_conf:
                continue
            key = (conf * sum(roms.values()) + sum(dr.values()), conf)
            if best is None or key > best[0]:
                best = (key, dict(urdu=u, rich=rich, deva=d2, conf=round(conf, 3),
                                  uw=sum(roms.values()), dw=sum(dr.values())))
        if best:
            out.append(best[1])
    return out


# ---------------------------------------------------------------------------
# Wikipedia: Hindi article titles and the Urdu article they link to (names of
# people, places, things -- the words a word list never has). CC BY-SA.
# ---------------------------------------------------------------------------

WIKI_TITLES = RAW / "wiki_hi_ur_titles.tsv"


def fetch_wiki_titles(path: Path = WIKI_TITLES) -> None:
    """Every Hindi Wikipedia article with an Urdu interlanguage link ->
    ``hindi title <TAB> urdu title`` (MediaWiki API, ~150 requests)."""
    import time
    import urllib.parse
    import urllib.request
    api = "https://hi.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "langbacklinks", "lbllang": "ur", "lbllimit": "500",
              "lblprop": "lllang|lltitle", "format": "json"}
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        while True:
            req = urllib.request.Request(api + "?" + urllib.parse.urlencode(params), headers={
                "User-Agent": "urdu-free-toolkit/1.0 (https://github.com/yugj-arch/Urdufreetoolkit; offline transliteration training data)"})
            data = json.loads(urllib.request.urlopen(req, timeout=60).read())
            for x in data["query"]["langbacklinks"]:
                if x.get("ns") == 0 and x.get("lltitle"):
                    fh.write(f"{x['title']}\t{x['lltitle']}\n")
                    n += 1
            if "continue" not in data:
                break
            params.update(data["continue"])
            time.sleep(0.2)
    print(f"{n} title pairs -> {path}")


_PAREN = re.compile(r"\s*[(（].*?[)）]\s*")


def wiki_word_pairs(path: Path = WIKI_TITLES):
    """(urdu word, devanagari word) from titles whose words line up one to one."""
    from urdu_nn.scheme import is_urdu_word, norm_urdu
    for line in path.read_text(encoding="utf-8").splitlines():
        hi, ur = line.split("\t")
        hi = unicodedata.normalize("NFC", _PAREN.sub(" ", hi)).replace("-", " ").split()
        ur = _PAREN.sub(" ", ur).replace("-", " ").split()
        if len(hi) != len(ur):
            continue
        for d, u in zip(hi, ur):
            u = norm_urdu(u)
            if _DEVA.match(d) and is_urdu_word(u) and len(u) >= 2:
                yield u, d


def wiki_pairs(urdu_casual, skip: set[str] = frozenset()) -> list[dict]:
    """Silver pairs from Wikipedia titles: the Hindi title word must spell the
    Urdu one (nuktas restored from it); its reading is picked with the human
    romanisations of the Urdu word when there are any."""
    u_rom: dict[str, collections.Counter] = {}
    for u, r, w in urdu_casual:
        u_rom.setdefault(u, collections.Counter())[r] += w
    votes: dict[str, collections.Counter] = {}
    for u, d in wiki_word_pairs():
        if u not in skip:
            votes.setdefault(u, collections.Counter())[d] += 1
    out = []
    for u, ds in votes.items():
        for d, _ in ds.most_common(2):
            d2 = spell_with_nuktas(u, d)
            if not d2:
                continue
            roms = u_rom.get(u, collections.Counter())
            rich, conf = pick_reading(u, d2, roms)
            if rich:
                out.append(dict(urdu=u, rich=rich, deva=d2, conf=round(conf, 3),
                                uw=sum(roms.values()), n=ds[d]))
                break
    return out


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["fetch-wiki"]:
        fetch_wiki_titles()
        return
    from training.translit.build_data import aksharantar, dakshina_lexicon
    from training.translit.build_dictionary import norm_nasals

    lex = json.loads((ROOT / "data" / "translit_ds" / "lexicon_full.json").read_text(encoding="utf-8"))
    rows = [(u, r, 1.0 if s in ("Dakshina", "Existing", "AK-Uni") else 0.5) for u, r, s in aksharantar()]
    rows += [(u, r, 1.0) for u, r in dakshina_lexicon()]
    gold_words = {u for u, v in lex.items() if v[0]}
    pairs = silver_pairs([x for x in rows if x[0] in gold_words])
    ok_r = ok_d = ok_both = 0
    bad = []
    for p in pairs:
        gd, gr = lex[p["urdu"]]
        dr = fold_roman(to_plain(p["rich"])) == fold_roman(to_plain(gr))
        dd = norm_nasals(p["deva"]) == norm_nasals(gd)
        ok_r += dr
        ok_d += dd
        ok_both += dr and dd
        if not (dr and dd):
            bad.append((p["urdu"], p["deva"], p["rich"], "gold", gd, gr, p["conf"]))
    n = len(pairs)
    print(f"gold words with deva: {len(gold_words)}  silver covered: {n}")
    print(f"R folded ok {ok_r / n:.3f}  deva ok {ok_d / n:.3f}  both {ok_both / n:.3f}")
    for b in bad[:40]:
        print("  ", *b)


if __name__ == "__main__":
    main()
