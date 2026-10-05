# -*- coding: utf-8 -*-
"""Rekhta's own spellings of words and izafat compounds, as rekhta.org
prints them -- the last step that makes a reading *look* like Rekhta's.

rekhta.org tags every word of a poem with an id from its dictionary, the
same id on the Urdu, Devanagari and Roman pages, and an izafat compound the
Urdu writes as two words (دل ناداں) is one id: one Devanagari word
(दिल-ए-नादाँ), one Roman word (dil-e-nādāñ). So a page lines its three
scripts up word by word, exactly. Those spellings follow conventions no rule
recovers: the ain apostrophe (दा'वा, mā'lūm -- but ya.anī), long vowels
doubled in some words and marked in others (yaañ, huuñ, aag, but āsmāñ,
kyā), hyphenated prefixes (बे-ज़ार, be-zār).

The lexicon only ever *spells* a reading the engine already chose: an entry
is used when its Devanagari is the engine's Devanagari up to spelling (ain
apostrophe, hyphens, nasal sign, nukta). Homographs (اس is/us) keep the
engine's choice.
"""
from __future__ import annotations

import collections
import gzip
import json
import re
import unicodedata
from pathlib import Path

ZER = "ِ"
MAX_PHRASE = 4
_EDGE = re.compile(r"^[^\wऀ-ॿ'’]+|[^\wऀ-ॿ]+$")


def urdu_key(words) -> str:
    """Urdu words -> lookup key (letter variants folded, harakat and the
    izafat zer dropped)."""
    from urdu_nn.rekhta_text import norm_word
    if isinstance(words, str):
        words = words.split()
    return " ".join(norm_word(w).rstrip(ZER) for w in words if norm_word(w).rstrip(ZER))


def deva_fold(deva: str, izafat: bool = True) -> str:
    """Devanagari up to spelling: no ain apostrophe, hyphens, spaces, nukta,
    and one nasal sign. ``izafat=False`` also drops the izafat ए."""
    d = unicodedata.normalize("NFC", deva).replace("’", "'")
    if not izafat:
        d = re.sub(r"-ए-|-ए$|-ए\s", "-", d)
    d = re.sub(r"[\s\-'़]", "", unicodedata.normalize("NFD", d))
    d = d.replace("ँ", "ं")
    d = re.sub(r"[ङञणनम]्(?=[क-ह])", "ं", d)
    d = re.sub(r"(^|्)यों", r"\1यूं", d)      # کیوں, یوں: kyoñ / kyūñ are one word said two ways
    return unicodedata.normalize("NFC", d)


def clean(word: str) -> str:
    """A word as printed on the page, punctuation and takhallus quotes off
    ('ग़ालिब', ग़ुबार-ए-'मीर', 'मीर'-जी); an ain apostrophe inside a word
    (मा'लूम) stays."""
    w = unicodedata.normalize("NFC", word.replace("’", "'").replace("‘", "'")).strip()
    w = _EDGE.sub("", w)
    return "-".join(p.strip("'") for p in w.split("-"))


class RekhtaLexicon:
    """urdu phrase key -> [(devanagari, roman, count)], most frequent first;
    and ``simple``: Rekhta's marked Roman word -> its simple Roman."""

    def __init__(self, entries: dict[str, list] | None = None, simple: dict[str, str] | None = None,
                 pairs: dict[str, list] | None = None, parts: dict[str, list] | None = None):
        self.entries = {k: [tuple(e) for e in v] for k, v in (entries or {}).items()}
        self.simple = dict(simple or {})
        self.pairs = dict(pairs or {})       # "w1 w2" -> [joined in one word, written apart]
        # words seen only inside compounds (اہل in اہل دل): their spelling there
        self.parts = {k: [tuple(e) for e in v] for k, v in (parts or {}).items()}
        self.max_phrase = max((k.count(" ") + 1 for k in self.entries), default=1)

    def __len__(self) -> int:
        return len(self.entries)

    # -- building -------------------------------------------------------------

    @staticmethod
    def units_of(row: dict):
        """(urdu, devanagari, roman) for every word of a gold ghazal whose
        three pages line up id for id."""
        u = row.get("units") or {}
        for lu, lh, lr in zip(u.get("ur", []), u.get("hi", []), u.get("roman", [])):
            if [a for a, _ in lu] != [a for a, _ in lh] or [a for a, _ in lu] != [a for a, _ in lr]:
                continue
            for (_, ur), (_, hi), (_, ro) in zip(lu, lh, lr):
                ur, hi, ro = urdu_key(ur), clean(hi), clean(ro).lower()
                if ur and hi and ro and ur.count(" ") < MAX_PHRASE:
                    yield ur, hi, ro

    @staticmethod
    def simple_pairs(row: dict):
        """(marked, simple) Roman for every word of a gold ghazal whose two
        Roman lines have the same words."""
        for a, b in zip(row.get("roman", []), row.get("simple", [])):
            wa, wb = a.split(), b.split()
            if len(wa) == len(wb):
                for x, y in zip(wa, wb):
                    x, y = clean(x).lower(), clean(y)
                    if x and y:
                        yield x, y

    @staticmethod
    def word_pairs(row: dict):
        """("w1 w2", joined) for every two neighbouring Urdu words of a gold
        ghazal: joined when Rekhta wrote them as one word (an izafat or
        hyphen compound, दिल-ए-नादाँ), else apart (दिल बेताब)."""
        for lu in (row.get("units") or {}).get("ur", []):
            words = [(k, urdu_key(w).split()) for k, (_, w) in enumerate(lu)]
            flat = [(k, w) for k, ws in words for w in ws]
            for (k1, a), (k2, b) in zip(flat, flat[1:]):
                yield f"{a} {b}", k1 == k2

    @staticmethod
    def word_parts(ur: str, hi: str, ro: str):
        """The words of a compound unit, each with its own spelling:
        اہل دل / अहल-ए-दिल / ahl-e-dil -> (اہل, अहल, ahl), (دل, दिल, dil). Only
        where the parts line up one for one."""
        words = ur.split()
        if len(words) < 2:
            return
        hs = [x for x in hi.split("-") if x and x != "ए"]
        rs = [x for x in ro.split("-") if x and x != "e"]
        if len(hs) == len(rs) == len(words):
            yield from zip(words, hs, rs)

    @classmethod
    def build(cls, rows: list[dict]) -> "RekhtaLexicon":
        c: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        s: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        p: dict[str, list] = collections.defaultdict(lambda: [0, 0])
        part: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for row in rows:
            for ur, hi, ro in cls.units_of(row):
                c[ur][(hi, ro)] += 1
                for w, h, r in cls.word_parts(ur, hi, ro):
                    part[w][(h, r)] += 1
            for x, y in cls.simple_pairs(row):
                s[x][y] += 1
            for key, joined in cls.word_pairs(row):
                p[key][0 if joined else 1] += 1
        # a pair only says something where its first word ever takes an izafat
        heads = {k.split()[0] for k, (j, _) in p.items() if j}
        return cls({k: [(h, r, n) for (h, r), n in v.most_common()] for k, v in c.items()},
                   {k: v.most_common(1)[0][0] for k, v in s.items()},
                   {k: v for k, v in p.items() if k.split()[0] in heads},
                   {k: [(h, r, n) for (h, r), n in v.most_common()] for k, v in part.items()})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps({"entries": self.entries, "simple": self.simple, "pairs": self.pairs,
                           "parts": self.parts},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        path.write_bytes(gzip.compress(data) if path.suffix == ".gz" else data)

    @classmethod
    def load(cls, path: Path | None) -> "RekhtaLexicon":
        if not path or not Path(path).exists():
            return cls()
        raw = Path(path).read_bytes()
        if str(path).endswith(".gz"):
            raw = gzip.decompress(raw)
        data = json.loads(raw.decode("utf-8"))
        if "entries" in data and isinstance(data["entries"], dict):
            return cls(data["entries"], data.get("simple"), data.get("pairs"), data.get("parts"))
        return cls(data)

    def agreement(self, words: list[str], units: list[list[str]], spans) -> tuple[float, float]:
        """How Rekhta-like a reading is: (word score, izafat score), each a
        sum of log shares -- for every Urdu word Rekhta has read, the share
        of its readings that are this one; for every neighbouring pair whose
        first word takes an izafat, the share of times Rekhta joined (or
        didn't join) them as this reading does. 0 where Rekhta is silent."""
        import math
        wsc = isc = 0.0
        for i0, i1, j0, j1 in spans:
            if i1 - i0 != 1:
                continue
            d = "".join(units[k][0] for k in range(j0, j1))
            cands = self.entries.get(urdu_key(words[i0]))
            if cands:
                f = deva_fold(d)
                tot = sum(n for *_, n in cands)
                hit = sum(n for h, _, n in cands if deva_fold(h) == f)
                wsc += math.log((hit + 0.5) / (tot + 1.0))
        span_of = {}
        for k, (i0, i1, j0, j1) in enumerate(spans):
            for i in range(i0, i1):
                span_of[i] = (k, j0, j1)
        for i in range(len(words) - 1):
            pr = self.pairs.get(urdu_key(words[i:i + 2]))
            if not pr:
                continue
            (k1, a0, a1), (k2, b0, b1) = span_of[i], span_of[i + 1]
            joined = k1 == k2 or (a1 == b0 and units[a1 - 1][1] != " ")
            j, s = pr
            isc += math.log(((j if joined else s) + 0.5) / (j + s + 1.0))
        return wsc, isc

    def to_simple(self, marked: str) -> str:
        """Rekhta's simple Roman for a run of its marked Roman: each word as
        Rekhta wrote it where known, else by rule."""
        from urdu_nn.rekhta_roman import to_simple
        out = []
        for w in re.split(r"(\s+)", marked):
            if not w or w.isspace():
                out.append(w)
                continue
            m = re.match(r"^(\W*)(.*?)(\W*)$", w)
            head, core, tail = m.groups()
            key = core.lower()
            if key in self.simple:
                out.append(head + self.simple[key] + tail)
            else:
                out.append(head + "-".join(to_simple(p) for p in core.split("-")) + tail)
        return "".join(out)

    # -- use --------------------------------------------------------------------

    def lookup(self, key: str, ours: str, prior: tuple[str, str] | None = None,
               loose: bool = True) -> tuple[str, str] | None:
        """Rekhta's (devanagari, roman) for the Urdu ``key`` when one of its
        readings is ours (``ours`` = our Devanagari for it), else None. A
        compound may differ from ours by the izafat alone (दिल-ए-नादाँ for
        our दिल नादाँ): Rekhta joined those words, so it read the -e-.

        Rekhta doesn't always spell a word one way (मालूम / मा'लूम, daaġh /
        dāġh), so each script takes its own majority over the matching
        entries; ``prior`` (the rules' spelling) breaks a tie."""
        cands = self.entries.get(key)
        if not cands:
            return None
        f = deva_fold(ours)
        match = [c for c in cands if deva_fold(c[0]) == f]
        if not match and loose and " " in key:
            f = deva_fold(ours, izafat=False)
            match = [c for c in cands if deva_fold(c[0], izafat=False) == f]
        if not match:
            return None
        votes: tuple[collections.Counter, collections.Counter] = (collections.Counter(), collections.Counter())
        for h, r, n in match:
            votes[0][h] += n
            votes[1][r] += n
        if prior:
            for v, p in zip(votes, prior):
                if p in v:
                    v[p] += 0.5
        return votes[0].most_common(1)[0][0], votes[1].most_common(1)[0][0]

    def lookup_part(self, key: str, ours: str) -> tuple[str, str] | None:
        """Like ``lookup``, from the spellings a word has inside Rekhta's
        compounds -- for a word it has never printed on its own."""
        if " " in key or key in self.entries or key not in self.parts:
            return None
        f = deva_fold(ours)
        match = [c for c in self.parts[key] if deva_fold(c[0]) == f]
        if not match:
            return None
        votes: tuple[collections.Counter, collections.Counter] = (collections.Counter(), collections.Counter())
        for h, r, n in match:
            votes[0][h] += n
            votes[1][r] += n
        return votes[0].most_common(1)[0][0], votes[1].most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Rekhta's Roman back to R (so the ASCII table and plain Roman follow it too)
# ---------------------------------------------------------------------------

_VOW = "aāiīuūeo"


def _from_deva(s: str, deva: str, roman_ch: str, deva_class: str, marked: str, mark: str) -> str:
    """Put back a distinction rekhta.org's Roman drops: the n-th ``roman_ch``
    is the n-th letter of ``deva_class`` in the Devanagari; where that is
    one of ``marked`` it becomes ``mark``."""
    letters = re.findall(f"[{deva_class}]", unicodedata.normalize("NFD", deva))
    pos = [m.start() for m in re.finditer(roman_ch, s)]
    if len(letters) != len(pos):
        return s
    s = list(s)
    for p, c in zip(pos, letters):
        if c in marked:
            s[p] = mark
    return "".join(s)


def rekhta_to_rich(roman: str, deva: str = "") -> str:
    """rekhta.org Roman (dil-e-nādāñ, yaañ, paḍhā, uthtā) -> R. Rekhta's
    Roman leaves out what its Devanagari keeps (ट is written t, ड d), so
    those come back from ``deva``, letter class by letter class in order."""
    s = unicodedata.normalize("NFC", roman.lower())
    s = s.replace("chh", "\x01").replace("ch", "c").replace("\x01", "ch")
    s = s.replace("ḳh", "x").replace("ġh", "ġ").replace("sh", "ś").replace("zh", "ž")
    s = s.replace("aa", "ā").replace("ii", "ī").replace("uu", "ū").replace("iye", "ie")
    s = re.sub(f"(?<=[{_VOW}])\\.(?=[{_VOW}])", "", s)
    s = re.sub(f"(?<=[{_VOW}])ñ", "~", s)
    s = s.replace("ḍ", "ṛ")                         # rekhta.org's ḍ is the flap ड़
    if deva:
        d = re.sub("[डढ]़", "", unicodedata.normalize("NFD", deva).replace("ड़", "").replace("ढ़", ""))
        s = _from_deva(s, deva, "t", "तथटठ", "टठ", "ṭ")
        s = _from_deva(s, d, "d", "दधडढ", "डढ", "ḍ")
    return s
