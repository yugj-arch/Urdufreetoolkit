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
    return unicodedata.normalize("NFC", d)


def clean(word: str) -> str:
    """A word as printed on the page, punctuation and takhallus quotes off."""
    w = unicodedata.normalize("NFC", word.replace("’", "'").replace("‘", "'")).strip()
    w = _EDGE.sub("", w)
    return w.strip("'") if w.count("'") and (w.startswith("'") or w.endswith("'")) else w


class RekhtaLexicon:
    """urdu phrase key -> [(devanagari, roman, count)], most frequent first."""

    def __init__(self, entries: dict[str, list] | None = None):
        self.entries = {k: [tuple(e) for e in v] for k, v in (entries or {}).items()}
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

    @classmethod
    def build(cls, rows: list[dict]) -> "RekhtaLexicon":
        c: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for row in rows:
            for ur, hi, ro in cls.units_of(row):
                c[ur][(hi, ro)] += 1
        return cls({k: [(h, r, n) for (h, r), n in v.most_common()] for k, v in c.items()})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.entries, ensure_ascii=False, sort_keys=True).encode("utf-8")
        path.write_bytes(gzip.compress(data) if path.suffix == ".gz" else data)

    @classmethod
    def load(cls, path: Path | None) -> "RekhtaLexicon":
        if not path or not Path(path).exists():
            return cls()
        raw = Path(path).read_bytes()
        if str(path).endswith(".gz"):
            raw = gzip.decompress(raw)
        return cls(json.loads(raw.decode("utf-8")))

    # -- use --------------------------------------------------------------------

    def lookup(self, key: str, ours: str, prior: tuple[str, str] | None = None) -> tuple[str, str] | None:
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
        if not match and " " in key:
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

    def confident(self, key: str, min_count: int) -> tuple[str, str] | None:
        """Rekhta's reading of ``key`` when it has only ever read it one way,
        at least ``min_count`` times -- whatever the engine read."""
        cands = self.entries.get(key)
        if not cands or sum(n for *_, n in cands) < min_count:
            return None
        if len({deva_fold(h) for h, *_ in cands}) != 1:
            return None
        return self.lookup(key, cands[0][0])


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
