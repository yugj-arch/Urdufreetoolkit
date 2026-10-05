# -*- coding: utf-8 -*-
"""Taqti: scanning a misra's reading into long and short syllables, and
finding a ghazal's meter (bahr) from its lines.

Every misra of a ghazal is in one meter, so where a reading leaves it
unsaid whether an izafat is there (शब-ए-दरमियाँ / शब दरमियाँ), whether a
consonant is doubled (लिखा / लिक्खा) or which of two words is meant (क्या /
किया), the meter often says -- as it does for Rekhta's editors.

A reading is an R line (``urdu_nn.scheme``: words, "-" joiners, the izafat
"e" and conjunctive "o" as words of their own). Its syllables follow the
Urdu prosody rules that matter here:

* a short vowel is short (u) unless a consonant closes it, a long vowel
  (ā ī ū e o ai au) is long (-);
* a long vowel closed by a consonant, or a short one by two, adds a short
  (yār = - u, dost = - u, dard = - u) -- except at the end of the misra,
  where that extra is dropped;
* a word's final consonant carries over to an izafat or conjunctive o (dil-e
  = di-le), and a final long vowel may be scanned short (ko, ke, se, hai,
  merā); so may the izafat e, the conjunctive o and the silent he (afsāna);
* the nasal ñ (~) never counts.

A meter is a string of "-" and "u". ``scans`` says whether a reading can be
read in it; ``infer_meter`` takes the meter most of a ghazal's lines can be
read in.
"""
from __future__ import annotations

import collections
import itertools
import re

LONG_V = {"ā", "ī", "ū", "e", "o", "ai", "au"}
_VOWELS = set("aāiīuūeo")
_ASP = set("kgcjṭḍtdpbṛ")
MAX_OPTS = 4096


def _units(word: str) -> list[tuple[str, str]]:
    """R word -> [("V", vowel) / ("C", consonant)]; diphthongs ai/au whole,
    aspirates (kh, ṭh) one consonant, nasal ~ and ain ' dropped."""
    out, i = [], 0
    word = word.replace("~", "").replace("'", "")
    while i < len(word):
        ch = word[i]
        if ch in _VOWELS:
            if ch == "a" and word[i + 1:i + 2] in ("i", "u") and not (i + 2 < len(word) and word[i + 2] in _VOWELS):
                out.append(("V", word[i:i + 2]))
                i += 2
            else:
                out.append(("V", ch))
                i += 1
        elif ch in _ASP and word[i + 1:i + 2] == "h":
            out.append(("C", word[i:i + 2]))
            i += 2
        elif ch.isalpha():
            out.append(("C", ch))
            i += 1
        else:
            i += 1
    return out


def syllables(rich_line: str) -> list[str]:
    """R line -> per syllable its possible weights: "-", "u" or "-u" (either).
    The last item may be "x": an extra short the end of a misra drops."""
    toks = [t for t in re.split(r"[ \-]+", rich_line.strip()) if t]
    words = [_units(t) for t in toks]
    # a final consonant carries over to a following izafat e / conjunctive o
    for k in range(len(words) - 1):
        if toks[k + 1] in ("e", "o") and words[k] and words[k][-1][0] == "C" \
                and any(u[0] == "V" for u in words[k]):
            words[k + 1] = [words[k][-1]] + words[k + 1]
            words[k] = words[k][:-1]
    out: list[str] = []
    for k, (tok, units) in enumerate(zip(toks, words)):
        vix = [n for n, u in enumerate(units) if u[0] == "V"]
        if not vix:
            continue
        last_word = k == len(words) - 1
        # a vowel-initial next word may take this word's last consonant (dil ab = di-lab)
        wasl = (not last_word and words[k + 1] and words[k + 1][0][0] == "V"
                and toks[k + 1] not in ("e", "o") and units[-1][0] == "C")
        for m, n in enumerate(vix):
            nxt = vix[m + 1] if m + 1 < len(vix) else len(units)
            cons = nxt - n - 1
            final = m == len(vix) - 1
            coda = cons if final else max(cons - 1, 0)
            v = units[n][1]
            long_v = v in LONG_V
            # Devanagari's e / o before h + consonant is a short vowel: तोहफ़ा tuh-fa, मेहफ़िल mah-fil
            if v in ("e", "o") and n + 2 < len(units) and units[n + 1] == ("C", "h") \
                    and units[n + 2][0] == "C":
                long_v = False
            w = "u" if not long_v and coda == 0 else "-"
            extra = (long_v and coda >= 1) or (not long_v and coda >= 2)
            if long_v and not final and cons == 0:
                w = "-u"              # a long vowel before a vowel may be short: ko-ī, hu-e
            if final and coda == 0:
                if long_v or tok in ("e", "o") or (v == "a" and len(vix) > 1):
                    w = "-u"          # ko / ke / merā, izafat e, afsāna: either
            if final and wasl:
                if extra:
                    out.append(w)
                    out.append("x")   # āb ab = ā-bab: the extra short may go
                    continue
                if not long_v and coda == 1:
                    w = "-u"
            out.append(w)
            if extra:
                # at the misra's end the extra short goes; inside a word a long
                # vowel before a cluster is often read without it (pūch-te)
                out.append("x" if (final and last_word) or (long_v and not final) else "u")
    return out


def _compatible(a: list[str], b: list[str]) -> bool:
    """Is there a meter string both scanned readings can be read as? Two
    patterns walked in step; "x" (an optional short) may also be skipped."""
    from functools import lru_cache

    def emits(s):
        return {"x": ("", "u"), "-u": ("-", "u")}.get(s, (s,))

    @lru_cache(maxsize=None)
    def go(i: int, j: int) -> bool:
        if i == len(a) and j == len(b):
            return True
        if i < len(a) and a[i] == "x" and go(i + 1, j):
            return True
        if j < len(b) and b[j] == "x" and go(i, j + 1):
            return True
        if i == len(a) or j == len(b):
            return False
        ea, eb = set(emits(a[i])) - {""}, set(emits(b[j])) - {""}
        return bool(ea & eb) and go(i + 1, j + 1)

    return go(0, 0)


def expansions(syl: list[str]) -> list[str]:
    """Every meter string a scanned reading can be read as."""
    opts = []
    for s in syl:
        if s == "x":
            opts.append(["", "u"])
        elif s == "-u":
            opts.append(["-", "u"])
        else:
            opts.append([s])
    n = 1
    for o in opts:
        n *= len(o)
    if n > MAX_OPTS:
        return []
    return ["".join(p) for p in itertools.product(*opts)]


def scans(syl: list[str], meter: str) -> bool:
    """Can a scanned reading be read in ``meter``? (dynamic programming, no
    expansion limit)"""
    states = {0}
    for s in syl:
        new = set()
        for i in states:
            choices = {"x": ("", "u"), "-u": ("-", "u")}.get(s, (s,))
            for c in choices:
                if meter.startswith(c, i):
                    new.add(i + len(c))
        states = new
        if not states:
            return False
    return len(meter) in states


class Ghazal:
    """The lines of one poem, scanned: how many of them a reading of one
    line can be read alongside. A ghazal's misras share a meter up to the
    variants its first and last feet allow, so a reading that fits no other
    line is very likely misread."""

    def __init__(self, lines_syl: list[list[str]]):
        self.lines = [s for s in lines_syl]
        self.n = sum(1 for s in self.lines if s)

    def support(self, syl: list[str], line: int | None = None) -> int:
        """How many of the other lines ``syl`` can share a scansion with."""
        if not syl:
            return 0
        return sum(1 for k, s in enumerate(self.lines)
                   if s and k != line and _compatible(syl, s))
