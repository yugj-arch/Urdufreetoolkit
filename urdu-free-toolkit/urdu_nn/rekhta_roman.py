# -*- coding: utf-8 -*-
"""Roman readings for the Rekhta-style engine, taken from its Devanagari.

Rekhta's Devanagari already spells every consonant the way Urdu means it
(ख़ for خ, क़ for ق, ड़ for ڑ), every long vowel, every nasal, the izafat
(ख़ौफ़-ए-रसन) and the ain (ए'लान). The one thing it leaves unsaid is which
inherent schwas are silent; ``urdu_nn.scheme.deva_to_rich_candidates`` lists
the schwa variants, and human evidence (our gold lexicon, the reviewed
dictionary, people's casual romanisations) picks among them. The result is
an R reading (see ``urdu_nn.scheme``) rendered three ways:

* ``plain``  -- the app's casual ASCII (``urdu_nn.scheme.to_plain``),
* ``rekhta`` -- Rekhta's diacritic Roman: ā ī ū, ñ, ḳh ġ, ṭ ḍ ṛ, a dot between
  vowels in hiatus (ga.e, ko.ī, aa.īna),
* ``ascii``  -- Rekhta's ASCII table: aa ii uu, ñ, KH G, T D, .D .Dh, zh
  (ghar, KHauf, Gam, pa.Dhaa.ii, aañkh, haiñ).
"""
from __future__ import annotations

import re
import unicodedata

from urdu_nn.scheme import R_VOWELS, deva_to_rich_candidates, fold_roman, to_plain

ZER = "ِ"
_ASPIRABLE = set("kgcjṭḍtdpbṛ")

# R consonant unit -> (rekhta diacritic, ascii table)
_CONS = {
    "c": ("ch", "ch"), "ch": ("chh", "chh"), "x": ("ḳh", "KH"), "ġ": ("ġ", "G"),
    "ś": ("sh", "sh"), "ṣ": ("sh", "sh"), "ž": ("zh", "zh"),
    "ṭ": ("ṭ", "T"), "ṭh": ("ṭh", "Th"), "ḍ": ("ḍ", "D"), "ḍh": ("ḍh", "Dh"),
    "ṛ": ("ṛ", ".D"), "ṛh": ("ṛh", ".Dh"),
    "ṇ": ("n", "n"), "ṅ": ("n", "n"), "ñ": ("n", "n"),
}
_VOWEL = {"ā": ("ā", "aa"), "ī": ("ī", "ii"), "ū": ("ū", "uu")}


def _units(word: str) -> list[str]:
    """R word -> consonant units (aspirates kept whole) and vowel runs (with
    a trailing ~), like ``scheme._units``."""
    out, i = [], 0
    while i < len(word):
        ch = word[i]
        if ch in R_VOWELS:
            j = i
            while j < len(word) and word[j] in R_VOWELS:
                j += 1
            if j < len(word) and word[j] == "~":
                j += 1
            out.append(word[i:j])
            i = j
        elif ch in _ASPIRABLE and word[i + 1:i + 2] == "h":
            out.append(word[i:i + 2])
            i += 2
        else:
            out.append(ch)
            i += 1
    return out


def _vowel_run(run: str, style: int) -> str:
    """One vowel run: diphthongs ai/au stay whole, any other vowel meeting a
    vowel is a hiatus and gets Rekhta's dot (ga.e, ko.ī, bhaa.ii)."""
    nasal = run.endswith("~")
    v = run.rstrip("~")
    parts, i = [], 0
    while i < len(v):
        if v[i] == "a" and v[i + 1:i + 2] in ("i", "u"):
            parts.append(v[i:i + 2])
            i += 2
            continue
        parts.append(v[i])
        i += 1
    out = ".".join(_VOWEL.get(p, (p, p))[style] for p in parts)
    return out + ("ñ" if nasal else "")


def _render_word(word: str, style: int) -> str:
    units = _units(word)
    out = []
    for k, u in enumerate(units):
        nxt = units[k + 1] if k + 1 < len(units) else ""
        if u[0] in R_VOWELS:
            out.append(_vowel_run(u, style))
        elif u == "ṃ":
            out.append("m" if nxt[:1] in ("p", "b", "m") else "n")
        elif u == "'":
            if 0 < k < len(units) - 1:
                out.append("'")
        elif u == "~":
            out.append("ñ")
        else:
            out.append(_CONS.get(u, (u, u))[style])
    return "".join(out)


def render(rich: str, style: str) -> str:
    """R line -> ``plain`` / ``rekhta`` / ``ascii`` Roman."""
    if style == "plain":
        return to_plain(rich)
    s = 0 if style == "rekhta" else 1
    parts = re.split(r"([ \-])", rich)
    return "".join(p if p in (" ", "-") or not p else _render_word(p, s) for p in parts)


_ASCII_IN = {".Dh": "ṛh", ".D": "ṛ", "KH": "x", "Th": "ṭh", "Dh": "ḍh", "chh": "ch", "ch": "c",
             "sh": "ś", "zh": "ž", "aa": "ā", "ii": "ī", "uu": "ū", "T": "ṭ", "D": "ḍ",
             "G": "ġ", "ñ": "~", ".": ""}


def ascii_to_rich(ascii_: str) -> str:
    """Rekhta's ASCII table back to R (a correction typed as ``KHauf-e-rasan``
    or ``pa.Dhaa.ii``), longest match first; everything else passes through."""
    out, i = [], 0
    while i < len(ascii_):
        for n in (3, 2, 1):
            piece = ascii_[i:i + n]
            if piece in _ASCII_IN:
                out.append(_ASCII_IN[piece])
                i += n
                break
        else:
            out.append(ascii_[i])
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Devanagari word -> R
# ---------------------------------------------------------------------------

def _loose(s: str) -> str:
    """Comparison key for choosing a schwa variant against evidence: every
    spelling difference except the vowels between consonants goes away."""
    s = fold_roman(s)
    for a, b in (("q", "k"), ("x", "kh"), ("ph", "f"), ("z", "j"), ("sh", "s"), ("y", "i")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s)


def _nasal_fold(deva: str) -> str:
    deva = deva.replace("ँ", "ं")
    return re.sub(r"[नम]्(?=[क-ह])", "ं", deva)


class WordModel:
    """The word-level neural transliterator (``urdu_nn.model``, trained on
    700k human romanisations) as a schwa voter: its beam of R readings for
    an Urdu word, with probabilities."""

    def __init__(self, path, device: str = "cpu", beam: int = 5):
        import torch
        from urdu_nn.model import Seq2Seq, Vocab
        ck = torch.load(path, map_location=device, weights_only=False)
        self.vocab = Vocab([])
        self.vocab.itos = ck["itos"]
        self.vocab.stoi = {s: i for i, s in enumerate(self.vocab.itos)}
        self.model = Seq2Seq(**ck["cfg"])
        self.model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
        self.model.to(device).eval()
        self.device, self.beam = device, beam
        self._cache: dict[str, list[tuple[str, float]]] = {}

    def readings(self, words: list[str]) -> dict[str, list[tuple[str, float]]]:
        import math
        import torch
        from urdu_nn.model import beam_search, pad_batch
        todo = [w for w in dict.fromkeys(words) if w not in self._cache
                and all(c in self.vocab.stoi for c in w)]
        for b in range(0, len(todo), 64):
            chunk = todo[b:b + 64]
            src = pad_batch([self.vocab.encode_src("j", w) for w in chunk], self.device)
            with torch.inference_mode():
                beams = beam_search(self.model, src, beam=self.beam, max_len=64)
            for w, hyps in zip(chunk, beams):
                got = []
                for ids, sc in hyps:
                    t = self.vocab.decode(ids)
                    if t.count("|") == 1:
                        got.append((t.split("|")[0], sc))
                z = sum(math.exp(s) for _, s in got) or 1.0
                self._cache[w] = [(r, math.exp(s) / z) for r, s in got]
        return {w: self._cache.get(w, []) for w in words}


class Reader:
    """Picks the R reading of a Devanagari word written for an Urdu word."""

    def __init__(self, lexicon: dict | None = None, dictionary: dict | None = None,
                 evidence: dict | None = None, word_model: WordModel | None = None):
        self.lexicon = lexicon or {}          # urdu -> (deva, R)   gold
        self.dictionary = dictionary or {}    # urdu -> (deva, plain, rekhta)   reviewed
        self.evidence = evidence or {}        # urdu -> {casual roman: weight}   human
        self.word_model = word_model
        self.model_weight = 1.5
        self._cache: dict[tuple[str, str], str] = {}

    def prefetch(self, urdu_words: list[str]) -> None:
        """Batch the word model over every word the tables can't settle."""
        if self.word_model:
            self.word_model.readings([w for w in urdu_words if w not in self.lexicon])

    def _cands(self, deva: str) -> list[str]:
        """Schwa variants of ``deva``; an ain apostrophe (ता'लीम) splits the
        word into syllables read separately, anything unmodelled is dropped."""
        parts = [p for p in deva.split("'")]
        if len(parts) > 1:
            heads = [self._cands(p) or [""] for p in parts]
            return ["'".join(h[0] for h in heads)]
        c = deva_to_rich_candidates(deva)
        if not c:
            clean = re.sub(r"[^ऀ-ॿ]", "", deva.replace("ॅ", "े").replace("ॉ", "ो"))
            c = deva_to_rich_candidates(clean) if clean else []
        return c

    def read(self, deva: str, urdu: str) -> str:
        key = (deva, urdu)
        if key in self._cache:
            return self._cache[key]
        deva = unicodedata.normalize("NFC", deva)
        urdu = urdu.rstrip(ZER)
        cands = self._cands(deva) or [deva]
        best = cands[0]
        if len(cands) > 1:
            best = self._pick(cands, deva, urdu)
        if "ँ" in deva and "ं" not in deva:
            best = best.replace("ṅ", "~")      # आँख is āñkh, a nasal vowel -- not āṅkh
        if urdu.endswith("ہ") and deva.endswith("ा") and best.endswith("ā"):
            best = best[:-1] + "a"             # silent he: afsāna, not afsānā
        self._cache[key] = best
        return best

    def _pick(self, cands: list[str], deva: str, urdu: str) -> str:
        lex = self.lexicon.get(urdu)
        if lex and lex[1]:
            if lex[1] in cands:
                return lex[1]
            want = _loose(to_plain(lex[1]))
            for c in cands:
                if _loose(to_plain(c)) == want:
                    return c
        votes: dict[str, float] = {}
        dic = self.dictionary.get(urdu)
        if dic and _nasal_fold(dic[0]) == _nasal_fold(deva):
            votes[_loose(dic[1])] = votes.get(_loose(dic[1]), 0.0) + 5.0
        for r, w in (self.evidence.get(urdu) or {}).items():
            votes[_loose(r)] = votes.get(_loose(r), 0.0) + w
        if self.word_model and urdu:
            for r, p in self.word_model.readings([urdu])[urdu]:
                k = _loose(to_plain(r))
                votes[k] = votes.get(k, 0.0) + self.model_weight * p
        if not votes:
            return cands[0]
        scored = [(votes.get(_loose(to_plain(c)), 0.0), -i, c) for i, c in enumerate(cands)]
        return max(scored)[2]


# ---------------------------------------------------------------------------
# Devanagari line <-> Urdu words
# ---------------------------------------------------------------------------

IZAFAT = {"ए", "-ए", "ए-"}
CONJ_O = "ओ"


def deva_units(deva_line: str) -> list[list[str]]:
    """Devanagari line -> [[word, joiner-after]]. Words split on spaces and
    hyphens; an izafat ए between hyphens becomes the joiner "-e-" of the word
    before it; the conjunctive ओ (संग-ओ-ख़िश्त) stays a word of its own -- it
    is the Urdu و."""
    units: list[list[str]] = []
    for w in deva_line.split():
        parts = w.split("-")
        for j, p in enumerate(parts):
            if not p:
                continue
            if p == "ए" and j > 0 and units:
                units[-1][1] = "-e-"
                continue
            if j > 0 and units and units[-1][1] == " ":
                units[-1][1] = "-"
            units.append([p, " "])
    return units


def align(urdu_words: list[str], units: list[list[str]]) -> list[int] | None:
    """Index of the Devanagari unit for each Urdu word, or None when the
    counts disagree (the caller then reads word by word)."""
    if len(urdu_words) != len(units):
        return None
    return list(range(len(units)))
