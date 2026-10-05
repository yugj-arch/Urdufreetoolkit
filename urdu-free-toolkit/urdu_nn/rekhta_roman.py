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

# R consonant unit -> (rekhta diacritic, ascii table). The diacritic column
# is what rekhta.org prints (checked against its pages): ġh for غ, the
# retroflex stops unmarked (uthtā, darte) and ḍ for the flap ड़ (chhoḍ, paḍhā)
_CONS = {
    "c": ("ch", "ch"), "ch": ("chh", "chh"), "x": ("ḳh", "KH"), "ġ": ("ġh", "G"),
    "ś": ("sh", "sh"), "ṣ": ("sh", "sh"), "ž": ("zh", "zh"),
    "ṭ": ("t", "T"), "ṭh": ("th", "Th"), "ḍ": ("d", "D"), "ḍh": ("dh", "Dh"),
    "ṛ": ("ḍ", ".D"), "ṛh": ("ḍh", ".Dh"),
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
    out = ""
    for k, p in enumerate(parts):
        # rekhta.org writes no dot after u or o (huā, hue, koī) -- only the ASCII table does
        if k and not (style == 0 and parts[k - 1] in ("u", "ū", "o")):
            out += "."
        out += _VOWEL.get(p, (p, p))[style]
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


_DOUBLE = {"ā": "aa", "ī": "ii", "ū": "uu"}


def _long(v: str, k: int, syl: list[dict]) -> bool:
    """Does rekhta.org double this long vowel (yaad, jaane, siine, huuñ) or
    mark it (kā, āḳhir, zamāne, bhūle)? Counted on its pages: a closed
    one-syllable word doubles; so does the open first syllable of a two-
    syllable word ending in a vowel (jaa-ne, paa-nī, sii-ne -- not ū), a
    first ā before a hiatus (jaa.e) and a word-initial ā before an open
    syllable (aarzū, aadmī). Everything else, and every part of a compound,
    is marked."""
    n, me = len(syl), syl[k]
    if n == 1:
        return me["closed"] or (v == "ā" and me["initial"])
    if n != 2 or k != 0 or v == "ū":
        return False
    nxt_open = not syl[1]["closed"]
    if v == "ī":
        return not me["closed"] and nxt_open
    return me["hiatus"] or (nxt_open and (not me["closed"] or me["initial"]))


def _rekhta_word(word: str, compound: bool) -> str:
    """One R word in rekhta.org's marked Roman."""
    toks: list[list] = []                       # ["V", vowel, nasal] / ["C", unit]
    for u in _units(word):
        if u[0] in R_VOWELS:
            nasal = u.endswith("~")
            v, i, parts = u.rstrip("~"), 0, []
            while i < len(v):
                step = 2 if v[i] == "a" and v[i + 1:i + 2] in ("i", "u") else 1
                parts.append(v[i:i + step])
                i += step
            toks += [["V", p, False] for p in parts]
            toks[-1][2] = nasal
        else:
            toks.append(["C", u])
    vix = [t for t, tok in enumerate(toks) if tok[0] == "V"]
    syl = []
    for k, t in enumerate(vix):
        nxt = vix[k + 1] if k + 1 < len(vix) else len(toks)
        cons = sum(1 for tok in toks[t + 1:nxt] if tok[0] == "C" and tok[1] != "'")
        last = k == len(vix) - 1
        syl.append({"closed": toks[t][2] or (cons > 0 if last else cons >= 2),
                    "hiatus": nxt == t + 1, "initial": t == 0})
    out = []
    for t, tok in enumerate(toks):
        nxt = toks[t + 1] if t + 1 < len(toks) else None
        if tok[0] == "C":
            u = tok[1]
            if u == "ṃ":
                out.append("m" if nxt and nxt[1][:1] in ("p", "b", "m") else "n")
            elif u == "'":
                if 0 < t < len(toks) - 1:
                    out.append("'")
            elif u == "~" or u == "ṅ":
                out.append("ñ")                  # hoñge, rañg, añgusht
            elif u == "ñ":                       # anusvara before ch/j: khīñchā, but ranj
                prev = toks[t - 1] if t else None
                out.append("ñ" if prev and prev[0] == "V" and prev[1] not in ("a", "i", "u") else "n")
            else:
                out.append(_CONS.get(u, (u, u))[0])
            continue
        v, k = tok[1], vix.index(t)
        prev = toks[t - 1] if t else None
        if prev and prev[0] == "V":
            if v == "e" and prev[1] in ("i", "ī"):
                v = "ye"                         # liye, kahiye, kījiye
            elif not (prev[1] in ("u", "ū") or v == "o" or (prev[1] == "o" and v in ("i", "ī"))):
                out.append(".")                  # ga.e, jaa.e, ro.eñge -- but huā, hue, koī, jaao
        if v in ("i", "ī") and nxt and nxt[0] == "V" and nxt[1] == "e":
            v = "i"
        elif v in _DOUBLE and not compound and _long(v, k, syl):
            v = _DOUBLE[v]
        out.append(v + ("ñ" if tok[2] else ""))
    return "".join(out)


_AIN_HEAD = re.compile("^([क-ह]़?)(ा|ो|अ)(?!')")


def ain_respell(urdu: str, deva: str, rich: str) -> tuple[str, str]:
    """rekhta.org's spelling of an ain right after a word's first letter and
    before a consonant (معلوم, یعنی, بعد, دعویٰ, وعدہ): the Devanagari keeps it
    as an apostrophe (मा'लूम, या'नी, बा'द) and the Roman as a hiatus (ma.alūm,
    ya.anī, ba.ad, sho.ala) -- Rekhta's commonest form, 3 to 1 on its pages.
    Other words come back unchanged."""
    u = urdu.rstrip(ZER)
    if len(u) < 3 or u[1] != "ع" or u[0] in "اآع" or u[2] in "اآیےہ":
        return deva, rich
    d = unicodedata.normalize("NFD", deva)
    m = _AIN_HEAD.match(d)
    if not m:
        return deva, rich
    deva = unicodedata.normalize("NFC", d[:m.end()] + "'" + d[m.end():])
    want = {"ा": "ā", "ो": "o", "अ": "a"}[m.group(2)]
    rm = re.match(f"^([^aāiīuūeo']+){want}'?", rich)
    if rm:
        rich = rm.group(1) + {"ā": "aa", "o": "oa", "a": "aa"}[want] + rich[rm.end():]
    return deva, rich


# rekhta.org's simple Roman (its Roman toggle: dil-e-nadan tujhe hua kya hai),
# from its marked Roman. Words it writes its own way:
_SIMPLE_WORDS = {
    "meñ": "mein", "maiñ": "main", "haiñ": "hain", "nahīñ": "nahin", "kahīñ": "kahin",
    "hameñ": "hamein", "tumheñ": "tumhein", "unheñ": "unhen", "inheñ": "inhen", "jinheñ": "jinhen",
    "kyuuñ": "kyun", "yuuñ": "yun", "huuñ": "hun", "ham": "hum", "vo": "wo", "tire": "tere",
    "mire": "mere", "tirī": "teri", "mirī": "meri", "tirā": "tera", "mirā": "mera", "ik": "ek",
    "kahūñ": "kahun", "jahāñ": "jahan", "yahāñ": "yahan", "vahāñ": "wahan",
}


def to_simple(word: str) -> str:
    """One word of rekhta.org's marked Roman -> its simple Roman: marks off
    (ā a, ḳh KH, ġh gh), ñ n (-eñ -en), v w, a word-initial ā kept long
    (aankh, aate) and a doubled vowel kept (yaad, raat)."""
    lw = word.lower()
    if lw in _SIMPLE_WORDS:
        return _SIMPLE_WORDS[lw]
    s = word.replace("ḳh", "KH").replace("ġh", "gh").replace("ḍ", "D").replace("Ḍ", "D")
    s = re.sub(r"(?<![\w'])ā", "aa", s)
    s = s.replace("a.a", "a").replace(".", "")
    s = s.replace("ii", "i").replace("uu", "u")
    s = s.replace("ā", "a").replace("ī", "i").replace("ū", "u")
    s = re.sub(r"eñ\b", "en", s)
    s = re.sub(r"aiñ\b", "ain", s).replace("ñ", "n")
    return re.sub(r"(?<![a-z])v|(?<=[aeiou-])v|(?<=n)v", "w", s)


def undouble(roman: str) -> str:
    """rekhta.org writes no doubled vowel inside a compound (hāl-e-dil, not haal)."""
    return roman.replace("aa", "ā").replace("ii", "ī").replace("uu", "ū")


def render(rich: str, style: str, compound: bool = False) -> str:
    """R line -> ``plain`` / ``rekhta`` / ``ascii`` Roman. ``compound``: the
    words are part of a hyphenated compound around them (rekhta.org marks
    every long vowel there)."""
    if style == "plain":
        return to_plain(rich)
    parts = re.split(r"([ \-])", rich)
    if style != "rekhta":
        return "".join(p if p in (" ", "-") or not p else _render_word(p, 1) for p in parts)
    out = []
    for i, p in enumerate(parts):
        if p in (" ", "-") or not p:
            out.append(p)
            continue
        near = (parts[i - 1] if i else "", parts[i + 1] if i + 1 < len(parts) else "")
        out.append(_rekhta_word(p, compound or "-" in near))
    return "".join(out)


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
        from urdu_nn.npnn import Checkpoint      # torch when installed, else numpy
        self.net = Checkpoint(path, device)
        self.vocab, self.beam = self.net.vocab, beam
        self._cache: dict[str, list[tuple[str, float]]] = {}

    def readings(self, words: list[str]) -> dict[str, list[tuple[str, float]]]:
        import math
        todo = [w for w in dict.fromkeys(words) if w not in self._cache
                and all(c in self.vocab.stoi for c in w)]
        for b in range(0, len(todo), 64):
            chunk = todo[b:b + 64]
            beams = self.net.beam_search([self.vocab.encode_src("j", w) for w in chunk],
                                         beam=self.beam, max_len=64)
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
        if urdu.rstrip("ٔ").endswith(("ہ", "ۂ")) and deva.endswith("ा") and best.endswith("ā"):
            best = best[:-1] + "a"             # silent he: afsāna, jalva-e-gul (جلوۂ گل)
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


def reading_ok(urdu: str, deva: str, back: str, min_back: float = 0.85) -> str:
    """"" when Rekhta's reading ``deva`` of the Urdu run ``urdu`` is sound,
    else why not: too long (a loop), Latin letters, words that don't line up
    with the Urdu (dropped, doubled), or a read-back (``back`` = its reverse
    model's Urdu for ``deva``) that isn't the Urdu written. The same test
    picks the teacher's training labels and, at runtime, when the engine
    gives Rekhta's reading verbatim."""
    from urdu_nn.rekhta_text import norm_word
    if not deva or len(deva) > 2 * len(urdu) + 12:
        return "length"
    if any(c.isascii() and c.isalpha() for c in deva):
        return "latin"
    if align_units(urdu.split(), deva_units(deva)) is None:
        return "words"

    def nb(s):
        return " ".join(norm_word(w).rstrip(ZER) for w in s.split())
    b, want = nb(back), nb(urdu)
    if b != want and _sim(b, want) < min_back:
        return "roundtrip"
    return ""


def _urdu_of(deva: str) -> str:
    """A likely Urdu spelling of one Devanagari word; Rekhta's ain apostrophe
    (मोअ'ला, ए'लान) becomes ع."""
    from urdu_nn.scheme import deva_to_urdu_candidates
    parts = []
    for p in re.sub("अ'", "'", deva).split("'"):
        d = re.sub(r"[^ऀ-ॿ]", "", p)
        c = deva_to_urdu_candidates(d, limit=1) if d else []
        parts.append(c[0] if c else "")
    return "ع".join(parts)


def _sim(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.normalized_similarity(a, b)


# Urdu letters that sound alike -- Devanagari writes each group with one
# letter (त for ت/ط, ह for ہ/ح, स for س/ص/ث, ज़ for ز/ذ/ض/ظ), so a spelling
# rebuilt from Devanagari can only be compared with the written Urdu by sound
_SOUND = str.maketrans({"ط": "ت", "ث": "س", "ص": "س", "ح": "ہ", "ذ": "ز", "ض": "ز",
                        "ظ": "ز", "ژ": "ز", "ق": "ک", "غ": "گ", "خ": "ک", "ۂ": "ہ", "ۃ": "ہ",
                        "ة": "ہ", "ئ": "ی", "ۓ": "ی", "ے": "ی", "ؤ": "و", "آ": "ا",
                        "أ": "ا", "ں": "ن", "ع": None, "ء": None, "ھ": None,
                        "ٔ": None, "ٰ": None, ZER: None})


def _sound(urdu: str) -> str:
    urdu = urdu.replace("یٰ", "ا")                  # اعلیٰ: the ی is read ā
    urdu = re.sub("ہ$", "ا", urdu)                 # silent final he reads like alif
    return urdu.translate(_SOUND)


def _skel(sound: str) -> str:
    """Consonants only: the vowel letters are where Urdu and Devanagari
    spellings of one word disagree most (عہدہ ओहदा, معلیٰ मोअ'ला)."""
    return re.sub("[اوی]", "", sound)


_IZAFAT_TAIL = re.compile("(ئے|ۓ|ۂ|ٔ|ے|ی)$")


def _word_sim(urdu: str, deva: str, izafat: bool = False) -> float:
    """How well a Devanagari reading fits the Urdu written -- by sound, and
    by consonant skeleton. With izafat the Urdu may spell the -e- itself
    (عطائے = अता-ए-, خانۂ = ख़ाना-ए-)."""
    d = _sound(_urdu_of(deva))
    forms = [urdu] + ([_IZAFAT_TAIL.sub("", urdu)] if izafat else [])
    best = 0.0
    for f in forms:
        u = _sound(f)
        best = max(best, _sim(u, d), _sim(_skel(u), _skel(d)) if _skel(u) or _skel(d) else 0.0)
    return best


_MOVES = ((1, 1, 0.0, 0.45), (2, 1, 0.1, 0.6), (1, 2, 0.1, 0.6), (3, 1, 0.2, 0.7),
          (1, 3, 0.2, 0.7))


def align_units(urdu_words: list[str], units: list[list[str]]):
    """Line up Urdu words with Devanagari units -> [(i0, i1, j0, j1)] spans
    (Urdu words [i0, i1) are Devanagari units [j0, j1)), or None.

    Mostly one to one, but Rekhta also writes two Urdu words as one
    (پاؤں گا -> पाऊँगा, گل زار -> गुलज़ार) and one as a hyphenated pair
    (سربسر -> सर-बसर). Each choice is scored by how close the Devanagari's
    Urdu spelling is to the Urdu actually written."""
    n, m = len(urdu_words), len(units)
    if not n or not m:
        return None
    words = [w.rstrip(ZER) for w in urdu_words]
    if n == m:
        sims = [_word_sim(words[i], units[i][0], units[i][1] == "-e-") for i in range(n)]
        if all(s >= 0.45 or len(words[i]) <= 2 for i, s in enumerate(sims)):
            return [(i, i + 1, i, i + 1) for i in range(n)]
    NEG = -1e9
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if best[i][j] == NEG:
                continue
            for di, dj, cost, floor in _MOVES:
                if i + di > n or j + dj > m:
                    continue
                if dj > 1 and any(units[k][1] != "-" for k in range(j, j + dj - 1)):
                    continue                     # only a plain hyphen splits one word
                u = "".join(words[i:i + di])
                d = "".join(units[k][0] for k in range(j, j + dj))
                s = _word_sim(u, d, units[j + dj - 1][1] == "-e-")
                if s < floor and not (di == dj == 1 and len(u) <= 2):
                    continue
                if best[i][j] + s - cost > best[i + di][j + dj]:
                    best[i + di][j + dj] = best[i][j] + s - cost
                    back[i + di][j + dj] = (di, dj)
    if best[n][m] == NEG:
        return None
    spans, i, j = [], n, m
    while i or j:
        di, dj = back[i][j]
        spans.append((i - di, i, j - dj, j))
        i, j = i - di, j - dj
    return spans[::-1]
