# -*- coding: utf-8 -*-
"""Spelling schemes shared by the neural transliterator's data builder,
trainer and runtime provider.

Three spellings meet here:

* **Urdu input** -- :func:`norm_urdu` folds letter variants (Arabic yeh/kaf,
  heh forms, digits, kashida, ZWNJ) so training keys and runtime tokens are
  compared on equal footing. Unlike ``transliterate.normalize_urdu`` it keeps
  hamza seats (ئ ؤ ء) -- they carry real reading information (جائے, گئے).

* **R ("rich") Roman** -- the ONE canonical Roman spelling the model learns.
  It is Wiktionary's romanisation cleaned into a single alphabet that loses
  nothing a reader needs:

      vowels      a ā i ī u ū e o  (ai au as two letters)
      nasal       "~" after the vowel run it nasalises   (hai~ = हैं)
      consonants  c = چ, ch = چھ, x = خ, ġ = غ, q = ق, ś = ش, ṣ = ष, ž = ژ,
                  ṭ ḍ ṛ retroflex, ṇ ṅ ñ ṃ nasal consonants, v = و,
                  ' = ain / hamza

  Every Roman string the app shows is *rendered* from R -- never learned
  separately -- so the Plain and Diacritic toggles can never disagree about
  a vowel.

* **Renders** -- :func:`to_plain` (the app's casual ASCII: kitaab, nahin,
  mein, kehna) and :func:`to_rekhta` (Rekhta-style: kitāb, nahīñ, meñ,
  ḳhayāl).

It also carries :func:`deva_to_urdu_candidates`, used by the data builder to
give Hindi-Wiktionary words an Urdu spelling (candidates are then checked
against a real Urdu corpus, so only attested spellings survive).
"""
from __future__ import annotations

import itertools
import re
import unicodedata

# ---------------------------------------------------------------------------
# Urdu input normalisation
# ---------------------------------------------------------------------------

HARAKAT = set("ًٌٍَُِّْ"
              "ٕٖٜٟٓٗ٘ٙٚٛٝٞ")
KHARI_ZABAR = "ٰ"      # superscript alif: common in running text (اعلیٰ)
HAMZA_ABOVE = "ٔ"      # izafat hamza on he/ye (خانۂ): kept, it's a reading cue

_URDU_FOLDS = {
    "ي": "ی",   # Arabic yeh -> Urdu yeh
    "ى": "ی",   # alif maqsura -> Urdu yeh
    "ك": "ک",   # Arabic kaf -> keheh
    "ه": "ہ",   # Arabic heh -> gol he
    "ە": "ہ",   # ae -> gol he
    "ة": "ہ",   # teh marbuta -> gol he
    "أ": "ا",   # alif hamza above -> alif
    "إ": "ا",   # alif hamza below -> alif
    "ٱ": "ا",   # alif wasla -> alif
    "ۀ": "ۂ",   # heh with yeh above -> heh goal with hamza
    "ـ": "",         # kashida
    "‌": "",         # ZWNJ
    "‍": "",         # ZWJ
    "‎": "", "‏": "",   # LRM / RLM
}
_URDU_FOLDS.update({chr(0x0660 + n): str(n) for n in range(10)})
_URDU_FOLDS.update({chr(0x06F0 + n): str(n) for n in range(10)})

URDU_LETTERS = set("ءآأؤإئابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوہۂھیےۓ")


def norm_urdu(text: str, keep_harakat: bool = False) -> str:
    """Fold letter variants; strip harakat unless ``keep_harakat``.

    Khari zabar (ٰ) and the izafat hamza (ٔ) are always kept -- they're common
    in ordinary text and change the reading."""
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        ch = _URDU_FOLDS.get(ch, ch)
        if not keep_harakat and ch in HARAKAT:
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def is_urdu_word(token: str) -> bool:
    return bool(token) and all(ch in URDU_LETTERS or ch in HARAKAT
                               or ch in (KHARI_ZABAR, HAMZA_ABOVE) for ch in token)


# ---------------------------------------------------------------------------
# Wiktionary romanisation -> R
# ---------------------------------------------------------------------------

_MACRON, _ACUTE, _GRAVE, _CIRC, _TILDE = "̄", "́", "̀", "̂", "̃"
_BREVE, _DOT_ABOVE, _DIAER, _CARON = "̆", "̇", "̈", "̌"
_DOT_BELOW, _DIAER_BELOW, _MACRON_BELOW, _DOUBLE_TILDE = "̣", "̤", "̱", "͠"
_APOSTROPHES = set("'ʻ’‘ʿʾʼ")

R_VOWELS = set("aāiīuūeo")
_REJECT = set(",~/()0123456789;:[]{}<>=+*&%$#@|\\\"")


def _base_to_r(base: str, marks: list[str], source: str, prev: str) -> str | None:
    m = set(marks)
    if base == "a":
        return "ā" if (_MACRON in m or _ACUTE in m) else "a"
    if base == "i":
        return "ī" if (_MACRON in m or _ACUTE in m) else "i"
    if base == "u":
        return "ū" if (_MACRON in m or _ACUTE in m) else "u"
    if base in "eo":
        return base
    if base == "s":
        if _ACUTE in m or _CARON in m:
            return "ś"
        if _DOT_BELOW in m:
            return "ṣ" if source == "hi" else "s"
        return "s"
    if base == "z":
        return "ž" if _CARON in m else "z"
    if base == "g":
        return "ġ" if (_BREVE in m or _DOT_ABOVE in m) else "g"
    if base == "n":
        if _DOT_ABOVE in m:
            return "ṅ"
        if _DOT_BELOW in m:
            return "ṇ"
        if _TILDE in m:
            return "ñ"
        return "n"
    if base == "t":
        return "ṭ" if _DOT_BELOW in m else "t"
    if base == "d":
        if _DOT_BELOW in m:
            return "ḍ"
        if _MACRON_BELOW in m:
            return "z"          # ḏ = ذ in scholarly schemes
        return "d"
    if base == "r":
        if _ACUTE in m or _DOT_BELOW in m and _MACRON in m:
            return "ri"         # vocalic r (ऋ)
        return "ṛ" if _DOT_BELOW in m else "r"
    if base == "h":
        if _MACRON_BELOW in m and prev == "x":
            return ""           # second half of ALA ḵẖ
        return "h"
    if base == "k":
        if _MACRON_BELOW in m:
            return "x"          # ALA ḵ(ẖ) = خ
        if _DOT_BELOW in m:
            return "q"
        return "k"
    if base == "m":
        return "ṃ" if _DOT_BELOW in m else "m"
    if base == "c":
        return "c"
    if base == "w":
        return "v"
    if base in "bfjlpqvxy":
        return base
    return None


def wik_to_rich(roman: str, urdu: str = "", source: str = "ur") -> str | None:
    """Convert a Wiktionary romanisation to R. ``urdu`` (the word's Urdu
    spelling, if known) resolves hand-written digraphs (``kh`` for خ, ``ch``
    for چ ...) into R's single letters. Returns None for entries that aren't a
    single clean reading (alternatives, notes, digits)."""
    if not roman:
        return None
    r = unicodedata.normalize("NFC", roman).strip().rstrip("?!.")
    if not r or any(ch in _REJECT for ch in r):
        return None
    if r.startswith("-") or r.endswith("-"):
        return None
    d = unicodedata.normalize("NFD", r.lower())

    units: list[str] = []          # R letters, " ", "-", "'"
    nasal_units: set[int] = set()  # index of a vowel unit carrying nasality
    pending_nasal = False          # double tilde on a consonant: nasalise next run
    i = 0
    while i < len(d):
        base = d[i]
        i += 1
        marks = []
        while i < len(d) and unicodedata.combining(d[i]):
            marks.append(d[i])
            i += 1
        if base in _APOSTROPHES:
            units.append("'")
            continue
        if base in " -":
            units.append(base)
            continue
        if base == "ı":        # dotless i
            base = "i"
        prev = units[-1] if units else ""
        out = _base_to_r(base, marks, source, prev)
        if out is None:
            return None
        nasal = (_TILDE in marks and base != "n") or _DOUBLE_TILDE in marks
        if out == "":
            continue
        for ch in out:
            units.append(ch)
        if nasal:
            if units[-1] in R_VOWELS:
                nasal_units.add(len(units) - 1)
            else:
                pending_nasal = True
        elif pending_nasal and units[-1] in R_VOWELS:
            nasal_units.add(len(units) - 1)
            pending_nasal = False

    # emit "~" once, at the end of each vowel run that carries nasality
    out_chars: list[str] = []
    run_nasal = False
    for idx, u in enumerate(units):
        out_chars.append(u)
        if u in R_VOWELS:
            run_nasal = run_nasal or idx in nasal_units
            nxt = units[idx + 1] if idx + 1 < len(units) else ""
            if nxt not in R_VOWELS:
                if run_nasal:
                    out_chars.append("~")
                run_nasal = False
    rich = "".join(out_chars).strip()
    rich = re.sub(r"\s+", " ", rich)
    if urdu:
        rich = _fix_digraphs(rich, norm_urdu(urdu))
    return unicodedata.normalize("NFC", rich) or None


def _fix_digraphs(rich: str, urdu: str) -> str:
    """Hand-typed Wiktionary entries often write خ/غ/ش/چ/ژ as digraphs; the
    Urdu spelling tells us which reading was meant."""
    if "ch" in rich and "چ" in urdu and "چھ" not in urdu:
        rich = rich.replace("chh", "\x00").replace("ch", "c").replace("\x00", "ch")
    if "sh" in rich and "ش" in urdu and not re.search("س[ہھ]", urdu):
        rich = rich.replace("sh", "ś")
    if "kh" in rich and "خ" in urdu and "کھ" not in urdu:
        rich = rich.replace("kh", "x")
    if "gh" in rich and "غ" in urdu and "گھ" not in urdu:
        rich = rich.replace("gh", "ġ")
    if "zh" in rich and "ژ" in urdu:
        rich = rich.replace("zh", "ž")
    return rich


# ---------------------------------------------------------------------------
# R -> display renders
# ---------------------------------------------------------------------------

_ASPIRABLE = set("kgcjṭḍtdpbṛ")


def _units(word: str) -> list[str]:
    """Split one R word into letters, keeping consonant+h aspirates and
    vowel runs (with a trailing ~) as single units."""
    out: list[str] = []
    i = 0
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
            continue
        if ch in _ASPIRABLE and i + 1 < len(word) and word[i + 1] == "h" \
                and not (i + 2 < len(word) and word[i + 2] in R_VOWELS and ch == "c" and False):
            out.append(word[i:i + 2])
            i += 2
            continue
        out.append(ch)
        i += 1
    return out


def _is_vowel_unit(u: str) -> bool:
    return bool(u) and u[0] in R_VOWELS


_PLAIN_CONS = {
    "c": "ch", "ch": "chh", "x": "kh", "ġ": "gh", "ś": "sh", "ṣ": "sh", "ž": "zh",
    "ṭ": "t", "ṭh": "th", "ḍ": "d", "ḍh": "dh", "ṛ": "d", "ṛh": "dh",
    "ṇ": "n", "ṅ": "n", "ñ": "n", "v": "w", "'": "",
}
_REKHTA_CONS = {
    "c": "ch", "ch": "chh", "x": "ḳh", "ś": "sh", "ṣ": "sh", "ž": "zh",
    "ṇ": "n", "ṅ": "n", "ñ": "n",
}


def _plain_vowels(run: str, final: bool, mono: bool) -> str:
    nasal = run.endswith("~")
    v = run.rstrip("~")
    if nasal:
        head, last = v[:-1], v[-1]
        if v.endswith("ai") or v.endswith("au"):
            head, last = v[:-2], v[-2:]
        head_s = "".join({"ā": "aa", "ī": "ee", "ū": "oo"}.get(c, c) for c in head)
        if head and last == "e":
            head_s += "y"                       # jāẽ -> jaayein
        tail = {
            "e": "ein", "ai": "ain", "au": "aun", "o": "on", "a": "an", "i": "in",
            "u": "un", "ū": "oon",
            "ā": "an" if (final and not mono) else "aan",
            "ī": "in" if final else "een",
        }[last]
        return head_s + tail
    out = []
    for k, c in enumerate(v):
        last = final and k == len(v) - 1
        if c == "ā":
            out.append("a" if last else "aa")
        elif c == "ī":
            out.append("i" if last else "ee")
        elif c == "ū":
            out.append("u" if last else "oo")
        elif c == "e" and k > 0:
            out.append("ye")                    # hiatus glide: jāe -> jaaye, gae -> gaye
        else:
            out.append(c)
    return "".join(out)


def _render_word(word: str, style: str) -> str:
    if not word:
        return word
    if style == "plain" and word == "ā":     # آ on its own: aa gaya, aa kar
        return "aa"
    units = _units(word)
    n_vowel_runs = sum(1 for u in units if _is_vowel_unit(u))
    mono = n_vowel_runs <= 1
    out: list[str] = []
    for k, u in enumerate(units):
        nxt = units[k + 1] if k + 1 < len(units) else ""
        final = k == len(units) - 1
        if _is_vowel_unit(u):
            if style == "plain":
                # "ah" + consonant -> "eh" (kahnā -> kehna, pahlā -> pehla)
                if (u == "a" and k > 0 and not _is_vowel_unit(units[k - 1])
                        and nxt == "h" and k + 2 < len(units)
                        and not _is_vowel_unit(units[k + 2])):
                    out.append("e")
                    continue
                out.append(_plain_vowels(u, final, mono))
            else:
                out.append(u.replace("~", "ñ"))
            continue
        if u == "ṃ":
            out.append("m" if nxt[:1] in ("p", "b", "m") else "n")
            continue
        if u == "'":
            if style == "plain" or k == 0 or final:
                continue
            out.append("'")
            continue
        if (style == "plain" and u == "v" and final and k > 0
                and _is_vowel_unit(units[k - 1])):
            out.append("v")                     # word-final: dev, shiv, dabaav -- not dew
            continue
        table = _PLAIN_CONS if style == "plain" else _REKHTA_CONS
        out.append(table.get(u, u))
    if style == "plain":            # a dropped ain can butt vowels: jamā'at -> jamaat
        return re.sub(r"([aeiou])\1{2,}", r"\1\1", "".join(out))
    return "".join(out)


def _render(rich: str, style: str) -> str:
    parts = re.split(r"([ \-])", rich)
    return "".join(p if p in (" ", "-") else _render_word(p, style) for p in parts)


def to_plain(rich: str) -> str:
    """R -> the app's casual ASCII Roman (kitaab, nahin, mein, kehna)."""
    return _render(rich, "plain")


def to_rekhta(rich: str) -> str:
    """R -> Rekhta-style diacritic Roman (kitāb, nahīñ, meñ, ḳhayāl)."""
    return _render(rich, "rekhta")


def fold_roman(s: str) -> str:
    """Style-agnostic comparison key for Roman strings from different systems
    (R renders, GPT casual, Dakshina casual): drops diacritics, vowel length,
    nasal spellings and the usual casual digraph variants. Used for eval only."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("ḳ", "k")
    s = re.sub(r"['’‘ʻʿ\-]", "", s)
    s = s.replace("w", "v").replace("chh", "ch").replace("ee", "i").replace("oo", "u")
    s = re.sub(r"([aeiou])\1+", r"\1", s)
    s = s.replace("ein", "en").replace("ain", "an")
    return s


_HALF_NASAL_D = re.compile("[ङञणनम]्(?=[क-ह])")


def hindi_fold(deva: str) -> str:
    """Comparison key for two Devanagari spellings of one word: blind to
    nukta, nasal notation and virama (कुर्सियाँ = कुरसियां, सम्बन्ध = संबंध).
    Keys the Hindi-word table the runtime reranks the model's beam with."""
    s = unicodedata.normalize("NFC", deva).replace("़", "").replace("ँ", "ं")
    s = _HALF_NASAL_D.sub("ं", s.replace("‍", "").replace("‌", ""))
    return s.replace("्", "")


# ---------------------------------------------------------------------------
# Devanagari -> Urdu spelling candidates (data building only)
# ---------------------------------------------------------------------------

_NUKTA = "़"
_VIRAMA = "्"
_ANUSVARA, _CANDRABINDU, _VISARGA = "ं", "ँ", "ः"

_D_CONS = {
    "क": ["ک", "ق"], "ख": ["کھ", "خ"], "ग": ["گ", "غ"], "घ": ["گھ"], "ङ": ["ن"],
    "च": ["چ"], "छ": ["چھ"], "ज": ["ج", "ز"], "झ": ["جھ"], "ञ": ["ن"],
    "ट": ["ٹ"], "ठ": ["ٹھ"], "ड": ["ڈ"], "ढ": ["ڈھ"], "ण": ["ن"],
    "त": ["ت", "ط"], "थ": ["تھ"], "द": ["د"], "ध": ["دھ"], "न": ["ن"],
    "प": ["پ"], "फ": ["پھ", "ف"], "ब": ["ب"], "भ": ["بھ"], "म": ["م"],
    "य": ["ی"], "र": ["ر"], "ल": ["ل"], "व": ["و"], "श": ["ش"], "ष": ["ش"],
    "स": ["س", "ص", "ث"], "ह": ["ہ", "ح"],
}
_D_NUKTA = {
    "क": ["ق"], "ख": ["خ"], "ग": ["غ"], "ज": ["ز", "ذ", "ض", "ظ"], "फ": ["ف"],
    "ड": ["ڑ"], "ढ": ["ڑھ"], "य": ["ی"], "झ": ["ژ"], "श": ["ش"], "स": ["ث", "س"],
}
_ASPIRATE_OF = {"क": "ख", "ग": "घ", "च": "छ", "ज": "झ", "ट": "ठ", "ड": "ढ",
                "त": "थ", "द": "ध", "प": "फ", "ब": "भ"}
_D_INDEP_INITIAL = {
    "अ": ["ا", "ع"], "आ": ["آ", "عا"], "इ": ["ا", "ع"], "ई": ["ای", "عی"],
    "उ": ["ا", "ع"], "ऊ": ["او"], "ए": ["ای", "اے", "ع"], "ऐ": ["ای", "اے", "عی"],
    "ओ": ["او"], "औ": ["او", "عو"], "ऋ": ["ر"], "ऑ": ["آ", "او"], "ऍ": ["ای"],
}
_D_INDEP_MEDIAL = {
    "अ": ["ا", "ع"], "आ": ["ا", "آ"], "इ": ["ئ", "ی"], "ई": ["ئی", "ی"],
    "उ": ["ؤ", "و"], "ऊ": ["ؤ", "و"], "ए": ["ئ", "ی"], "ऐ": ["ئ", "ی"],
    "ओ": ["ؤ", "و"], "औ": ["ؤ", "و"], "ऋ": ["ر"],
}
_D_MATRA = {
    "ा": ["ا"], "ि": [""], "ी": ["ی"], "ु": [""], "ू": ["و"], "े": ["ی"],
    "ै": ["ی"], "ो": ["و"], "ौ": ["و"], "ृ": ["ر"], "ॉ": ["ا", "و"], "ॅ": ["ی"],
}


def deva_to_urdu_candidates(deva: str, limit: int = 96) -> list[str]:
    """All plausible Urdu spellings of a single Devanagari word, most likely
    first (first-listed letter options win ties). Empty list if the word has
    anything this mapper doesn't model (digits, Vedic marks, ...)."""
    deva = unicodedata.normalize("NFC", deva.strip())
    if not deva or " " in deva:
        return []
    # tokenise into (kind, value, nukta)
    toks: list[tuple[str, str, bool]] = []
    for ch in deva:
        if ch == _NUKTA:
            if not toks or toks[-1][0] != "C":
                return []
            k, v, _ = toks[-1]
            toks[-1] = (k, v, True)
        elif ch in _D_CONS:
            toks.append(("C", ch, False))
        elif ch in _D_MATRA:
            toks.append(("M", ch, False))
        elif ch in _D_INDEP_INITIAL:
            toks.append(("V", ch, False))
        elif ch == _VIRAMA:
            toks.append(("H", ch, False))
        elif ch in (_ANUSVARA, _CANDRABINDU):
            toks.append(("N", ch, False))
        elif ch == _VISARGA:
            toks.append(("S", ch, False))
        else:
            return []
    slots: list[list[str]] = []
    n = len(toks)
    for i, (kind, val, nukta) in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < n else None
        rest_is_nasal_only = all(t[0] == "N" for t in toks[i + 1:])
        at_end = i == n - 1 or rest_is_nasal_only
        if kind == "C":
            # C + virama + same / aspirated-same consonant -> written once
            if nxt and nxt[0] == "H" and i + 2 < n and toks[i + 2][0] == "C":
                c2 = toks[i + 2][1]
                if (c2 == val or _ASPIRATE_OF.get(val) == c2) and not nukta:
                    slots.append([""])
                    continue
            if val == "ज" and nxt and nxt[0] == "H" and i + 2 < n and toks[i + 2][1] == "ञ":
                slots.append(["گ"])
                continue
            if val == "ञ" and i >= 2 and toks[i - 2][1] == "ज" and toks[i - 1][0] == "H":
                slots.append(["ی"])
                continue
            opts = _D_NUKTA.get(val, _D_CONS[val]) if nukta else _D_CONS[val]
            # a consonant with no matra/virama after it at word end: inherent
            # vowel is silent in Hindustani -- nothing extra to write
            slots.append(list(opts))
        elif kind == "M":
            if val in ("े", "ै") and at_end:
                has_nasal = i + 1 < n
                slots.append(["ی"] if has_nasal else ["ے"])
            elif val == "ा" and at_end and i + 1 == n:
                slots.append(["ا", "ہ"])
            elif val == "ि" and at_end:
                slots.append(["ی"])
            elif val == "ु" and at_end:
                slots.append(["و"])
            else:
                slots.append(list(_D_MATRA[val]))
        elif kind == "V":
            prev = toks[i - 1] if i else None
            if i == 0 or (prev and prev[0] in ("N",) and i == 1):
                opts = _D_INDEP_INITIAL[val]
            elif prev and prev[0] in ("M", "V") or (prev and prev[0] == "C" and False):
                opts = _D_INDEP_MEDIAL.get(val, ["ا"])
                if val in ("ए", "ऐ") and at_end and i + 1 == n:
                    opts = ["ئے", "ے"]
                elif val in ("ए", "ऐ") and at_end:
                    opts = ["ئی", "ئ"]
            else:
                # vowel letter right after a consonant (schwa + vowel hiatus)
                opts = _D_INDEP_MEDIAL.get(val, ["ا"])
                if val == "अ":
                    opts = ["ع", "ا"]
                elif val in ("ए", "ऐ") and at_end and i + 1 == n:
                    opts = ["ئے", "ے"]
                elif val in ("ए", "ऐ") and at_end:
                    opts = ["ئی", "ئ"]
            slots.append(list(opts))
        elif kind == "H":
            slots.append([""])
        elif kind == "N":
            if i == n - 1:
                slots.append(["ں"])
            else:
                nxt_c = nxt[1] if nxt and nxt[0] == "C" else ""
                slots.append(["م", "ن"] if nxt_c in ("ब", "प", "म") else ["ن"])
        elif kind == "S":
            slots.append(["ہ"])
    # best-first: the all-first-choice spelling, then every spelling that
    # deviates from it in one slot, then in two, then three
    multi = [k for k, s in enumerate(slots) if len(s) > 1]
    base = [s[0] for s in slots]
    out: list[str] = []
    seen = set()
    for depth in range(0, 4):
        for pos in itertools.combinations(multi, depth):
            for choice in itertools.product(*[slots[p][1:] for p in pos]):
                combo = list(base)
                for p, c in zip(pos, choice):
                    combo[p] = c
                s = "".join(combo)
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
                    if len(out) >= limit:
                        return out
    return out


# ---------------------------------------------------------------------------
# Devanagari -> R readings (data building only)
# ---------------------------------------------------------------------------

_R_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ṅ", "च": "c", "छ": "ch", "ज": "j",
    "झ": "jh", "ञ": "ñ", "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ", "त": "t",
    "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh",
    "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "ś", "ष": "ṣ", "स": "s", "ह": "h",
}
_R_NUKTA = {"क": "q", "ख": "x", "ग": "ġ", "ज": "z", "झ": "ž", "फ": "f", "ड": "ṛ", "ढ": "ṛh",
            "य": "y"}
_R_INDEP = {"अ": "a", "आ": "ā", "इ": "i", "ई": "ī", "उ": "u", "ऊ": "ū", "ए": "e", "ऐ": "ai",
            "ओ": "o", "औ": "au", "ऋ": "ri", "ऑ": "o", "ऍ": "e"}
_R_MATRA = {"ा": "ā", "ि": "i", "ी": "ī", "ु": "u", "ू": "ū", "े": "e", "ै": "ai", "ो": "o",
            "ौ": "au", "ृ": "ri", "ॉ": "o", "ॅ": "e"}


def _anusvara_before(cons: str) -> str:
    """R for an anusvara followed by consonant ``cons`` (homorganic nasal;
    a nasal vowel ``~`` before semivowels, as Hindi Wiktionary writes them)."""
    c = cons[:1]
    if c in "kgqxġ":
        return "ṅ"
    if c in "cjś":
        return "ñ"
    if c in "ṭḍṛ":
        return "ṇ"
    if c in "pbm":
        return "m"
    if c in "yrlv":
        return "~"
    return "n"


def deva_to_rich_candidates(deva: str, limit: int = 16) -> list[str]:
    """Plausible R readings of one Devanagari word, most likely first.

    Devanagari spells every vowel except the inherent schwa, so the only real
    ambiguity is which schwas are silent. The first candidate follows the
    standard Hindi schwa-deletion rule (right to left, delete in V C _ C V;
    word-final schwa silent); the rest flip one, two, ... of the deletable
    schwas, so a caller holding human romanisations (adaalaton vs adaalton)
    can pick the one people actually say. Empty list if the word has
    anything this mapper doesn't model."""
    deva = unicodedata.normalize("NFC", deva.strip()).replace("ज्ञ", "ग्य")
    if not deva or " " in deva:
        return []
    # items: [kind, r] with kind C (consonant), V (vowel), S (schwa slot),
    # F (schwa slot that is always pronounced), N (anusvara), M (candrabindu)
    items: list[list[str]] = []
    i, n = 0, len(deva)
    while i < n:
        ch = deva[i]
        if ch in _R_CONS:
            r = _R_CONS[ch]
            if i + 1 < n and deva[i + 1] == _NUKTA:
                r = _R_NUKTA.get(ch, r)
                i += 1
            items.append(["C", r])
            nx = deva[i + 1] if i + 1 < n else ""
            if nx in _R_MATRA:
                items.append(["V", _R_MATRA[nx]])
                i += 2
                continue
            if nx == _VIRAMA:
                i += 2
                continue
            items.append(["S", "a"])
            i += 1
            continue
        if ch in _R_INDEP:
            if items and items[-1][0] == "S":
                items[-1][0] = "F"          # kaī, gae: the schwa is heard before a vowel letter
            items.append(["V", _R_INDEP[ch]])
        elif ch in (_ANUSVARA, _CANDRABINDU):
            if items and items[-1][0] == "S":
                items[-1][0] = "F"          # hans, ha~s: a nasalised schwa is heard
            items.append(["N" if ch == _ANUSVARA else "M", ""])
        elif ch == _VISARGA:
            items.append(["C", "h"])
        else:
            return []
        i += 1
    if not any(k in ("V", "S", "F") for k, _ in items):
        return []

    # the first syllable's schwa is always heard (kar-nā, pra-kār)
    for k, it in enumerate(items):
        if it[0] in ("V", "F"):
            break
        if it[0] == "S":
            it[0] = "F"
            break
    slots = [k for k, it in enumerate(items) if it[0] == "S"]
    last = max((k for k, it in enumerate(items) if it[0] not in ("N", "M")), default=-1)

    def is_vowel(k, keep):
        return 0 <= k < len(items) and (items[k][0] in ("V", "F")
                                          or (items[k][0] == "S" and keep.get(k, True)))

    # default: final schwa silent, then Ohala's right-to-left V C _ C V rule
    keep: dict[int, bool] = {}
    for k in reversed(slots):
        if k == last:
            keep[k] = False
            continue
        vc = (k >= 2 and items[k - 1][0] == "C" and is_vowel(k - 2, keep))
        cv = (k + 2 < len(items) and items[k + 1][0] == "C" and is_vowel(k + 2, keep))
        keep[k] = not (vc and cv)
    # the final schwa is only really optional after a cluster (vrikṣ ~ svapna)
    free = [k for k in slots if k != last
            or (k >= 2 and items[k - 1][0] == "C" and items[k - 2][0] == "C")]

    def render(kp):
        out = []
        for k, (kind, r) in enumerate(items):
            if kind in ("C", "V", "F"):
                out.append(r)
            elif kind == "S":
                out.append("a" if kp[k] else "")
            elif kind == "M":   # candrabindu: nasal vowel, but a velar nasal before k/g
                nxt = items[k + 1] if k + 1 < len(items) else None
                out.append("ṅ" if nxt and nxt[0] == "C" and nxt[1][:1] in "kgqxġ" else "~")
            else:   # anusvara: homorganic nasal before a consonant, else nasal vowel
                nxt = items[k + 1] if k + 1 < len(items) else None
                out.append(_anusvara_before(nxt[1]) if nxt and nxt[0] == "C" else "~")
        return "".join(out)

    cands: list[str] = []
    for depth in range(len(free) + 1):
        for flip in itertools.combinations(free, depth):
            kp = dict(keep)
            for k in flip:
                kp[k] = not kp[k]
            s = render(kp)
            if s not in cands:
                cands.append(s)
                if len(cands) >= limit:
                    return cands
    return cands
