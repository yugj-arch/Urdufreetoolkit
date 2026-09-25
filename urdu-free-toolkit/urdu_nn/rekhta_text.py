# -*- coding: utf-8 -*-
"""How a line of Urdu is cut up for the line-level (Rekhta-style) models.

The models only ever see *runs of Urdu words*: letters with single spaces
between them, never punctuation, digits or Latin text. Those are handled
around the model, so a comma or a year can never make it hallucinate. The
same cutting is used when the teacher labels the corpus, when the student is
trained, and at runtime -- so all three see identical inputs.
"""
from __future__ import annotations

import re
import unicodedata

from urdu_nn.scheme import HARAKAT, HAMZA_ABOVE, KHARI_ZABAR, URDU_LETTERS, norm_urdu

ZER = "ِ"
MAX_RUN = 45            # chars per model input: Rekhta's teacher was trained on misras
                        # (99% of ghazal lines are <= 50 chars) and loops on longer input
_HONORIFIC = re.compile("[ؐ-ؚ]")       # ؐ ؑ ؒ ؓ ...: marks after holy names, not read
# Arabic letter variants, kashida and joiners fold away inside norm_urdu
_VARIANTS = set("يىكهەةأإٱۀـ‌‍")
_WORD_CH = URDU_LETTERS | HARAKAT | {KHARI_ZABAR, HAMZA_ABOVE} | _VARIANTS
_WORD_RE = re.compile("[" + re.escape("".join(sorted(_WORD_CH))) + "]+")


def norm_word(word: str) -> str:
    """One Urdu word as the models key it: letter variants folded, harakat
    dropped -- except a final zer, the izafat the writer spelled out (دلِ)."""
    word = _HONORIFIC.sub("", unicodedata.normalize("NFC", word))
    zer = word.endswith(ZER) and len(word) > 1
    w = norm_urdu(word)
    return w + ZER if zer and w else w


def _chunks(words: list[str]) -> list[str]:
    """Cut a long run into nearly equal pieces of <= MAX_RUN chars (a
    trailing one-word scrap would lose its context)."""
    total = len(" ".join(words))
    if total <= MAX_RUN:
        return [" ".join(words)] if words else []
    target = total / -(-total // MAX_RUN)
    out, cur = [], []
    for w in words:
        if cur and (len(" ".join(cur)) >= target or len(" ".join(cur + [w])) > MAX_RUN):
            out.append(" ".join(cur))
            cur = []
        cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


def segments(line: str) -> list[tuple[str, str]]:
    """``line`` -> [(kind, text)]: kind "u" is a run of normalised Urdu words
    (single-spaced, <= MAX_RUN chars), kind "x" is whatever sits between runs,
    verbatim (spaces, punctuation, digits, Latin). A lone ء after a year
    (1828ء) counts as "x"."""
    line = _HONORIFIC.sub("", unicodedata.normalize("NFC", line))
    out: list[tuple[str, str]] = []
    run: list[str] = []
    gap, pos = "", 0

    def flush():
        for k, c in enumerate(_chunks(run)):
            if k:
                out.append(("x", " "))
            out.append(("u", c))
        run.clear()

    for m in _WORD_RE.finditer(line):
        gap += line[pos:m.start()]
        pos = m.end()
        w = norm_word(m.group())
        if not w or w == "ء":
            gap += m.group()
            continue
        if run and (not gap or gap.isspace()):
            run.append(w)
            gap = ""
            continue
        flush()
        if gap:
            out.append(("x", gap))
        gap = ""
        run.append(w)
    gap += line[pos:]
    flush()
    if gap:
        out.append(("x", gap))
    return out


def urdu_runs(line: str) -> list[str]:
    return [t for k, t in segments(line) if k == "u"]
