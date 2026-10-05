# -*- coding: utf-8 -*-
"""Rekhta-style Urdu transliteration: Devanagari as Rekhta writes it, Roman
in Rekhta's own scheme.

Pipeline, per line of input:

1. ``urdu_nn.rekhta_text.segments`` cuts the line into runs of Urdu words
   (<= 45 chars, a misra's length) and everything else (punctuation, digits,
   Latin), which is carried through untouched.
2. Each run is read *as a whole* by our line model (``data/rekhta_model/``,
   trained by ``training/rekhta`` on labels from Rekhta Labs' published
   poetry model, checked by its reverse model). Context matters: izafat
   (ख़ौफ़-ए-रसन), the conjunctive و (संग-ओ-ख़िश्त), کہ/کے, poetic forms. Until
   the line model is trained, Rekhta's own model stands in.
3. Every Devanagari word is lined up with its Urdu word and read into R
   (``urdu_nn.rekhta_roman``): the schwas Devanagari leaves unsaid are chosen
   by our gold lexicon, the reviewed dictionary, human romanisations and the
   word model. R is rendered as Rekhta's ASCII table (aa ii uu, KH G, T D,
   .D, ñ) and as Rekhta's diacritic Roman (ā ī ū, ḳh ġ, ṭ ḍ ṛ, ñ).

    python rekhta_translit.py "دل ہی تو ہے نہ سنگ و خشت درد سے بھر نہ آئے کیوں"
"""
from __future__ import annotations

import gzip
import json
import logging
import re
import threading
from pathlib import Path

from urdu_nn.rekhta_roman import (Reader, WordModel, ain_respell, align_units, deva_units,
                                  reading_ok, render)
from urdu_nn.rekhta_text import segments

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data" / "rekhta_model" / "model.pt"
CORRECTIONS_PATH = ROOT / "data" / "rekhta_corrections.tsv"
# Rekhta's own spellings (urdu_nn.rekhta_lexicon), built by `training.rekhta.gold
# lexicon` from rekhta.org pages; the local gold build is used when present
LEXICON_PATHS = (ROOT / "data" / "rekhta_lexicon.json.gz", ROOT / "data" / "rekhta_gold" / "lexicon.json.gz")
WORD_DIR = ROOT / "data" / "translit_model"
_DIGITS = {ord(a): str(i) for i, a in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGITS.update({ord(a): str(i) for i, a in enumerate("٠١٢٣٤٥٦٧٨٩")})
_PUNCT_DEVA = {"۔": "।", "،": ",", "؟": "?", "؛": ";", "٪": "%", "ء": ""}
_PUNCT_ROMAN = {"۔": ".", "،": ",", "؟": "?", "؛": ";", "٪": "%", "ء": ""}


def _punct(text: str, table: dict) -> str:
    text = text.translate(_DIGITS)
    return "".join(table.get(ch, ch) for ch in text)


class LineModel:
    """Our trained line model (``training/rekhta/train.py``); torch when
    installed, else the numpy runtime (``urdu_nn.npnn``)."""

    def __init__(self, path: Path, device: str = "cpu", beam: int = 4):
        from urdu_nn.npnn import Checkpoint
        self.net = Checkpoint(path, device)
        self.vocab, self.beam = self.net.vocab, beam
        self.max_len = self.net.cfg.get("max_len", 128) - 2
        ck = self.net.meta
        self.info = {"epoch": ck.get("epoch"), "dev": ck.get("dev_beam4") or ck.get("dev")}

    def nbest(self, runs: list[str]) -> list[list[tuple[str, float]]]:
        """Per run, the beam's readings best-first as (text, normalised logprob)."""
        from urdu_nn.vocab import UNK
        tag = self.vocab.stoi["<l>"]
        out = []
        for b in range(0, len(runs), 64):
            chunk = runs[b:b + 64]
            hyps = self.net.beam_search([[tag] + [self.vocab.stoi.get(c, UNK) for c in r] for r in chunk],
                                        beam=self.beam, max_len=self.max_len)
            out += [[(self.vocab.decode(ids), sc) for ids, sc in h] for h in hyps]
        return out

    def __call__(self, runs: list[str]) -> list[str]:
        return [h[0][0] for h in self.nbest(runs)]


def _clean(deva: str) -> str:
    return re.sub(r"\s+", " ", deva).strip()


def _lines_up(run: str, deva: str) -> bool:
    return align_units(run.split(), deva_units(deva)) is not None


class Ensemble:
    """Rekhta's reading, verbatim, wherever it is sound; our model where it breaks.

    Rekhta's forward model reads the run and its reverse model reads that
    Devanagari back into Urdu. If the reading passes ``reading_ok`` -- the
    same test that picked our training labels: its words line up with the
    Urdu, nothing looped, and it reads back as the Urdu written -- it is
    the answer, unchanged. Otherwise (a loop, a dropped, doubled or misread
    word) our model's beam joins in and every candidate is scored by the
    read-back log P(urdu | devanagari), plus a small weight on our model's
    own score."""

    def __init__(self, student: LineModel, fwd, back, prior: float = 0.3, margin: float = 0.3):
        self.student, self.fwd, self.back = student, fwd, back
        self.prior, self.margin = prior, margin
        self.stats = {"rekhta": 0, "refereed": 0}

    def choose(self, runs: list[str]) -> list[str]:
        out: list[str | None] = [None] * len(runs)
        tg = [_clean(t) for t in self.fwd(runs)]
        backs = self.back(tg)
        for i, (r, t, b) in enumerate(zip(runs, tg, backs)):
            if not reading_ok(r, t, b):
                out[i] = t
        ok = [i for i, (r, t) in enumerate(zip(runs, tg)) if out[i] is None and t and _lines_up(r, t)]
        lps = self.back.logprob([tg[i] for i in ok], [runs[i] for i in ok]) if ok else []
        t_lp = dict(zip(ok, lps))
        todo = [i for i in range(len(runs)) if out[i] is None]
        self.stats["rekhta"] += len(runs) - len(todo)
        self.stats["refereed"] += len(todo)
        if not todo:
            return out
        nb = self.student.nbest([runs[i] for i in todo])
        flat: list[tuple[int, str, float]] = []
        for i, hyps in zip(todo, nb):
            seen: dict[str, float] = {}
            for h, s in hyps:
                h = _clean(h)
                if h and _lines_up(runs[i], h):
                    seen.setdefault(h, s)
            if not seen:                            # nothing lines up: our best guess stands
                out[i] = _clean(hyps[0][0]) if hyps else tg[i]
                continue
            flat += [(i, c, s) for c, s in seen.items()]
        lps = self.back.logprob([c for _, c, _ in flat], [runs[i] for i, _, _ in flat]) if flat else []
        best: dict[int, tuple[float, str, float]] = {}
        for (i, c, s), lp in zip(flat, lps):
            sc = lp + self.prior * s
            if i not in best or sc > best[i][0]:
                best[i] = (sc, c, lp)
        for i in todo:
            if out[i] is None:
                _, cand, lp = best[i]
                # the read-back can't hear short vowels (सताए and सिताए both read
                # back as ستائے): Rekhta's reading only loses when it reads back
                # clearly worse -- a dropped, doubled or misread word
                if i in t_lp and t_lp[i] >= lp - self.margin:
                    cand = tg[i]
                out[i] = cand
        return out

    __call__ = choose


def load_corrections(path: Path = None) -> dict[str, tuple[str, str]]:
    """Reviewed fixes: urdu word -> (devanagari, R or ""). File lines are
    ``urdu <TAB> devanagari [<TAB> roman in Rekhta's ASCII table]``."""
    path = path or CORRECTIONS_PATH
    out: dict[str, tuple[str, str]] = {}
    if not Path(path).exists():
        return out
    from urdu_nn.rekhta_roman import ascii_to_rich
    from urdu_nn.rekhta_text import norm_word
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            rich = ascii_to_rich(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else ""
            out[norm_word(parts[0].strip())] = (parts[1].strip(), rich)
    return out


def save_correction(urdu: str, deva: str, roman_ascii: str = "", path: Path = None) -> None:
    """Add or replace one fix (the newest line for a word wins)."""
    from urdu_nn.rekhta_text import norm_word
    path = Path(path or CORRECTIONS_PATH)
    key = norm_word(urdu.strip())
    keep = []
    if path.exists():
        keep = [l for l in path.read_text(encoding="utf-8").splitlines()
                if not l.strip() or l.startswith("#") or norm_word(l.split("\t")[0].strip()) != key]
    else:
        keep = ["# urdu <TAB> devanagari [<TAB> roman in Rekhta's ASCII table] -- always wins"]
    keep.append("\t".join([key, deva.strip()] + ([roman_ascii.strip()] if roman_ascii.strip() else [])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    eng = _ENGINE
    if eng is not None:
        eng.corrections = load_corrections(path)
        eng._cache.clear()


class RekhtaTransliterator:
    """``mode``: "ensemble" (our model + Rekhta's, refereed by the reverse
    model; the default when both are present), "student" or "teacher"."""

    def __init__(self, model_path: Path = MODEL_PATH, device: str = "cpu", beam: int = 4,
                 word_dir: Path = WORD_DIR, use_teacher: bool | None = None,
                 mode: str | None = None, corrections_path: Path | None = None,
                 lexicon_path: Path | None = None):
        from urdu_nn.npnn import use_torch
        from urdu_nn.rekhta_models import load_teacher
        if use_torch():
            import torch
            torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        have_student = Path(model_path).exists()
        if use_teacher:
            mode = "teacher"
        teachers = None
        if mode in (None, "ensemble") and have_student:
            # Rekhta's models aren't ours to ship, so a fresh checkout or the
            # deployed site fetches them here; if that fails, ours reads alone
            try:
                teachers = load_teacher("ur2hi", device), load_teacher("hi2ur", device)
            except Exception as e:
                if mode == "ensemble":
                    raise
                log.warning("Rekhta's models unavailable (%s); line model only", e)
            mode = "ensemble" if teachers else "student"
        mode = mode or "teacher"
        if mode == "teacher":
            self.model = load_teacher("ur2hi", device)
        elif mode == "student":
            self.model = LineModel(Path(model_path), device, beam)
        else:
            self.model = Ensemble(LineModel(Path(model_path), device, beam), *teachers)
        self.mode = mode
        self.corrections = load_corrections(corrections_path)
        from urdu_nn.rekhta_lexicon import RekhtaLexicon
        if lexicon_path is None:
            lexicon_path = next((p for p in LEXICON_PATHS if p.exists()), None)
        self.lexicon = RekhtaLexicon.load(lexicon_path)
        self.lex_override = 0       # >0: Rekhta's only reading of a word, seen this often, wins
        self.reader = Reader(*self._tables(word_dir))
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}

    @staticmethod
    def _tables(word_dir: Path):
        from urdu_nn.scheme import norm_urdu
        import transliterate as rule
        lex, ev, wm = {}, {}, None
        p = word_dir / "lexicon.json.gz"
        if p.exists():
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                lex = {k: (v[0], v[1]) for k, v in json.load(fh).items()}
        p = word_dir / "evidence.json.gz"
        if p.exists():
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                ev = json.load(fh)
        dic = {norm_urdu(k): v for k, v in rule.load_exact_dictionary(rule.EXACT_DICTIONARY_PATH).items()}
        if (word_dir / "model.pt").exists():
            try:
                wm = WordModel(word_dir / "model.pt")
            except Exception:       # incompatible checkpoint: tables + schwa rule only
                wm = None
        return lex, dic, ev, wm

    def deva(self, runs: list[str]) -> dict[str, str]:
        todo = [r for r in dict.fromkeys(runs) if r not in self._cache]
        if todo:
            with self._lock:
                for r, d in zip(todo, self.model(todo)):
                    self._cache[r] = re.sub(r"\s+", " ", d).strip()
        return {r: self._cache[r] for r in runs}

    @staticmethod
    def _deva_of(units: list[list[str]], j0: int, j1: int) -> str:
        """Devanagari units [j0, j1) with the joiners between them."""
        return "".join(units[k][0] + (units[k][1].replace("-e-", "-ए-") if k < j1 - 1 else "")
                       for k in range(j0, j1))

    def _lex_hits(self, words: list[str], spans, units) -> dict[int, tuple[int, str, str]]:
        """Rekhta's spellings for this run: first span -> (last span, its
        devanagari, its roman), longest Urdu phrase first, never across a
        span (a word the model joined or split stays one piece)."""
        hits: dict[int, tuple[int, str, str]] = {}
        if not len(self.lexicon):
            return hits
        from urdu_nn.rekhta_lexicon import urdu_key
        end_at = {s[1]: k for k, s in enumerate(spans)}
        k = 0
        while k < len(spans):
            i0, hit = spans[k][0], None
            for i1 in range(min(len(words), i0 + self.lexicon.max_phrase), i0, -1):
                k1 = end_at.get(i1)
                if k1 is None or k1 < k:
                    continue
                ours = self._deva_of(units, spans[k][2], spans[k1][3])
                key, prior = urdu_key(words[i0:i1]), None
                if k1 == k and i1 - i0 == 1 and spans[k][3] - spans[k][2] == 1:
                    d0, r0 = ain_respell(words[i0], ours, self.reader.read(ours, words[i0]))
                    prior = (d0, render(r0, "rekhta"))
                got = self.lexicon.lookup(key, ours, prior)
                if not got and self.lex_override:
                    got = self.lexicon.confident(key, self.lex_override)
                if got:
                    hit = (k1, *got)
                    break
            if hit:
                hits[k] = hit
                k = hit[0] + 1
            else:
                k += 1
        return hits

    def _read_run(self, run: str, deva: str) -> tuple[str, list]:
        """(Devanagari, pieces) for one run; pieces alternate words --
        (R, Rekhta's own Roman or "") -- and the joiners between them.

        The model's words are lined up with the Urdu words (one to one, or
        Rekhta's joins पाऊँगा and splits सर-बसर); if they can't be, every
        word is read on its own. Where Rekhta's lexicon has the same reading
        its spelling is used, both scripts; a reviewed correction replaces
        whatever the model said for that word."""
        from urdu_nn.rekhta_lexicon import rekhta_to_rich
        words = run.split()
        units = deva_units(deva)
        spans = align_units(words, units)
        if spans is None:
            singles = self.deva(words)
            units = [[(deva_units(singles[w]) or [[singles[w], " "]])[0][0], " "] for w in words]
            spans = [(i, i + 1, i, i + 1) for i in range(len(words))]
        hits = self._lex_hits(words, spans, units)
        out_deva, pieces = [], []
        k = 0
        while k < len(spans):
            i0, i1, j0, j1 = spans[k]
            if k in hits:
                k1, d, roman = hits[k]
                i1, j1 = spans[k1][1], spans[k1][3]
                r, verbatim = rekhta_to_rich(roman, d), roman
                k = k1
            else:
                urdu = "".join(words[i0:i1])
                fix = (self.corrections.get(words[i0]) or self.corrections.get(words[i0].rstrip("ِ"))
                       if i1 - i0 == 1 else None)
                verbatim = ""
                if fix:                               # replaces a hyphenated pair too
                    d = fix[0]
                    r = fix[1] or self.reader.read(fix[0], urdu)
                else:                                 # सर-बसर: each half read on its own
                    d = self._deva_of(units, j0, j1)
                    r = "-".join(self.reader.read(units[m][0], urdu) if j1 - j0 == 1
                                 else self.reader.read(units[m][0], "") for m in range(j0, j1))
                    if i1 - i0 == 1 and j1 - j0 == 1:
                        d, r = ain_respell(urdu, d, r)     # मा'लूम, ma.alūm
            joiner = units[j1 - 1][1] if k < len(spans) - 1 else ""
            out_deva.append(d + joiner.replace("-e-", "-ए-"))
            pieces.append((r, verbatim))
            if joiner:
                pieces.append(joiner)
            k += 1
        return "".join(out_deva), pieces

    @staticmethod
    def _render_pieces(pieces: list, style: str) -> str:
        from urdu_nn.rekhta_roman import undouble
        out = []
        for n, p in enumerate(pieces):
            if isinstance(p, str):
                out.append(p)
                continue
            r, verbatim = p
            near = [pieces[m] for m in (n - 1, n + 1) if 0 <= m < len(pieces)]
            compound = any(isinstance(j, str) and "-" in j for j in near)
            if not (verbatim and style == "rekhta"):
                out.append(render(r, style, compound))
                continue
            if compound:
                verbatim = undouble(verbatim)       # haal alone, hāl-e-dil in a compound
            out.append(verbatim)
        return "".join(out)

    def transliterate_full(self, text: str) -> dict[str, str]:
        """-> {devanagari, roman (Rekhta ASCII table), roman_diacritic, plain}."""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segs = [segments(line) for line in lines]
        runs = [t for s in segs for k, t in s if k == "u"]
        devas = self.deva(runs)
        self.reader.prefetch([w for r in runs for w in r.split()])
        out = {"devanagari": [], "roman": [], "roman_diacritic": [], "plain": []}
        for s in segs:
            d_line, r_line = [], []
            prev = ""
            for kind, t in s:
                if kind == "x":
                    d_line.append(_punct(t, _PUNCT_DEVA))
                    r_line.append(("x", _punct(t, _PUNCT_ROMAN)))
                    prev = t
                    continue
                d = devas[t]
                if re.search(r"[\d۰-۹٠-٩]\s*ء?\s*$", prev) and t.startswith("میں") \
                        and d.startswith("मैं"):
                    d = "में" + d[3:]          # "1947ء میں": the model never saw the year
                d, pieces = self._read_run(t, d)
                d_line.append(d)
                r_line.append(("r", pieces))
            out["devanagari"].append("".join(d_line).rstrip())
            for key, style in (("roman", "ascii"), ("roman_diacritic", "rekhta"), ("plain", "plain")):
                out[key].append("".join(self._render_pieces(t, style) if k == "r" else t
                                        for k, t in r_line).rstrip())
        return {k: "\n".join(v) for k, v in out.items()}

    def transliterate(self, text: str) -> tuple[str, str, str]:
        r = self.transliterate_full(text)
        return r["devanagari"], r["roman"], r["roman_diacritic"]


_ENGINE: RekhtaTransliterator | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> RekhtaTransliterator:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = RekhtaTransliterator()
        return _ENGINE


def transliterate(text: str) -> tuple[str, str, str]:
    """-> (devanagari, roman in Rekhta's ASCII table, Rekhta diacritic roman)."""
    return get_engine().transliterate(text)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    r = get_engine().transliterate_full(" ".join(sys.argv[1:]) or sys.stdin.read())
    for k, v in r.items():
        print(f"{k:16s} {v}")
