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
import unicodedata
from pathlib import Path

from urdu_nn.rekhta_roman import (Reader, WordModel, ain_respell, align_units, deva_units,
                                  reading_ok, render)
from urdu_nn.rekhta_text import segments, takhallus_marks

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

    def logprob(self, runs: list[str], devas: list[str]) -> list[float]:
        """Mean per-token log P(devanagari | run) under this model."""
        from urdu_nn.vocab import UNK
        tag = self.vocab.stoi["<l>"]
        srcs = [[tag] + [self.vocab.stoi.get(c, UNK) for c in r] for r in runs]
        tgts = [self.vocab.encode_tgt(d)[:self.max_len + 2] for d in devas]
        return self.net.logprob(srcs, tgts)


_HALF_NASAL = re.compile("[ङञणन]्(?=[कखगघचछजझटठडढतथदध])")


def nasal_convention(deva: str) -> str:
    """rekhta.org writes a nasal before a consonant of its own class with the
    anusvara (ज़िंदगी, रंग, मंज़िल -- 1450 times to 60 on its pages); before
    ब/प it is split (जुम्बिश, संभाल), so that stays as read."""
    return unicodedata.normalize("NFC", _HALF_NASAL.sub("ं", unicodedata.normalize("NFD", deva)))


def _quote(word: str, wrap: str) -> str:
    """A pen name in Rekhta's quotes: 'ग़ालिब', ग़ुबार-ए-'मीर', 'मीर'-जी."""
    if not wrap or not word:
        return word
    if wrap == "last" and "-" in word:
        head, tail = word.rsplit("-", 1)
        return f"{head}-'{tail}'"
    if wrap == "first" and "-" in word:
        head, tail = word.split("-", 1)
        return f"'{head}'-{tail}"
    return f"'{word}'"


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
    from urdu_nn.rekhta_text import norm_word
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            rich = roman_to_rich(parts[2].strip(), parts[1].strip()) if len(parts) > 2 and parts[2].strip() else ""
            out[norm_word(parts[0].strip())] = (parts[1].strip(), rich)
    return out


def roman_to_rich(roman: str, deva: str = "") -> str:
    """A Roman spelling typed in either of Rekhta's schemes -> R: marked
    (ḳhayāl, mā'lūm) or the ASCII table (KHayaal, pa.Dhaa.ii)."""
    from urdu_nn.rekhta_lexicon import rekhta_to_rich
    from urdu_nn.rekhta_roman import ascii_to_rich
    if re.search("[āīūñḳġḍṭṛ]", roman):
        return rekhta_to_rich(roman, deva)
    return ascii_to_rich(roman)


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
        self.use_meter = True
        # measured on rekhta.org's test ghazals: Rekhta's model spells a poem's
        # Devanagari better than the lexicon's majority does (word 94.2 vs 93.7),
        # so the lexicon spells the Roman only, and only where the readings
        # match exactly (an izafat-blind match would add an -e- to the Roman alone)
        self.lex_loose, self.lex_deva = False, False
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

    def _lex_hits(self, words: list[str], spans, units) -> dict[int, tuple[int, str, str, bool]]:
        """Rekhta's spellings for this run: first span -> (last span, its
        devanagari, its roman, whether the roman came from inside a compound),
        longest Urdu phrase first, never across a span (a word the model
        joined or split stays one piece)."""
        hits: dict[int, tuple[int, str, str, bool]] = {}
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
                got = self.lexicon.lookup(key, ours, prior, self.lex_loose)
                if got:
                    hit = (k1, *got, False)
                    break
                if prior and (got := self.lexicon.lookup_part(key, ours)):
                    hit = (k1, *got, True)       # اہل as in اہل دل: ahl, not ahal
                    break
            if hit:
                hits[k] = hit
                k = hit[0] + 1
            else:
                k += 1
        return hits

    def _read_run(self, run: str, deva: str, marks: list[bool] | None = None) -> tuple[str, list]:
        """(Devanagari, pieces) for one run; pieces alternate words --
        (R, Rekhta's own Roman or "", takhallus quoting) -- and the joiners
        between them.

        The model's words are lined up with the Urdu words (one to one, or
        Rekhta's joins पाऊँगा and splits सर-बसर); if they can't be, every
        word is read on its own. Where Rekhta's lexicon has the same reading
        its spelling is used, both scripts; a reviewed correction replaces
        whatever the model said for that word. A pen name (``marks``: the
        Urdu word carries ؔ) is quoted as Rekhta prints it: 'ग़ालिब'."""
        from urdu_nn.rekhta_lexicon import rekhta_to_rich
        words = run.split()
        if not marks or len(marks) != len(words):
            marks = [False] * len(words)
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
                k1, d, roman, part = hits[k]
                i1, j1 = spans[k1][1], spans[k1][3]
                # a spelling from inside a compound is re-rendered for where it stands
                r, verbatim = rekhta_to_rich(roman, d), "" if part else roman
                if not self.lex_deva:
                    d = self._deva_of(units, j0, j1)
                    if i1 - i0 == 1 and j1 - j0 == 1:
                        d = ain_respell("".join(words[i0:i1]), d, "")[0]
                    d = nasal_convention(d)
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
                    d = nasal_convention(d)                 # ज़िन्दगी -> ज़िंदगी
            m = marks[i0:i1]
            wrap = "" if not any(m) else "all" if len(m) == 1 else "last" if m[-1] else "first"
            joiner = units[j1 - 1][1] if k < len(spans) - 1 else ""
            out_deva.append(_quote(d, wrap) + joiner.replace("-e-", "-ए-"))
            pieces.append((r, verbatim, wrap))
            if joiner:
                pieces.append(joiner)
            k += 1
        return "".join(out_deva), pieces

    def _render_pieces(self, pieces: list, style: str) -> str:
        """``style``: "rekhta" (rekhta.org's marked Roman), "simple" (its
        simple Roman), "ascii" (its ASCII table) or "plain" (casual)."""
        from urdu_nn.rekhta_roman import undouble
        out = []
        for n, p in enumerate(pieces):
            if isinstance(p, str):
                out.append(p)
                continue
            r, verbatim, wrap = p
            near = [pieces[m] for m in (n - 1, n + 1) if 0 <= m < len(pieces)]
            compound = any(isinstance(j, str) and "-" in j for j in near)
            if style in ("rekhta", "simple"):
                if verbatim:
                    s = undouble(verbatim) if compound else verbatim   # haal alone, hāl-e-dil
                else:
                    s = render(r, "rekhta", compound)
                if style == "simple":
                    s = self.lexicon.to_simple(s)
            else:
                s = render(r, style, compound)
            out.append(_quote(s, wrap))
        return "".join(out)

    # -- the meter: a misread line doesn't scan with the rest of its ghazal ------

    METER_TAU, METER_DELTA, METER_GAP = 0.3, 0.1, 4     # fitted on rekhta.org's dev ghazals

    def rich_line(self, run: str, deva: str) -> str | None:
        """R for a reading of a run (the reader's schwas, no respelling) --
        what ``urdu_nn.meter`` scans."""
        words, units = run.split(), deva_units(deva)
        spans = align_units(words, units)
        if spans is None:
            return None
        parts = []
        for n, (i0, i1, j0, j1) in enumerate(spans):
            urdu = "".join(words[i0:i1])
            parts.append("-".join(self.reader.read(units[m][0], urdu if j1 - j0 == 1 else "")
                                  for m in range(j0, j1)))
            if n < len(spans) - 1:
                parts.append(units[j1 - 1][1])
        return "".join(parts)

    @staticmethod
    def _izafat_toggles(run: str, deva: str) -> list[str]:
        """The reading with one izafat added or taken away."""
        words, units = run.split(), deva_units(deva)
        spans = align_units(words, units)
        if spans is None:
            return []
        out = []
        for n in range(len(spans) - 1):
            j = spans[n][3] - 1
            if units[j][1] not in (" ", "-e-"):
                continue
            alt = [list(u) for u in units]
            alt[j][1] = " " if units[j][1] == "-e-" else "-e-"
            out.append("".join(u[0] + (u[1].replace("-e-", "-ए-") if k < len(alt) - 1 else "")
                               for k, u in enumerate(alt)))
        return out

    def _scorer(self):
        m = self.model
        return m.fwd.logprob if isinstance(m, Ensemble) else getattr(m, "logprob", None)

    def by_meter(self, runs: list[str], devas: list[str]) -> list[str]:
        """Re-read the lines of a poem that don't scan with the rest of it.

        A line keeps its reading unless that reading can be read in the
        meter of few of the other lines while one close to it -- an izafat
        added or dropped, or one word read differently by our student --
        that Rekhta's model finds nearly as likely, fits clearly more of
        them. Prose, or under four lines, is left alone."""
        from urdu_nn.meter import Ghazal, syllables
        if len(runs) < 4:
            return devas
        syl = [syllables(x) if (x := self.rich_line(r, d)) else [] for r, d in zip(runs, devas)]
        g = Ghazal(syl)
        others = g.n - 1
        if others < 3:
            return devas
        sup = [g.support(s, i) if s else 0 for i, s in enumerate(syl)]
        todo = [i for i in range(len(runs)) if syl[i] and sup[i] < self.METER_TAU * others]
        if not todo:
            return devas
        score = self._scorer()
        if score is None:
            return devas
        cands = {i: self._izafat_toggles(runs[i], devas[i]) for i in todo}
        if isinstance(self.model, Ensemble):
            for i, hyps in zip(todo, self.model.student.nbest([runs[i] for i in todo])):
                cands[i] += [_clean(h) for h, _ in hyps]
        from urdu_nn.rekhta_lexicon import deva_fold

        def close(t, c):
            if deva_fold(t, izafat=False) == deva_fold(c, izafat=False):
                return True
            a, b = t.replace("-", " ").split(), c.replace("-", " ").split()
            return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1

        flat = [(i, c) for i in todo for c in dict.fromkeys(cands[i])
                if c and c != devas[i] and close(devas[i], c) and _lines_up(runs[i], c)]
        if not flat:
            return devas
        lp = score([runs[i] for i, _ in flat] + [runs[i] for i in todo],
                   [c for _, c in flat] + [devas[i] for i in todo])
        own = dict(zip(todo, lp[len(flat):]))
        best: dict[int, tuple[int, float, str]] = {}
        for (i, c), p in zip(flat, lp):
            if p < own[i] - self.METER_DELTA:
                continue
            s = syllables(self.rich_line(runs[i], c) or "")
            k = g.support(s, i) if s else 0
            if k >= max(self.METER_TAU * others, sup[i] + self.METER_GAP) and \
                    (i not in best or (k, p) > best[i][:2]):
                best[i] = (k, p, c)
        out = list(devas)
        for i, (_, _, c) in best.items():
            out[i] = c
        return out

    def transliterate_full(self, text: str) -> dict[str, str]:
        """-> {devanagari, roman (rekhta.org's simple Roman), roman_diacritic
        (its marked Roman), ascii (its ASCII table), plain (casual)}."""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segs = [segments(line) for line in lines]
        runs = [t for s in segs for k, t in s if k == "u"]
        devas = self.deva(runs)
        self.reader.prefetch([w for r in runs for w in r.split()])
        # a poem's misras (one run each) are read together, against its meter
        verse = [n for n, s in enumerate(segs) if sum(k == "u" for k, _ in s) == 1]
        line_deva: dict[int, str] = {}
        if self.use_meter and len(verse) >= 4:
            vr = [next(t for k, t in segs[n] if k == "u") for n in verse]
            line_deva = dict(zip(verse, self.by_meter(vr, [devas[t] for t in vr])))
        styles = (("roman", "simple"), ("roman_diacritic", "rekhta"), ("ascii", "ascii"), ("plain", "plain"))
        out = {"devanagari": [], **{k: [] for k, _ in styles}}
        for ln, (line, s) in enumerate(zip(lines, segs)):
            marks = takhallus_marks(line)
            if len(marks) != sum(len(t.split()) for k, t in s if k == "u"):
                marks = []
            d_line, r_line = [], []
            prev = ""
            for kind, t in s:
                if kind == "x":
                    d_line.append(_punct(t, _PUNCT_DEVA))
                    r_line.append(("x", _punct(t, _PUNCT_ROMAN)))
                    prev = t
                    continue
                d = line_deva.get(ln, devas[t])
                if re.search(r"[\d۰-۹٠-٩]\s*ء?\s*$", prev) and t.startswith("میں") \
                        and d.startswith("मैं"):
                    d = "में" + d[3:]          # "1947ء میں": the model never saw the year
                n = len(t.split())
                d, pieces = self._read_run(t, d, marks[:n])
                marks = marks[n:]
                d_line.append(d)
                r_line.append(("r", pieces))
            out["devanagari"].append("".join(d_line).rstrip())
            for key, style in styles:
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
