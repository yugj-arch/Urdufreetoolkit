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
import re
import threading
from pathlib import Path

from urdu_nn.rekhta_roman import Reader, WordModel, deva_units, render
from urdu_nn.rekhta_text import segments

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data" / "rekhta_model" / "model.pt"
CORRECTIONS_PATH = ROOT / "data" / "rekhta_corrections.tsv"
WORD_DIR = ROOT / "data" / "translit_model"
_DIGITS = {ord(a): str(i) for i, a in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGITS.update({ord(a): str(i) for i, a in enumerate("٠١٢٣٤٥٦٧٨٩")})
_PUNCT_DEVA = {"۔": "।", "،": ",", "؟": "?", "؛": ";", "٪": "%", "ء": ""}
_PUNCT_ROMAN = {"۔": ".", "،": ",", "؟": "?", "؛": ";", "٪": "%", "ء": ""}


def _punct(text: str, table: dict) -> str:
    text = text.translate(_DIGITS)
    return "".join(table.get(ch, ch) for ch in text)


class LineModel:
    """Our trained line model (``training/rekhta/train.py``)."""

    def __init__(self, path: Path, device: str = "cpu", beam: int = 4):
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
        self.max_len = ck["cfg"].get("max_len", 128) - 2
        self.info = {"epoch": ck.get("epoch"), "dev": ck.get("dev_beam4") or ck.get("dev")}

    def nbest(self, runs: list[str]) -> list[list[tuple[str, float]]]:
        """Per run, the beam's readings best-first as (text, normalised logprob)."""
        import torch
        from urdu_nn.model import UNK, beam_search, pad_batch
        tag = self.vocab.stoi["<l>"]
        out = []
        for b in range(0, len(runs), 64):
            chunk = runs[b:b + 64]
            src = pad_batch([[tag] + [self.vocab.stoi.get(c, UNK) for c in r] for r in chunk],
                            self.device)
            with torch.inference_mode():
                hyps = beam_search(self.model, src, beam=self.beam, max_len=self.max_len)
            out += [[(self.vocab.decode(ids), sc) for ids, sc in h] for h in hyps]
        return out

    def __call__(self, runs: list[str]) -> list[str]:
        return [h[0][0] for h in self.nbest(runs)]


def _clean(deva: str) -> str:
    return re.sub(r"\s+", " ", deva).strip()


class Ensemble:
    """Our line model and Rekhta's own, refereed by Rekhta's reverse model.

    Candidates for a run are our model's beam plus Rekhta's greedy reading.
    Each is scored by how well it reads back into the Urdu that was written,
    log P(urdu | devanagari) under ``hi2ur`` (a candidate that drops, adds or
    misreads a word reads back to different Urdu), plus a small weight on our
    model's own score. A candidate whose words don't line up one-to-one with
    the Urdu is out."""

    def __init__(self, student: LineModel, fwd, back, prior: float = 0.3, teacher_prior: float = -0.05):
        self.student, self.fwd, self.back = student, fwd, back
        self.prior, self.teacher_prior = prior, teacher_prior

    def choose(self, runs: list[str]) -> list[str]:
        nb = self.student.nbest(runs)
        tg = self.fwd(runs)
        cands: list[list[tuple[str, float]]] = []
        for run, hyps, t in zip(runs, nb, tg):
            n = len(run.split())
            seen: dict[str, float] = {}
            for h, s in hyps:
                h = _clean(h)
                if h and len(deva_units(h)) == n:
                    seen.setdefault(h, s)
            t = _clean(t)
            if t and len(deva_units(t)) == n:
                seen.setdefault(t, (max(seen.values()) if seen else 0.0) + self.teacher_prior)
            if not seen:                            # nobody lines up: our best guess stands
                seen[_clean(hyps[0][0]) if hyps else t] = 0.0
            cands.append(list(seen.items()))
        flat = [(i, c, s) for i, cs in enumerate(cands) for c, s in cs]
        lps = self.back.logprob([c for _, c, _ in flat], [runs[i] for i, _, _ in flat])
        best: dict[int, tuple[float, str]] = {}
        for (i, c, s), lp in zip(flat, lps):
            sc = lp + self.prior * s
            if i not in best or sc > best[i][0]:
                best[i] = (sc, c)
        return [best[i][1] for i in range(len(runs))]

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
                 mode: str | None = None, corrections_path: Path | None = None):
        import torch
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        from urdu_nn import rekhta_teacher as rt
        have_student = Path(model_path).exists()
        have_teacher = (rt.MODELS / "ur-2-hi" / rt.FILES["ur2hi"][2]).exists()
        if use_teacher:
            mode = "teacher"
        mode = mode or ("ensemble" if have_student and have_teacher
                        else "student" if have_student else "teacher")
        if mode == "teacher":
            self.model = rt.Teacher("ur2hi", device)
        elif mode == "student":
            self.model = LineModel(Path(model_path), device, beam)
        else:
            self.model = Ensemble(LineModel(Path(model_path), device, beam),
                                  rt.Teacher("ur2hi", device), rt.Teacher("hi2ur", device))
        self.mode = mode
        self.corrections = load_corrections(corrections_path)
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

    def _read_run(self, run: str, deva: str) -> tuple[str, str]:
        """(Devanagari, R) for one run. When the model's words don't line up
        one-to-one with the Urdu words, every word is read on its own."""
        words = run.split()
        units = deva_units(deva)
        if len(units) != len(words):
            singles = self.deva(words)
            units = []
            for w in words:
                u = deva_units(singles[w]) or [[singles[w], " "]]
                units.append([u[0][0], " "])
            deva = " ".join(u[0] for u in units)
        rich = []
        fixed = False
        for k, ((dw, join), uw) in enumerate(zip(units, words)):
            fix = self.corrections.get(uw) or self.corrections.get(uw.rstrip("ِ"))
            if fix:
                units[k][0] = dw = fix[0]
                fixed = True
            rich.append(fix[1] if fix and fix[1] else self.reader.read(dw, uw))
            if k < len(units) - 1:
                rich.append(join)
        if fixed:
            deva = "".join(u[0] + (u[1].replace("-e-", "-ए-") if k < len(units) - 1 else "")
                           for k, u in enumerate(units))
        return deva, "".join(rich)

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
                d, rich = self._read_run(t, d)
                d_line.append(d)
                r_line.append(("r", rich))
            out["devanagari"].append("".join(d_line).rstrip())
            for key, style in (("roman", "ascii"), ("roman_diacritic", "rekhta"), ("plain", "plain")):
                out[key].append("".join(render(t, style) if k == "r" else t
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
