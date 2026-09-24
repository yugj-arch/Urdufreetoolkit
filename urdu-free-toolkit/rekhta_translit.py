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

    def __call__(self, runs: list[str]) -> list[str]:
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
            out += [self.vocab.decode(h[0][0]) for h in hyps]
        return out


class RekhtaTransliterator:
    def __init__(self, model_path: Path = MODEL_PATH, device: str = "cpu", beam: int = 4,
                 word_dir: Path = WORD_DIR, use_teacher: bool | None = None):
        import torch
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        if use_teacher or (use_teacher is None and not Path(model_path).exists()):
            from urdu_nn.rekhta_teacher import Teacher
            self.model, self.source = Teacher("ur2hi", device), "rekhtalabs/ur-2-hi-translit"
        else:
            self.model, self.source = LineModel(Path(model_path), device, beam), "rekhta-line"
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
        for k, ((dw, join), uw) in enumerate(zip(units, words)):
            rich.append(self.reader.read(dw, uw))
            if k < len(units) - 1:
                rich.append(join)
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
