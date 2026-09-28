# -*- coding: utf-8 -*-
"""Rekhta Labs' two published transliterators: where their files live, and a
torch-free runner for the deployed site.

``urdu_nn.rekhta_teacher.Teacher`` runs them on torch (and is what training
uses); ``NumpyTeacher`` is the same weights, the same greedy decode and the
same read-back scores on ``urdu_nn.npnn``. ``load_teacher`` picks whichever
can run here.

The files are Rekhta's (their Hugging Face repos, licence "other"), so we
don't re-host them: a fresh checkout or the deployed site fetches them on
first use -- into ``data/rekhta_models/``, or the temp dir where the code
folder is read-only (Vercel).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data" / "rekhta_models"
CACHE = Path(tempfile.gettempdir()) / "rekhta_models"
REPOS = {"ur2hi": "rekhtalabs/ur-2-hi-translit", "hi2ur": "rekhtalabs/hi-2-ur-translit"}
FILES = {   # direction -> (source spm, target spm, checkpoint)
    "ur2hi": ("nastaaliq_char.model", "devanagari_char.model", "transformer_transliteration_final.pt"),
    "hi2ur": ("devanagari_bpe.model", "nastaaliq_bpe.model", "h2u_2.0.pt"),
}
PAD, BOS, EOS = 0, 2, 3
MAX_LEN = 128            # both models were trained on <= 128 pieces


def model_dir(direction: str, fetch: bool = False) -> Path | None:
    """The folder holding this direction's files, or None. With ``fetch``,
    download whatever is missing from Hugging Face first."""
    name = direction.replace("2", "-2-")
    homes = (MODELS / name, CACHE / name)
    for d in homes:
        if all((d / f).exists() for f in FILES[direction]):
            return d
    if not fetch:
        return None
    for d in homes:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue                # read-only code folder: use the temp dir
        if not os.access(d, os.W_OK):
            continue
        for f in FILES[direction]:
            if not (d / f).exists():
                _download(f"https://huggingface.co/{REPOS[direction]}/resolve/main/{f}", d / f)
        return d
    return None


def _download(url: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    os.replace(part, dest)          # never leave a half-written model behind


class NumpyTeacher:
    """``urdu_nn.rekhta_teacher.Teacher`` without torch (same interface)."""

    def __init__(self, direction: str = "ur2hi"):
        import sentencepiece as spm
        from urdu_nn.npnn import RekhtaNet, load_pt
        d = model_dir(direction, fetch=True)
        if d is None:
            raise FileNotFoundError(f"Rekhta {direction} model: no writable place to fetch it to")
        src_m, tgt_m, ckpt = FILES[direction]
        self.direction = direction
        self.src_sp = spm.SentencePieceProcessor(model_file=str(d / src_m))
        self.tgt_sp = spm.SentencePieceProcessor(model_file=str(d / tgt_m))
        self.model = RekhtaNet(load_pt(d / ckpt)["model_state_dict"])

    def _src(self, texts: list[str]):
        from urdu_nn.npnn import pad_batch
        return pad_batch([[BOS] + self.src_sp.encode(unicodedata.normalize("NFC", t))[:MAX_LEN - 2]
                          + [EOS] for t in texts])

    def __call__(self, texts: list[str], batch_size: int = 64, with_score: bool = False) -> list:
        """Greedy transliteration of each text (the model cards' decoding);
        ``with_score`` also returns the mean token log-probability."""
        import numpy as np
        from urdu_nn.npnn import log_softmax
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        res: list = [None] * len(texts)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            st = self.model.start(self._src([texts[i] for i in idx]))
            B = len(idx)
            last = np.full(B, BOS, dtype=np.int64)
            done = np.zeros(B, bool)
            lp, n = np.zeros(B, np.float32), np.zeros(B, np.float32)
            toks = []
            for _ in range(MAX_LEN):
                logp = log_softmax(self.model.step(last, st))
                nxt = logp.argmax(-1)
                lp += np.where(done, np.float32(0), logp[np.arange(B), nxt])
                n += (~done).astype(np.float32)
                nxt = np.where(done, EOS, nxt)
                toks.append(nxt)
                done |= nxt == EOS
                if done.all():
                    break
                last = nxt
            rows = np.stack(toks, 1).tolist()
            for k, i in enumerate(idx):
                ids = rows[k]
                ids = ids[:ids.index(EOS)] if EOS in ids else ids
                text = unicodedata.normalize("NFC", self.tgt_sp.decode(ids))
                res[i] = (text, float(lp[k] / max(n[k], 1))) if with_score else text
        return res

    def logprob(self, sources: list[str], targets: list[str], batch_size: int = 64) -> list[float]:
        """Mean per-token log P(target | source) (round-trip checks, reranking)."""
        import numpy as np
        from urdu_nn.npnn import log_softmax, pad_batch
        out: list[float] = [0.0] * len(sources)
        for b in range(0, len(sources), batch_size):
            idx = list(range(b, min(b + batch_size, len(sources))))
            t = pad_batch([[BOS] + self.tgt_sp.encode(unicodedata.normalize("NFC", targets[i]))[:MAX_LEN - 2]
                           + [EOS] for i in idx])
            # target padding only ever trails, and those rows are zeroed below,
            # so the causal mask alone reads every real token as torch does
            logp = log_softmax(self.model.decode(t[:, :-1], self.model.start(self._src([sources[i] for i in idx]))))
            gold = t[:, 1:]
            tok = np.where(gold == PAD, 0, np.take_along_axis(logp, gold[..., None], -1)[..., 0])
            lens = np.maximum((gold != PAD).sum(1), 1)
            for k, i in enumerate(idx):
                out[i] = float(tok[k].sum() / lens[k])
        return out


def load_teacher(direction: str, device: str | None = None):
    """Rekhta's model for ``direction``: on torch when installed, else numpy."""
    from urdu_nn.npnn import use_torch
    if use_torch():
        from urdu_nn.rekhta_teacher import Teacher
        return Teacher(direction, device)
    return NumpyTeacher(direction)
