# -*- coding: utf-8 -*-
"""Rekhta Labs' published Urdu<->Devanagari poetry transliterators, run fast.

Two small (~4M param) character Transformers from Hugging Face
(``rekhtalabs/ur-2-hi-translit`` trained on ~800k Urdu->Hindi poetry pairs,
``rekhtalabs/hi-2-ur-translit`` on ~1.3M the other way). Their model cards
decode one line at a time and re-run the whole decoder prefix every step;
here the same weights decode a whole batch at once with padding masks (so a
batched line reads exactly as it would alone) and a per-layer cache of the
decoder's self-attention inputs.

They are the *teacher* for ``training/rekhta``: their Devanagari labels our
Urdu corpus, the reverse model round-trips each label back to Urdu as a
quality check, and our own student model learns from what survives.

    python -m urdu_nn.rekhta_teacher "وسوسے دل میں نہ رکھ خوف رسن لے کے نہ چل"
"""
from __future__ import annotations

import math
import unicodedata
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data" / "rekhta_models"
REPOS = {"ur2hi": "rekhtalabs/ur-2-hi-translit", "hi2ur": "rekhtalabs/hi-2-ur-translit"}
FILES = {   # direction -> (source spm, target spm, checkpoint)
    "ur2hi": ("nastaaliq_char.model", "devanagari_char.model", "transformer_transliteration_final.pt"),
    "hi2ur": ("devanagari_bpe.model", "nastaaliq_bpe.model", "h2u_2.0.pt"),
}
PAD, BOS, EOS = 0, 2, 3
MAX_LEN = 128            # both models were trained on <= 128 pieces


class _Sinusoid(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x, start: int = 0):
        return x + self.pe[:, start:start + x.size(1)]


class RekhtaTransformer(nn.Module):
    """Same parameter names as the model cards' ``Transformer`` so their
    checkpoints load unchanged (post-norm ``nn.Transformer``, sinusoidal
    positions, no embedding scaling)."""

    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 512):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoder = _Sinusoid(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=dim_feedforward, batch_first=True)
        self.out = nn.Linear(d_model, tgt_vocab)

    def encode(self, src):
        mask = src == PAD
        mem = self.transformer.encoder(self.pos_encoder(self.src_tok_emb(src)),
                                       src_key_padding_mask=mask)
        return mem, mask

    def step(self, last, pos: int, mem, mem_mask, cache: list):
        """Logits for the newest target token. ``cache[l]`` holds layer l's
        inputs at every earlier position -- in a post-norm layer those are
        exactly the self-attention keys/values, and causal masking means they
        never change once computed."""
        x = self.pos_encoder(self.tgt_tok_emb(last), start=pos)
        for li, layer in enumerate(self.transformer.decoder.layers):
            cache[li] = x if cache[li] is None else torch.cat([cache[li], x], 1)
            x = layer.norm1(x + layer.self_attn(x, cache[li], cache[li], need_weights=False)[0])
            x = layer.norm2(x + layer.multihead_attn(x, mem, mem, key_padding_mask=mem_mask,
                                                     need_weights=False)[0])
            x = layer.norm3(x + layer.linear2(layer.activation(layer.linear1(x))))
        return self.out(self.transformer.decoder.norm(x))[:, -1]

    def score(self, src, tgt_in):
        """Teacher-forced logits for whole target prefixes (padding-masked)."""
        mem, mask = self.encode(src)
        L = tgt_in.size(1)
        causal = torch.triu(torch.full((L, L), float("-inf"), device=src.device), 1)
        h = self.transformer.decoder(self.pos_encoder(self.tgt_tok_emb(tgt_in)), mem,
                                     tgt_mask=causal, tgt_key_padding_mask=tgt_in == PAD,
                                     memory_key_padding_mask=mask)
        return self.out(h)


def download(direction: str) -> Path:
    from huggingface_hub import snapshot_download
    d = MODELS / direction.replace("2", "-2-")
    snapshot_download(repo_id=REPOS[direction], local_dir=str(d))
    return d


class Teacher:
    """One direction (``ur2hi`` or ``hi2ur``) of the Rekhta models."""

    def __init__(self, direction: str = "ur2hi", device: str | None = None):
        import sentencepiece as spm
        self.direction = direction
        d = MODELS / direction.replace("2", "-2-")
        src_m, tgt_m, ckpt = FILES[direction]
        if not (d / ckpt).exists():
            download(direction)
        self.src_sp = spm.SentencePieceProcessor(model_file=str(d / src_m))
        self.tgt_sp = spm.SentencePieceProcessor(model_file=str(d / tgt_m))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = RekhtaTransformer(self.src_sp.get_piece_size(), self.tgt_sp.get_piece_size())
        state = torch.load(d / ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def _src(self, texts: list[str]) -> torch.Tensor:
        seqs = [[BOS] + self.src_sp.encode(unicodedata.normalize("NFC", t))[:MAX_LEN - 2] + [EOS]
                for t in texts]
        L = max(len(s) for s in seqs)
        out = torch.full((len(seqs), L), PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s)
        return out.to(self.device)

    @torch.no_grad()
    def __call__(self, texts: list[str], batch_size: int = 64,
                 with_score: bool = False) -> list:
        """Greedy transliteration (exactly the model cards' decoding) of each
        text. ``with_score`` also returns the mean token log-probability."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        res: list = [None] * len(texts)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            src = self._src([texts[i] for i in idx])
            mem, mask = self.model.encode(src)
            B = src.size(0)
            cache: list = [None] * len(self.model.transformer.decoder.layers)
            last = torch.full((B, 1), BOS, dtype=torch.long, device=self.device)
            done = torch.zeros(B, dtype=torch.bool, device=self.device)
            toks, lp = [], torch.zeros(B, device=self.device)
            n = torch.zeros(B, device=self.device)
            for t in range(MAX_LEN):
                logp = torch.log_softmax(self.model.step(last, t, mem, mask, cache).float(), -1)
                nxt = logp.argmax(-1)
                lp += torch.where(done, 0.0, logp.gather(1, nxt[:, None])[:, 0])
                n += (~done).float()
                nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
                toks.append(nxt)
                done |= nxt == EOS
                if done.all():
                    break
                last = nxt[:, None]
            toks = torch.stack(toks, 1).tolist()
            for k, i in enumerate(idx):
                ids = toks[k]
                ids = ids[:ids.index(EOS)] if EOS in ids else ids
                text = unicodedata.normalize("NFC", self.tgt_sp.decode(ids))
                res[i] = (text, (lp[k] / n[k].clamp(min=1)).item()) if with_score else text
        return res

    @torch.no_grad()
    def logprob(self, sources: list[str], targets: list[str], batch_size: int = 64) -> list[float]:
        """Mean per-token log P(target | source) under this model (for
        round-trip checks and reranking)."""
        out: list[float] = [0.0] * len(sources)
        for b in range(0, len(sources), batch_size):
            idx = list(range(b, min(b + batch_size, len(sources))))
            src = self._src([sources[i] for i in idx])
            tg = [[BOS] + self.tgt_sp.encode(unicodedata.normalize("NFC", targets[i]))[:MAX_LEN - 2]
                  + [EOS] for i in idx]
            L = max(len(s) for s in tg)
            t = torch.full((len(tg), L), PAD, dtype=torch.long)
            for k, s in enumerate(tg):
                t[k, :len(s)] = torch.tensor(s)
            t = t.to(self.device)
            logp = torch.log_softmax(self.model.score(src, t[:, :-1]).float(), -1)
            gold = t[:, 1:]
            tok = logp.gather(2, gold[..., None])[..., 0].masked_fill(gold == PAD, 0.0)
            lens = (gold != PAD).sum(1).clamp(min=1)
            for k, i in enumerate(idx):
                out[i] = (tok[k].sum() / lens[k]).item()
        return out


def main(argv=None) -> int:
    import argparse
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--dir", default="ur2hi", choices=sorted(FILES))
    a = ap.parse_args(argv)
    for line in Teacher(a.dir, device="cpu")(a.text):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
