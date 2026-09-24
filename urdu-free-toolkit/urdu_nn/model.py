# -*- coding: utf-8 -*-
"""Character-level Transformer seq2seq for Urdu word transliteration.

One model, several tasks, selected by a tag token at the start of the source:

    <j>  urdu -> "R|devanagari"      (joint: Devanagari follows from the Roman reading)
    <r>  urdu -> R
    <c>  urdu -> casual roman         (auxiliary: teaches short vowels from 600k pairs)
    <h>  R -> devanagari             (Hindi orthography for a known reading)

Trained by ``training/translit/train.py``; used at runtime by ``neural_translit.py``.

Small on purpose (~8M params): trains in minutes on a laptop GPU and decodes
fast on CPU. The runtime provider only calls it for words the lexicon misses.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>", "<j>", "<r>", "<c>", "<h>"]
TASK_TOKEN = {"j": "<j>", "r": "<r>", "c": "<c>", "h": "<h>"}


class Vocab:
    def __init__(self, chars: list[str]):
        self.itos = list(SPECIALS) + [c for c in chars if c not in SPECIALS]
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode_src(self, task: str, text: str) -> list[int]:
        return [self.stoi[TASK_TOKEN[task]]] + [self.stoi.get(c, UNK) for c in text]

    def encode_tgt(self, text: str) -> list[int]:
        return [BOS] + [self.stoi.get(c, UNK) for c in text] + [EOS]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i >= len(SPECIALS):
                out.append(self.itos[i])
        return "".join(out)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 enc_layers: int = 4, dec_layers: int = 4, ff: int = 1024,
                 dropout: float = 0.1, max_len: int = 96):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                        enc_layers=enc_layers, dec_layers=dec_layers, ff=ff,
                        dropout=dropout, max_len=max_len)
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_src = nn.Embedding(max_len, d_model)
        self.pos_tgt = nn.Embedding(max_len, d_model)
        self.tf = nn.Transformer(d_model, nhead, enc_layers, dec_layers, ff, dropout,
                                 batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, vocab_size)
        self.out.weight = self.emb.weight   # tied
        nn.init.normal_(self.emb.weight, std=d_model ** -0.5)
        nn.init.normal_(self.pos_src.weight, std=0.02)
        nn.init.normal_(self.pos_tgt.weight, std=0.02)
        self.scale = math.sqrt(d_model)

    def _embed(self, ids, pos):
        p = torch.arange(ids.size(1), device=ids.device)
        return self.emb(ids) * self.scale + pos(p)[None]

    def encode(self, src):
        mask = src == PAD
        mem = self.tf.encoder(self._embed(src, self.pos_src), src_key_padding_mask=mask)
        return mem, mask

    def decode(self, tgt, mem, mem_mask):
        L = tgt.size(1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tgt.device), 1)
        h = self.tf.decoder(self._embed(tgt, self.pos_tgt), mem, tgt_mask=causal,
                            tgt_key_padding_mask=tgt == PAD,
                            memory_key_padding_mask=mem_mask)
        return self.out(h)

    def forward(self, src, tgt_in):
        mem, mask = self.encode(src)
        return self.decode(tgt_in, mem, mask)


def pad_batch(seqs: list[list[int]], device=None) -> torch.Tensor:
    L = max(len(s) for s in seqs)
    t = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return t.to(device) if device is not None else t


def _decode_step(model: Seq2Seq, last, pos: int, mem, mem_mask, cache: list):
    """One incremental decoder step for the newest token only. ``cache`` holds,
    per layer, the normed inputs of every earlier position (the self-attention
    keys/values), so a step costs O(t) instead of re-running the whole prefix."""
    x = model.emb(last) * model.scale + model.pos_tgt.weight[pos]
    for li, layer in enumerate(model.tf.decoder.layers):
        h = layer.norm1(x)
        cache[li] = h if cache[li] is None else torch.cat([cache[li], h], 1)
        x = x + layer.self_attn(h, cache[li], cache[li], need_weights=False)[0]
        h = layer.norm2(x)
        x = x + layer.multihead_attn(h, mem, mem, key_padding_mask=mem_mask,
                                     need_weights=False)[0]
        h = layer.norm3(x)
        x = x + layer.linear2(layer.activation(layer.linear1(h)))
    return model.out(model.tf.decoder.norm(x))[:, -1]


@torch.no_grad()
def beam_search(model: Seq2Seq, src: torch.Tensor, beam: int = 4, max_len: int = 64,
                len_alpha: float = 0.6):
    """Batched, KV-cached beam search. Returns, per source row, a list of
    (token_ids, normalised_logprob) best-first."""
    model.eval()
    B = src.size(0)
    mem, mask = model.encode(src)
    mem = mem.repeat_interleave(beam, 0)
    mask = mask.repeat_interleave(beam, 0)
    cache: list = [None] * len(model.tf.decoder.layers)
    seqs = torch.full((B * beam, 1), BOS, dtype=torch.long, device=src.device)
    scores = torch.zeros(B, beam, device=src.device)
    scores[:, 1:] = float("-inf")                 # all beams start identical
    finished: list[list[tuple[list[int], float]]] = [[] for _ in range(B)]
    alive = torch.ones(B, beam, dtype=torch.bool, device=src.device)
    base = (torch.arange(B, device=src.device) * beam).unsqueeze(-1)
    for step in range(max_len):
        logits = _decode_step(model, seqs[:, -1:], step, mem, mask, cache).float()
        logp = torch.log_softmax(logits, -1).view(B, beam, -1)
        V = logp.size(-1)
        cand = (scores.unsqueeze(-1) + logp)
        cand = cand.masked_fill(~alive.unsqueeze(-1), float("-inf"))
        top, idx = cand.view(B, -1).topk(beam, -1)
        beam_idx, tok = idx // V, idx % V
        reorder = (base + beam_idx).view(-1)
        seqs = torch.cat([seqs[reorder], tok.view(-1, 1)], 1)
        cache = [c[reorder] for c in cache]
        scores = top
        is_eos = (tok == EOS) & torch.isfinite(top)
        if is_eos.any():
            length = seqs.size(1) - 1
            lp = ((5 + length) / 6) ** len_alpha
            for b, k in is_eos.nonzero().tolist():
                finished[b].append((seqs[b * beam + k, 1:].tolist(), scores[b, k].item() / lp))
            scores = scores.masked_fill(is_eos, float("-inf"))
        alive = torch.isfinite(scores)
        if not alive.any() or all(len(finished[b]) >= beam for b in range(B)):
            break
    for b in range(B):
        if not finished[b]:
            k = int(scores[b].argmax())
            finished[b].append((seqs[b * beam + k, 1:].tolist(), scores[b, k].item()))
        finished[b].sort(key=lambda x: -x[1])
    return finished
