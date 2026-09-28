# -*- coding: utf-8 -*-
"""Character vocabulary and special tokens of ``urdu_nn.model``.

Torch-free, so the numpy runtime (``urdu_nn.npnn``) can read the same
checkpoints on the deployed site; ``urdu_nn.model`` re-exports all of it.
"""
from __future__ import annotations

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>", "<j>", "<r>", "<c>", "<h>"]
TASK_TOKEN = {"j": "<j>", "r": "<r>", "c": "<c>", "h": "<h>"}


class Vocab:
    def __init__(self, chars: list[str]):
        self.itos = list(SPECIALS) + [c for c in chars if c not in SPECIALS]
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def from_itos(cls, itos: list[str]) -> "Vocab":
        """The vocabulary a checkpoint was trained with (its saved ``itos``)."""
        v = cls([])
        v.itos = list(itos)
        v.stoi = {s: i for i, s in enumerate(v.itos)}
        return v

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
