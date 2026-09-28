# -*- coding: utf-8 -*-
"""Our Transformers without PyTorch: numpy inference for the deployed site.

Vercel can't ship torch (it alone outweighs what a function may hold), so this
runs the very same checkpoints with the same math:

* ``load_pt`` reads a ``torch.save`` file straight out of its zip, allowing
  only the three pickle globals a state dict needs.
* ``Transformer`` is ``nn.Transformer`` inference (batch-first, ReLU, pre- or
  post-norm) with an incremental decoder that caches each layer's
  self-attention keys/values.
* ``Seq2Seq`` + ``beam_search`` mirror ``urdu_nn.model`` (our word and line
  models); ``RekhtaNet`` mirrors ``urdu_nn.rekhta_teacher.RekhtaTransformer``.
* ``Checkpoint`` loads one of our models on torch when it's installed, else
  here -- the tests hold the two backends to the same readings.
"""
from __future__ import annotations

import math
import os
import pickle
import zipfile
from collections import OrderedDict

import numpy as np

from urdu_nn.vocab import BOS, EOS, PAD, Vocab

_NEG = np.float32(-1e9)      # a masked attention score: exp() is exactly 0, never NaN


def use_torch() -> bool:
    """Torch when it's installed, unless ``URDU_NN_BACKEND=numpy`` forces numpy."""
    if os.environ.get("URDU_NN_BACKEND", "").lower() == "numpy":
        return False
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


# -- reading torch.save files ----------------------------------------------------

_STORAGE = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
            "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
            "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_}


def _rebuild_tensor(storage, offset, size, stride, *_):
    view = np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                           strides=tuple(s * storage.itemsize for s in stride))
    return np.array(view)


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh, zf: zipfile.ZipFile, prefix: str):
        super().__init__(fh)
        self._zf, self._prefix, self._storages = zf, prefix, {}

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name in _STORAGE:
            return _STORAGE[name]
        raise pickle.UnpicklingError(f"load_pt: {module}.{name} is not allowed")

    def persistent_load(self, pid):
        _, dtype, key, _location, _numel = pid
        if key not in self._storages:
            raw = self._zf.read(f"{self._prefix}/data/{key}")
            self._storages[key] = np.frombuffer(raw, dtype=dtype)
        return self._storages[key]


def load_pt(path) -> dict:
    """``torch.load`` for our checkpoints, without torch: tensors come back as
    numpy arrays, everything else as the plain Python it was saved as."""
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl") and n.count("/") == 1)
        with zf.open(pkl) as fh:
            return _Unpickler(fh, zf, pkl.rsplit("/", 1)[0]).load()


# -- the network -----------------------------------------------------------------

def pad_batch(seqs: list[list[int]]) -> np.ndarray:
    out = np.full((len(seqs), max(len(s) for s in seqs)), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _ln(x, p):
    m = x.mean(-1, keepdims=True)
    v = np.square(x - m).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * p[0] + p[1]


def _pad_bias(pad: np.ndarray) -> np.ndarray:
    """Key-padding mask [B, L] -> additive scores [B, 1, 1, L]."""
    return np.where(pad, _NEG, np.float32(0))[:, None, None, :]


class _Attention:
    """``nn.MultiheadAttention`` (batch-first), its packed in-projection split."""

    def __init__(self, st: dict, pre: str, heads: int):
        w, b = st[pre + "in_proj_weight"], st[pre + "in_proj_bias"]
        d = w.shape[1]
        self.h, self.dh = heads, d // heads
        self.wq, self.bq = w[:d].T.copy(), b[:d]
        self.wkv, self.bkv = w[d:].T.copy(), b[d:]          # keys and values in one matmul
        self.wo, self.bo = st[pre + "out_proj.weight"].T.copy(), st[pre + "out_proj.bias"]
        self.scale = np.float32(1 / math.sqrt(self.dh))

    def _heads(self, x):                                     # [B, L, d] -> [B, h, L, dh]
        B, L, _ = x.shape
        return x.reshape(B, L, self.h, self.dh).transpose(0, 2, 1, 3)

    def keys(self, x):
        """Keys and values for inputs ``x`` -- projected once, then cached."""
        kv = x @ self.wkv + self.bkv
        d = kv.shape[-1] // 2
        return self._heads(kv[..., :d]), self._heads(kv[..., d:])

    def __call__(self, x, k, v, bias=None):
        s = (self._heads(x @ self.wq + self.bq) * self.scale) @ k.transpose(0, 1, 3, 2)
        if bias is not None:
            s = s + bias
        a = _softmax(s) @ v
        B, _, L, _ = a.shape
        return a.transpose(0, 2, 1, 3).reshape(B, L, -1) @ self.wo + self.bo


class _Layer:
    """One ``nn.TransformerEncoderLayer`` or ``nn.TransformerDecoderLayer`` (ReLU)."""

    def __init__(self, st: dict, pre: str, heads: int, cross: bool):
        self.sa = _Attention(st, pre + "self_attn.", heads)
        self.ca = _Attention(st, pre + "multihead_attn.", heads) if cross else None
        self.w1, self.b1 = st[pre + "linear1.weight"].T.copy(), st[pre + "linear1.bias"]
        self.w2, self.b2 = st[pre + "linear2.weight"].T.copy(), st[pre + "linear2.bias"]
        self.norm = [(st[f"{pre}norm{i}.weight"], st[f"{pre}norm{i}.bias"])
                     for i in range(1, 4 if cross else 3)]

    def ff(self, x):
        return np.maximum(x @ self.w1 + self.b1, 0) @ self.w2 + self.b2


class DecodeState:
    """Per decoder layer: the memory's keys/values (fixed) and the keys/values
    of every target position decoded so far (causal: they never change)."""

    def __init__(self, cross: list, mem_bias: np.ndarray):
        self.cross, self.mem_bias = cross, mem_bias
        self.self_kv: list = [None] * len(cross)
        self.length = 0

    def add(self, layer: int, k, v):
        if self.self_kv[layer] is not None:
            k0, v0 = self.self_kv[layer]
            k, v = np.concatenate([k0, k], 2), np.concatenate([v0, v], 2)
        self.self_kv[layer] = (k, v)

    def reorder(self, rows: np.ndarray):
        """Beam search: row i continues from row ``rows[i]`` (same source, so
        the memory's rows need no moving)."""
        self.self_kv = [(k[rows], v[rows]) for k, v in self.self_kv]


class Transformer:
    """``nn.Transformer`` inference under ``prefix`` in a state dict."""

    def __init__(self, st: dict, prefix: str, heads: int, pre_norm: bool):
        def count(part):
            n = 0
            while f"{prefix}{part}.layers.{n}.norm1.weight" in st:
                n += 1
            return n
        self.pre_norm = pre_norm
        self.enc = [_Layer(st, f"{prefix}encoder.layers.{i}.", heads, False)
                    for i in range(count("encoder"))]
        self.dec = [_Layer(st, f"{prefix}decoder.layers.{i}.", heads, True)
                    for i in range(count("decoder"))]
        self.enc_norm = (st[prefix + "encoder.norm.weight"], st[prefix + "encoder.norm.bias"])
        self.dec_norm = (st[prefix + "decoder.norm.weight"], st[prefix + "decoder.norm.bias"])

    def encode(self, x: np.ndarray, pad: np.ndarray) -> np.ndarray:
        """Embedded source [B, L, d] + its padding [B, L] -> memory [B, L, d]."""
        bias = _pad_bias(pad)
        for l in self.enc:
            if self.pre_norm:
                h = _ln(x, l.norm[0])
                x = x + l.sa(h, *l.sa.keys(h), bias)
                x = x + l.ff(_ln(x, l.norm[1]))
            else:
                x = _ln(x + l.sa(x, *l.sa.keys(x), bias), l.norm[0])
                x = _ln(x + l.ff(x), l.norm[1])
        return _ln(x, self.enc_norm)

    def start(self, mem: np.ndarray, pad: np.ndarray) -> DecodeState:
        return DecodeState([l.ca.keys(mem) for l in self.dec], _pad_bias(pad))

    def decode(self, x: np.ndarray, st: DecodeState) -> np.ndarray:
        """Embedded new target positions [B, T, d] (T = 1 while generating;
        the whole prefix when scoring) -> final-normed states [B, T, d]."""
        T = x.shape[1]
        causal = None
        if T > 1:
            q = np.arange(T)[:, None] + st.length
            causal = np.where(np.arange(st.length + T)[None, :] > q, _NEG, np.float32(0))
        for i, l in enumerate(self.dec):
            h = _ln(x, l.norm[0]) if self.pre_norm else x
            st.add(i, *l.sa.keys(h))
            k, v = st.self_kv[i]
            if self.pre_norm:
                x = x + l.sa(h, k, v, causal)
                x = x + l.ca(_ln(x, l.norm[1]), *st.cross[i], st.mem_bias)
                x = x + l.ff(_ln(x, l.norm[2]))
            else:
                x = _ln(x + l.sa(x, k, v, causal), l.norm[0])
                x = _ln(x + l.ca(x, *st.cross[i], st.mem_bias), l.norm[1])
                x = _ln(x + l.ff(x), l.norm[2])
        st.length += T
        return _ln(x, self.dec_norm)


class _EncDec:
    """Embeddings + ``Transformer`` + output layer; subclasses embed."""
    tf: Transformer
    out_w: np.ndarray
    out_b: np.ndarray

    def _embed_src(self, src):
        raise NotImplementedError

    def _embed_tgt(self, ids, start: int):
        raise NotImplementedError

    def start(self, src: np.ndarray, repeat: int = 1) -> DecodeState:
        """Encode padded source ids [B, L]; ``repeat`` copies each row (beams)."""
        pad = src == PAD
        mem = self.tf.encode(self._embed_src(src), pad)
        if repeat > 1:
            mem, pad = np.repeat(mem, repeat, 0), np.repeat(pad, repeat, 0)
        return self.tf.start(mem, pad)

    def decode(self, ids: np.ndarray, st: DecodeState) -> np.ndarray:
        """Logits [B, T, V] for target ids [B, T] following what ``st`` has seen."""
        return self.tf.decode(self._embed_tgt(ids, st.length), st) @ self.out_w + self.out_b

    def step(self, last: np.ndarray, st: DecodeState) -> np.ndarray:
        """Logits [B, V] for the token after ``last`` [B]."""
        return self.decode(last[:, None], st)[:, -1]


def _float32(state: dict) -> dict:
    return {k.replace("module.", ""): np.asarray(v, dtype=np.float32) for k, v in state.items()}


class Seq2Seq(_EncDec):
    """``urdu_nn.model.Seq2Seq``: pre-norm, learned positions, scaled and tied
    embeddings."""

    def __init__(self, state: dict, cfg: dict):
        st = _float32(state)
        self.cfg = cfg
        self.scale = np.float32(math.sqrt(cfg["d_model"]))
        self.emb, self.pos_src, self.pos_tgt = st["emb.weight"], st["pos_src.weight"], st["pos_tgt.weight"]
        self.out_w, self.out_b = st.get("out.weight", self.emb).T.copy(), st["out.bias"]
        self.tf = Transformer(st, "tf.", cfg["nhead"], pre_norm=True)

    def _embed_src(self, src):
        return self.emb[src] * self.scale + self.pos_src[:src.shape[1]]

    def _embed_tgt(self, ids, start):
        return self.emb[ids] * self.scale + self.pos_tgt[start:start + ids.shape[1]]


class RekhtaNet(_EncDec):
    """``urdu_nn.rekhta_teacher.RekhtaTransformer``: post-norm, sinusoidal
    positions, no embedding scaling."""

    def __init__(self, state: dict, nhead: int = 4, max_len: int = 512):
        st = _float32(state)
        self.src_emb, self.tgt_emb = st["src_tok_emb.weight"], st["tgt_tok_emb.weight"]
        d = self.src_emb.shape[1]
        pos = np.arange(max_len, dtype=np.float32)[:, None]
        div = np.exp(np.arange(0, d, 2, dtype=np.float32) * np.float32(-math.log(10000.0) / d))
        self.pe = np.zeros((max_len, d), np.float32)
        self.pe[:, 0::2], self.pe[:, 1::2] = np.sin(pos * div), np.cos(pos * div)
        self.out_w, self.out_b = st["out.weight"].T.copy(), st["out.bias"]
        self.tf = Transformer(st, "transformer.", nhead, pre_norm=False)

    def _embed_src(self, src):
        return self.src_emb[src] + self.pe[:src.shape[1]]

    def _embed_tgt(self, ids, start):
        return self.tgt_emb[ids] + self.pe[start:start + ids.shape[1]]


def beam_search(model: Seq2Seq, seqs: list[list[int]], beam: int = 4, max_len: int = 64,
                len_alpha: float = 0.6) -> list[list[tuple[list[int], float]]]:
    """``urdu_nn.model.beam_search`` on numpy. Per source (a list of token
    ids), (token_ids, normalised_logprob) best-first."""
    B = len(seqs)
    st = model.start(pad_batch(seqs), repeat=beam)
    out = np.full((B * beam, 1), BOS, dtype=np.int64)
    scores = np.zeros((B, beam), np.float32)
    scores[:, 1:] = -np.inf                                   # all beams start identical
    finished: list[list[tuple[list[int], float]]] = [[] for _ in range(B)]
    alive = np.ones((B, beam), bool)
    base = (np.arange(B) * beam)[:, None]
    for _ in range(max_len):
        logp = log_softmax(model.step(out[:, -1], st)).reshape(B, beam, -1)
        V = logp.shape[-1]
        cand = np.where(alive[..., None], scores[..., None] + logp, -np.inf).reshape(B, -1)
        idx = np.argsort(-cand, axis=-1, kind="stable")[:, :beam]
        top = np.take_along_axis(cand, idx, -1)
        beam_idx, tok = idx // V, idx % V
        rows = (base + beam_idx).reshape(-1)
        out = np.concatenate([out[rows], tok.reshape(-1, 1)], 1)
        st.reorder(rows)
        scores = top
        is_eos = (tok == EOS) & np.isfinite(top)
        if is_eos.any():
            lp = ((5 + out.shape[1] - 1) / 6) ** len_alpha
            for b, k in zip(*np.nonzero(is_eos)):
                finished[b].append((out[b * beam + k, 1:].tolist(), float(scores[b, k]) / lp))
            scores = np.where(is_eos, -np.inf, scores)
        alive = np.isfinite(scores)
        if not alive.any() or all(len(f) >= beam for f in finished):
            break
    for b in range(B):
        if not finished[b]:
            k = int(scores[b].argmax())
            finished[b].append((out[b * beam + k, 1:].tolist(), float(scores[b, k])))
        finished[b].sort(key=lambda x: -x[1])
    return finished


class Checkpoint:
    """One of our trained ``urdu_nn.model`` checkpoints, ready to beam-search:
    on torch when it's installed, else on this numpy runtime."""

    def __init__(self, path, device: str = "cpu"):
        self.torch, self.device = use_torch(), device
        if self.torch:
            import torch
            from urdu_nn.model import Seq2Seq as TorchSeq2Seq
            ck = torch.load(path, map_location=device, weights_only=False)
            self.model = TorchSeq2Seq(**ck["cfg"])
            self.model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
            self.model.to(device).eval()
        else:
            ck = load_pt(path)
            self.model = Seq2Seq(ck["state"], ck["cfg"])
        self.cfg = ck["cfg"]
        self.meta = {k: v for k, v in ck.items() if k != "state"}
        self.vocab = Vocab.from_itos(ck["itos"])

    def beam_search(self, seqs: list[list[int]], beam: int, max_len: int):
        if not self.torch:
            return beam_search(self.model, seqs, beam=beam, max_len=max_len)
        import torch
        from urdu_nn.model import beam_search as torch_beam_search, pad_batch as torch_pad
        with torch.inference_mode():
            return torch_beam_search(self.model, torch_pad(seqs, self.device),
                                     beam=beam, max_len=max_len)
