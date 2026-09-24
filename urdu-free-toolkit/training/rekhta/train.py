# -*- coding: utf-8 -*-
"""Train our own line-level Urdu -> Devanagari model on the distilled labels.

    python -m training.rekhta.label        # teacher labels  (data/rekhta_ds/labels.jsonl)
    python -m training.rekhta.dataset      # filter + split  (data/rekhta_ds/{train,dev,test}.tsv)
    python -m training.rekhta.train        # -> data/rekhta_model/model.pt

Same architecture family as the word model (``urdu_nn.model.Seq2Seq``:
pre-norm Transformer, tied embeddings, KV-cached beam search), bigger by
default, reading whole runs of up to 45 Urdu characters so it sees the
context Rekhta's teacher used (izafat, کہ vs کے, poetic forms).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from urdu_nn.model import PAD, Seq2Seq, Vocab, beam_search, pad_batch

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "rekhta_ds"
OUT = ROOT / "data" / "rekhta_model"
LINE_TAG = "<l>"
MAX_POS = 128


def load(split: str, kept_only: bool = True) -> list[tuple[str, str, str, float]]:
    rows = []
    path = DS / f"{split}.tsv"
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        src, ur, hi, exact, kept = line.split("\t")
        if kept_only and kept != "1":
            continue
        rows.append((src, ur, hi, 1.0 if exact == "1" else 0.7))
    return rows


def encode_src(vocab: Vocab, text: str) -> list[int]:
    from urdu_nn.model import UNK
    return [vocab.stoi[LINE_TAG]] + [vocab.stoi.get(c, UNK) for c in text]


def batches(rows, vocab, bs, device, shuffle=True):
    rows = sorted(rows, key=lambda r: len(r[1]) + len(r[2]))
    chunks = [rows[i:i + bs] for i in range(0, len(rows), bs)]
    if shuffle:
        random.shuffle(chunks)
    for ch in chunks:
        src = pad_batch([encode_src(vocab, r[1]) for r in ch], device)
        tgt = pad_batch([vocab.encode_tgt(r[2]) for r in ch], device)
        w = torch.tensor([r[3] for r in ch], device=device)
        yield src, tgt, w


@torch.no_grad()
def evaluate(model, vocab, rows, device, n=3000, beam=1) -> dict:
    """Line exact-match and word accuracy against the teacher's labels."""
    model.eval()
    rows = rows[:n]
    line_ok = word_ok = words = 0
    for i in range(0, len(rows), 256):
        ch = rows[i:i + 256]
        src = pad_batch([encode_src(vocab, r[1]) for r in ch], device)
        hyps = beam_search(model, src, beam=beam, max_len=MAX_POS - 2)
        for h, r in zip(hyps, ch):
            got = vocab.decode(h[0][0])
            line_ok += got == r[2]
            g, w = got.split(), r[2].split()
            words += len(w)
            word_ok += sum(a == b for a, b in zip(g, w)) if len(g) == len(w) else 0
    model.train()
    return {"line": round(line_ok / max(len(rows), 1), 4), "word": round(word_ok / max(words, 1), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--avg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=0, help="train rows (0 = all), for smoke runs")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda when present)")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train, dev = load("train"), load("dev")
    random.shuffle(train)
    random.shuffle(dev)
    if args.limit:
        train = train[:args.limit]
    chars = set()
    for r in train + dev:
        chars.update(r[1])
        chars.update(r[2])
    vocab = Vocab(sorted(chars) + [LINE_TAG])
    print(f"device={device} train={len(train)} dev={len(dev)} vocab={len(vocab)}", flush=True)
    model = Seq2Seq(len(vocab), d_model=args.d_model, nhead=args.heads, enc_layers=args.layers,
                    dec_layers=args.layers, ff=args.d_model * 4, max_len=MAX_POS).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    total = math.ceil(len(train) / args.bs) * args.epochs
    warm = min(args.warmup, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    use_amp = device == "cuda"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    best, step, t0 = -1.0, 0, time.time()
    recent: list[dict] = []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for src, tgt, w in batches(train, vocab, args.bs, device):
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(src, tgt[:, :-1])
                gold = tgt[:, 1:]
                loss_tok = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                           gold.reshape(-1), ignore_index=PAD,
                                           label_smoothing=args.label_smoothing,
                                           reduction="none").view(gold.shape)
                ntok = (gold != PAD).sum(1).clamp(min=1)
                loss = ((loss_tok.sum(1) / ntok) * w).sum() / w.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print("  (OOM: skipped a batch)", flush=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tot += loss.item()
            n += 1
            if step % 500 == 0:
                print(f"  step {step} loss {tot / n:.4f} {time.time() - t0:.0f}s", flush=True)
        acc = evaluate(model, vocab, dev, device)
        print(f"ep {ep:3d} step {step} loss {tot / max(n, 1):.4f} dev {acc} "
              f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ck = {"state": state, "cfg": model.cfg, "itos": vocab.itos, "epoch": ep, "dev": acc,
              "kind": "rekhta-line"}
        torch.save(ck, out / "model_last.pt")
        recent = (recent + [state])[-args.avg:]
        if acc["line"] > best:
            best = acc["line"]
            torch.save(ck, out / "model_best.pt")

    cands = {"last": torch.load(out / "model_last.pt", map_location=device)}
    if (out / "model_best.pt").exists():
        cands["best"] = torch.load(out / "model_best.pt", map_location=device)
    if len(recent) > 1:
        avg = {k: (sum(r[k].float() for r in recent) / len(recent)).to(recent[-1][k].dtype)
               for k in recent[-1]}
        cands["avg"] = dict(cands["last"], state=avg, epoch=f"avg{len(recent)}")
    scored = {}
    for name, c in cands.items():
        model.load_state_dict(c["state"])
        scored[name] = evaluate(model, vocab, dev, device, beam=4)
        print(f"{name} (epoch {c['epoch']}) dev(beam4) {scored[name]}", flush=True)
    pick = max(scored, key=lambda k: (scored[k]["line"], scored[k]["word"]))
    ck = cands[pick]
    ck["state"] = {k: v.half() for k, v in ck["state"].items()}
    ck["dev_beam4"] = scored[pick]
    torch.save(ck, out / "model.pt")
    print(f"shipped {pick} (epoch {ck['epoch']}) dev(beam4) {scored[pick]}", flush=True)
    (out / "train_log.json").write_text(json.dumps(
        {"shipped": pick, "epoch": ck["epoch"], "dev_beam4": scored, "args": vars(args)},
        indent=1, default=str))


if __name__ == "__main__":
    main()
