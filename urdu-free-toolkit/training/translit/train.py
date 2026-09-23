# -*- coding: utf-8 -*-
"""Train the neural transliterator on ``data/translit_ds`` (see build_data.py).

    python -m training.translit.train                  # full run (GPU if present)
    python -m training.translit.train --epochs 2       # smoke run
    python -m training.translit.train --resume --epochs 16 --lr 6e-4 --warmup 200
                                                       # continue from model.pt

Writes the best checkpoint (by dev joint exact-match) to
``data/translit_model/model.pt``.
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
DS = ROOT / "data" / "translit_ds"
OUT = ROOT / "data" / "translit_model"


def load(split):
    rows = []
    for line in (DS / f"{split}.tsv").read_text(encoding="utf-8").splitlines():
        t, s, g, w = line.split("\t")
        rows.append((t, s, g, float(w)))
    return rows


def build_vocab(rows) -> Vocab:
    chars = set()
    for _, s, g, _ in rows:
        chars.update(s)
        chars.update(g)
    return Vocab(sorted(chars))


def epoch_mix(train, rng, casual_per_epoch, main_repeat, hindi_per_epoch=40_000):
    main = [r for r in train if r[0] in "jr"]
    casual = [r for r in train if r[0] == "c"]
    hindi = [r for r in train if r[0] == "h"]
    mix = (main * main_repeat + rng.sample(casual, min(casual_per_epoch, len(casual)))
           + rng.sample(hindi, min(hindi_per_epoch, len(hindi))))
    rng.shuffle(mix)
    return mix


def batches(rows, vocab, bs, device):
    # bucket by length for less padding
    rows = sorted(rows, key=lambda r: len(r[1]) + len(r[2]))
    chunks = [rows[i:i + bs] for i in range(0, len(rows), bs)]
    random.shuffle(chunks)
    for ch in chunks:
        src = pad_batch([vocab.encode_src(t, s) for t, s, _, _ in ch], device)
        tgt = pad_batch([vocab.encode_tgt(g) for _, _, g, _ in ch], device)
        w = torch.tensor([x[3] for x in ch], device=device)
        yield src, tgt, w


@torch.no_grad()
def evaluate(model, vocab, dev, device, n=1500, beam=1):
    model.eval()
    res = {}
    for task in "jrch":
        rows = [r for r in dev if r[0] == task][:n]
        if not rows:
            continue
        ok = 0
        for i in range(0, len(rows), 256):
            ch = rows[i:i + 256]
            src = pad_batch([vocab.encode_src(t, s) for t, s, _, _ in ch], device)
            if beam == 1:
                hyps = greedy(model, src)
            else:
                hyps = [h[0][0] for h in beam_search(model, src, beam=beam)]
            for h, (_, _, g, _) in zip(hyps, ch):
                ok += vocab.decode(h) == g
        res[task] = ok / len(rows)
    model.train()
    return res


@torch.no_grad()
def greedy(model, src, max_len=64):
    return [h[0][0] for h in beam_search(model, src, beam=1, max_len=max_len)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--casual-per-epoch", type=int, default=160_000)
    ap.add_argument("--main-repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--resume", action="store_true",
                    help="start from data/translit_model/model.pt (fresh optimizer)")
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--avg", type=int, default=4,
                    help="also try the weight average of the last N epochs")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, dev = load("train"), load("dev")
    vocab = build_vocab(train + dev)
    resume = torch.load(OUT / "model.pt", map_location="cpu") if args.resume else None
    if resume:
        vocab.itos = resume["itos"]
        vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    print(f"device={device} train={len(train)} dev={len(dev)} vocab={len(vocab)}", flush=True)

    model = Seq2Seq(len(vocab), d_model=args.d_model, enc_layers=args.layers,
                    dec_layers=args.layers, ff=args.d_model * 4).to(device)
    if resume:
        model.load_state_dict({k: v.float() for k, v in resume["state"].items()})
        print(f"resumed from epoch {resume['epoch']} dev {resume['dev']}", flush=True)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    steps_per_epoch = math.ceil((sum(1 for r in train if r[0] in "jr") * args.main_repeat
                                 + args.casual_per_epoch + 40_000) / args.bs)
    total = steps_per_epoch * args.epochs
    warm = min(args.warmup, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    use_amp = device == "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    best, step, t0 = -1.0, 0, time.time()
    if resume:   # a resumed run must beat the checkpoint it started from
        best = resume["dev"].get("j", 0)
    recent: list[dict] = []   # last --avg epochs' weights, for averaging
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for src, tgt, w in batches(epoch_mix(train, rng, args.casual_per_epoch, args.main_repeat),
                                   vocab, args.bs, device):
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(src, tgt[:, :-1])
                gold = tgt[:, 1:]
                loss_tok = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                           gold.reshape(-1), ignore_index=PAD, label_smoothing=0.1,
                                           reduction="none").view(gold.shape)
                ntok = (gold != PAD).sum(1).clamp(min=1)
                loss = ((loss_tok.sum(1) / ntok) * w).sum() / w.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
            except torch.OutOfMemoryError:
                # another process grabbed the GPU (e.g. an OCR model): skip this batch
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
        acc = evaluate(model, vocab, dev, device)
        print(f"ep {ep:3d} step {step} loss {tot / n:.4f} dev {acc} "
              f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        # select on the joint task only: its dev set is ~5x the others (less noise)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ck = {"state": state, "cfg": model.cfg, "itos": vocab.itos, "epoch": ep, "dev": acc}
        torch.save(ck, OUT / "model_last.pt")
        recent = (recent + [state])[-args.avg:]
        if acc.get("j", 0) > best:
            best = acc["j"]
            torch.save(ck, OUT / "model_best.pt")

    # final pick: best epoch vs last epoch vs average of the last N, by beam-4 dev
    cands = {"last": torch.load(OUT / "model_last.pt", map_location=device)}
    if (OUT / "model_best.pt").exists():
        cands["best"] = torch.load(OUT / "model_best.pt", map_location=device)
    if len(recent) > 1:
        avg = {k: (sum(r[k].float() for r in recent) / len(recent)).to(recent[-1][k].dtype)
               for k in recent[-1]}
        cands["avg"] = dict(cands["last"], state=avg, epoch=f"avg{len(recent)}")
    scored = {}
    for name, c in cands.items():
        model.load_state_dict(c["state"])
        scored[name] = evaluate(model, vocab, dev, device, beam=4)
        print(f"{name} (epoch {c['epoch']}) dev(beam4) {scored[name]}", flush=True)
    pick = max(scored, key=lambda n: (scored[n].get("j", 0), scored[n].get("h", 0)))
    ck = cands[pick]
    # ship half-precision weights (the runtime upcasts on load): half the size
    ck["state"] = {k: v.half() for k, v in ck["state"].items()}
    ck["dev_beam4"] = scored[pick]
    torch.save(ck, OUT / "model.pt")
    print(f"shipped {pick} (epoch {ck['epoch']}) dev(beam4) {scored[pick]}", flush=True)
    (OUT / "train_log.json").write_text(json.dumps(
        {"shipped": pick, "epoch": ck["epoch"], "dev_beam4": scored, "args": vars(args)},
        indent=1, default=str))


if __name__ == "__main__":
    main()
