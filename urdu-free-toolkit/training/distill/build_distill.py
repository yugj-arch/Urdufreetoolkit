# -*- coding: utf-8 -*-
"""Write aligned (crop image, GPT text) pairs into a train/val dataset in the
same layout build_synth uses (out_dir/{train,val}/*.png + gt.txt).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def build(pairs, out_dir, val_frac: float = 0.1, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)
    for i, (img, text) in enumerate(pairs):
        arr = np.array(img.convert("L")) if hasattr(img, "convert") else np.asarray(img)
        split = "val" if rng.random() < val_frac else "train"
        rel = f"{i:06d}.png"
        cv2.imwrite(str(out_dir / split / rel), arr)
        with (out_dir / split / "gt.txt").open("a", encoding="utf-8") as fh:
            fh.write(f"{rel}\t{text}\n")
        counts[split] += 1
    return {"train": counts["train"], "val": counts["val"]}
