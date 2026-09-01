# -*- coding: utf-8 -*-
"""Assemble a synthetic OCR training set: sentences x fonts x augmentations ->
out_dir/{train,val}/*.png + gt.txt (tab-separated: <relpath>\\t<text>).

    python -m training.synth.build_synth --sentences training/corpus/sample_sentences.txt \\
        --fonts training/synth/fonts --out data/synth --per-sentence 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from training.synth import augment as _aug
from training.synth import render as _render


def build(sentences: list[str], font_paths: list[str], out_dir, per_sentence: int = 1,
          val_frac: float = 0.1, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    counts = {"train": 0, "val": 0}
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    idx = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        for _ in range(per_sentence):
            fp = font_paths[int(rng.integers(0, len(font_paths)))]
            img = _render.render_line(s, fp)
            arr = _aug.augment(np.array(img), rng)
            split = "val" if rng.random() < val_frac else "train"
            rel = f"{idx:06d}.png"
            cv2.imwrite(str(out_dir / split / rel), arr)
            with (out_dir / split / "gt.txt").open("a", encoding="utf-8") as fh:
                fh.write(f"{rel}\t{s}\n")
            counts[split] += 1
            idx += 1
    return {"train": counts["train"], "val": counts["val"], "gt": out_dir / "train" / "gt.txt"}


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-sentence", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sentences = Path(a.sentences).read_text(encoding="utf-8").splitlines()
    fonts = _render.find_fonts(Path(a.fonts))
    if not fonts:
        raise SystemExit(f"no .ttf/.otf fonts in {a.fonts}")
    rep = build(sentences, fonts, Path(a.out), a.per_sentence, a.val_frac, a.seed)
    print(rep)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
