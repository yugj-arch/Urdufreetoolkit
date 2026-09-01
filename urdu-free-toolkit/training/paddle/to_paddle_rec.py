# -*- coding: utf-8 -*-
"""Convert a gt.txt dataset to a PaddleOCR recognition label file.

PaddleOCR rec expects lines of ``<image_path>\\t<transcription>`` where
image_path is relative to the ``data_dir`` set in the training YAML. See
training/paddle/train.md.
"""
from __future__ import annotations

from pathlib import Path


def convert(gt_txt, out_txt, image_root: str) -> int:
    n = 0
    with Path(out_txt).open("w", encoding="utf-8") as fh:
        for line in Path(gt_txt).read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            rel, text = line.split("\t", 1)
            fh.write(f"{image_root.rstrip('/')}/{rel.strip()}\t{text}\n")
            n += 1
    return n
