# -*- coding: utf-8 -*-
"""Convert a gt.txt dataset to the layout Clova deep-text-recognition-benchmark
(EasyOCR's recognizer trainer) expects. See training/easyocr/train.md.
"""
from __future__ import annotations

from pathlib import Path


def _read_gt(gt_txt) -> list[tuple[str, str]]:
    rows = []
    for line in Path(gt_txt).read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        rel, text = line.split("\t", 1)
        rows.append((rel.strip(), text))
    return rows


def gt_to_dtrb_txt(gt_txt, out_txt, image_dir_prefix: str = "") -> int:
    rows = _read_gt(gt_txt)
    with Path(out_txt).open("w", encoding="utf-8") as fh:
        for rel, text in rows:
            path = f"{image_dir_prefix.rstrip('/')}/{rel}" if image_dir_prefix else rel
            fh.write(f"{path}\t{text}\n")
    return len(rows)


def write_lmdb(gt_txt, image_dir, out_lmdb) -> int:
    try:
        import lmdb
    except Exception as e:  # pragma: no cover - documented, not run in CI
        raise RuntimeError("pip install lmdb  (needed for DTRB LMDB datasets)") from e
    rows = _read_gt(gt_txt)
    env = lmdb.open(str(out_lmdb), map_size=1 << 40)
    with env.begin(write=True) as txn:
        cnt = 0
        for rel, text in rows:
            img = (Path(image_dir) / rel).read_bytes()
            cnt += 1
            txn.put(f"image-{cnt:09d}".encode(), img)
            txn.put(f"label-{cnt:09d}".encode(), text.encode("utf-8"))
        txn.put(b"num-samples", str(cnt).encode())
    return cnt
