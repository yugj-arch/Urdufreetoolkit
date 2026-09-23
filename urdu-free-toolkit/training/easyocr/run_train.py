#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch the real DTRB training run from ``config.yaml``.

``clovaai/deep-text-recognition-benchmark``'s ``train.py`` is pure argparse
CLI -- it has no ``--config`` flag. This script is the translation layer: it
reads this directory's ``config.yaml`` (the single source of truth for the
architecture params, verified against the installed ``easyocr`` package --
see ``train.md`` step 2) and invokes ``train.py`` with the equivalent CLI
arguments, via ``subprocess`` with an argument *list* (never a shell string)
so the Urdu/Arabic ``character`` set's quotes and backslashes never need
shell-escaping.

    python training/easyocr/run_train.py --dtrb-dir /path/to/deep-text-recognition-benchmark
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent  # urdu-free-toolkit/


def _resolve(path_str: str) -> str:
    p = Path(path_str)
    return str(p if p.is_absolute() else (_ROOT / path_str).resolve())


def build_argv(cfg: dict, dtrb_dir: Path) -> list[str]:
    argv = [sys.executable, str(dtrb_dir / "train.py")]
    string_opts = [
        "experiment_name", "train_data", "valid_data", "saved_model",
        "Transformation", "FeatureExtraction", "SequenceModeling", "Prediction",
        "character", "select_data", "batch_ratio",
    ]
    int_opts = [
        "manualSeed", "workers", "batch_size", "num_iter", "valInterval",
        "input_channel", "output_channel", "hidden_size", "imgH", "imgW",
        "batch_max_length",
    ]
    float_opts = ["lr"]
    bool_flags = ["FT", "adam", "PAD", "data_filtering_off", "sensitive", "rgb"]

    path_keyed = {"train_data", "valid_data", "saved_model"}
    for key in string_opts:
        if key not in cfg or cfg[key] in (None, ""):
            continue
        value = _resolve(cfg[key]) if key in path_keyed else cfg[key]
        flag = "--exp_name" if key == "experiment_name" else f"--{key}"
        argv += [flag, str(value)]
    for key in int_opts:
        if key in cfg and cfg[key] is not None:
            argv += [f"--{key}", str(cfg[key])]
    for key in float_opts:
        if key in cfg and cfg[key] is not None:
            argv += [f"--{key}", str(cfg[key])]
    for key in bool_flags:
        if cfg.get(key):
            argv.append(f"--{key}")
    return argv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=_HERE / "config.yaml")
    ap.add_argument("--dtrb-dir", type=Path, required=True,
                     help="path to a clone of github.com/clovaai/deep-text-recognition-benchmark")
    ap.add_argument("--dry-run", action="store_true", help="print the argv, don't run it")
    args = ap.parse_args()

    if not (args.dtrb_dir / "train.py").exists():
        raise SystemExit(f"{args.dtrb_dir}/train.py not found -- wrong --dtrb-dir?")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not cfg.get("saved_model"):
        default_ckpt = Path.home() / ".EasyOCR" / "model" / "arabic.pth"
        if default_ckpt.exists():
            cfg["saved_model"] = str(default_ckpt)
    argv = build_argv(cfg, args.dtrb_dir)

    if args.dry_run:
        print(argv)
        return

    Path(_resolve(cfg["train_data"])).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(argv, cwd=str(args.dtrb_dir), check=True)


if __name__ == "__main__":
    main()
