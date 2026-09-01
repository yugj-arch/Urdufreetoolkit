# -*- coding: utf-8 -*-
"""Score OCR engines against cached GPT silver references.

    cd urdu-free-toolkit
    python -m eval.run_eval --engines gpt,paddle,easyocr
    python -m eval.run_eval --engines paddle,easyocr --check          # CI regression gate
    python -m eval.run_eval --refresh                                 # regenerate silver refs (needs OPENAI_API_KEY)
    python -m eval.run_eval --engines paddle,easyocr --write-baseline
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval import refs
from eval.score import aggregate, cer, score_pair, wer
from providers.ocr._normalize import normalize_urdu
from providers.ocr._types import OcrConfig

_DEFAULT_BASELINE = Path(__file__).resolve().parent / "baseline.json"


def _resolve_ocr(engine_ids: list[str]) -> dict:
    from providers import registry
    registry.reset_cache()
    reg = registry.discover(package="providers")
    out = {}
    for eid in engine_ids:
        prov = reg.get(f"ocr:{eid}") or reg.get(eid)
        if prov is None:
            raise SystemExit(f"unknown OCR engine: {eid}")
        out[eid] = prov.ocr
    return out


def _score(ref: str, hyp: str, spellfix: bool) -> dict:
    row = score_pair(ref, hyp)
    if spellfix:
        cfg = OcrConfig(spellfix=True)
        rn, hn = normalize_urdu(ref, cfg), normalize_urdu(hyp, cfg)
        row["cer_norm"] = cer(rn, hn)
        row["wer_norm"] = wer(rn, hn)
    return row


def evaluate(engines: list[str], fixtures_dir=None, spellfix: bool = False,
             ocr_map: dict | None = None) -> dict:
    fixtures_dir = Path(fixtures_dir) if fixtures_dir else refs.FIXTURES_DIR
    ocr_map = ocr_map or _resolve_ocr(engines)
    pngs = refs.list_fixtures(fixtures_dir)
    results: dict = {}
    for eid in engines:
        rows, t0 = [], time.monotonic()
        for png in pngs:
            ref = refs.load_ref(png)
            if not ref:
                continue
            res = ocr_map[eid](png.read_bytes())
            hyp = getattr(res, "text", "") if getattr(res, "ok", False) else ""
            row = _score(ref["urdu"], hyp, spellfix)
            row["fixture"] = png.stem
            rows.append(row)
        ms = int((time.monotonic() - t0) * 1000 / max(1, len(pngs)))
        results[eid] = {"rows": rows, "agg": aggregate(rows), "ms": ms}
    return results


def format_table(results: dict) -> str:
    head = ("| engine | CER | WER | CER(norm) | WER(norm) | n | ms/img |\n"
            "|---|---|---|---|---|---|---|")
    lines = [head]
    for eid, r in results.items():
        a = r["agg"]
        lines.append(f"| {eid} | {a['cer']:.3f} | {a['wer']:.3f} | {a['cer_norm']:.3f} "
                     f"| {a['wer_norm']:.3f} | {a.get('n', 0)} | {r.get('ms', 0)} |")
    return "\n".join(lines)


def check_regression(results: dict, baseline: dict, tol: float = 0.01) -> list[str]:
    msgs = []
    for eid, r in results.items():
        base = baseline.get(eid)
        if not base:
            continue
        curv = r["agg"]["cer_norm"]
        if curv > base["cer_norm"] + tol:
            msgs.append(f"{eid}: CER(norm) {curv:.3f} > baseline {base['cer_norm']:.3f} + {tol}")
    return msgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eval.run_eval")
    ap.add_argument("--engines", default="gpt,paddle,easyocr")
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--baseline", default=str(_DEFAULT_BASELINE))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--spellfix", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args(argv)

    fixtures_dir = Path(args.fixtures) if args.fixtures else refs.FIXTURES_DIR
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    if args.refresh:
        rep = refs.ensure_refs(fixtures_dir, refresh=True)
        print(f"silver refs: wrote {rep['written']}, skipped {rep['skipped']}")

    results = evaluate(engines, fixtures_dir=fixtures_dir, spellfix=args.spellfix)
    print(format_table(results))

    if args.write_baseline:
        data = {eid: {"cer_norm": r["agg"]["cer_norm"], "wer_norm": r["agg"]["wer_norm"]}
                for eid, r in results.items()}
        Path(args.baseline).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote baseline -> {args.baseline}")
        return 0

    if args.check:
        try:
            baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        except Exception:
            baseline = {}
        msgs = check_regression(results, baseline)
        for m in msgs:
            print("REGRESSION:", m)
        return 1 if msgs else 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
