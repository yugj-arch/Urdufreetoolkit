"""RUN_HEAVY=1 only. Runs the real paddle/easyocr engines over the committed
OCR fixtures (cached silver refs, no network) and asserts CER(norm) has not
regressed past eval/baseline.json. Skips cleanly when there are no fixtures or
no baseline entries yet."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy


def test_offline_ocr_no_cer_regression():
    from eval import refs, run_eval

    if not refs.list_fixtures():
        pytest.skip("no OCR fixtures committed yet")
    baseline_path = Path(run_eval._DEFAULT_BASELINE)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    engines = [e for e in ("paddle", "easyocr") if e in baseline]
    if not engines:
        pytest.skip("no baseline entries for paddle/easyocr yet")

    results = run_eval.evaluate(engines)
    msgs = run_eval.check_regression(results, baseline, tol=0.01)
    assert not msgs, "\n".join(msgs)
