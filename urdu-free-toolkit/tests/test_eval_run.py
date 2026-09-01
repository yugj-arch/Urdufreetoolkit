import json

from PIL import Image

from eval import run_eval


def _png(p):
    Image.new("RGB", (30, 16), "white").save(p, format="PNG")


class _Res:
    def __init__(self, text):
        self.ok, self.text, self.meta = True, text, {}


def _fixtures(tmp_path):
    _png(tmp_path / "one.png")
    (tmp_path / "one.gpt.json").write_text(
        json.dumps({"urdu": "ہے ٹھیک", "model": "gpt"}), encoding="utf-8")
    return tmp_path


def test_evaluate_scores_each_engine(tmp_path):
    d = _fixtures(tmp_path)
    ocr_map = {
        "perfect": lambda b: _Res("ہے ٹھیک"),
        "bad": lambda b: _Res("xxx"),
    }
    res = run_eval.evaluate(["perfect", "bad"], fixtures_dir=d, ocr_map=ocr_map)
    assert res["perfect"]["agg"]["cer_norm"] == 0.0
    assert res["bad"]["agg"]["cer_norm"] > 0.5


def test_check_regression_flags_worse_cer():
    results = {"paddle": {"agg": {"cer_norm": 0.30}}}
    assert run_eval.check_regression(results, {"paddle": {"cer_norm": 0.20}}, tol=0.01)
    assert not run_eval.check_regression(results, {"paddle": {"cer_norm": 0.30}}, tol=0.01)


def test_format_table_has_headers():
    t = run_eval.format_table({"paddle": {
        "agg": {"cer": 0.1, "wer": 0.2, "cer_norm": 0.05, "wer_norm": 0.1, "n": 3},
        "ms": 12}})
    assert "engine" in t and "CER" in t and "paddle" in t


def test_main_check_returns_nonzero_on_regression(tmp_path, monkeypatch):
    d = _fixtures(tmp_path)
    monkeypatch.setattr(run_eval, "_resolve_ocr", lambda ids: {"paddle": lambda b: _Res("xxx")})
    (d / "baseline.json").write_text('{"paddle": {"cer_norm": 0.0}}', encoding="utf-8")
    rc = run_eval.main(["--engines", "paddle", "--fixtures", str(d),
                        "--baseline", str(d / "baseline.json"), "--check"])
    assert rc == 1


def test_main_check_passes_when_within_tolerance(tmp_path, monkeypatch):
    d = _fixtures(tmp_path)
    monkeypatch.setattr(run_eval, "_resolve_ocr", lambda ids: {"paddle": lambda b: _Res("ہے ٹھیک")})
    (d / "baseline.json").write_text('{"paddle": {"cer_norm": 0.0}}', encoding="utf-8")
    rc = run_eval.main(["--engines", "paddle", "--fixtures", str(d),
                        "--baseline", str(d / "baseline.json"), "--check"])
    assert rc == 0
