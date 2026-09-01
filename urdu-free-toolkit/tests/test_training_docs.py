from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent / "training"


def test_runbook_and_configs_present_and_parse():
    for rel in ["README.md", "easyocr/train.md", "paddle/train.md"]:
        p = _ROOT / rel
        assert p.exists() and len(p.read_text(encoding="utf-8")) > 400, rel
    for rel in ["easyocr/config.yaml", "paddle/arabic_rec_ft.yml"]:
        p = _ROOT / rel
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and data, rel


def test_readme_links_the_eval_loop():
    txt = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "run_eval" in txt and "OCR_PADDLE_REC_DIR" in txt and "user_network" in txt
