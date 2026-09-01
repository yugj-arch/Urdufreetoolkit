# -*- coding: utf-8 -*-
"""Container deploy (Hugging Face Spaces / any Docker host) — the full-engine
target where PaddleOCR + EasyOCR actually run, unlike the Vercel deployment.
"""
from pathlib import Path

APP = Path(__file__).resolve().parents[1]          # urdu-free-toolkit/
REPO = APP.parent                                  # repo root


def test_dockerfile_serves_the_flask_app_on_the_hf_port():
    df = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "7860" in df                             # HF Spaces convention
    assert "gunicorn" in df
    assert "app:app" in df                          # the WSGI callable
    assert "requirements-docker.txt" in df
    # CPU-only torch keeps the image small enough to build
    assert "download.pytorch.org/whl/cpu" in df


def test_requirements_docker_has_the_offline_ocr_engines():
    req = (APP / "requirements-docker.txt").read_text(encoding="utf-8").lower()
    for pkg in ("easyocr", "paddleocr", "paddlepaddle", "gunicorn"):
        assert pkg in req, f"{pkg} missing from requirements-docker.txt"
    # still needs the app + cloud engines
    for pkg in ("flask", "openai"):
        assert pkg in req


def test_root_readme_is_a_hf_space_card():
    fm = (REPO / "README.md").read_text(encoding="utf-8")
    assert fm.lstrip().startswith("---")           # YAML frontmatter
    assert "sdk: docker" in fm
    assert "app_port: 7860" in fm


def test_dockerignore_keeps_secrets_and_test_junk_out():
    di = {ln.strip() for ln in (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()}
    assert ".env" in di
    assert ".git" in di


def test_root_requirements_still_untouched():
    # the container path must not have edited the user's local requirements.txt
    root_req = (APP / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "easyocr" in root_req and "surya-ocr" in root_req and "pytest" in root_req
