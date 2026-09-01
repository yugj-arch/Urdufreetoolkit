# -*- coding: utf-8 -*-
"""Behaviour that changes when the app runs on Vercel.

Vercel injects ``VERCEL=1``, serves a read-only filesystem, and caps the
request body at ~4.5 MB. ``config`` is the single place those facts turn into
switches; these tests pin each switch and the endpoints/UI that read it.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# config module                                                              #
# --------------------------------------------------------------------------- #
def test_on_vercel_false_without_env(monkeypatch):
    import config
    monkeypatch.delenv("VERCEL", raising=False)
    assert config.on_vercel() is False


def test_on_vercel_true_with_env(monkeypatch):
    import config
    monkeypatch.setenv("VERCEL", "1")
    assert config.on_vercel() is True


def test_batch_max_files_is_30_locally(monkeypatch):
    import config
    monkeypatch.delenv("VERCEL", raising=False)
    assert config.batch_max_files() == 30


def test_batch_max_files_drops_to_4_on_vercel(monkeypatch):
    import config
    monkeypatch.setenv("VERCEL", "1")
    assert config.batch_max_files() == 4


def test_settings_readonly_only_on_vercel(monkeypatch):
    import config
    monkeypatch.delenv("VERCEL", raising=False)
    assert config.settings_readonly() is False
    monkeypatch.setenv("VERCEL", "1")
    assert config.settings_readonly() is True


def test_max_content_length_tightens_on_vercel(monkeypatch):
    import config
    monkeypatch.delenv("VERCEL", raising=False)
    local = config.max_content_length()
    monkeypatch.setenv("VERCEL", "1")
    assert config.max_content_length() < local
    assert config.max_content_length() <= 4 * 1024 * 1024


# --------------------------------------------------------------------------- #
# /api/config endpoint                                                        #
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(monkeypatch):
    from providers import registry
    registry.reset_cache()
    registry.discover(package="tests.fakes")
    import app as appmod
    appmod.app.config["TESTING"] = True
    yield appmod.app.test_client()
    registry.reset_cache()


def test_api_config_reports_local_defaults(client, monkeypatch):
    monkeypatch.delenv("VERCEL", raising=False)
    data = client.get("/api/config").get_json()
    assert data["on_vercel"] is False
    assert data["batch_max_files"] == 30
    assert data["settings_readonly"] is False


def test_api_config_reflects_vercel(client, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    data = client.get("/api/config").get_json()
    assert data["on_vercel"] is True
    assert data["batch_max_files"] == 4
    assert data["settings_readonly"] is True


# --------------------------------------------------------------------------- #
# settings write guard                                                        #
# --------------------------------------------------------------------------- #
def test_settings_save_is_noop_on_vercel(tmp_path, monkeypatch):
    import settings
    envf = tmp_path / ".env"
    monkeypatch.setattr(settings, "env_path", lambda: envf)
    monkeypatch.setenv("VERCEL", "1")
    written = settings.save({"OPENAI_API_KEY": "sk-should-not-persist"})
    assert written == []
    assert not envf.exists()


def test_api_settings_post_reports_readonly_on_vercel(client, tmp_path, monkeypatch):
    import settings
    monkeypatch.setattr(settings, "env_path", lambda: tmp_path / ".env")
    monkeypatch.setenv("VERCEL", "1")
    body = client.post("/api/settings", json={"OPENAI_API_KEY": "sk-x"}).get_json()
    assert body["readonly"] is True
    assert body["saved"] == []


# --------------------------------------------------------------------------- #
# deployment config files                                                     #
# --------------------------------------------------------------------------- #
def test_vercel_json_declares_flask_function_duration():
    cfg = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    assert cfg["functions"]["app.py"]["maxDuration"] >= 60


def test_pyproject_carries_minimal_runtime_deps():
    import tomllib
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = " ".join(data["project"]["dependencies"]).lower()
    for pkg in ("flask", "pillow", "openai", "python-dotenv"):
        assert pkg in deps, f"{pkg} missing from pyproject deps"
    # heavy offline engines must NOT be pulled into the serverless bundle
    for heavy in ("torch", "easyocr", "surya", "paddle", "transformers"):
        assert heavy not in deps, f"{heavy} would blow the 250 MB function limit"
    assert data["tool"]["vercel"]["entrypoint"] == "app:app"


def test_vercelignore_excludes_the_heavy_requirements_file():
    lines = {ln.strip() for ln in (ROOT / ".vercelignore").read_text(encoding="utf-8").splitlines()}
    assert "requirements.txt" in lines
    assert "tests/" in lines or "tests" in lines


def test_env_example_lists_the_provider_keys():
    txt = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
                "GOOGLE_VISION_KEY", "GROQ_API_KEY"):
        assert key in txt
