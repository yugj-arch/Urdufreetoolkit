import os

import settings


def test_save_and_status(tmp_path, monkeypatch):
    envf = tmp_path / ".env"
    monkeypatch.setattr(settings, "env_path", lambda: envf)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    written = settings.save({"OPENAI_API_KEY": "sk-abc", "UNKNOWN": "x"})
    assert written == ["OPENAI_API_KEY"]
    assert "OPENAI_API_KEY=sk-abc" in envf.read_text()
    assert os.environ["OPENAI_API_KEY"] == "sk-abc"
    assert settings.status()["OPENAI_API_KEY"] is True


def test_save_preserves_other_lines(tmp_path, monkeypatch):
    envf = tmp_path / ".env"
    envf.write_text("# comment\nOTHER=keep\nOPENAI_API_KEY=old\n")
    monkeypatch.setattr(settings, "env_path", lambda: envf)
    settings.save({"OPENAI_API_KEY": "new"})
    text = envf.read_text()
    assert "OTHER=keep" in text and "# comment" in text
    assert "OPENAI_API_KEY=new" in text and "old" not in text


def test_model_override_keys_are_known():
    assert "ANTHROPIC_MODEL" in settings.KNOWN_KEYS
    assert "GEMINI_MODEL" in settings.KNOWN_KEYS


def test_empty_value_removes(tmp_path, monkeypatch):
    envf = tmp_path / ".env"
    envf.write_text("OPENAI_API_KEY=old\n")
    monkeypatch.setattr(settings, "env_path", lambda: envf)
    monkeypatch.setenv("OPENAI_API_KEY", "old")
    settings.save({"OPENAI_API_KEY": ""})
    assert "OPENAI_API_KEY" not in envf.read_text()
    assert os.environ.get("OPENAI_API_KEY") is None
