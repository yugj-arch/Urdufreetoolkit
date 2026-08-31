import io
import os

import pytest
from PIL import Image

from providers import registry


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    registry.reset_cache()
    registry.discover(package="tests.fakes")  # deterministic provider set
    import app as appmod
    appmod.app.config["TESTING"] = True
    yield appmod.app.test_client()
    registry.reset_cache()


def test_providers_lists_fake(client):
    data = client.get("/api/providers").get_json()
    ocr_ids = [r["id"] for r in data["ocr"]]
    translit_ids = [r["id"] for r in data["translit"]]
    assert "fake_ok" in ocr_ids
    assert "tr_fake" in translit_ids


def test_ocr_requires_image(client):
    r = client.post("/api/ocr", data={"providers": "fake_ok"})
    assert r.status_code == 400


def test_ocr_requires_provider(client):
    r = client.post(
        "/api/ocr",
        data={"image": (io.BytesIO(_png_bytes()), "x.png"), "providers": ""},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_ocr_streams_results(client):
    r = client.post(
        "/api/ocr",
        data={"image": (io.BytesIO(_png_bytes()), "x.png"), "providers": "fake_ok"},
        content_type="multipart/form-data",
    )
    body = r.get_data(as_text=True)
    assert "text/event-stream" in r.content_type
    assert "salaam" in body
    assert '"done": true' in body


def test_transliterate_returns_row(client):
    r = client.post("/api/transliterate", json={"text": "میں", "providers": ["tr_fake"]})
    rows = r.get_json()["results"]
    assert rows[0]["provider_id"] == "tr_fake"
    assert rows[0]["devanagari"] == "देव" and rows[0]["roman"] == "dev"


def test_translate_stub_returns_empty(client):
    r = client.post("/api/translate", json={"text": "میں", "providers": []})
    assert r.get_json() == {"results": []}


def test_settings_roundtrip(client, monkeypatch, tmp_path):
    import settings
    monkeypatch.setattr(settings, "env_path", lambda: tmp_path / ".env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/api/settings", json={"OPENAI_API_KEY": "sk-xyz"})
    assert r.get_json()["saved"] == ["OPENAI_API_KEY"]
    assert client.get("/api/settings").get_json()["OPENAI_API_KEY"] is True
    os.environ.pop("OPENAI_API_KEY", None)
