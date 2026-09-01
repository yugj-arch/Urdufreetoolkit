import json

from PIL import Image

from eval import refs


def _png(p):
    Image.new("RGB", (30, 16), "white").save(p, format="PNG")


class _FakeOcr:
    def __init__(self, text):
        self.text, self.ok, self.meta = text, True, {"model": "fake"}


def test_ensure_refs_writes_missing_then_is_idempotent(tmp_path):
    _png(tmp_path / "a.png")
    calls = []

    def fake_ocr(b):
        calls.append(1)
        return _FakeOcr("ہے")

    r1 = refs.ensure_refs(tmp_path, ocr=fake_ocr)
    assert r1["written"] == ["a"] and (tmp_path / "a.gpt.json").exists()
    data = json.loads((tmp_path / "a.gpt.json").read_text(encoding="utf-8"))
    assert data["urdu"] == "ہے" and data["model"] == "fake"

    r2 = refs.ensure_refs(tmp_path, ocr=fake_ocr)
    assert r2["written"] == [] and r2["have"] == ["a"] and len(calls) == 1


def test_ensure_refs_skips_when_ocr_unavailable(tmp_path):
    _png(tmp_path / "b.png")

    def broken(b):
        raise RuntimeError("OPENAI_API_KEY not set")

    r = refs.ensure_refs(tmp_path, ocr=broken)
    assert r["skipped"] == ["b"] and not (tmp_path / "b.gpt.json").exists()


def test_refresh_overwrites(tmp_path):
    _png(tmp_path / "c.png")
    (tmp_path / "c.gpt.json").write_text('{"urdu":"old","model":"x"}', encoding="utf-8")
    refs.ensure_refs(tmp_path, refresh=True, ocr=lambda b: _FakeOcr("new"))
    assert json.loads((tmp_path / "c.gpt.json").read_text(encoding="utf-8"))["urdu"] == "new"
