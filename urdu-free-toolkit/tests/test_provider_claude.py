import io
import json
import types

from PIL import Image


def _fake_anthropic(monkeypatch, payload: dict):
    """Patch _anthropic_common so no real client / key / network is touched."""
    import providers._anthropic_common as common

    class _Block:
        type = "text"
        text = json.dumps(payload)

    class _Resp:
        content = [_Block()]

    class _Messages:
        @staticmethod
        def create(**kw):
            return _Resp()

    client = types.SimpleNamespace(messages=_Messages())
    monkeypatch.setattr(common, "get_client", lambda: client)
    monkeypatch.setattr(common, "have_key", lambda: True)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


_PNG = _png_bytes()


def test_claude_ocr_unavailable_without_key(monkeypatch):
    import providers._anthropic_common as common
    monkeypatch.setattr(common, "have_key", lambda: False)
    from providers.ocr.claude import PROVIDER
    ok, _reason = PROVIDER.available()
    assert ok is False


def test_claude_ocr_parses_payload(monkeypatch):
    _fake_anthropic(monkeypatch, {"urdu": "محبت", "devanagari": "मोहब्बत",
                                  "roman": "mohabbat", "notes": ""})
    from providers.ocr.claude import PROVIDER
    r = PROVIDER.ocr(_PNG)
    assert r.ok is True and r.text == "محبت"
    assert r.meta["devanagari"] == "मोहब्बत" and r.meta["roman"] == "mohabbat"


def test_claude_translit_parses_payload(monkeypatch):
    _fake_anthropic(monkeypatch, {"devanagari": "मैं ठीक हूँ",
                                  "roman": "main theek hoon", "notes": ""})
    from providers.translit.claude import PROVIDER
    from providers.base import TranslitOpts
    r = PROVIDER.translit("میں ٹھیک ہوں", TranslitOpts())
    assert r.ok is True and r.devanagari == "मैं ठीक हूँ" and r.roman == "main theek hoon"


def test_parse_json_strips_markdown_fence():
    import providers._anthropic_common as common
    assert common._parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert common._parse_json('here you go: {"b": 2} thanks') == {"b": 2}
