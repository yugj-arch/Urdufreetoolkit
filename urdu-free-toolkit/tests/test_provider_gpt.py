import io
import json
import types

from PIL import Image


def _fake_openai(monkeypatch, payload: dict):
    import providers._openai_common as common

    class _Msg:
        content = json.dumps(payload)

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Chat:
        class completions:
            @staticmethod
            def create(**kw):
                return _Resp()

    client = types.SimpleNamespace(chat=_Chat())
    monkeypatch.setattr(common, "get_client", lambda: client)
    monkeypatch.setattr(common, "have_key", lambda: True)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


_PNG_1x1 = _png_bytes()


def test_gpt_ocr_available_without_key(monkeypatch):
    import providers._openai_common as common
    monkeypatch.setattr(common, "have_key", lambda: False)
    from providers.ocr.gpt import PROVIDER
    ok, reason = PROVIDER.available()
    assert ok is False and "OPENAI_API_KEY" in reason


def test_gpt_ocr_parses_payload(monkeypatch):
    _fake_openai(monkeypatch, {"urdu": "محبت", "devanagari": "मोहब्बत",
                               "roman": "mohabbat", "notes": ""})
    from providers.ocr.gpt import PROVIDER
    r = PROVIDER.ocr(_PNG_1x1)
    assert r.ok is True and r.text == "محبت"
    assert r.meta["devanagari"] == "मोहब्बत" and r.meta["roman"] == "mohabbat"


def test_gpt_translit_parses_payload(monkeypatch):
    _fake_openai(monkeypatch, {"devanagari": "मैं ठीक हूँ",
                               "roman": "main theek hoon", "notes": ""})
    from providers.translit.gpt import PROVIDER
    from providers.base import TranslitOpts
    r = PROVIDER.translit("میں ٹھیک ہوں", TranslitOpts())
    assert r.ok is True and r.devanagari == "मैं ठीक हूँ" and r.roman == "main theek hoon"
