import json
import types


def _fake_groq(monkeypatch, payload: dict):
    """Patch _groq_common so no real client / key / network is touched."""
    import providers._groq_common as gq

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
    monkeypatch.setattr(gq, "get_client", lambda: client)
    monkeypatch.setattr(gq, "have_key", lambda: True)


def test_groq_translit_unavailable_without_key(monkeypatch):
    import providers._groq_common as gq
    monkeypatch.setattr(gq, "have_key", lambda: False)
    from providers.translit.groq import PROVIDER
    ok, _reason = PROVIDER.available()
    assert ok is False


def test_groq_translit_parses_payload(monkeypatch):
    _fake_groq(monkeypatch, {"devanagari": "मैं ठीक हूँ",
                             "roman": "main theek hoon", "notes": ""})
    from providers.translit.groq import PROVIDER
    from providers.base import TranslitOpts
    r = PROVIDER.translit("میں ٹھیک ہوں", TranslitOpts())
    assert r.ok is True and r.devanagari == "मैं ठीक हूँ" and r.roman == "main theek hoon"


def test_groq_parse_strips_markdown_fence():
    import providers._groq_common as gq
    assert gq._loads('```json\n{"a": 1}\n```') == {"a": 1}
    assert gq._loads('sure: {"b": 2}') == {"b": 2}
