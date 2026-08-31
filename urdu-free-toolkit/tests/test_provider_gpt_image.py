import base64
import io
import types

from PIL import Image

from providers.base import Capability


def _png_bytes(size=(16, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


def test_render_info_and_gate(monkeypatch):
    import providers._openai_common as common
    monkeypatch.setattr(common, "have_key", lambda: False)
    from providers.render.gpt_image import PROVIDER
    assert PROVIDER.info.capability == Capability.RENDER
    ok, reason = PROVIDER.available()
    assert ok is False and "OPENAI_API_KEY" in reason


def test_render_returns_png(monkeypatch):
    import providers._openai_common as common
    src = _png_bytes()

    class _Data:
        b64_json = base64.b64encode(src).decode()

    class _Resp:
        data = [_Data()]

    class _Images:
        @staticmethod
        def edit(**kw):
            return _Resp()

    monkeypatch.setattr(common, "have_key", lambda: True)
    monkeypatch.setattr(common, "get_client",
                        lambda: types.SimpleNamespace(images=_Images()))
    from providers.render.gpt_image import PROVIDER
    r = PROVIDER.render(src, ["mohabbat ek ehsaas hai"])
    assert r.ok is True and r.png[:8] == b"\x89PNG\r\n\x1a\n"
