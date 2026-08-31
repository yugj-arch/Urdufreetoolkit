from providers.base import (
    Capability, ProviderInfo, OcrResult, TranslitResult, TranslitOpts, BaseProvider,
)


def test_capability_values():
    assert Capability.OCR == "ocr"
    assert Capability.TRANSLATE == "translate"


def test_result_defaults():
    r = OcrResult(provider_id="x")
    assert r.ok is True and r.error == "" and r.ms == 0 and r.text == "" and r.boxes is None
    assert r.meta == {}


def test_translit_opts_defaults():
    o = TranslitOpts()
    assert o.roman_style == "natural"
    assert o.targets == ("devanagari", "roman")


def test_base_provider_available_default():
    p = BaseProvider()
    p.info = ProviderInfo(id="x", label="X", capability=Capability.OCR, kind="offline")
    assert p.available() == (True, "")


def test_timed_catches_exception():
    p = BaseProvider()

    def boom():
        raise ValueError("nope")

    out = p._timed(boom)
    assert out["ok"] is False
    assert "nope" in out["error"]
    assert isinstance(out["ms"], int)
