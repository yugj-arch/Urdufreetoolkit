"""Regressions in the PaddleOCR engine construction.

1. Language code: PaddleOCR 2.x accepted the bucket string ``lang="arabic"``;
   3.x replaced it with ISO-style codes and raises
   ``ValueError: No models are available for lang='arabic' ...`` instead.
   ``_get_engine`` only catches ``TypeError``, so a wrong ``lang`` turns every
   PaddleOCR row into a failure.

2. ``enable_mkldnn=False``: paddlepaddle 3.3.x's PIR executor crashes inside the
   oneDNN backend (``ConvertPirAttribute2RuntimeAttribute not support``) on
   these models. The plain CPU path is slower but portable.
"""
from __future__ import annotations

# Codes PaddleOCR 3.x resolves to the Arabic-script recognizer (covers Urdu).
_OK_LANGS = {"ur", "ar", "fa"}


def test_paddle_engine_construction_is_version_safe(monkeypatch):
    import providers.ocr.paddle as paddle

    calls: list[dict] = []

    class _FakePaddleOCR:
        def __init__(self, *, lang=None, **kw):
            calls.append({"lang": lang, **kw})
            if lang not in _OK_LANGS:
                raise ValueError(
                    f"No models are available for lang={lang!r} and ocr_version=None."
                )

    monkeypatch.setattr(paddle, "PaddleOCR", _FakePaddleOCR)
    monkeypatch.setattr(paddle, "_engine", None)

    engine = paddle._get_engine()

    assert engine is not None
    assert calls, "PaddleOCR was never constructed"
    assert calls[0]["lang"] in _OK_LANGS
    assert calls[0].get("enable_mkldnn") is False


def test_unknown_kwargs_never_reach_constructor(monkeypatch):
    """A version whose __init__ has no **kwargs must not receive tuning kwargs
    it doesn't declare (would be a TypeError -> every row fails)."""
    import providers.ocr.paddle as paddle

    seen = {}

    class _FakePaddleOCR:
        def __init__(self, *, lang=None, enable_mkldnn=None, use_textline_orientation=None):
            seen.update(lang=lang, enable_mkldnn=enable_mkldnn)

    monkeypatch.setattr(paddle, "PaddleOCR", _FakePaddleOCR)
    monkeypatch.setattr(paddle, "_engine", None)
    paddle._get_engine()
    assert seen["lang"] in {"ur", "ar", "fa"} and seen["enable_mkldnn"] is False
