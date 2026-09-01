"""Regressions in the PaddleOCR engine construction.

1. Language code: PaddleOCR 2.x accepted the bucket string ``lang="arabic"``;
   3.x replaced it with ISO-style codes and raises
   ``ValueError: No models are available for lang='arabic' ...`` instead.
   ``_get_engine`` only catches ``TypeError``, so a wrong ``lang`` turns every
   PaddleOCR row into a failure.

2. ``enable_mkldnn=False``: paddlepaddle 3.3.x's PIR executor crashes inside the
   oneDNN backend (``ConvertPirAttribute2RuntimeAttribute not support``) on
   these models. The plain CPU path is slower but portable.

3. Renamed tuning kwarg: PaddleOCR 3.7 renamed ``drop_score`` to
   ``text_rec_score_thresh`` (confirmed against the installed 3.7.0 via a real
   end-to-end eval run — every row failed with
   ``ValueError: Unknown argument: drop_score``). Because 3.x's ``__init__``
   takes ``**kwargs``, ``_supported_kwargs`` cannot filter this away (no
   ``TypeError``); ``_get_engine`` falls back to a small version-stable kwarg
   set instead.
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


def test_falls_back_when_var_keyword_init_rejects_kwarg_internally(monkeypatch):
    """PaddleOCR 3.7: __init__ has **kwargs (no TypeError to catch) but raises
    its own ValueError for a kwarg it doesn't recognize. _get_engine must retry
    with the small safe set instead of failing every row."""
    import providers.ocr.paddle as paddle

    attempts: list[dict] = []
    _KNOWN = {"lang", "enable_mkldnn", "use_textline_orientation"}

    class _FakePaddleOCR37:
        def __init__(self, **kw):
            attempts.append(dict(kw))
            unknown = set(kw) - _KNOWN
            if unknown:
                raise ValueError(f"Unknown argument: {sorted(unknown)[0]}")

    monkeypatch.setattr(paddle, "PaddleOCR", _FakePaddleOCR37)
    monkeypatch.setattr(paddle, "_engine", None)

    engine = paddle._get_engine()

    assert engine is not None
    assert len(attempts) >= 2, "expected a first failing attempt then a safe retry"
    assert attempts[0].get("text_rec_score_thresh") == 0.4   # the tuned kwarg was tried first
    assert set(attempts[-1]) <= _KNOWN                        # the retry stayed inside the safe set
