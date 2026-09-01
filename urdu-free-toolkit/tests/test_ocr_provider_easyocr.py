import io

from PIL import Image


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (80, 40), "white").save(buf, format="PNG")
    return buf.getvalue()


class _FakeReader:
    def __init__(self, found):
        self._found = found

    def readtext(self, arr, **kw):
        return self._found


def test_easyocr_ocr_returns_pipeline_result(monkeypatch):
    import providers.ocr.easyocr_p as mod

    found = [
        ([[60, 5], [78, 5], [78, 22], [60, 22]], "الف", 0.9),   # right
        ([[5, 5], [30, 5], [30, 22], [5, 22]], "بے", 0.8),       # left
    ]
    monkeypatch.setattr(mod, "easyocr", object())               # pass availability gate
    monkeypatch.setattr(mod, "_get_reader", lambda: _FakeReader(found))

    r = mod.PROVIDER.ocr(_png())
    assert r.ok is True
    assert r.provider_id == "easyocr"
    assert r.text.split() == ["الف", "بے"]                       # RTL: rightmost first
    assert "devanagari" in r.meta and "roman" in r.meta
    assert r.meta["reading_order"] == "rtl-pipeline"
    assert isinstance(r.boxes, list) and r.boxes and set(r.boxes[0]) == {"box", "text", "conf"}


def test_easyocr_missing_lib_is_graceful(monkeypatch):
    import providers.ocr.easyocr_p as mod
    monkeypatch.setattr(mod, "easyocr", None)
    r = mod.PROVIDER.ocr(_png())
    assert r.ok is False and "not installed" in r.error
