import io

from PIL import Image


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    return buf.getvalue()


_PNG = _png()

# One image's worth of a DOCUMENT_TEXT_DETECTION response: annotation [0] is the
# whole block, [1:] are per-word with vertices.
_FAKE_RESPONSE = {
    "fullTextAnnotation": {"text": "عید مبارک\nمحبت\n"},
    "textAnnotations": [
        {"locale": "ur", "description": "عید مبارک\nمحبت"},
        {"description": "عید", "boundingPoly": {"vertices": [
            {"x": 10, "y": 4}, {"x": 40, "y": 4}, {"x": 40, "y": 22}, {"x": 10, "y": 22}]}},
        {"description": "مبارک", "boundingPoly": {"vertices": [
            {"x": 42, "y": 4}, {"x": 90, "y": 4}, {"x": 90, "y": 22}, {"x": 42, "y": 22}]}},
    ],
}


def test_gcv_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_VISION_KEY", raising=False)
    from providers.ocr.google_vision import PROVIDER
    ok, reason = PROVIDER.available()
    assert ok is False and "GOOGLE_VISION_KEY" in reason


def test_gcv_parses_text_and_boxes(monkeypatch):
    monkeypatch.setenv("GOOGLE_VISION_KEY", "test-key")
    import providers.ocr.google_vision as gv

    seen = {}

    def _fake_annotate(image_b64, api_key):
        seen["b64"], seen["key"] = image_b64, api_key
        return _FAKE_RESPONSE

    monkeypatch.setattr(gv, "_annotate", _fake_annotate)
    r = gv.PROVIDER.ocr(_PNG)

    assert r.ok is True
    assert r.text == "عید مبارک\nمحبت"
    assert seen["key"] == "test-key" and seen["b64"]
    assert [b["text"] for b in r.boxes] == ["عید", "مبارک"]
    assert r.boxes[0]["box"] == [[10, 4], [40, 4], [40, 22], [10, 22]]


def test_gcv_api_error_is_a_failed_row(monkeypatch):
    monkeypatch.setenv("GOOGLE_VISION_KEY", "test-key")
    import providers.ocr.google_vision as gv

    def _boom(image_b64, api_key):
        raise RuntimeError("HTTP 403: PERMISSION_DENIED")

    monkeypatch.setattr(gv, "_annotate", _boom)
    r = gv.PROVIDER.ocr(_PNG)

    assert r.ok is False and "403" in r.error
