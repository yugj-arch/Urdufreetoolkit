from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_static_files_exist():
    assert (ROOT / "static/app.css").is_file()
    assert (ROOT / "static/app.js").is_file()


def test_index_references_static_and_no_cdn():
    html = (ROOT / "templates/index.html").read_text(encoding="utf-8")
    assert "app.js" in html and "app.css" in html
    assert "http://" not in html and "https://" not in html  # no CDN / external hosts


def test_app_js_hits_endpoints():
    js = (ROOT / "static/app.js").read_text(encoding="utf-8")
    for ep in ("/api/providers", "/api/ocr", "/api/transliterate", "/api/batch", "/api/settings"):
        assert ep in js


def test_app_js_reads_runtime_config():
    js = (ROOT / "static/app.js").read_text(encoding="utf-8")
    assert "/api/config" in js
    # the Batch cap comes from the server, not a hard-coded 30
    assert "batch_max_files" in js
    # the Settings panel reacts to the read-only flag Vercel sets
    assert "settings_readonly" in js
