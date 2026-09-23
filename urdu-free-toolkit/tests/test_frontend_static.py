from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_static_files_exist():
    assert (ROOT / "static/app.css").is_file()
    assert (ROOT / "static/app.js").is_file()


def test_index_references_static_and_no_cdn():
    html = (ROOT / "templates/index.html").read_text(encoding="utf-8")
    assert "app.js" in html and "app.css" in html
    # no external resources (scripts / stylesheets / fonts) loaded from CDN
    import re
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=["\']https?://', html, re.I)


def test_app_js_hits_endpoints():
    js = (ROOT / "static/app.js").read_text(encoding="utf-8")
    for ep in ("/api/providers", "/api/ocr", "/api/transliterate", "/api/settings"):
        assert ep in js


def test_app_js_reads_runtime_config():
    js = (ROOT / "static/app.js").read_text(encoding="utf-8")
    assert "/api/config" in js
    # the Settings panel reacts to the read-only flag Vercel sets
    assert "settings_readonly" in js


def test_engine_pickers_are_present_and_wired():
    html = (ROOT / "templates/index.html").read_text(encoding="utf-8")
    js = (ROOT / "static/app.js").read_text(encoding="utf-8")
    # each step has a multi-select engine list + the quick-pick chips
    assert 'id="ocr-engines"' in html and 'id="translit-engines"' in html
    assert 'data-pick="ocr:recommended"' in html and 'data-pick="ocr:none"' in html
    # the list is filled from /api/providers, the picked set is remembered,
    # and more than one engine can be ticked per step
    assert "renderEngineList" in js
    assert "urdu.engines" in js
    assert "state.engines" in js


def test_reader_compares_engines():
    html = (ROOT / "templates/index.html").read_text(encoding="utf-8")
    js = (ROOT / "static/app.js").read_text(encoding="utf-8")
    # the ghazal reader can hold several engines' results and switch between them
    assert 'id="reader-engines"' in html
    assert "state.results" in js and "activeEngine" in js


def test_diacritics_toggle_is_present_and_wired():
    html = (ROOT / "templates/index.html").read_text(encoding="utf-8")
    js = (ROOT / "static/app.js").read_text(encoding="utf-8")
    # the Plain / Diacritics switch lives in the reader control bar
    assert "dia-switch" in html
    # the dead "Roman style" selects are gone, replaced by the toggle
    assert "roman-style" not in html and "batch-roman-style" not in html
    # each result carries both Roman spellings; flipping the switch repaints the
    # reader from stored lines with no re-fetch
    assert "roman_diacritic" in js
    assert "urdu.diacritics" in js and "repaintReader" in js
