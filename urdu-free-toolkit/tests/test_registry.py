from providers import registry
from providers.base import Capability


def setup_function():
    registry.reset_cache()


def teardown_function():
    registry.reset_cache()


def test_discover_loads_fake_package():
    reg = registry.discover(package="tests.fakes")
    ids = {p.info.id for p in reg.values()}
    assert "fake_ok" in ids and "fake_no" in ids


def test_get_resolves_by_capability_and_id():
    registry.discover(package="tests.fakes")
    p = registry.get(Capability.OCR, "fake_ok")
    assert p.info.label == "Fake OK"


def test_get_unknown_raises():
    registry.discover(package="tests.fakes")
    try:
        registry.get(Capability.OCR, "nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_for_ui_badges():
    registry.discover(package="tests.fakes")
    rows = {r["id"]: r for r in registry.for_ui(Capability.OCR)}
    assert rows["fake_ok"]["available"] is True
    assert rows["fake_ok"]["badge"] == "offline"
    assert rows["fake_no"]["available"] is False
    assert rows["fake_no"]["badge"] == "needs key"
    assert rows["fake_no"]["reason"] == "FAKE_KEY not set"


def test_for_ui_filters_by_capability():
    registry.discover(package="tests.fakes")
    rows = registry.for_ui(Capability.OCR)
    assert rows and all(r["capability"] == "ocr" for r in rows)
    assert "tr_fake" not in {r["id"] for r in rows}


def test_for_ui_shows_only_curated_engines_in_order():
    registry.discover(package="providers")
    from providers import curation
    for cap in Capability:
        rows = registry.for_ui(cap)
        ids = [r["id"] for r in rows]
        curated_installed = [i for i in curation.FEATURED.get(cap.value, []) if i in ids]
        if curated_installed:  # (fallback path only triggers when none are installed)
            assert ids == curated_installed
        for r in rows:
            assert "price" in r


def test_for_ui_include_hidden_is_a_superset():
    registry.discover(package="providers")
    shown = registry.for_ui(Capability.OCR)
    everything = registry.for_ui(Capability.OCR, include_hidden=True)
    assert len(everything) >= len(shown)
    assert {r["id"] for r in shown} <= {r["id"] for r in everything}
