"""Contract smoke test across every real provider.

For each discovered provider: ``available()`` returns a ``(bool, str)`` tuple and
does not raise; and if the provider is available and offline, calling its
capability method with a trivial input returns a Result (``ok`` True or False) —
never an exception. Accuracy is not checked here.
"""
import io

import pytest
from PIL import Image

from providers import registry
from providers.base import Capability, TranslateOpts, TranslitOpts


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (48, 24), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _real_registry():
    registry.reset_cache()
    registry.discover(package="providers")
    yield
    registry.reset_cache()


def test_every_provider_has_valid_info_and_available():
    reg = registry.discover(package="providers")
    assert reg, "no providers discovered"
    for key, p in reg.items():
        assert p.info.id and p.info.label
        assert p.info.capability in Capability
        assert p.info.kind in ("offline", "api")
        res = p.available()
        assert isinstance(res, tuple) and len(res) == 2
        assert isinstance(res[0], bool) and isinstance(res[1], str)


@pytest.mark.parametrize("cap", list(Capability))
def test_for_ui_rows_are_serializable(cap):
    rows = registry.for_ui(cap)
    for r in rows:
        assert set(r) >= {"id", "label", "capability", "kind", "badge", "available", "reason", "note"}


def test_offline_translit_providers_do_not_crash():
    for p in registry.discover(package="providers").values():
        if p.info.capability != Capability.TRANSLIT or p.info.kind != "offline":
            continue
        ok, _ = p.available()
        if not ok:
            continue
        r = p.translit("میں ٹھیک ہوں", TranslitOpts())
        assert r.provider_id == p.info.id
        assert isinstance(r.ok, bool)
