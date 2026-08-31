from providers.translit.rule import PROVIDER
from providers.base import TranslitOpts, Capability


def test_info():
    assert PROVIDER.info.id == "rule"
    assert PROVIDER.info.capability == Capability.TRANSLIT
    assert PROVIDER.available() == (True, "")


def test_translit_known_words():
    r = PROVIDER.translit("میں ٹھیک ہوں", TranslitOpts())
    assert r.ok is True
    assert "में" in r.devanagari or "मैं" in r.devanagari  # rule engine defaults to "mein"
    assert r.roman.startswith(("mein", "main"))
    assert r.ms >= 0
