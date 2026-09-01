from providers.ocr._translit_fill import rule_translit


def test_blank_in_blank_out():
    assert rule_translit("   ") == ("", "")


def test_returns_two_strings_for_real_text():
    deva, roman = rule_translit("میں ٹھیک ہوں")
    assert isinstance(deva, str) and isinstance(roman, str)
    assert deva and roman


def test_never_raises_on_engine_error(monkeypatch):
    import providers.ocr._translit_fill as tf
    monkeypatch.setattr(tf._rule, "transliterate",
                        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("x")))
    assert rule_translit("abc") == ("", "")
