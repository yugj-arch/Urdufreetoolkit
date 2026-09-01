from dataclasses import FrozenInstanceError

import pytest

from providers.ocr._types import Word, Line, OcrConfig, PipelineOut


def test_word_is_frozen():
    w = Word(text="ہے", box=[(0, 0), (10, 0), (10, 8), (0, 8)], conf=0.9)
    with pytest.raises(FrozenInstanceError):
        w.text = "x"


def test_ocrconfig_defaults():
    c = OcrConfig()
    assert c.grayscale is True
    assert c.binarize == "sauvola"
    assert c.deskew is True
    assert c.map_digits_to == "ascii"
    assert c.fold_arabic_heh is False
    assert c.spellfix is False
    assert c.engine == {}


def test_ocrconfig_engine_dict_is_per_instance():
    a, b = OcrConfig(), OcrConfig()
    assert a.engine is not b.engine


def test_pipelineout_shape():
    out = PipelineOut(text="ہے", lines=[], meta={"reading_order": "rtl-pipeline"})
    assert out.text == "ہے" and out.lines == [] and out.meta["reading_order"] == "rtl-pipeline"
