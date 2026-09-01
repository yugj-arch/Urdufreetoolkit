import numpy as np

import providers.ocr.paddle as paddle


def test_supported_kwargs_filters_unknown():
    class Fake:
        def __init__(self, *, lang=None, enable_mkldnn=None, text_det_thresh=None):
            pass

    got = paddle._supported_kwargs(Fake, {
        "lang": "ur", "enable_mkldnn": False, "text_det_thresh": 0.3,
        "bogus_kwarg": 1, "drop_score": 0.4,
    })
    assert got == {"lang": "ur", "enable_mkldnn": False, "text_det_thresh": 0.3}


def test_supported_kwargs_passes_all_when_var_keyword():
    class Fake:
        def __init__(self, *, lang=None, **kw):
            pass

    want = {"lang": "ur", "drop_score": 0.4, "anything": 1}
    assert paddle._supported_kwargs(Fake, want) == want


def test_extract_normalizes_dict_shape():
    result = [{
        "rec_texts": ["الف", "بے"],
        "rec_polys": [np.array([[60, 5], [78, 5], [78, 22], [60, 22]]),
                      np.array([[5, 5], [30, 5], [30, 22], [5, 22]])],
        "rec_scores": [0.9, 0.8],
    }]
    words = paddle._extract(result)
    assert [w.text for w in words] == ["الف", "بے"]
    assert words[0].conf == 0.9 and len(words[0].box) == 4
    assert words[0].box[0] == (60, 5)
