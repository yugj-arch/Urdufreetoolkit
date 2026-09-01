from providers.ocr import _model_hooks as mh


def test_easyocr_network_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.delenv("OCR_EASYOCR_RECOG_NETWORK", raising=False)
    monkeypatch.setattr(mh, "_EASYOCR_USER_DIR", tmp_path)
    assert mh.easyocr_recog_network() is None


def test_easyocr_network_from_env(monkeypatch, tmp_path):
    net = "urdu_ft"
    (tmp_path / f"{net}.py").write_text("# stub")
    monkeypatch.setattr(mh, "_EASYOCR_USER_DIR", tmp_path)
    monkeypatch.setenv("OCR_EASYOCR_RECOG_NETWORK", net)
    assert mh.easyocr_recog_network() == net


def test_easyocr_network_autodetected(monkeypatch, tmp_path):
    (tmp_path / "urdu_ft.py").write_text("# stub")
    (tmp_path / "urdu_ft.yaml").write_text("x: 1")
    monkeypatch.setattr(mh, "_EASYOCR_USER_DIR", tmp_path)
    monkeypatch.delenv("OCR_EASYOCR_RECOG_NETWORK", raising=False)
    assert mh.easyocr_recog_network() == "urdu_ft"


def test_paddle_rec_dir_only_when_real(monkeypatch, tmp_path):
    monkeypatch.delenv("OCR_PADDLE_REC_DIR", raising=False)
    assert mh.paddle_rec_dir() is None
    monkeypatch.setenv("OCR_PADDLE_REC_DIR", str(tmp_path / "nope"))
    assert mh.paddle_rec_dir() is None
    real = tmp_path / "rec"
    real.mkdir()
    monkeypatch.setenv("OCR_PADDLE_REC_DIR", str(real))
    assert mh.paddle_rec_dir() == str(real)
