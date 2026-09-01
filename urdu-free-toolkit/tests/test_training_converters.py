from training.easyocr.to_lmdb import gt_to_dtrb_txt
from training.paddle.to_paddle_rec import convert


def _gt(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    (d / "gt.txt").write_text("000000.png\tہے\n000001.png\tٹھیک\n", encoding="utf-8")
    (d / "000000.png").write_bytes(b"x")
    (d / "000001.png").write_bytes(b"x")
    return d / "gt.txt", d


def test_paddle_rec_label_format(tmp_path):
    gt, d = _gt(tmp_path)
    out = tmp_path / "paddle_train.txt"
    n = convert(gt, out, image_root="dataset/urdu")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert n == 2
    assert lines[0] == "dataset/urdu/000000.png\tہے"


def test_dtrb_txt_format(tmp_path):
    gt, d = _gt(tmp_path)
    out = tmp_path / "dtrb_gt.txt"
    n = gt_to_dtrb_txt(gt, out, image_dir_prefix="train")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert n == 2 and lines[0].split("\t")[0].endswith("train/000000.png")
    assert lines[0].split("\t")[1] == "ہے"
