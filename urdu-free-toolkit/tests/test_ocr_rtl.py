from providers.ocr._rtl import group_into_lines, order_lines, lines_to_text
from providers.ocr._types import Word, OcrConfig


def _w(text, x, y, w=20, h=16, conf=0.9):
    return Word(text=text, box=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)], conf=conf)


def test_words_within_a_line_are_rtl():
    # visually: [ C ][ B ][ A ]  -> reading order A B C (right to left)
    words = [_w("A", 200, 0), _w("B", 100, 0), _w("C", 0, 0)]
    lines = group_into_lines(words, OcrConfig())
    assert len(lines) == 1
    assert lines[0].text == "A B C"


def test_two_lines_split_vertically():
    words = [_w("A", 100, 0), _w("B", 0, 0), _w("C", 0, 40), _w("D", 100, 40)]
    lines = group_into_lines(words, OcrConfig())
    assert [l.text for l in lines] == ["A B", "D C"]


def test_two_columns_right_first():
    # left column x~0, right column x~500; Urdu reads right column first
    left = [_w("L1", 0, 0), _w("L2", 0, 40)]
    right = [_w("R1", 500, 0), _w("R2", 500, 40)]
    lines = order_lines(group_into_lines(left + right, OcrConfig()))
    assert [l.text for l in lines] == ["R1", "R2", "L1", "L2"]


def test_inline_latin_run_kept_ltr():
    # Urdu ... "OpenAI GPT" ... Urdu ; Latin island is visually OpenAI(left) GPT(right)
    words = [_w("یہ", 300, 0), _w("GPT", 200, 0), _w("OpenAI", 130, 0), _w("ہے", 0, 0)]
    lines = group_into_lines(words, OcrConfig())
    assert "OpenAI GPT" in lines[0].text
    assert lines[0].text.startswith("یہ")
    assert lines[0].text.endswith("ہے")


def test_lines_to_text_joins_with_newlines():
    words = [_w("A", 0, 0), _w("B", 0, 40)]
    assert lines_to_text(group_into_lines(words, OcrConfig())) == "A\nB"
