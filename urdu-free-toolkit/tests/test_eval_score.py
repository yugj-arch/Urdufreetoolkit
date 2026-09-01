import pytest

from eval.score import cer, wer, score_pair, aggregate


def test_cer_identical_is_zero():
    assert cer("ہے", "ہے") == 0.0


def test_cer_one_substitution():
    assert cer("abcd", "abxd") == pytest.approx(0.25)


def test_wer_counts_word_edits():
    assert wer("a b c", "a x c") == pytest.approx(1 / 3)


def test_score_pair_normalization_erases_encoding_only_diff():
    # ك (U+0643) vs ک (U+06A9): raw differs, normalized identical
    row = score_pair("كتاب", "کتاب")
    assert row["cer"] > 0
    assert row["cer_norm"] == 0.0


def test_aggregate_means():
    rows = [{"cer": 0.0, "wer": 0.0, "cer_norm": 0.0, "wer_norm": 0.0},
            {"cer": 0.5, "wer": 1.0, "cer_norm": 0.1, "wer_norm": 0.2}]
    agg = aggregate(rows)
    assert agg["cer"] == pytest.approx(0.25) and agg["n"] == 2


def test_aggregate_empty():
    assert aggregate([])["n"] == 0
