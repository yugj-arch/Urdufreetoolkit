from training.corpus.fetch_corpus import clean_sentences, fetch


def test_clean_keeps_urdu_lines_in_word_bounds():
    raw = "\n".join([
        "یہ ایک اچھی مثال ہے",     # ok
        "too short",                # not urdu / too short
        "ایک",                      # 1 word -> drop
        "یہ ایک اچھی مثال ہے",     # dup -> drop
    ])
    out = clean_sentences(raw, min_words=3, max_words=18)
    assert out == ["یہ ایک اچھی مثال ہے"]


def test_fetch_sample_writes_file(tmp_path):
    out = tmp_path / "sentences.txt"
    n = fetch(out, source="sample")
    assert n >= 5 and out.read_text(encoding="utf-8").strip()
