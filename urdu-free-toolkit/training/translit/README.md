# Neural Urdu transliterator -- training runbook

Offline, free Urdu -> Devanagari + Roman (plain `kitaab` and Rekhta-style
`kitāb`). Runtime: `neural_translit.py` / provider `neural`. Everything here
is CPU-side except `train.py`, which uses a GPU when present (an RTX 3050
laptop does a full run in about an hour; CPU works but is slow).

## 1. Raw data (free, human-authored -- see `data/SOURCES.md`)

Put these in `data/translit_raw/` (gitignored):

```bash
cd urdu-free-toolkit/data/translit_raw
curl -L -o kaikki_ur.jsonl https://kaikki.org/dictionary/Urdu/kaikki.org-dictionary-Urdu.jsonl
curl -L -o kaikki_hi.jsonl https://kaikki.org/dictionary/Hindi/kaikki.org-dictionary-Hindi.jsonl
curl -L -o aksharantar_urd.zip https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/urd.zip
# Dakshina: stream the 2 GB tarball, keep only Urdu
curl -L https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar \
  | tar -x --wildcards 'dakshina_dataset_v1.0/ur/*'
```

## 2. Build the dataset

```bash
python -m training.translit.build_data        # ~1 min -> data/translit_ds/
```

Produces the multi-task training set (`j` urdu->"R|devanagari", `r`
urdu->R, `c` urdu->casual roman, `h` R->devanagari), a held-out gold test set
split by lemma, and the gold lexicon (`lexicon_full.json`, `lexicon_train.json`).
Hindi-Wiktionary words get Urdu spellings from `urdu_nn.scheme.deva_to_urdu_candidates`
and are kept only when Dakshina's Urdu Wikipedia attests the spelling.

## 3. Train

```bash
python -m training.translit.train --epochs 40   # best checkpoint -> data/translit_model/model.pt
```

## 4. Ship the runtime tables + benchmark

```bash
python -m training.translit.ship      # lexicon / evidence / English tables -> data/translit_model/
python -m training.translit.evaluate --systems rule,neural,gpt --words 500 --sents 120
```

The benchmark scores held-out Wiktionary words (Devanagari exact + style-folded
Roman) and Dakshina test sentences with human-typed Roman. The neural system
is evaluated with `lexicon_train` (test words removed), so its score is on
words it has never seen. GPT outputs are cached in `data/translit_model/gpt_cache_*.json`.
