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
curl -L -o aksharantar_hin.zip https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/hin.zip
# Dakshina: stream the 2 GB tarball, keep only Urdu
curl -L https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar \
  | tar -x --wildcards 'dakshina_dataset_v1.0/ur/*'
```

Plus the Hindi Wikipedia titles that link to an Urdu article (~56k pairs, ~6 min):

```bash
python -m training.translit.silver fetch-wiki   # -> data/translit_raw/wiki_hi_ur_titles.tsv
```

## 2. Build the dataset

```bash
python -m training.translit.build_data        # ~12 min -> data/translit_ds/  (--no-silver: ~1 min)
```

Produces the multi-task training set (`j` urdu->"R|devanagari", `r`
urdu->R, `c` urdu->casual roman, `h` R->devanagari), a held-out gold test set
split by lemma, and the gold lexicon (`lexicon_full.json`, `lexicon_train.json`).
Hindi-Wiktionary words get Urdu spellings from `urdu_nn.scheme.deva_to_urdu_candidates`
and are kept only when Dakshina's Urdu Wikipedia attests the spelling.
The reviewed exact dictionary (section 5) adds its readings as `j` rows.

`silver.tsv` holds silver `j` pairs for Urdu words no gold source covers
(`training/translit/silver.py`): an Urdu word and a Hindi word that people
romanise identically (Aksharantar), where the Hindi letters really spell the
Urdu word (`deva_to_urdu_candidates`), give the Urdu word's vowelled
Devanagari; its R reading comes from `deva_to_rich_candidates`, the schwa
variant picked by those same romanisations. Wikipedia title words add names.
Against the gold lexicon the kept tiers read like gold ~84-94% of the time
(`python -m training.translit.silver` prints the precision report).

## 3. Train

```bash
python -m training.translit.train --epochs 40   # best checkpoint -> data/translit_model/model.pt
```

Candidates train into their own folder and only replace the shipped model
after the benchmark (section 4) says they beat it:

```bash
# stage 1: 25M-param model from scratch on gold + dictionary + silver (~2 h on an RTX 3050)
python -m training.translit.train --d-model 384 --layers 6 --epochs 30 --lr 7e-4     --out data/translit_model/runs/big1
# stage 2: continue at a lower rate (--init takes any checkpoint; its size wins)
python -m training.translit.train --init data/translit_model/runs/big1/model.pt     --epochs 10 --lr 2e-4 --warmup 200 --label-smoothing 0.05 --out data/translit_model/runs/big2
python -m training.translit.evaluate --systems neural+dict --words 0 --sents 300     --model data/translit_model/runs/big2/model.pt --bench-out /tmp/bench.json
```

`--silver-per-epoch 0` trains on gold only.

## 4. Ship the runtime tables + benchmark

```bash
python -m training.translit.ship      # lexicon / evidence / English / Hindi-forms tables -> data/translit_model/
python -m training.translit.evaluate --systems rule,neural,gpt --words 500 --sents 120
```

The benchmark scores held-out Wiktionary words (Devanagari exact + style-folded
Roman) and Dakshina test sentences with human-typed Roman. The neural system
is evaluated with `lexicon_train` (test words removed), so its score is on
words it has never seen. GPT outputs are cached in `data/translit_model/gpt_cache_*.json`.

The runtime reranks the model's beam (`NeuralTransliterator._rerank`) with
human romanisations of the word when there are any, else the model's own
casual-Roman beam, plus a bonus for readings that are real Hindi words
spelling that Urdu word (`hindi_forms.json.gz`). `--casual-weight 0
--hindi-bonus 0` benchmarks without the last two.

## 5. The exact dictionary (top 10,000 words, reviewed)

`data/translit_dictionary.tsv` holds the 10,000 most frequent Urdu word forms
(about 92% of running Urdu text), each with a reviewed Devanagari, plain Roman
and Rekhta-style Roman spelling. Every offline engine looks a word up there
before guessing: `neural` right after its sentence-context rules
(میں main/mein, کیا kyā/kiyā, بن, سو, جلد, گر, کل), the rule engine (and so
uroman / aksharamukha) right after its curated `COMMON_WORDS`. Words written
with harakat skip it in the rule engine: the writer spelled the vowels out.

```bash
python -m training.translit.build_dictionary draft   # ~7 min: frequency list + engine readings
python -m training.translit.build_dictionary build   # draft + dictionary_fixes.txt -> the TSV
```

Corrections live in `training/translit/dictionary_fixes.txt`, one per line:
`urdu | devanagari | R [| plain [| rekhta]]` (see the file header). To fix a
word, add or edit its line and run `build`; never hand-edit the TSV. The builder
also normalises Devanagari nasals to the modern anusvara spelling
(हिन्दी -> हिंदी).

Benchmark with and without it (held-out words are removed from the dictionary
for the word test, so no system is scored on a word it was handed):

```bash
python -m training.translit.evaluate --systems rule,rule+dict,neural,neural+dict --words 0 --sents 0
```

## 6. GPT-distilled tier (match the GPT column exactly)

`neural` consults `data/translit_model/gpt.json.gz` before anything else: a line
GPT has already read comes back verbatim, and otherwise GPT's own context
readings, majority word spellings (per script) and joins (hyphens, unwritten
izafat `taaqat-e-bedaad`, pairs written as one word `iska`, `royenge`) win over
the dictionary / lexicon / model. The tables come from running real text
through the app's own GPT call (same model, `TEXT_SYSTEM` prompt, sampling):

```bash
python -m training.translit.gpt_distill fetch              # Urdu Wikisource <poem> pages -> data/translit_raw/
python -m training.translit.gpt_distill label --calls 300  # needs OPENAI_API_KEY; cached, only new chunks are paid for
python -m training.translit.gpt_distill build              # -> data/translit_model/gpt.json.gz
python -m training.translit.gpt_distill eval               # neural vs GPT on held-out chunks (tables from train only)
python -m training.translit.gpt_distill eval --no-tables   # the same without the tier (baseline)
```

Chunks are whole ghazals (<= 16 lines) or 8 Dakshina prose sentences; ~8% of
them (by poem title) are held out for `eval`. Replies are cached in
`data/translit_ds/gpt_labels.jsonl`, so `build` / `eval` never call the API.
