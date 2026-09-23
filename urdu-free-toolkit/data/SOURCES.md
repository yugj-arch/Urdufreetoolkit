# data/urdu_lexicon.json.gz -- source & provenance

This file documents where the bundled "breadth layer" lexicon
(`data/urdu_lexicon.json.gz`, tier 2 of `transliterate.py`'s lookup --
see `_load_lexicon` / `_LEXICON` there) comes from, its license, and how
to rebuild or extend it.

## Source

- **Site:** [en.wiktionary.org](https://en.wiktionary.org)
- **Category:** [Category:Urdu lemmas](https://en.wiktionary.org/wiki/Category:Urdu_lemmas)
  (~7,900 headwords at time of pull)
- **License:** Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0) and GNU Free Documentation License (GFDL), per the
  [Wiktionary copyright terms](https://en.wiktionary.org/wiki/Wiktionary:Copyrights).
  Both licenses require attribution and, for CC BY-SA, ShareAlike on
  redistributed/derived text -- this document and the generating script
  (`scripts/build_lexicon.py`) are that attribution, and the derived data
  file is itself offered under the same CC BY-SA 4.0 terms.
- **What was explicitly NOT used:** rekhta.org or any other site
  publishing copyrighted poetry/song lyrics/literary transliterations.
  Only Wiktionary's dictionary headword data was fetched. See the
  module docstring in `transliterate.py` and the project's own
  contribution notes for why.

## Date pulled

2026-09-23 (America/Denver / UTC, per the MediaWiki API responses at
the time this script was run -- re-run `scripts/build_lexicon.py` and
update this line if you regenerate the file from scratch).

## Extraction method

1. `action=query&generator=categorymembers&gcmtitle=Category:Urdu lemmas`
   (paginated via `gcmcontinue`, 500 titles/page) enumerates every page
   in the category.
2. Titles containing a space are dropped: this app's `transliterate_word_with`
   only ever looks up one *contiguous run of Perso-Arabic characters* at a
   time (see `WORD_RE` in `transliterate.py`), so a multi-word phrase key
   (e.g. "آئس کریم") could never match a real lookup and is useless weight.
3. For each remaining single-word title, `action=parse&page=<title>&prop=text`
   renders the page to HTML. The HTML is split into per-language sections
   on `<h2 id="...">`, and the Urdu section's own headword line
   (`<span class="headword-line">...<span class="headword-tr tr Latn">`)
   is read for its Latin-script transliteration. This is scoped strictly
   to the headword line -- earlier attempts that searched the whole Urdu
   section for any `tr Latn` span sometimes picked up a synonym's or an
   etymology ancestor's transliteration instead of the entry's own.
4. Wiktionary's academic Urdu-Latin scheme (macrons `ā ī ū`, a combining
   tilde for nasalisation, `ś` for ش, bare `x`/`c` for خ/چ, dotted/underlined
   emphatics for retroflex/pharyngeal letters, ...) is mechanically folded
   down to this project's plain-ASCII Roman convention (long vowels
   doubled -- `aa`/`ii`/`oo` -- digraphs `kh`/`gh`/`sh`/`ch`, nasalisation
   folded to a trailing `n`). See `normalize_roman()` in
   `scripts/build_lexicon.py`.
5. A Devanagari spelling is then derived **from that normalised Roman
   string** (not from the raw Urdu spelling) with a small deterministic
   Roman-to-Devanagari syllabifier, `roman_to_devanagari()` in the same
   script. Deriving it from Roman rather than Urdu matters: Urdu's
   unwritten short vowels are exactly as ambiguous for Devanagari as for
   Roman (both need an explicit vowel choice) -- e.g. running raw کتاب
   through the existing character-level fallback yields "कताब" (wrong,
   "kataab"), whereas converting the correct Roman "kitab" gives "किताब".
   This mechanical Devanagari does not attempt to recover retroflex/
   dental or nukta/non-nukta distinctions that the plain Roman string
   itself has already collapsed (e.g. "kh" could be ख़ or ख; it defaults
   to the non-nukta form ख/ग/...). It is a reasonable, deterministic
   fallback, not a linguistically perfect transcription.
6. Both `[devanagari, roman]` are written to
   `data/urdu_lexicon.json.gz` keyed by the (NFC-normalized) Urdu
   spelling, matching the `{urdu_spelling: [devanagari, roman]}` format
   `_load_lexicon()` expects.

## Entry count

See the `entry_count` recorded by the script's final log line
(`wrote N entries to data/urdu_lexicon.json.gz`) for the exact figure at
build time -- run:

```
python -c "import transliterate; print(len(transliterate._LEXICON))"
```

to check the count actually loaded by the app.

## How to re-run / extend

```
# grow the lexicon further (resumes from the existing checkpoint, only
# fetches titles not already attempted):
python scripts/build_lexicon.py --limit 12000

# start completely over:
python scripts/build_lexicon.py --no-resume --limit 6000
```

The script paginates the category listing itself (cached to
`data/.lexicon_titles.json`) and checkpoints per-title progress to
`data/.lexicon_checkpoint.json` after every page (both gitignored --
they're local working state, not shipped data), so an interrupted run
can simply be re-launched with the same arguments. It is polite to the
MediaWiki API by default: single-threaded, paced at `--delay` seconds
(default 1.0s) between page fetches, with exponential backoff and
`Retry-After` handling on 429/5xx responses. The `User-Agent` sent
(`UrduFreeToolkitLexiconBot/1.0 (offline lexicon builder; github.com repo)`)
carries no personal contact information.

Category:Urdu lemmas has roughly 7,900 entries in total (minus the
space-containing phrases filtered out in step 2 above); re-running with a
higher `--limit` will keep pulling from the ones not yet attempted until
the category is exhausted.

---

# data/urdu_lexicon_llm.json.gz -- source & provenance

The second, complementary "breadth layer" lexicon -- same
`{urdu_spelling: [devanagari, roman]}` shape, merged with the file above at
load time (`transliterate._LEXICON`, Wiktionary wins on overlap).

## Why a second file

`urdu_lexicon.json.gz` (above) only covers Wiktionary's dictionary
*citation forms* -- e.g. it has the lemma but not necessarily every common
inflected/plural/oblique form a real sentence actually uses. This file
instead targets real running-text **frequency**: rank Urdu Wikipedia
vocabulary by how often it actually occurs, then resolve each word's short
vowels with an LLM instead of a second dictionary source.

## Source & method

1. **Word list:** `training/corpus/fetch_corpus.py --source wikipedia`
   pulls plain-text extracts live from the `ur.wikipedia.org` API
   (`action=query&generator=random&prop=extracts`) -- CC BY-SA 3.0,
   attribution "Wikipedia contributors, ur.wikipedia.org". Same
   not-Rekhta constraint as the file above: only Wikipedia's own API is
   used, see the `transliterate.py` module docstring.
2. **Ranking:** `scripts/llm_distill_lexicon.py` tokenizes that corpus,
   counts word frequency, and drops anything already covered by
   `transliterate.COMMON_WORDS` or either existing bundled lexicon -- so
   this file only ever adds words, never re-derives ones we already have
   an exact answer for.
3. **Resolution:** the remaining words, most-frequent first, are sent to
   the OpenAI API (same client/model the app's own `gpt` transliteration
   provider uses -- see `providers/_openai_common.py`) in numbered
   batches of 40, with a system prompt that asks for each word's
   Devanagari + plain-ASCII-Roman reading in this project's exact
   convention (long vowels doubled, `kh`/`gh`/`sh`/`ch` digraphs, trailing
   `n` for nasalisation -- the same target `normalize_roman()` produces
   for the Wiktionary file).
4. Written to `data/urdu_lexicon_llm.json.gz`, keyed by (NFC-normalized)
   Urdu spelling.

Unlike the Wiktionary file, entries here are an LLM's single best reading,
not a human-authored dictionary source -- that is why `transliterate.py`
merges the Wiktionary file on top (it wins on any key collision).

## How to re-run / extend

```
# 1. (re)build the word-frequency source corpus (regenerable, gitignored):
python -m training.corpus.fetch_corpus \
    --out training/corpus/sentences_wiki.txt --source wikipedia --limit 4000

# 2. distill the top N still-uncovered words:
python scripts/llm_distill_lexicon.py --limit 3000
python scripts/llm_distill_lexicon.py --limit 8000   # resumes, does more
```

Needs `OPENAI_API_KEY` (loaded from `.env`, same as the app). Checkpointed
to `data/.lexicon_llm_checkpoint.json` (gitignored) after every batch, so
an interrupted run can simply be re-launched with the same arguments.

---

# data/translit_model/ -- the neural transliterator (model.pt + lexicon.json.gz)

Trained by `training/translit/` (runbook: `training/translit/README.md`),
loaded at runtime by `neural_translit.py` (provider id `neural`). No LLM
output was used as training data -- every label is human-authored.

| Source | What we use | License / attribution |
|---|---|---|
| [kaikki.org](https://kaikki.org) Wiktionary extracts -- Urdu and Hindi dictionaries (wiktextract) | headwords, inflection tables, romanisations, Hindi/Urdu cross-spellings, usage examples | CC BY-SA 4.0 + GFDL, "Wiktionary contributors" |
| [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar) (AI4Bharat), Urdu split | ~700k Urdu word -> casual Roman pairs (auxiliary task, homograph tie-breaks) | CC BY 4.0, "AI4Bharat, Madhani et al. 2022" |
| [Dakshina](https://github.com/google-research-datasets/dakshina) (Google Research), Urdu | romanisation lexicon, 10k word-aligned romanised sentences (test split held out for eval), Wikipedia text as the attestation corpus for generated Urdu spellings | CC BY-SA 4.0, "Roark et al. 2020" |

`lexicon.json.gz` (Urdu spelling -> [Devanagari, R-reading]) and the model
weights are derived works of the above and are offered under CC BY-SA 4.0.
Rekhta or any other copyrighted poetry site was NOT used.
