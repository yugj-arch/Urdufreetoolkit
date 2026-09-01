# Urdu Toolkit — pick your engines, compare, transliterate

A local web app that pulls **Urdu text out of images**, **transliterates** it to
Devanagari (Hindi script) and Roman, and lets you **run several engines side by
side and compare** the results.

**Free / offline first.** The offline engines need no API key and no internet
once installed. Paid / free-tier APIs (OpenAI, Anthropic, Google Gemini, Google
Cloud Vision) are *optional* extras you switch on in Settings.

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

## How it's built

Every step is a list of small **provider** modules under `providers/<capability>/`:

| Capability | What it does | Shown in the picker (curated — max 4 per step) |
|---|---|---|
| **OCR** | image → Urdu text | OpenAI GPT vision, Google Cloud Vision *(API)*, PaddleOCR, EasyOCR *(offline)* |
| **Transliteration** | Urdu → Devanagari + Roman | GPT *(API)*, Rule engine, Aksharamukha, uroman *(all offline, no key)* |

The picker shows a **curated shortlist** (at most four per step) with a rough
**price badge** on each engine — offline engines are free; API estimates assume
one short image / line and drift with provider pricing. Every other engine still
ships (Gemini vision, Claude vision + transliteration, Surya, RapidOCR, docTR,
TrOCR, Tesseract, uroman, PyICU) and is re-enabled by editing
`providers/curation.py`. API model ids are overridable via `OPENAI_MODEL`,
`ANTHROPIC_MODEL`, `GEMINI_MODEL`.

A registry discovers providers at runtime and reports which can run right now
(installed? key set?). A parallel **runner** executes the ones you tick, isolating
any that fail or time out into their own result column. The OCR endpoint streams
results over SSE so columns fill in as each engine finishes.

### OCR pipeline (offline engines)

`paddle` and `easyocr` don't call the model raw — they run through
`providers/ocr/_pipeline.py`: preprocess (grayscale, upscale, denoise, CLAHE,
deskew, binarize) → the engine's detect+recognize → RTL reading-order
reconstruction → Urdu text normalization → a rule-engine transliteration fill,
so each offline column shows `urdu + devanagari + roman` like the GPT column.

**Measuring accuracy.** `eval/` scores every OCR engine against cached GPT
"silver reference" transcriptions:

```bash
cd urdu-free-toolkit
python -m eval.run_eval --engines gpt,paddle,easyocr          # scoreboard (CER/WER)
python -m eval.run_eval --refresh                             # regenerate silver refs (needs OPENAI_API_KEY)
python -m eval.run_eval --engines paddle,easyocr --check      # CI regression gate vs eval/baseline.json
```

Fixtures live in `tests/fixtures/ocr/` — see that folder's README to add one.

**Fine-tuning the offline recognizers** (optional, needs a GPU): the CPU-side
dataset tooling is in `training/` (synthetic Nastaliq/Naskh renderer +
GPT-distillation aligner + dataset converters); the training itself is a
copy-paste runbook in `training/README.md`. Once a model exists, point the
provider at it via `OCR_EASYOCR_RECOG_NETWORK` / `OCR_PADDLE_REC_DIR` — with no
model set, the stock weights are used.

### Modes

- **Single image** — upload, tick engines, compare columns, pick the best, then
  transliterate.
- **Paste text** — skip OCR, go straight to comparing transliteration engines.
- **Batch** — queue up to 30 images (4 on the Vercel deployment), tick OCR
  engines (plus optional transliteration engines), and run them all. Results
  stream into a table as each file finishes, with a **Download CSV** button.
  Compare mode is kept: tick several engines and each image gets a row per engine
  combination.

## Settings

Click **Settings** in the header to paste API keys. They're written to a local
`.env` file next to `app.py` (git-ignored), applied immediately, and only the key
*names* are ever sent back to the browser. Offline engines ignore all of this.
On a read-only host (Vercel) the panel is read-only — keys come from the
project's environment variables and it just shows which are set.

## Deploy (Vercel)

`git push` → live site, cloud engines only. One-time dashboard setup (import the
repo, Root Directory = `urdu-free-toolkit`, add keys) is in **[DEPLOY.md](DEPLOY.md)**.
`api/index.py` re-exports the Flask app; `requirements.txt` is left untouched and
Vercel installs the trimmed set from `api/requirements.txt` (`.vercelignore`
hides the root `requirements.txt` from the build).

## Honest limitations

- **Vowel restoration is fundamentally a guess.** Urdu script omits short vowels;
  Devanagari and Roman need them. Offline engines fill them with dictionaries and
  heuristics and will be wrong on uncommon words, proper nouns, and poetry. Only
  a context-aware LLM (the GPT / Claude / Gemini providers) genuinely reasons
  them out. Always check
  the extracted Urdu before trusting the output — that's what the compare view is
  for.
- **`python app.py` is the dev server** (`app.run(debug=True)`). The Vercel
  deployment runs the same `app` under the platform's WSGI instead — see
  [DEPLOY.md](DEPLOY.md) for what's different there (cloud engines only, buffered
  SSE, ~4.5 MB upload cap).
- **Offline engines are heavy to install.** Later phases pull in PyTorch and
  several model downloads (multiple GB, first run only). Each provider degrades
  gracefully — if its library or binary isn't present it just shows as "not
  installed" and the app still runs.

## Repo layout

```
app.py                 Flask routes over the provider pipeline
config.py              where-am-I-running switches (Vercel: batch cap, read-only settings)
runner.py              run N providers in parallel (+ SSE stream), per-provider timeout
settings.py            read/write API keys to .env
transliterate.py       the offline rule-based transliteration engine
api/index.py           Vercel entrypoint — re-exports the Flask app
vercel.json / api/requirements.txt / .vercelignore   deploy config — see DEPLOY.md
providers/
  base.py              Capability enum, Result dataclasses, BaseProvider (+ price)
  registry.py          discovery + availability + UI metadata
  curation.py          the curated shortlist each picker shows (FEATURED)
  _openai_common.py    shared OpenAI plumbing for the gpt providers
  _anthropic_common.py shared Anthropic plumbing for the claude providers
  _gemini_common.py    shared Google Gemini plumbing for the gemini providers
  ocr/gpt.py           OpenAI GPT vision OCR   (ocr/claude.py, ocr/gemini.py, ...)
  ocr/_pipeline.py     shared preprocess -> recognize -> RTL -> normalize -> translit-fill
  ocr/paddle.py        ocr/easyocr_p.py       both run through _pipeline.py
  translit/rule.py     wraps transliterate.py
  translit/gpt.py      OpenAI GPT transliteration
static/                app.css, app.js  (no CDN — system fonts only)
templates/index.html
tests/                 pytest; heavy providers gated behind RUN_HEAVY=1
eval/                  CER/WER scoring of every OCR engine vs cached GPT refs
training/              Phase-2 OCR fine-tune dataset tooling + GPU runbook
docs/superpowers/       design spec + phased implementation plans
```
