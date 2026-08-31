# Urdu Toolkit — pick your engines, compare, transliterate

A local web app that pulls **Urdu text out of images**, **transliterates** it to
Devanagari (Hindi script) and Roman, and lets you **run several engines side by
side and compare** the results.

**Free / offline first.** The offline engines need no API key and no internet
once installed. Paid APIs (OpenAI now; Anthropic / Google / Azure / OCR.space in
later phases) are *optional* extras you switch on in Settings.

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

## How it's built

Every step is a list of small **provider** modules under `providers/<capability>/`:

| Capability | What it does | Providers now | Providers coming |
|---|---|---|---|
| **OCR** | image → Urdu text | OpenAI GPT-4o vision *(API)* | Surya, EasyOCR, PaddleOCR, RapidOCR, docTR, TrOCR, Tesseract *(all offline)* + Claude / Gemini / OCR.space |
| **Transliteration** | Urdu → Devanagari + Roman | Rule engine *(offline)*, GPT-4o *(API)* | Aksharamukha, uroman, PyICU, IndicXlit, python-hutrans *(offline)* |
| **Translation** | Urdu → English / Hindi | — | IndicTrans2, NLLB-200, M2M100, Argos *(offline)* + deep-translator + APIs |
| **Redraw image** | erase Urdu, draw transliteration in place | gpt-image-1 *(API)* | PIL box-cover, OpenCV inpaint, LaMa *(offline)* |

A registry discovers providers at runtime and reports which can run right now
(installed? key set?). A parallel **runner** executes the ones you tick, isolating
any that fail or time out into their own result column. The OCR endpoint streams
results over SSE so columns fill in as each engine finishes.

### Modes

- **Single image** — upload, tick engines, compare columns, pick the best, then
  transliterate and (optionally) redraw the image.
- **Paste text** — skip OCR, go straight to comparing transliteration engines.
- **Batch (100–200 images)** — *coming in a later phase*: a job queue with
  progress, resume, and CSV / XLSX / ZIP export.

## Settings

Click **Settings** in the header to paste API keys. They're written to a local
`.env` file next to `app.py` (git-ignored), applied immediately, and only the key
*names* are ever sent back to the browser. Offline engines ignore all of this.

## Honest limitations

- **Vowel restoration is fundamentally a guess.** Urdu script omits short vowels;
  Devanagari and Roman need them. Offline engines fill them with dictionaries and
  heuristics and will be wrong on uncommon words, proper nouns, and poetry. Only
  a context-aware LLM (the GPT provider) genuinely reasons them out. Always check
  the extracted Urdu before trusting the output — that's what the compare view is
  for.
- **Not a production server.** `app.run(debug=True)` is Flask's dev server.
- **Offline engines are heavy to install.** Later phases pull in PyTorch and
  several model downloads (multiple GB, first run only). Each provider degrades
  gracefully — if its library or binary isn't present it just shows as "not
  installed" and the app still runs.
- **Redraw is an AI re-render, not a pixel patch** (for the `gpt_image`
  provider): layout and colour match well, fine details and the typeface do not.

## Repo layout

```
app.py                 Flask routes over the provider pipeline
runner.py              run N providers in parallel (+ SSE stream), per-provider timeout
settings.py            read/write API keys to .env
transliterate.py       the offline rule-based transliteration engine
providers/
  base.py              Capability enum, Result dataclasses, BaseProvider
  registry.py          discovery + availability + UI metadata
  _openai_common.py    shared OpenAI plumbing for the gpt providers
  ocr/gpt.py           OpenAI GPT-4o vision OCR
  translit/rule.py     wraps transliterate.py
  translit/gpt.py      OpenAI GPT-4o transliteration
  render/gpt_image.py  gpt-image-1 in-place redraw
static/                app.css, app.js  (no CDN — system fonts only)
templates/index.html
tests/                 pytest; heavy providers gated behind RUN_HEAVY=1
docs/superpowers/       design spec + phased implementation plans
```
