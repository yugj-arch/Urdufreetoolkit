# Urdu → Hindi / Roman Toolkit (AI-powered)

**Uses the OpenAI API. Needs an `OPENAI_API_KEY`. ~1–2¢ per image for OCR; ~4–8¢ more if you also generate the edited image.**

This is a complete, working, standalone app you can run on your own
machine that:
1. Extracts Urdu text from an uploaded image using **OpenAI GPT-4o vision**
   (reads Nastaliq far better than free offline OCR could)
2. Transliterates that Urdu into Devanagari (Hindi script) and Roman
   script in the **same API call**, restoring the short vowels Urdu omits
   from sentence context
3. Optionally produces an **edited copy of the image** (via OpenAI's
   `gpt-image-1`) with the Urdu erased and its Roman or Devanagari
   transliteration painted back in place — download as PNG

> **History:** this started as a 100%-free offline toolkit (Tesseract +
> a rule-based transliteration engine). Tesseract's accuracy on Urdu
> Nastaliq was too low to be useful, so the OCR + transliteration path
> was switched to the OpenAI API. The old rule-based engine is still in
> the repo as `transliterate.py` (now unused) if you want the free path
> back.

---

## Stack

| Layer | Tool | Cost |
|---|---|---|
| OCR + transliteration | OpenAI **gpt-4o** vision (one call per image), via the `openai` SDK | ~1–2¢ per image |
| Image editing | OpenAI **gpt-image-1** edit call — erases the Urdu, repaints the transliteration in place | ~4–8¢ per image |
| Image handling | Pillow (PIL) — validate, downscale, resize result to original size | Free |
| Config | `python-dotenv` (`.env` holds the API key) | Free |
| Web framework | Flask | Free |
| Frontend | Plain HTML/CSS/JS, no CDN | Free |

Swapping providers (Claude, Gemini): change `OPENAI_MODEL` in `.env`, or
replace the two client calls in `ocr.py` — the JSON contract the
functions return is provider-agnostic.

---

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Add your OpenAI API key
#    Create a file named .env next to app.py containing:
#      OPENAI_API_KEY=sk-...
#    (.env is gitignored. Optionally add OPENAI_MODEL=gpt-4o to override.)

# 3. Run it
python app.py
```

Then open **http://localhost:5000** in your browser.

Get an API key at <https://platform.openai.com/api-keys>. Never commit
`.env` or paste the key anywhere public — if it leaks, revoke it on that
page and issue a new one.

---

## Files in this download

```
urdu-free-toolkit/
├── app.py              # Flask web server — the whole app, run this
├── ocr.py              # OpenAI GPT-4o calls: image → Urdu → Devanagari + Roman
├── imgedit.py          # gpt-image-1: edited image with the transliteration in place
├── transliterate.py    # OLD rule-based engine — no longer used by the app
├── templates/
│   └── index.html      # The web page (upload, check, results, edited image)
├── requirements.txt
├── .env                # your OPENAI_API_KEY (create this; gitignored)
└── README.md           # This file
```

The prompts are at the top of `ocr.py` (`_VISION_SYSTEM`, `_TEXT_SYSTEM`)
and `imgedit.py` (`_EDIT_PROMPT`) — edit them there.

---

## What was tested

Run for real against the live OpenAI API on this exact code:

- `ocr.extract_and_transliterate()` on a rendered Urdu image →
  transcribed the Urdu correctly and returned matching Devanagari + Roman
  (`محبت ایک خوبصورت احساس ہے` → `मोहब्बत एक खूबसूरत एहसास है` →
  `mohabbat ek khoobsurat ehsaas hai`).
- `ocr.transliterate_text()` on plain Urdu text
  (`میں ٹھیک ہوں، آپ کیسے ہیں؟` → `मैं ठीक हूँ, आप कैसे हैं?` →
  `main theek hoon, aap kaise hain?`).
- `app.py` `/api/extract`, `/api/transliterate` and `/api/render-image`
  via Flask's test client → `200` with correct JSON / `image/png`;
  missing/empty file → `400`; bad `script` → `400`; missing
  `OPENAI_API_KEY` → a clear `RuntimeError`.
- `imgedit.render_transliterated_image()` on a 2-line poster image →
  gpt-image-1 erased the Urdu and wrote the Roman transliteration in
  place (~26s), background preserved, output resized back to the original
  1100×420. It did reflow one long line onto two — see Limitations.

Honest gap: the test image was rendered in a Naskh font, not Nastaliq
(no Nastaliq font was available locally). GPT-4o reads Nastaliq well in
practice, but check accuracy yourself on a real photo the first time.

---

## Limitations

- **Vowel restoration is still a guess.** Urdu script omits short vowels;
  the model infers them from context. It's right most of the time but can
  be wrong on uncommon words, proper nouns, and poetry. Always check the
  extracted Urdu in step 2 before trusting the output.
- **Cost and network.** Every image and every paste-text run is an API
  call (~1–2¢/image) and needs internet. No offline mode.
- **Key security.** The key sits in `.env` in plaintext. Keep `.env` out
  of version control (it's in `.gitignore`) and revoke a leaked key at
  <https://platform.openai.com/api-keys>.
- **Not a production web server.** `app.run(debug=True)` is Flask's dev
  server. For real traffic run it behind `gunicorn app:app`.
- **No self-learning.** Corrections you make in the UI aren't saved.
- **Edited image is an AI re-render, not a pixel-exact patch.**
  `gpt-image-1` erases the Urdu cleanly and keeps the background, but it
  does not fully obey layout instructions: when the transliteration is
  longer than the Urdu (it usually is) it may wrap onto an extra line or
  shift the alignment, and the typeface is a generic sans, not the
  original's. Costs ~4–8¢ per image, takes 30–90s, and needs
  `gpt-image-1` enabled on the API account (OpenAI may require
  organisation verification).
