# OCR eval fixtures

Each `<slug>.png` is a small Urdu image; `<slug>.gpt.json` is its cached GPT
"silver reference" transcription
(`{"urdu": "...", "model": "...", "generated_at": "..."}`).

## Add a fixture

1. Drop a redistributable `<slug>.png` here — a synthetic render from
   `training/synth`, or an image you have the right to commit. Keep it small.
2. From `urdu-free-toolkit/`:
   `OPENAI_API_KEY=... python -m eval.run_eval --refresh`
   to generate `<slug>.gpt.json`.
3. Eyeball the JSON — GPT misreads too. Hand-fix the `urdu` value if needed;
   it's just JSON.
4. Commit both files.

The eval and its regression test (`tests/test_eval_regression.py`) read the
committed JSON and never call the network.
