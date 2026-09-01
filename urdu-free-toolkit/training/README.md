# Fine-tuning the offline OCR recognizers

This is the Phase 2 runbook from
`docs/superpowers/specs/2026-09-01-ocr-parity-pipeline-design.md`. Phase 1
(the shared preprocessing/RTL/normalize pipeline and the `eval/` CER/WER
harness) already ships and needs no GPU. This directory holds the CPU-side
dataset tooling — built and unit-tested in this repo — plus the runbooks for
the GPU training step, which you run in a separate session when you're ready.

## Prerequisites

- A cloud GPU session (Colab, RunPod, Lambda, or similar). A single T4 is
  enough for both trainers below.
- Python + this repo's `requirements.txt` installed for the dataset-building
  steps (`training/synth`, `training/distill`, `training/*/to_*`).
- The two upstream trainer repos, cloned where you run the GPU job:
  [`JaidedAI/deep-text-recognition-benchmark`](https://github.com/JaidedAI/deep-text-recognition-benchmark)
  (EasyOCR) and [`PaddlePaddle/PaddleOCR`](https://github.com/PaddlePaddle/PaddleOCR).
- `OPENAI_API_KEY` set if you want the GPT-distillation step (optional —
  synthetic data alone is enough to get started).

## 1. Build the dataset

**Bulk, synthetic (always do this first — it's free and fast):**

```bash
cd urdu-free-toolkit
python -m training.corpus.fetch_corpus --out training/corpus/sentences.txt --source sample
# Drop real Urdu fonts (Noto Nastaliq Urdu, Noto Naskh Arabic — both OFL — or
# your own) into training/synth/fonts/ first; see training/synth/fonts.json.
python -m training.synth.build_synth \
    --sentences training/corpus/sentences.txt \
    --fonts training/synth/fonts \
    --out data/synth --per-sentence 8
```

**Domain fit, from your real images (optional, needs `OPENAI_API_KEY`):**

```python
from providers.ocr.easyocr_p import _recognize as easyocr_recognize   # or paddle's
from training.distill.collect import crops_from_image
from training.distill.align import align
from training.distill.build_distill import build
from providers.ocr.gpt import PROVIDER as gpt

pairs = []
for path in real_image_paths:
    image = open(path, "rb").read()
    crops = crops_from_image(image, easyocr_recognize)
    gpt_text = gpt.ocr(image).text
    for word, text in align(crops, gpt_text):
        # crop the original image to word.box, pair with `text`
        pairs.append((crop_pil_image, text))
build(pairs, "data/distill")
```

**Merge** the two `gt.txt` files (concatenate `data/synth/train/gt.txt` and
`data/distill/train/gt.txt`, same for `val`) before training.

## 2. Train

- EasyOCR: `training/easyocr/train.md`
- PaddleOCR: `training/paddle/train.md`

Both are copy-paste runbooks with Colab and RunPod starting cells.

## 3. Wire the fine-tuned model in

- EasyOCR: drop the converted model at
  `~/.EasyOCR/user_network/urdu_ft.{py,yaml,pth}`, or set
  `OCR_EASYOCR_RECOG_NETWORK=urdu_ft`.
- PaddleOCR: set `OCR_PADDLE_REC_DIR=/path/to/exported/inference/dir`.

With neither set, both providers use their stock weights — nothing here is
required for the app to work.

## 4. Acceptance

```bash
cd urdu-free-toolkit
python -m eval.run_eval --engines gpt,paddle,easyocr           # before
# ...wire in the fine-tuned model(s)...
python -m eval.run_eval --engines gpt,paddle,easyocr           # after
python -m eval.run_eval --engines paddle,easyocr --write-baseline
git add eval/baseline.json
```

Commit the new `baseline.json` and paste the before/after table into the
commit message so the improvement is measured, not asserted.

## Handwriting (not covered by this pass)

The synthetic renderer's font list can include handwriting-style fonts
(tag them in `training/synth/fonts.json`), but handwriting is a materially
different recognition target from printed Nastaliq/Naskh. Treat it as a
separate dataset + separate acceptance run, not an extension of this one.
