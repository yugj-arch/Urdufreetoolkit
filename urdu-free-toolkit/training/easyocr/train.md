# Fine-tuning EasyOCR's Urdu recognizer

This is the GPU half of the OCR parity plan — the dataset tooling
(`training/synth`, `training/distill`, `training/easyocr/to_lmdb.py`) is
built and unit-tested in-repo; the training run itself needs a GPU and is
this copy-paste runbook.

**Expected wall-clock:** ~2–4 hours on a single T4 for 30k iterations of the
config below. Scale `num_iter` down for a quick sanity run.

## 0. Build the dataset

From `urdu-free-toolkit/`, with your merged `gt.txt` (synthetic + distilled):

```bash
python -m training.easyocr.to_lmdb  # or call gt_to_dtrb_txt / write_lmdb from a script:
python - <<'PY'
from training.easyocr.to_lmdb import write_lmdb
write_lmdb("data/synth/train/gt.txt", "data/synth/train", "data/lmdb/train")
write_lmdb("data/synth/val/gt.txt", "data/synth/val", "data/lmdb/val")
PY
```

(`write_lmdb` needs `pip install lmdb`.)

## 1. Clone the trainer

```bash
git clone https://github.com/JaidedAI/deep-text-recognition-benchmark
cd deep-text-recognition-benchmark
pip install -r requirements.txt
```

## 2. VERIFY the architecture + character set

EasyOCR's Urdu model is the "arabic_g2" (generation 2) recognizer. Before
training, confirm against your installed `easyocr` version:

```bash
python - <<'PY'
import easyocr, json, pathlib
# Locate the arabic_g2 weights EasyOCR downloaded on first run:
print(pathlib.Path.home() / ".EasyOCR" / "model")
# Read easyocr's own config for the arabic recognizer to get the exact
# architecture block and character string, then paste them into config.yaml.
import easyocr.config as c
print(getattr(c, "recognition_models", None))
PY
```

Update `training/easyocr/config.yaml`'s `character:` field and the
`Transformation` / `FeatureExtraction` / `SequenceModeling` / `Prediction`
block to match exactly what you find — a mismatch trains a model EasyOCR's
`Reader` can't load.

Set `saved_model:` in `config.yaml` to the `.pth` path you found, so training
fine-tunes from it instead of from scratch.

## 3. Train

```bash
python train.py --config /path/to/urdu-free-toolkit/training/easyocr/config.yaml
```

## 4. Convert the checkpoint into an EasyOCR custom model

EasyOCR loads custom recognizers from `~/.EasyOCR/user_network/<name>.{py,yaml,pth}`:

```bash
mkdir -p ~/.EasyOCR/user_network
cp saved_models/urdu_ft/best_accuracy.pth ~/.EasyOCR/user_network/urdu_ft.pth
cp /path/to/urdu-free-toolkit/training/easyocr/config.yaml ~/.EasyOCR/user_network/urdu_ft.yaml
# urdu_ft.py: copy the matching model class from deep-text-recognition-benchmark's
# model.py (or EasyOCR's own recognizer module) — it must match the arch you
# trained in step 2/3.
```

## 5. Verify

```bash
cd urdu-free-toolkit
OCR_EASYOCR_RECOG_NETWORK=urdu_ft python -m eval.run_eval --engines easyocr,gpt
```

Compare against the stock run (`python -m eval.run_eval --engines easyocr,gpt`
without the env var). If CER(norm) improved, commit the new baseline:

```bash
python -m eval.run_eval --engines easyocr --write-baseline
```

## Colab cell (starting point)

```python
!pip install -q lmdb
!git clone -q https://github.com/JaidedAI/deep-text-recognition-benchmark
%cd deep-text-recognition-benchmark
!pip install -q -r requirements.txt
!python train.py --config /content/config.yaml
```

## RunPod

Use a PyTorch base image (CUDA 11.8+), mount your `data/lmdb` volume, run the
same `train.py` command inside a tmux/screen session so it survives a dropped
connection.
