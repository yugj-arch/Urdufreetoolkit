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
git clone https://github.com/clovaai/deep-text-recognition-benchmark
cd deep-text-recognition-benchmark
pip install -r requirements.txt
pip install lmdb fire   # lmdb dataset writer + create_lmdb_dataset.py's CLI
```

(Not `JaidedAI/deep-text-recognition-benchmark` — that repo doesn't exist.
EasyOCR's own org is JaidedAI, but the trainer it was built on top of is
Clova AI Research's original repo, above.)

## 2. VERIFIED architecture + character set

EasyOCR's Urdu model is `arabic_g1` -- **generation 1**, not generation 2.
Confirmed 2026-09-23 against the installed `easyocr` package
(`easyocr.easyocr.Reader.__init__`, the `arabic_lang_list` branch, and
`easyocr.recognition.get_recognizer`'s `'generation1'` case): generation 1
uses `easyocr/model/model.py` (**ResNet** feature extractor, not VGG --
VGG is generation 2's `vgg_model.py`), `network_params =
{"input_channel": 1, "output_channel": 512, "hidden_size": 512}`, and
`imgH=32` (the `AlignCollate` default). `training/easyocr/config.yaml` is
already filled in with these values plus the exact `character` string from
`easyocr.config.recognition_models["gen1"]["arabic_g1"]["characters"]`
(order matters -- it fixes the pretrained checkpoint's output-layer class
indices). If you're on a different `easyocr` version, re-verify:

```bash
python - <<'PY'
import easyocr.config as c
print(c.recognition_models["gen1"]["arabic_g1"])
PY
```

Locate the weights EasyOCR downloaded on first run (`arabic.pth`) and set
`saved_model:` in `config.yaml` to that path, so training fine-tunes from it
instead of from scratch:

```bash
python -c "import pathlib; print(pathlib.Path.home() / '.EasyOCR' / 'model' / 'arabic.pth')"
```

## 3. Train

`train.py` in the Clova repo is plain argparse -- it has **no `--config`
flag**. Use this project's `run_train.py`, which reads `config.yaml` and
invokes `train.py` with the equivalent CLI args (as a real argument list,
not a shell string, so the Urdu/Arabic `character` set never needs
shell-escaping):

```bash
cd urdu-free-toolkit
python training/easyocr/run_train.py --dtrb-dir /path/to/deep-text-recognition-benchmark
# sanity-check the argv first without running it:
python training/easyocr/run_train.py --dtrb-dir /path/to/deep-text-recognition-benchmark --dry-run
```

`config.yaml`'s `batch_size: 32` is sized for a 6GB card; raise it if you
have more VRAM (a T4's 16GB comfortably fits DTRB's own default of 192).

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
!pip install -q lmdb fire pyyaml
!git clone -q https://github.com/clovaai/deep-text-recognition-benchmark
!python /content/urdu-free-toolkit/training/easyocr/run_train.py \
    --dtrb-dir /content/deep-text-recognition-benchmark
```

## RunPod

Use a PyTorch base image (CUDA 11.8+), mount your `data/lmdb` volume, run the
same `run_train.py` command inside a tmux/screen session so it survives a
dropped connection.
