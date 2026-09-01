# Fine-tuning PaddleOCR's Urdu recognizer

The dataset tooling (`training/synth`, `training/distill`,
`training/paddle/to_paddle_rec.py`) is built and unit-tested in-repo; this is
the GPU-side runbook for the actual training run.

**Expected wall-clock:** ~3–6 hours on a single T4 for 60 epochs of the config
below, dataset-size dependent.

## 1. Clone PaddleOCR and install

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR
cd PaddleOCR
pip install -r requirements.txt
```

## 2. Download the pretrained Arabic recognizer

```bash
mkdir -p pretrain
wget -P pretrain https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/arabic_PP-OCRv4_rec_train.tar
tar -xf pretrain/arabic_PP-OCRv4_rec_train.tar -C pretrain
```

(Check the PaddleOCR docs for the current URL if this one has moved — model
hosting paths change between releases.)

## 3. Build the label files

From `urdu-free-toolkit/`, with your merged `gt.txt` (synthetic + distilled):

```bash
python - <<'PY'
from training.paddle.to_paddle_rec import convert
convert("data/synth/train/gt.txt", "PaddleOCR/data/urdu/paddle_train.txt", image_root="train")
convert("data/synth/val/gt.txt",   "PaddleOCR/data/urdu/paddle_val.txt",   image_root="val")
PY
# copy the actual image files alongside:
mkdir -p PaddleOCR/data/urdu/train PaddleOCR/data/urdu/val
cp data/synth/train/*.png PaddleOCR/data/urdu/train/
cp data/synth/val/*.png   PaddleOCR/data/urdu/val/
```

## 4. Train

```bash
cd PaddleOCR
python tools/train.py -c /path/to/urdu-free-toolkit/training/paddle/arabic_rec_ft.yml
```

## 5. Export for inference

```bash
python tools/export_model.py \
  -c /path/to/urdu-free-toolkit/training/paddle/arabic_rec_ft.yml \
  -o Global.pretrained_model=output/urdu_rec_ft/best_accuracy \
     Global.save_inference_dir=output/urdu_rec_infer
```

## 6. Verify

```bash
cd urdu-free-toolkit
OCR_PADDLE_REC_DIR=/abs/path/to/PaddleOCR/output/urdu_rec_infer python -m eval.run_eval --engines paddle,gpt
```

Compare against the stock run (no env var). If CER(norm) improved, commit the
new baseline:

```bash
python -m eval.run_eval --engines paddle --write-baseline
```

## Colab cell (starting point)

```python
!git clone -q https://github.com/PaddlePaddle/PaddleOCR
%cd PaddleOCR
!pip install -q -r requirements.txt
!python tools/train.py -c /content/arabic_rec_ft.yml
```

## RunPod

Use a PaddlePaddle-GPU base image (or `pip install paddlepaddle-gpu` matching
your CUDA version), mount the dataset volume, run inside tmux/screen.
