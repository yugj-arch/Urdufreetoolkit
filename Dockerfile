# Full-engine container image (Hugging Face Spaces / any Docker host).
# All engines run here — cloud APIs plus offline PaddleOCR + EasyOCR — unlike the
# Vercel deployment, which is cloud-only because torch can't fit its 250 MB limit.

FROM python:3.12-slim

# system libs opencv / torch / paddle / onnxruntime link against
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as uid 1000 — install and run as that user so the
# model-weight caches (~/.EasyOCR, ~/.paddlex) land somewhere writable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /home/user/app
COPY --chown=user urdu-free-toolkit/ /home/user/app/

# CPU-only torch/torchvision first (small wheels), then the leaner full set.
RUN pip install --user --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user --no-cache-dir -r requirements-docker.txt

# HF Spaces routes to port 7860. One worker (models load once — keep memory
# sane), threads for the parallel runner + SSE, long timeout for slow OCR.
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8", \
     "--timeout", "180", "--graceful-timeout", "180", "app:app"]
