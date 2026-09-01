# Deploy to Hugging Face Spaces (all engines)

The Vercel deploy is cloud-only — PyTorch/PaddleOCR can't fit its 250 MB
function limit. A **Docker Space** on Hugging Face has no such limit (16 GB RAM
on the free CPU tier), so **PaddleOCR + EasyOCR run there too**, alongside the
cloud engines. Deploy is `git push`, same as Vercel.

## What builds

- Root **`Dockerfile`** — `python:3.12-slim`, runs as uid 1000 (HF requirement),
  CPU-only torch/torchvision, then `urdu-free-toolkit/requirements-docker.txt`
  (app + cloud engines + offline translit + PaddleOCR/EasyOCR). Served by
  gunicorn on port 7860.
- Root **`README.md`** — the Space card (`sdk: docker`, `app_port: 7860`).
- `.dockerignore` keeps `.git`, tests, `.env`, training/eval out of the image.
- The repo's `requirements.txt` is untouched.

## One-time setup

1. [huggingface.co/new-space](https://huggingface.co/new-space) → **SDK: Docker**
   (blank template) → create. Note the URL
   `huggingface.co/spaces/<you>/<space>`.
2. Add it as a git remote and push (the Space builds from its `main`):
   ```bash
   git remote add hf https://huggingface.co/spaces/<you>/<space>
   git push hf feat/vercel-deploy:main
   ```
   (HF asks for your username + an access token as the password — make one at
   huggingface.co/settings/tokens with *write* scope.)
3. Space → **Settings → Variables and secrets** → add as **Secrets**:
   `OPENAI_API_KEY`, `GOOGLE_VISION_KEY`, and optionally `ANTHROPIC_API_KEY`,
   `GOOGLE_API_KEY`, `GROQ_API_KEY`.
4. First build takes ~8–15 min (torch + torchvision + paddle). The first OCR
   call then downloads the Urdu model weights (~1 min) and caches them under
   `~/.EasyOCR` / `~/.paddlex` for the container's life.

After that, every `git push hf <branch>:main` redeploys.

## Notes

- Free Spaces **sleep after ~48 h idle** and wake on the next visit (cold start
  ~30 s, plus model re-download if the container was rebuilt).
- `VERCEL`-gated behaviour (batch cap 4, read-only Settings, 4.5 MB upload cap)
  does **not** apply here — full 30-image batch, editable Settings, 64 MB
  uploads.
- Settings keys typed in the UI are written to `.env` inside the container and
  last only until the next rebuild — use HF Secrets for anything permanent.
- Keep or delete the Vercel project as you like; the two are independent.
