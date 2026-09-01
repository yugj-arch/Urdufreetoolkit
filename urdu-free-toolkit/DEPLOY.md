# Deploy to Vercel

The repo is deploy-ready. You connect it to Vercel once; after that every push
to the production branch ships automatically.

## What runs on Vercel

The deployed site is **cloud-engine only** — Vercel's 250 MB function limit
can't hold PyTorch/PaddleOCR.

| Step | Live on Vercel | Local only |
|---|---|---|
| OCR | GPT vision, Claude vision, Gemini vision, Google Cloud Vision | EasyOCR, PaddleOCR, Surya, … (show as "not installed") |
| Transliteration | GPT, Groq, Rule engine, Aksharamukha, uroman | ICU |

Dependencies come from **`pyproject.toml`** (the cloud set). `requirements.txt`
stays the full local set and is hidden from the build by `.vercelignore` — it is
never touched.

## One-time setup (Vercel dashboard)

1. Push this branch (already on `origin/feat/vercel-deploy`), then fold it into
   whatever branch Vercel treats as production (usually `master`/`main`):
   ```bash
   git push
   git checkout master && git merge feat/vercel-deploy && git push
   ```
   Pushing the branch alone gets you a **preview URL**; merging to the production
   branch updates the live site.
2. [vercel.com/new](https://vercel.com/new) → **Import** this Git repo.
3. **Root Directory** → click *Edit* → choose **`urdu-free-toolkit`**. (Vercel
   then auto-detects Flask from `app.py`; leave Framework Preset alone.)
4. **Environment Variables** → add the keys you want live (see `.env.example`):
   `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_VISION_KEY`,
   `GROQ_API_KEY`, plus any `*_MODEL` overrides. Every key is optional.
5. **Deploy.**

Done. From now on `git push` to the production branch = new deployment. PR
branches get preview URLs automatically.

## CLI alternative

```bash
npm i -g vercel
cd urdu-free-toolkit
vercel          # links the project; accept "./" as the root
vercel --prod
```

## Known limits on the deployed site

- **Uploads capped at ~4.5 MB** (Vercel request-body limit). Large photos fail.
- **Batch tab capped at 4 images** on Vercel (vs 30 locally) to stay under that
  cap — set automatically via `VERCEL` env, surfaced by `GET /api/config`.
- **`maxDuration` is 60 s** (`vercel.json`). Enough for a single vision call;
  raise to `300` on a Pro plan for heavier batch runs.
- **SSE arrives buffered** — result columns fill in one shot at the end, not
  progressively. Functionally identical.
- **Settings panel is read-only.** Keys live in the Vercel project's env vars;
  the panel just shows which are set. Change them in the dashboard, then redeploy.
