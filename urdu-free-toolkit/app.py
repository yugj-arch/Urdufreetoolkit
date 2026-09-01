# -*- coding: utf-8 -*-
"""Flask app: pick engines, run them in parallel, compare.

Providers live in ``providers/<capability>/`` and are discovered at runtime.
Free/offline providers work with no API key; API providers light up once their
key is saved in Settings. One engine failing among several is reported in its
own result row, never as an HTTP 500.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000
"""
from __future__ import annotations

import json

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

import runner
import settings
from providers import registry
from providers.base import Capability, TranslitOpts

load_dotenv(settings.env_path())

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB — batch tab uploads many images

BATCH_MAX_FILES = 30


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/providers")
def api_providers():
    """Every provider grouped by capability, with a badge and an availability
    flag the UI uses to enable/grey each checkbox."""
    return jsonify({c.value: registry.for_ui(c) for c in Capability})


@app.post("/api/ocr")
def api_ocr():
    """Multipart: `image`, `providers` (comma list). Streams one SSE `data:`
    line per provider as it finishes, then `{"done": true}`."""
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "No image uploaded."}), 400
    ids = [s for s in (request.form.get("providers") or "").split(",") if s]
    if not ids:
        return jsonify({"error": "Pick at least one OCR engine."}), 400
    image = request.files["image"].read()
    if not image:
        return jsonify({"error": "Uploaded file is empty."}), 400

    def gen():
        for res in runner.stream(Capability.OCR, ids,
                                 lambda p: p.ocr(image), timeout_s=120):
            row = {
                "provider_id": res.provider_id, "ok": res.ok, "error": res.error,
                "ms": res.ms, "text": getattr(res, "text", ""),
                "notes": getattr(res, "notes", ""),
                "devanagari": res.meta.get("devanagari", ""),
                "roman": res.meta.get("roman", ""),
            }
            yield f"data: {json.dumps(row, ensure_ascii=False)}\n\n"
        yield 'data: {"done": true}\n\n'

    return Response(gen(), mimetype="text/event-stream")


@app.post("/api/transliterate")
def api_transliterate():
    """JSON: `text`, `providers`, `roman_style`, `targets`. Returns one row per
    provider."""
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    ids = data.get("providers") or []
    if not text:
        return jsonify({"error": "No text provided."}), 400
    opts = TranslitOpts(
        roman_style=data.get("roman_style", "natural"),
        targets=tuple(data.get("targets") or ("devanagari", "roman")),
    )
    results = runner.run(Capability.TRANSLIT, ids,
                         lambda p: p.translit(text, opts), timeout_s=120)
    return jsonify({"results": [
        {"provider_id": r.provider_id, "ok": r.ok, "error": r.error, "ms": r.ms,
         "devanagari": getattr(r, "devanagari", ""), "roman": getattr(r, "roman", "")}
        for r in results]})


@app.post("/api/batch")
def api_batch():
    """Multipart: `images` (repeated), `ocr_providers` + `translit_providers`
    (comma lists), `roman_style`. Streams one SSE `data:` row per
    (file x OCR engine x transliteration engine), a `{"progress", "total"}` line
    after each file, then `{"done": true}`. A single engine failing shows up as
    an ``error`` on its row, never as an HTTP 500."""
    files = [f for f in request.files.getlist("images") if f.filename]
    if not files:
        return jsonify({"error": "No images uploaded."}), 400
    ocr_ids = [s for s in (request.form.get("ocr_providers") or "").split(",") if s]
    if not ocr_ids:
        return jsonify({"error": "Pick at least one OCR engine."}), 400
    tr_ids = [s for s in (request.form.get("translit_providers") or "").split(",") if s]
    opts = TranslitOpts(roman_style=request.form.get("roman_style", "natural"))
    images = [(f.filename, f.read()) for f in files[:BATCH_MAX_FILES]]
    total = len(images)

    def emit(row: dict) -> str:
        return f"data: {json.dumps(row, ensure_ascii=False)}\n\n"

    def gen():
        for i, (name, blob) in enumerate(images, 1):
            for o in runner.run(Capability.OCR, ocr_ids,
                                lambda p, b=blob: p.ocr(b), timeout_s=120):
                text = (getattr(o, "text", "") or "").strip()
                base = {
                    "file": name, "ocr_engine": o.provider_id, "urdu": text,
                    "translit_engine": "", "devanagari": "", "roman": "",
                    "ms": o.ms, "error": "" if o.ok else o.error,
                }
                if not o.ok or not text or not tr_ids:
                    yield emit(base)
                    continue
                for t in runner.run(Capability.TRANSLIT, tr_ids,
                                    lambda p, s=text: p.translit(s, opts), timeout_s=120):
                    yield emit({**base,
                                "translit_engine": t.provider_id,
                                "ms": o.ms + t.ms,
                                "devanagari": getattr(t, "devanagari", "") if t.ok else "",
                                "roman": getattr(t, "roman", "") if t.ok else "",
                                "error": "" if t.ok else t.error})
            yield emit({"progress": i, "total": total})
        yield 'data: {"done": true}\n\n'

    return Response(gen(), mimetype="text/event-stream")


@app.get("/api/settings")
def api_settings_get():
    return jsonify(settings.status())


@app.post("/api/settings")
def api_settings_post():
    """JSON key->value. Writes `.env`, applies to the environment, and busts the
    provider cache so API providers re-check their keys. Echoes only names."""
    data = request.get_json(force=True) or {}
    saved = settings.save({k: v for k, v in data.items() if isinstance(v, str)})
    registry.reset_cache()
    return jsonify({"saved": saved})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=True)

