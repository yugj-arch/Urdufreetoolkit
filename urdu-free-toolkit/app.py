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
from providers.base import Capability, TranslateOpts, TranslitOpts

load_dotenv(settings.env_path())

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB (chunked batch uploads)


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


@app.post("/api/translate")
def api_translate():
    """JSON: `text`, `providers`, `targets`. Returns one row per provider.
    (No translate providers exist until Phase 3 — the contract is stable now.)"""
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    ids = data.get("providers") or []
    if not text:
        return jsonify({"error": "No text provided."}), 400
    opts = TranslateOpts(targets=tuple(data.get("targets") or ("english",)))
    results = runner.run(Capability.TRANSLATE, ids,
                         lambda p: p.translate(text, opts), timeout_s=180)
    return jsonify({"results": [
        {"provider_id": r.provider_id, "ok": r.ok, "error": r.error, "ms": r.ms,
         "english": getattr(r, "english", ""), "hindi": getattr(r, "hindi", "")}
        for r in results]})


@app.post("/api/render")
def api_render():
    """Multipart: `image`, `provider`, `lines` (newline-separated). Returns a PNG,
    or 4xx/5xx JSON on failure."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    pid = request.form.get("provider") or ""
    lines = [ln for ln in (request.form.get("lines") or "").split("\n") if ln.strip()]
    image = request.files["image"].read()
    if not lines:
        return jsonify({"error": "No transliteration lines given."}), 400
    try:
        prov = registry.get(Capability.RENDER, pid)
    except KeyError:
        return jsonify({"error": f"Unknown render provider: {pid}"}), 400
    res = prov.render(image, lines)
    if not res.ok:
        return jsonify({"error": res.error}), 500
    return Response(res.png, mimetype="image/png")


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
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
