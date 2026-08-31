# -*- coding: utf-8 -*-
"""
app.py -- Flask web app. Image -> Urdu text -> Hindi/Roman, powered by
the OpenAI GPT-4o vision API (see ocr.py).

Requires OPENAI_API_KEY in a .env file next to this file.

Run:
    pip install -r requirements.txt
    python app.py

Then open http://localhost:5000 in a browser.
"""

from flask import Flask, request, render_template, jsonify, Response

from ocr import extract_and_transliterate, transliterate_text
from imgedit import render_transliterated_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extract", methods=["POST"])
def api_extract():
    """One GPT-4o vision call: uploaded image -> extracted Urdu text plus
    its Devanagari and Roman transliteration."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        result = extract_and_transliterate(image_bytes)
        return jsonify({
            "text": result["urdu"],
            "devanagari": result["devanagari"],
            "roman": result["roman"],
            "notes": result["notes"],
            "model": result["model"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/render-image", methods=["POST"])
def api_render_image():
    """Return an edited copy of the uploaded image with each line of Urdu
    covered over and its transliteration drawn in place. Form fields:
    `image` (file) and `script` ('roman' or 'devanagari')."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image_bytes = request.files["image"].read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    script = (request.form.get("script") or "roman").lower()
    if script not in ("roman", "devanagari"):
        return jsonify({"error": "script must be 'roman' or 'devanagari'."}), 400

    try:
        png = render_transliterated_image(image_bytes, script)
        return Response(png, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transliterate", methods=["POST"])
def api_transliterate():
    """Text-only GPT call for the 'paste text' path: Urdu text -> its
    Devanagari and Roman transliteration."""
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        result = transliterate_text(text)
        return jsonify({
            "devanagari": result["devanagari"],
            "roman": result["roman"],
            "notes": result["notes"],
            "model": result["model"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # threaded: the gpt-image-1 edit can take 30-90s; don't block the whole
    # server (and the page's other requests) while one runs.
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
