# -*- coding: utf-8 -*-
"""Google Cloud Vision as an OCR provider (paid API).

Google's document OCR is one of the most accurate readers of Urdu Nastaliq. This
hits the REST endpoint directly with an API key (no SDK, stdlib HTTP only),
asking for DOCUMENT_TEXT_DETECTION with an Urdu language hint. Unlike the
Gemini/GPT providers it only transcribes -- the app's transliterate step fills in
Devanagari/Roman afterwards, so this returns ``text`` + word ``boxes`` like the
offline engines. Enable the Vision API on a GCP project, make an API key, and
paste it in Settings as ``GOOGLE_VISION_KEY``.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from providers._openai_common import prepare_image
from providers.base import BaseProvider, Capability, OcrResult, ProviderInfo

_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
_TIMEOUT = 30


def _key() -> str:
    return os.environ.get("GOOGLE_VISION_KEY", "")


def _annotate(image_b64: str, api_key: str) -> dict:
    """POST one image to Cloud Vision; return its single response object.

    Raises ``RuntimeError`` on a transport failure or an API-level error so the
    provider folds it into a failed result row rather than crashing.
    """
    body = json.dumps({
        "requests": [{
            "image": {"content": image_b64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            "imageContext": {"languageHints": ["ur"]},
        }]
    }).encode()
    req = urllib.request.Request(
        f"{_ENDPOINT}?key={api_key}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("error", {}).get("message", detail)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cloud Vision request failed: {e.reason}") from None

    if data.get("error"):
        raise RuntimeError(str(data["error"].get("message", data["error"])))
    resp = (data.get("responses") or [{}])[0]
    if resp.get("error"):
        raise RuntimeError(str(resp["error"].get("message", resp["error"])))
    return resp


class GoogleVisionOcr(BaseProvider):
    info = ProviderInfo(
        id="gcv",
        label="Google Cloud Vision (API)",
        capability=Capability.OCR,
        kind="api",
        needs=["GOOGLE_VISION_KEY"],
        note="GCP Vision API key. DOCUMENT_TEXT_DETECTION with an Urdu hint; strong on Nastaliq.",
        price="~$1.50 / 1k images",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if _key() else (False, "GOOGLE_VISION_KEY not set")

    def ocr(self, image: bytes) -> OcrResult:
        key = _key()
        if not key:
            return OcrResult(provider_id=self.info.id, ok=False,
                             error="GOOGLE_VISION_KEY is not set. Add it in Settings.")

        def _call():
            b64, _mime = prepare_image(image)
            resp = _annotate(b64, key)
            text = (resp.get("fullTextAnnotation", {}).get("text") or "").strip()
            boxes = []
            for ann in (resp.get("textAnnotations") or [])[1:]:
                verts = ann.get("boundingPoly", {}).get("vertices") or []
                pts = [[int(v.get("x", 0)), int(v.get("y", 0))] for v in verts]
                boxes.append({"box": pts, "text": ann.get("description", ""),
                              "conf": float(ann.get("confidence", 0.0) or 0.0)})
            return text, boxes

        t = self._timed(_call)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        text, boxes = t["value"]
        return OcrResult(provider_id=self.info.id, ok=True, ms=t["ms"], text=text, boxes=boxes,
                         meta={"mode": "DOCUMENT_TEXT_DETECTION"})


PROVIDER = GoogleVisionOcr()
