# -*- coding: utf-8 -*-
"""M2M100 (418M) as a translation provider (offline).

Lighter than NLLB (~1.9 GB) and lower quality, but a useful second opinion in
the compare view. Runs offline on CPU after the first download.
"""
from __future__ import annotations

from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslateOpts,
    TranslateResult,
)

try:
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
except Exception:  # noqa: BLE001
    M2M100ForConditionalGeneration = None
    M2M100Tokenizer = None

_MODEL = "facebook/m2m100_418M"
_pair = None


def _get():
    global _pair
    if _pair is None:
        _pair = (
            M2M100Tokenizer.from_pretrained(_MODEL),
            M2M100ForConditionalGeneration.from_pretrained(_MODEL),
        )
    return _pair


class M2m100Translate(BaseProvider):
    info = ProviderInfo(
        id="m2m100",
        label="M2M100 418M (offline)",
        capability=Capability.TRANSLATE,
        kind="offline",
        note="Lighter multilingual NMT (~1.9 GB). Lower quality than NLLB.",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if M2M100Tokenizer is not None else (False, "pip install transformers torch")

    def translate(self, text: str, opts: TranslateOpts) -> TranslateResult:
        if M2M100Tokenizer is None:
            return TranslateResult(provider_id=self.info.id, ok=False,
                                   error="transformers not installed")

        def _one(tok, model, target_lang: str) -> str:
            tok.src_lang = "ur"
            enc = tok(text, return_tensors="pt")
            gen = model.generate(**enc, forced_bos_token_id=tok.get_lang_id(target_lang),
                                 max_new_tokens=256)
            return tok.batch_decode(gen, skip_special_tokens=True)[0]

        def _call():
            tok, model = _get()
            out = {}
            if "english" in opts.targets:
                out["english"] = _one(tok, model, "en")
            if "hindi" in opts.targets:
                out["hindi"] = _one(tok, model, "hi")
            return out

        t = self._timed(_call)
        if not t["ok"]:
            return TranslateResult(provider_id=self.info.id, ok=False,
                                   error=t["error"], ms=t["ms"])
        v = t["value"]
        return TranslateResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                               english=v.get("english", ""), hindi=v.get("hindi", ""))


PROVIDER = M2m100Translate()
