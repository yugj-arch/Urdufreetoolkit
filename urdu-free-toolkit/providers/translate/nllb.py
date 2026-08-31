# -*- coding: utf-8 -*-
"""NLLB-200 (distilled 600M) as a translation provider (offline).

Strong multilingual model with real Perso-Arabic Urdu support (``urd_Arab``).
First call downloads ~2.4 GB from the Hugging Face hub, then runs offline on CPU.
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
    from transformers import pipeline
except Exception:  # noqa: BLE001
    pipeline = None

_MODEL = "facebook/nllb-200-distilled-600M"
_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = pipeline("translation", model=_MODEL)
    return _pipe


class NllbTranslate(BaseProvider):
    info = ProviderInfo(
        id="nllb",
        label="NLLB-200 600M (offline)",
        capability=Capability.TRANSLATE,
        kind="offline",
        note="Meta NLLB, real Urdu support. First run downloads ~2.4 GB.",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if pipeline is not None else (False, "pip install transformers torch")

    def translate(self, text: str, opts: TranslateOpts) -> TranslateResult:
        if pipeline is None:
            return TranslateResult(provider_id=self.info.id, ok=False,
                                   error="transformers not installed")

        def _call():
            pipe = _get_pipe()
            out = {}
            if "english" in opts.targets:
                out["english"] = pipe(text, src_lang="urd_Arab", tgt_lang="eng_Latn")[0]["translation_text"]
            if "hindi" in opts.targets:
                out["hindi"] = pipe(text, src_lang="urd_Arab", tgt_lang="hin_Deva")[0]["translation_text"]
            return out

        t = self._timed(_call)
        if not t["ok"]:
            return TranslateResult(provider_id=self.info.id, ok=False,
                                   error=t["error"], ms=t["ms"])
        v = t["value"]
        return TranslateResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                               english=v.get("english", ""), hindi=v.get("hindi", ""))


PROVIDER = NllbTranslate()
