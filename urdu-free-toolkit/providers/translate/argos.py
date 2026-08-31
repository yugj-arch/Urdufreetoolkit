# -*- coding: utf-8 -*-
"""Argos Translate as a translation provider (offline after first download).

Small CPU-friendly NMT. The first Urdu translation downloads the ur->en model
package (~100 MB); Hindi is produced by pivoting through English when a direct
ur->hi package is not available.
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
    import argostranslate.package
    import argostranslate.translate
except Exception:  # noqa: BLE001
    argostranslate = None

_ready = {"ur_en": False}


def _ensure(from_code: str, to_code: str) -> None:
    langs = {(l.code) for l in argostranslate.translate.get_installed_languages()}
    if from_code in langs and to_code in langs:
        return
    argostranslate.package.update_package_index()
    for pkg in argostranslate.package.get_available_packages():
        if pkg.from_code == from_code and pkg.to_code == to_code:
            argostranslate.package.install_from_path(pkg.download())
            return


def _translate(text: str, from_code: str, to_code: str) -> str:
    _ensure(from_code, to_code)
    return argostranslate.translate.translate(text, from_code, to_code)


class ArgosTranslate(BaseProvider):
    info = ProviderInfo(
        id="argos",
        label="Argos Translate (offline)",
        capability=Capability.TRANSLATE,
        kind="offline",
        note="Small offline NMT. First run downloads the ur->en model (~100 MB).",
    )

    def available(self) -> tuple[bool, str]:
        return (True, "") if argostranslate is not None else (False, "pip install argostranslate")

    def translate(self, text: str, opts: TranslateOpts) -> TranslateResult:
        if argostranslate is None:
            return TranslateResult(provider_id=self.info.id, ok=False,
                                   error="argostranslate not installed")

        def _call():
            out = {}
            en = _translate(text, "ur", "en")
            if "english" in opts.targets:
                out["english"] = en
            if "hindi" in opts.targets:
                try:
                    out["hindi"] = _translate(text, "ur", "hi")
                except Exception:  # noqa: BLE001 - no direct package: pivot via English
                    out["hindi"] = _translate(en, "en", "hi")
            return out

        t = self._timed(_call)
        if not t["ok"]:
            return TranslateResult(provider_id=self.info.id, ok=False,
                                   error=t["error"], ms=t["ms"])
        v = t["value"]
        return TranslateResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                               english=v.get("english", ""), hindi=v.get("hindi", ""))


PROVIDER = ArgosTranslate()
