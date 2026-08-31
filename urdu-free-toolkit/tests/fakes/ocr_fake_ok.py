from providers.base import BaseProvider, ProviderInfo, Capability, OcrResult


class _Fake(BaseProvider):
    info = ProviderInfo(id="fake_ok", label="Fake OK", capability=Capability.OCR, kind="offline")

    def ocr(self, image: bytes) -> OcrResult:
        return OcrResult(provider_id="fake_ok", text="salaam")


PROVIDER = _Fake()
