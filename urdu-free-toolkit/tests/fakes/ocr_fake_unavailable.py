from providers.base import BaseProvider, ProviderInfo, Capability


class _Fake(BaseProvider):
    info = ProviderInfo(id="fake_no", label="Fake No", capability=Capability.OCR,
                        kind="api", needs=["FAKE_KEY"])

    def available(self):
        return (False, "FAKE_KEY not set")


PROVIDER = _Fake()
