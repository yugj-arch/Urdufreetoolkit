from providers.base import BaseProvider, ProviderInfo, Capability


class _Fake(BaseProvider):
    """available() itself blows up — like an engine whose check imports a
    library the deploy doesn't have (torch on Vercel)."""
    info = ProviderInfo(id="fake_broken", label="Fake Broken", capability=Capability.OCR,
                        kind="offline")

    def available(self):
        import torch_not_installed_anywhere  # noqa: F401
        return (True, "")


PROVIDER = _Fake()
