from providers.base import (
    BaseProvider,
    Capability,
    ProviderInfo,
    TranslitOpts,
    TranslitResult,
)


class _Fake(BaseProvider):
    info = ProviderInfo(id="tr_fake", label="Fake Translit", capability=Capability.TRANSLIT,
                        kind="offline")

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        return TranslitResult(provider_id="tr_fake", devanagari="देव",
                              roman="dev", roman_diacritic="dev-dia")


PROVIDER = _Fake()
