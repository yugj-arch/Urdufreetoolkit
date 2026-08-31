import time

from providers import registry
from providers.base import Capability, Result, BaseProvider, ProviderInfo
import runner


class _P(BaseProvider):
    def __init__(self, pid, behavior, capability=Capability.OCR):
        self.info = ProviderInfo(id=pid, label=pid, capability=capability, kind="offline")
        self.behavior = behavior

    def go(self):
        if self.behavior == "ok":
            return Result(provider_id=self.info.id, ok=True)
        if self.behavior == "raise":
            raise RuntimeError("kaboom")
        if self.behavior == "slow":
            time.sleep(2.0)
            return Result(provider_id=self.info.id, ok=True)


def _install(monkeypatch, providers):
    reg = {(p.info.capability, p.info.id): p for p in providers}
    monkeypatch.setattr(registry, "get", lambda cap, pid: reg[(cap, pid)])


def test_run_isolates_failure(monkeypatch):
    _install(monkeypatch, [_P("a", "ok"), _P("b", "raise"), _P("c", "ok")])
    out = runner.run(Capability.OCR, ["a", "b", "c"], lambda p: p.go(), timeout_s=5)
    assert [r.provider_id for r in out] == ["a", "b", "c"]
    assert out[0].ok and not out[1].ok and out[2].ok
    assert "kaboom" in out[1].error


def test_run_timeout(monkeypatch):
    _install(monkeypatch, [_P("a", "ok"), _P("s", "slow")])
    out = runner.run(Capability.OCR, ["a", "s"], lambda p: p.go(), timeout_s=0.5)
    assert out[0].ok is True
    assert out[1].ok is False and "timed out" in out[1].error


def test_run_unknown_provider(monkeypatch):
    reg = {(Capability.OCR, "a"): _P("a", "ok")}
    monkeypatch.setattr(registry, "get", lambda cap, pid: reg[(cap, pid)])
    out = runner.run(Capability.OCR, ["a", "ghost"], lambda p: p.go(), timeout_s=2)
    assert out[0].ok is True
    assert out[1].ok is False and "unknown provider" in out[1].error


def test_stream_yields_as_completed(monkeypatch):
    _install(monkeypatch, [_P("fast", "ok"), _P("slow", "slow")])
    seen = [r.provider_id for r in runner.stream(
        Capability.OCR, ["slow", "fast"], lambda p: p.go(), timeout_s=5)]
    assert seen[0] == "fast"  # completes first despite being listed second
    assert set(seen) == {"fast", "slow"}
