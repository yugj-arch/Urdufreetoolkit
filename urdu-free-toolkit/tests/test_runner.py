import threading
import time
from concurrent.futures import ThreadPoolExecutor

from providers import registry
from providers.base import Capability, Result, BaseProvider, ProviderInfo
import runner


class _P(BaseProvider):
    def __init__(self, pid, behavior, capability=Capability.OCR, kind="offline"):
        self.info = ProviderInfo(id=pid, label=pid, capability=capability, kind=kind)
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
    # "fast" is an API engine so the offline-serialization lock can't reorder it.
    _install(monkeypatch, [_P("fast", "ok", kind="api"), _P("slow", "slow")])
    seen = [r.provider_id for r in runner.stream(
        Capability.OCR, ["slow", "fast"], lambda p: p.go(), timeout_s=5)]
    assert seen[0] == "fast"  # completes first despite being listed second
    assert set(seen) == {"fast", "slow"}


class _Overlap(BaseProvider):
    """Records whether its window ever overlaps another _Overlap instance."""

    live = 0
    max_live = 0
    _guard = threading.Lock()

    def __init__(self, pid, kind, secs=0.3):
        self.info = ProviderInfo(id=pid, label=pid, capability=Capability.OCR, kind=kind)
        self.secs = secs

    def go(self):
        with _Overlap._guard:
            _Overlap.live += 1
            _Overlap.max_live = max(_Overlap.max_live, _Overlap.live)
        try:
            time.sleep(self.secs)
        finally:
            with _Overlap._guard:
                _Overlap.live -= 1
        return Result(provider_id=self.info.id, ok=True)


def test_offline_engines_never_run_concurrently(monkeypatch):
    """Two heavy native engines in one batch must not do inference at the same
    time — that combination deadlocks the process in the real app."""
    _Overlap.live = _Overlap.max_live = 0
    _install(monkeypatch, [_Overlap("p", "offline"), _Overlap("e", "offline")])
    out = runner.run(Capability.OCR, ["p", "e"], lambda p: p.go(), timeout_s=5)
    assert all(r.ok for r in out)
    assert _Overlap.max_live == 1  # serialized


def test_api_engine_still_overlaps_offline(monkeypatch):
    """The offline lock must not stall API engines — they keep running in
    parallel with (and alongside) an offline engine."""
    _Overlap.live = _Overlap.max_live = 0
    _install(monkeypatch, [_Overlap("offline", "offline"), _Overlap("api", "api")])
    start = time.monotonic()
    out = runner.run(Capability.OCR, ["offline", "api"], lambda p: p.go(), timeout_s=5)
    elapsed = time.monotonic() - start
    assert all(r.ok for r in out)
    assert _Overlap.max_live == 2          # ran together
    assert elapsed < 0.6                   # ~0.3s, not 0.6s serialized


class _OverlapTr(_Overlap):
    """An _Overlap whose capability is TRANSLIT rather than OCR."""

    def __init__(self, pid, kind, secs=0.3):
        super().__init__(pid, kind, secs)
        self.info = ProviderInfo(id=pid, label=pid,
                                 capability=Capability.TRANSLIT, kind=kind)


def test_offline_translit_engines_are_not_serialized(monkeypatch):
    """Offline transliteration engines are plain string transforms with no ML
    runtime — they must run in parallel, not queue on the OCR inference lock."""
    _Overlap.live = _Overlap.max_live = 0
    _install(monkeypatch, [_OverlapTr("rule", "offline"), _OverlapTr("uroman", "offline")])
    out = runner.run(Capability.TRANSLIT, ["rule", "uroman"], lambda p: p.go(), timeout_s=5)
    assert all(r.ok for r in out)
    assert _Overlap.max_live == 2          # ran together, lock not taken


def test_offline_translit_overlaps_offline_ocr(monkeypatch):
    """A trivial offline transliteration must not wait out a slow offline OCR
    model load holding _OFFLINE_LOCK — the button greys for the whole wait."""
    _Overlap.live = _Overlap.max_live = 0
    _install(monkeypatch, [_Overlap("paddle", "offline", secs=0.5),
                           _OverlapTr("rule", "offline", secs=0.05)])
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_ocr = ex.submit(runner.run, Capability.OCR, ["paddle"],
                          lambda p: p.go(), 5)
        time.sleep(0.05)  # let the OCR worker take the lock first
        f_tr = ex.submit(runner.run, Capability.TRANSLIT, ["rule"],
                         lambda p: p.go(), 5)
        assert f_tr.result()[0].ok
        tr_done = time.monotonic() - start
        assert f_ocr.result()[0].ok
    assert tr_done < 0.3  # translit returned while OCR (0.5s) was still running
