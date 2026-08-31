# Urdu Toolkit — Phase 1: Provider Foundation + Compare UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the API-only pipeline with a pluggable provider registry + parallel runner, wrap the three things that already work (rule transliterator, GPT OCR/translit, GPT image render) as providers, and ship a single-image "run several engines and compare side by side" UI.

**Architecture:** Every capability (OCR, transliteration, translation, render) is a list of small provider modules under `providers/<capability>/`. A registry auto-discovers them and reports which are usable right now (`available()`). A runner executes a chosen set in parallel with per-provider timeouts, isolating failures to their own result row. Flask exposes `/api/providers` + per-capability endpoints; the OCR endpoint streams results over SSE so the UI fills comparison columns as each engine finishes.

**Tech Stack:** Python 3.11+, Flask, Pillow, python-dotenv, `openai` SDK, `concurrent.futures`, server-sent events, vanilla JS/CSS (no CDN).

**Spec:** `docs/superpowers/specs/2026-08-31-urdu-toolkit-multi-provider-design.md`

## Global Constraints

- No external fonts / CDNs in the frontend — system font stacks only.
- Every provider must run **CPU-only**; GPU is never required.
- A provider that cannot import its library, is missing an API key, or is
  missing a binary must return `available() -> (False, reason)` and must
  **never** raise into a request path.
- Provider failure is data: results carry `ok: bool` / `error: str`, never
  propagate as HTTP 500 for one engine among several.
- API keys are read from `.env` next to `app.py`; never logged, never returned
  to the client after being saved.
- `MAX_CONTENT_LENGTH` = 32 MB.
- Python 3.11+ (uses `tuple[bool, str]` / `list[...]` builtins generics, `X | None`).
- Tests must pass without any heavy ML library installed — Phase 1 depends only
  on `openai` (mocked in tests) plus stdlib + Flask + Pillow.

---

### Task 1: Provider base types

**Files:**
- Create: `urdu-free-toolkit/providers/__init__.py` (empty)
- Create: `urdu-free-toolkit/providers/base.py`
- Test: `urdu-free-toolkit/tests/test_base.py`
- Create: `urdu-free-toolkit/tests/__init__.py` (empty)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class Capability(str, Enum)` with members `OCR="ocr"`, `TRANSLIT="translit"`, `TRANSLATE="translate"`, `RENDER="render"`.
  - `@dataclass ProviderInfo` fields: `id: str`, `label: str`, `capability: Capability`, `kind: str` (`"offline"|"api"`), `needs: list[str]` (default `[]`), `note: str` (default `""`).
  - `@dataclass class Result` base fields: `provider_id: str`, `ok: bool = True`, `error: str = ""`, `ms: int = 0`, `meta: dict = field(default_factory=dict)`.
  - `@dataclass class OcrResult(Result)`: `text: str = ""`, `boxes: list | None = None`, `notes: str = ""`.
  - `@dataclass class TranslitResult(Result)`: `devanagari: str = ""`, `roman: str = ""`.
  - `@dataclass class TranslateResult(Result)`: `english: str = ""`, `hindi: str = ""`.
  - `@dataclass class RenderResult(Result)`: `png: bytes = b""`.
  - `@dataclass TranslitOpts`: `roman_style: str = "natural"`, `targets: tuple[str, ...] = ("devanagari", "roman")`.
  - `@dataclass TranslateOpts`: `targets: tuple[str, ...] = ("english",)`.
  - `class BaseProvider`: holds `info: ProviderInfo`; default `available(self) -> tuple[bool, str]` returns `(True, "")`; helper `_timed(self, fn, *a, **kw)` that runs `fn`, catches `Exception`, and returns a partially-filled result dict `{"ok":..., "error":..., "ms":...}` — subclasses merge this into their concrete Result.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base.py
from providers.base import (
    Capability, ProviderInfo, OcrResult, TranslitResult, TranslitOpts, BaseProvider,
)

def test_capability_values():
    assert Capability.OCR == "ocr"
    assert Capability.TRANSLATE == "translate"

def test_result_defaults():
    r = OcrResult(provider_id="x")
    assert r.ok is True and r.error == "" and r.ms == 0 and r.text == "" and r.boxes is None
    assert r.meta == {}

def test_translit_opts_defaults():
    o = TranslitOpts()
    assert o.roman_style == "natural"
    assert o.targets == ("devanagari", "roman")

def test_base_provider_available_default():
    p = BaseProvider()
    p.info = ProviderInfo(id="x", label="X", capability=Capability.OCR, kind="offline")
    assert p.available() == (True, "")

def test_timed_catches_exception():
    p = BaseProvider()
    def boom():
        raise ValueError("nope")
    out = p._timed(boom)
    assert out["ok"] is False
    assert "nope" in out["error"]
    assert isinstance(out["ms"], int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers.base'`

- [ ] **Step 3: Write minimal implementation**

```python
# providers/base.py
from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum


class Capability(str, Enum):
    OCR = "ocr"
    TRANSLIT = "translit"
    TRANSLATE = "translate"
    RENDER = "render"


@dataclass
class ProviderInfo:
    id: str
    label: str
    capability: Capability
    kind: str  # "offline" | "api"
    needs: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class Result:
    provider_id: str
    ok: bool = True
    error: str = ""
    ms: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class OcrResult(Result):
    text: str = ""
    boxes: list | None = None
    notes: str = ""


@dataclass
class TranslitResult(Result):
    devanagari: str = ""
    roman: str = ""


@dataclass
class TranslateResult(Result):
    english: str = ""
    hindi: str = ""


@dataclass
class RenderResult(Result):
    png: bytes = b""


@dataclass
class TranslitOpts:
    roman_style: str = "natural"
    targets: tuple[str, ...] = ("devanagari", "roman")


@dataclass
class TranslateOpts:
    targets: tuple[str, ...] = ("english",)


class BaseProvider:
    info: ProviderInfo

    def available(self) -> tuple[bool, str]:
        return (True, "")

    def _timed(self, fn, *args, **kwargs) -> dict:
        start = time.monotonic()
        try:
            value = fn(*args, **kwargs)
            return {"ok": True, "error": "", "ms": int((time.monotonic() - start) * 1000), "value": value}
        except Exception as e:  # noqa: BLE001 - provider failures are data
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "ms": int((time.monotonic() - start) * 1000), "value": None}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_base.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/__init__.py urdu-free-toolkit/providers/base.py urdu-free-toolkit/tests/__init__.py urdu-free-toolkit/tests/test_base.py
git commit -m "feat: provider base types (Capability, Result dataclasses, BaseProvider)"
```

---

### Task 2: Provider registry

**Files:**
- Create: `urdu-free-toolkit/providers/registry.py`
- Test: `urdu-free-toolkit/tests/test_registry.py`
- Create: `urdu-free-toolkit/tests/fakes/__init__.py` (empty)
- Create: `urdu-free-toolkit/tests/fakes/ocr_fake_ok.py`
- Create: `urdu-free-toolkit/tests/fakes/ocr_fake_unavailable.py`

**Interfaces:**
- Consumes: `providers.base` (`Capability`, `ProviderInfo`, `BaseProvider`).
- Produces:
  - `discover(package: str = "providers") -> dict[str, BaseProvider]` — imports every
    submodule under `providers/ocr`, `providers/translit`, `providers/translate`,
    `providers/render`; each provider module defines a module-level `PROVIDER`
    instance (a `BaseProvider` subclass with `.info`). Returns `{info.id: instance}`.
    A module that raises `ImportError` at import time is skipped silently (its lib
    isn't installed); any other exception is re-raised.
  - `get(capability: Capability, provider_id: str) -> BaseProvider` — raises `KeyError`
    if unknown. Two-arg because the same `id` (e.g. `"gpt"`) legitimately exists for
    more than one capability; the registry is keyed by `"<capability>:<id>"`.
  - `for_ui(capability: Capability | None = None) -> list[dict]` — each dict:
    `{"id","label","capability","kind","note","available":bool,"reason":str,"badge":str}`
    where `badge` is `"offline"` / `"needs key"` / `"free (net)"` / `"not installed"`
    derived from `kind` + `needs` + `available()`.
  - Discovery result is cached in a module global; `reset_cache()` clears it (used
    by tests and by the settings endpoint after a key is saved).

- [ ] **Step 1: Write the failing test**

```python
# tests/fakes/ocr_fake_ok.py
from providers.base import BaseProvider, ProviderInfo, Capability, OcrResult

class _Fake(BaseProvider):
    info = ProviderInfo(id="fake_ok", label="Fake OK", capability=Capability.OCR, kind="offline")
    def ocr(self, image: bytes) -> OcrResult:
        return OcrResult(provider_id="fake_ok", text="salaam")

PROVIDER = _Fake()
```

```python
# tests/fakes/ocr_fake_unavailable.py
from providers.base import BaseProvider, ProviderInfo, Capability

class _Fake(BaseProvider):
    info = ProviderInfo(id="fake_no", label="Fake No", capability=Capability.OCR,
                        kind="api", needs=["FAKE_KEY"])
    def available(self):
        return (False, "FAKE_KEY not set")

PROVIDER = _Fake()
```

```python
# tests/test_registry.py
import importlib
from providers import registry
from providers.base import Capability

def setup_function():
    registry.reset_cache()

def test_discover_loads_fake_package(monkeypatch):
    reg = registry.discover(package="tests.fakes")
    assert "fake_ok" in reg and "fake_no" in reg

def test_get_unknown_raises():
    registry.discover(package="tests.fakes")
    try:
        registry.get("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass

def test_for_ui_badges():
    registry.discover(package="tests.fakes")
    rows = {r["id"]: r for r in registry.for_ui(Capability.OCR)}
    assert rows["fake_ok"]["available"] is True
    assert rows["fake_ok"]["badge"] == "offline"
    assert rows["fake_no"]["available"] is False
    assert rows["fake_no"]["badge"] == "needs key"
    assert rows["fake_no"]["reason"] == "FAKE_KEY not set"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# providers/registry.py
from __future__ import annotations
import importlib
import pkgutil

from providers.base import BaseProvider, Capability

_SUBPACKAGES = ("ocr", "translit", "translate", "render")
_cache: dict[str, BaseProvider] | None = None


def reset_cache() -> None:
    global _cache
    _cache = None


def _iter_modules(package: str):
    for sub in _SUBPACKAGES:
        name = f"{package}.{sub}"
        try:
            pkg = importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        for m in pkgutil.iter_modules(pkg.__path__):
            if m.name.startswith("_"):
                continue
            yield f"{name}.{m.name}"


def discover(package: str = "providers") -> dict[str, BaseProvider]:
    global _cache
    if _cache is not None and package == "providers":
        return _cache
    found: dict[str, BaseProvider] = {}
    for mod_name in _iter_modules(package):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue  # library for this provider isn't installed
        provider = getattr(mod, "PROVIDER", None)
        if provider is None:
            continue
        found[provider.info.id] = provider
    if package == "providers":
        _cache = found
    return found


def get(provider_id: str) -> BaseProvider:
    reg = discover()
    return reg[provider_id]


def _badge(p: BaseProvider, ok: bool) -> str:
    if ok:
        return "offline" if p.info.kind == "offline" else (
            "free (net)" if not p.info.needs else "needs key")
    if p.info.kind == "api":
        return "needs key" if p.info.needs else "free (net)"
    return "not installed"


def for_ui(capability: Capability | None = None) -> list[dict]:
    reg = discover(package="tests.fakes") if _cache is None and False else discover(
        package="providers") if capability is None or _cache is not None else discover()
    # simple, predictable: always use whatever discover() cached
    reg = _cache if _cache is not None else discover()
    rows: list[dict] = []
    for p in reg.values():
        if capability is not None and p.info.capability != capability:
            continue
        ok, reason = p.available()
        rows.append({
            "id": p.info.id, "label": p.info.label,
            "capability": p.info.capability.value, "kind": p.info.kind,
            "note": p.info.note, "available": ok, "reason": reason,
            "badge": _badge(p, ok),
        })
    rows.sort(key=lambda r: (not r["available"], r["label"].lower()))
    return rows
```

> Note for implementer: the `for_ui` body above has a deliberately simplified
> cache lookup — replace the first two lines of its body with just
> `reg = discover(package="providers")` if `_cache` is set by the test via
> `discover(package="tests.fakes")`. To keep tests and prod aligned, change
> `discover` so that when called with a non-default package it ALSO sets
> `_cache` (guard: only when `_cache is None`). Then `for_ui` is simply
> `reg = _cache or discover()`. Apply that simplification now.

- [ ] **Step 3b: Apply the cache simplification**

Edit `discover` so a non-default package populates the cache when empty:

```python
    if package == "providers" or _cache is None:
        _cache = found
    return found
```

Replace the messy first lines of `for_ui` body with:

```python
    reg = _cache if _cache is not None else discover()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_registry.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/registry.py urdu-free-toolkit/tests/test_registry.py urdu-free-toolkit/tests/fakes/
git commit -m "feat: provider registry with discovery, availability gate, UI metadata"
```

---

### Task 3: Parallel runner

**Files:**
- Create: `urdu-free-toolkit/runner.py`
- Test: `urdu-free-toolkit/tests/test_runner.py`

**Interfaces:**
- Consumes: `providers.registry.get`, `providers.base` result types.
- Produces:
  - `run(capability: Capability, provider_ids: list[str], call: Callable[[BaseProvider], Result], timeout_s: float = 60.0) -> list[Result]`
    - resolves each id via `registry.get(capability, id)`; unknown id → a `Result(provider_id=id, ok=False, error="unknown provider")`.
    - runs `call(provider)` for each in a `ThreadPoolExecutor(max_workers=max(1,len(ids)))`.
    - each future awaited with `timeout_s`; on `TimeoutError` →
      `Result(provider_id=id, ok=False, error=f"timed out after {timeout_s:.0f}s")`.
    - any exception from `call` → `Result(provider_id=id, ok=False, error="...")`.
    - returns results in the **same order** as `provider_ids`.
  - `stream(capability, provider_ids, call, timeout_s=60.0) -> Iterator[Result]`
    - same as `run` but `yield`s each `Result` as it completes (order = completion order). Used by the SSE endpoint.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
import time
from providers import registry
from providers.base import Capability, Result, BaseProvider, ProviderInfo
import runner

class _P(BaseProvider):
    def __init__(self, pid, behavior):
        self.info = ProviderInfo(id=pid, label=pid, capability=Capability.OCR, kind="offline")
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
    reg = {p.info.id: p for p in providers}
    monkeypatch.setattr(registry, "get", lambda pid: reg[pid])

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
    _install(monkeypatch, [_P("a", "ok")])
    monkeypatch.setattr(registry, "get", lambda pid: {"a": _P("a", "ok")}[pid])
    out = runner.run(Capability.OCR, ["a", "ghost"], lambda p: p.go(), timeout_s=2)
    assert out[0].ok is True
    assert out[1].ok is False and "unknown provider" in out[1].error

def test_stream_yields_as_completed(monkeypatch):
    _install(monkeypatch, [_P("fast", "ok"), _P("slow", "slow")])
    seen = [r.provider_id for r in runner.stream(Capability.OCR, ["slow", "fast"], lambda p: p.go(), timeout_s=5)]
    assert seen[0] == "fast"  # completes first despite being listed second
    assert set(seen) == {"fast", "slow"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runner'`

- [ ] **Step 3: Write minimal implementation**

```python
# runner.py
from __future__ import annotations
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from concurrent.futures import TimeoutError as FutTimeout

from providers import registry
from providers.base import Capability, Result, BaseProvider


def _resolve(pid: str) -> BaseProvider | None:
    try:
        return registry.get(pid)
    except KeyError:
        return None


def run(capability: Capability, provider_ids: list[str],
        call: Callable[[BaseProvider], Result], timeout_s: float = 60.0) -> list[Result]:
    by_id: dict[str, Result] = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, len(provider_ids))) as pool:
        for pid in provider_ids:
            prov = _resolve(pid)
            if prov is None:
                by_id[pid] = Result(provider_id=pid, ok=False, error="unknown provider")
                continue
            futures[pool.submit(call, prov)] = pid
        for fut, pid in futures.items():
            try:
                by_id[pid] = fut.result(timeout=timeout_s)
            except FutTimeout:
                by_id[pid] = Result(provider_id=pid, ok=False,
                                    error=f"timed out after {timeout_s:.0f}s")
            except Exception as e:  # noqa: BLE001
                by_id[pid] = Result(provider_id=pid, ok=False,
                                    error=f"{type(e).__name__}: {e}")
    return [by_id[pid] for pid in provider_ids]


def stream(capability: Capability, provider_ids: list[str],
           call: Callable[[BaseProvider], Result], timeout_s: float = 60.0) -> Iterator[Result]:
    with ThreadPoolExecutor(max_workers=max(1, len(provider_ids))) as pool:
        futures = {}
        for pid in provider_ids:
            prov = _resolve(pid)
            if prov is None:
                yield Result(provider_id=pid, ok=False, error="unknown provider")
                continue
            futures[pool.submit(call, prov)] = pid
        pending = set(futures)
        deadline_map = {f: timeout_s for f in pending}
        while pending:
            done, pending = wait(pending, timeout=timeout_s, return_when=FIRST_COMPLETED)
            if not done:  # everything left has exceeded the wall clock
                for f in pending:
                    yield Result(provider_id=futures[f], ok=False,
                                 error=f"timed out after {timeout_s:.0f}s")
                    f.cancel()
                return
            for f in done:
                pid = futures[f]
                try:
                    yield f.result(timeout=0)
                except Exception as e:  # noqa: BLE001
                    yield Result(provider_id=pid, ok=False, error=f"{type(e).__name__}: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_runner.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/runner.py urdu-free-toolkit/tests/test_runner.py
git commit -m "feat: parallel provider runner with per-provider timeout + SSE stream"
```

---

### Task 4: Wrap the rule-based transliterator as a provider

**Files:**
- Create: `urdu-free-toolkit/providers/translit/__init__.py` (empty)
- Create: `urdu-free-toolkit/providers/translit/rule.py`
- Test: `urdu-free-toolkit/tests/test_translit_rule.py`

**Interfaces:**
- Consumes: existing `transliterate.transliterate(text) -> (deva, roman)`; `providers.base`.
- Produces: module-level `PROVIDER` — `info.id = "rule"`, `capability = TRANSLIT`,
  `kind = "offline"`, `label = "Rule engine (offline)"`.
  Method `translit(self, text: str, opts: TranslitOpts) -> TranslitResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_translit_rule.py
from providers.translit.rule import PROVIDER
from providers.base import TranslitOpts, Capability

def test_info():
    assert PROVIDER.info.id == "rule"
    assert PROVIDER.info.capability == Capability.TRANSLIT
    assert PROVIDER.available() == (True, "")

def test_translit_known_words():
    r = PROVIDER.translit("میں ٹھیک ہوں", TranslitOpts())
    assert r.ok is True
    assert "में" in r.devanagari or "मैं" in r.devanagari  # rule engine defaults to "mein"
    assert r.roman.startswith(("mein", "main"))
    assert r.ms >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_translit_rule.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers.translit.rule'`

- [ ] **Step 3: Write minimal implementation**

```python
# providers/translit/rule.py
from __future__ import annotations

from providers.base import BaseProvider, ProviderInfo, Capability, TranslitResult, TranslitOpts
import transliterate as _rule


class RuleTranslit(BaseProvider):
    info = ProviderInfo(
        id="rule", label="Rule engine (offline)", capability=Capability.TRANSLIT,
        kind="offline", note="Dictionary + heuristic. Free, deterministic, no vowel restoration.",
    )

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        t = self._timed(_rule.transliterate, text)
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        deva, roman = t["value"]
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari=deva, roman=roman)


PROVIDER = RuleTranslit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_translit_rule.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/translit/__init__.py urdu-free-toolkit/providers/translit/rule.py urdu-free-toolkit/tests/test_translit_rule.py
git commit -m "feat: wrap rule-based transliterator as 'rule' provider"
```

---

### Task 5: Wrap GPT OCR + GPT transliteration as providers

**Files:**
- Create: `urdu-free-toolkit/providers/ocr/__init__.py` (empty)
- Create: `urdu-free-toolkit/providers/_openai_common.py`
- Create: `urdu-free-toolkit/providers/ocr/gpt.py`
- Create: `urdu-free-toolkit/providers/translit/gpt.py`
- Test: `urdu-free-toolkit/tests/test_provider_gpt.py`

**Interfaces:**
- Consumes: `openai` SDK; `providers.base`. Reuses the prompts + `_prepare_image`
  + `_parse_json` logic currently in `ocr.py` (copy them into `_openai_common.py`).
- Produces:
  - `providers/_openai_common.py`: `get_client()`, `prepare_image(bytes) -> (b64, mime)`,
    `parse_json(str) -> dict`, `VISION_SYSTEM`, `TEXT_SYSTEM`, `MODEL` (env `OPENAI_MODEL`, default `"gpt-4o"`), `have_key() -> bool`.
  - `providers/ocr/gpt.py`: `PROVIDER` — `info.id="gpt"`, `capability=OCR`, `kind="api"`,
    `needs=["OPENAI_API_KEY"]`. `available()` → `(False, "OPENAI_API_KEY not set")` when
    `not have_key()`. `ocr(self, image: bytes) -> OcrResult` — one vision call, fills
    `text` + `notes`, `meta={"model": MODEL, "devanagari":..., "roman":...}` (the vision
    call also returns translit; stash it so the UI can show it without a second call).
  - `providers/translit/gpt.py`: `PROVIDER` — `info.id="gpt"`, `capability=TRANSLIT`,
    `kind="api"`, `needs=["OPENAI_API_KEY"]`. `translit(text, opts) -> TranslitResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provider_gpt.py
import json
import types
import pytest
from providers.base import TranslitOpts, Capability

def _fake_openai(monkeypatch, payload: dict):
    import providers._openai_common as common
    class _Msg: content = json.dumps(payload)
    class _Choice: message = _Msg()
    class _Resp: choices = [_Choice()]
    class _Chat:
        class completions:
            @staticmethod
            def create(**kw): return _Resp()
    client = types.SimpleNamespace(chat=_Chat())
    monkeypatch.setattr(common, "get_client", lambda: client)
    monkeypatch.setattr(common, "have_key", lambda: True)

def test_gpt_ocr_available_without_key(monkeypatch):
    import providers._openai_common as common
    monkeypatch.setattr(common, "have_key", lambda: False)
    from providers.ocr.gpt import PROVIDER
    ok, reason = PROVIDER.available()
    assert ok is False and "OPENAI_API_KEY" in reason

def test_gpt_ocr_parses_payload(monkeypatch):
    _fake_openai(monkeypatch, {"urdu": "محبت", "devanagari": "मोहब्बत", "roman": "mohabbat", "notes": ""})
    from providers.ocr.gpt import PROVIDER
    # 1x1 png
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000154a24f5f0000000049454e44ae426082")
    r = PROVIDER.ocr(png)
    assert r.ok is True and r.text == "محبت"
    assert r.meta["devanagari"] == "मोहब्बत" and r.meta["roman"] == "mohabbat"

def test_gpt_translit_parses_payload(monkeypatch):
    _fake_openai(monkeypatch, {"devanagari": "मैं ठीक हूँ", "roman": "main theek hoon", "notes": ""})
    from providers.translit.gpt import PROVIDER
    r = PROVIDER.translit("میں ٹھیک ہوں", TranslitOpts())
    assert r.ok is True and r.devanagari == "मैं ठीक हूँ" and r.roman == "main theek hoon"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_provider_gpt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers._openai_common'`

- [ ] **Step 3: Write minimal implementation**

Copy `_VISION_SYSTEM`, `_TEXT_SYSTEM`, `_prepare_image`, `_parse_json`, `_get_client`,
`OPENAI_MODEL` out of `ocr.py` into `_openai_common.py`, renamed as the interface lists.
Add:

```python
# providers/_openai_common.py  (excerpt of the new parts)
import os

def have_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))
```

```python
# providers/ocr/gpt.py
from __future__ import annotations
from providers.base import BaseProvider, ProviderInfo, Capability, OcrResult
from providers import _openai_common as common


class GptOcr(BaseProvider):
    info = ProviderInfo(id="gpt", label="OpenAI GPT-4o vision", capability=Capability.OCR,
                        kind="api", needs=["OPENAI_API_KEY"],
                        note="Best overall; restores short vowels from context. ~1-2¢/image.")

    def available(self):
        return (True, "") if common.have_key() else (False, "OPENAI_API_KEY not set")

    def ocr(self, image: bytes) -> OcrResult:
        def _call():
            b64, mime = common.prepare_image(image)
            client = common.get_client()
            resp = client.chat.completions.create(
                model=common.MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": common.VISION_SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Transcribe and transliterate the Urdu in this image."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ]},
                ],
            )
            return common.parse_json(resp.choices[0].message.content)
        t = self._timed(_call)
        if not t["ok"]:
            return OcrResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        d = t["value"]
        return OcrResult(
            provider_id=self.info.id, ok=True, ms=t["ms"],
            text=(d.get("urdu") or "").strip(), notes=(d.get("notes") or "").strip(),
            meta={"model": common.MODEL,
                  "devanagari": (d.get("devanagari") or "").strip(),
                  "roman": (d.get("roman") or "").strip()},
        )


PROVIDER = GptOcr()
```

```python
# providers/translit/gpt.py
from __future__ import annotations
from providers.base import BaseProvider, ProviderInfo, Capability, TranslitResult, TranslitOpts
from providers import _openai_common as common


class GptTranslit(BaseProvider):
    info = ProviderInfo(id="gpt", label="OpenAI GPT-4o", capability=Capability.TRANSLIT,
                        kind="api", needs=["OPENAI_API_KEY"],
                        note="Context-aware vowel restoration — the tie-breaker.")

    def available(self):
        return (True, "") if common.have_key() else (False, "OPENAI_API_KEY not set")

    def translit(self, text: str, opts: TranslitOpts) -> TranslitResult:
        def _call():
            client = common.get_client()
            resp = client.chat.completions.create(
                model=common.MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": common.TEXT_SYSTEM},
                    {"role": "user", "content": text},
                ],
            )
            return common.parse_json(resp.choices[0].message.content)
        t = self._timed(_call)
        if not t["ok"]:
            return TranslitResult(provider_id=self.info.id, ok=False, error=t["error"], ms=t["ms"])
        d = t["value"]
        return TranslitResult(provider_id=self.info.id, ok=True, ms=t["ms"],
                              devanagari=(d.get("devanagari") or "").strip(),
                              roman=(d.get("roman") or "").strip())


PROVIDER = GptTranslit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_provider_gpt.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/ocr/__init__.py urdu-free-toolkit/providers/_openai_common.py urdu-free-toolkit/providers/ocr/gpt.py urdu-free-toolkit/providers/translit/gpt.py urdu-free-toolkit/tests/test_provider_gpt.py
git commit -m "feat: wrap GPT-4o OCR + transliteration as 'gpt' providers"
```

---

### Task 6: Wrap GPT image-edit as a render provider

**Files:**
- Create: `urdu-free-toolkit/providers/render/__init__.py` (empty)
- Create: `urdu-free-toolkit/providers/render/gpt_image.py`
- Test: `urdu-free-toolkit/tests/test_provider_gpt_image.py`

**Interfaces:**
- Consumes: `openai` SDK; `providers._openai_common.get_client/have_key`;
  the `_EDIT_PROMPT` + `_png_for_upload` logic from `imgedit.py`.
- Produces: `PROVIDER` — `info.id="gpt_image"`, `capability=RENDER`, `kind="api"`,
  `needs=["OPENAI_API_KEY"]`. `render(self, image: bytes, lines: list[str]) -> RenderResult`
  where `lines` are the already-transliterated lines to burn in.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provider_gpt_image.py
import base64, types
from providers.base import Capability

def test_render_info_and_gate(monkeypatch):
    import providers._openai_common as common
    monkeypatch.setattr(common, "have_key", lambda: False)
    from providers.render.gpt_image import PROVIDER
    assert PROVIDER.info.capability == Capability.RENDER
    ok, reason = PROVIDER.available()
    assert ok is False and "OPENAI_API_KEY" in reason

def test_render_returns_png(monkeypatch):
    import providers._openai_common as common
    png1 = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000154a24f5f0000000049454e44ae426082")
    class _Data: b64_json = base64.b64encode(png1).decode()
    class _Resp: data = [_Data()]
    class _Images:
        @staticmethod
        def edit(**kw): return _Resp()
    monkeypatch.setattr(common, "have_key", lambda: True)
    monkeypatch.setattr(common, "get_client", lambda: types.SimpleNamespace(images=_Images()))
    from providers.render.gpt_image import PROVIDER
    r = PROVIDER.render(png1, ["mohabbat ek ehsaas hai"])
    assert r.ok is True and r.png[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_provider_gpt_image.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'providers.render.gpt_image'`

- [ ] **Step 3: Write minimal implementation**

Port `imgedit.py` logic. `render()` builds the prompt from `"\n".join(lines)`,
calls `client.images.edit(...)` (same kwargs as `imgedit.py`), decodes
`resp.data[0].b64_json`, returns `RenderResult(png=...)`. On any error →
`ok=False, error=...` (keep the "403 → org verification" hint in the message).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_provider_gpt_image.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/providers/render/__init__.py urdu-free-toolkit/providers/render/gpt_image.py urdu-free-toolkit/tests/test_provider_gpt_image.py
git commit -m "feat: wrap gpt-image-1 edit as 'gpt_image' render provider"
```

---

### Task 7: Settings module (.env read/write for API keys)

**Files:**
- Create: `urdu-free-toolkit/settings.py`
- Test: `urdu-free-toolkit/tests/test_settings.py`

**Interfaces:**
- Consumes: stdlib only.
- Produces:
  - `KNOWN_KEYS: tuple[str, ...]` = `("OPENAI_API_KEY","ANTHROPIC_API_KEY","GOOGLE_API_KEY","OCRSPACE_API_KEY","AZURE_VISION_KEY","AZURE_VISION_ENDPOINT","OPENAI_MODEL","HF_TROCR_CKPT")`.
  - `env_path() -> Path` — `.env` next to `app.py` (i.e. repo `urdu-free-toolkit/.env`).
  - `save(values: dict[str, str]) -> list[str]` — upsert each `KNOWN_KEYS` entry present
    in `values` into `.env` (preserve other lines, preserve order, append new keys),
    also `os.environ[k] = v` so it takes effect without restart. Returns the list of
    key **names** written. Ignores unknown keys. Empty string value → remove that line
    and `os.environ.pop`.
  - `status() -> dict[str, bool]` — `{key: bool(os.environ.get(key))}` for `KNOWN_KEYS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_settings.py
import os
import importlib
import settings

def test_save_and_status(tmp_path, monkeypatch):
    envf = tmp_path / ".env"
    monkeypatch.setattr(settings, "env_path", lambda: envf)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    written = settings.save({"OPENAI_API_KEY": "sk-abc", "UNKNOWN": "x"})
    assert written == ["OPENAI_API_KEY"]
    assert "OPENAI_API_KEY=sk-abc" in envf.read_text()
    assert os.environ["OPENAI_API_KEY"] == "sk-abc"
    assert settings.status()["OPENAI_API_KEY"] is True

def test_save_preserves_other_lines(tmp_path, monkeypatch):
    envf = tmp_path / ".env"
    envf.write_text("# comment\nOTHER=keep\nOPENAI_API_KEY=old\n")
    monkeypatch.setattr(settings, "env_path", lambda: envf)
    settings.save({"OPENAI_API_KEY": "new"})
    text = envf.read_text()
    assert "OTHER=keep" in text and "# comment" in text
    assert "OPENAI_API_KEY=new" in text and "old" not in text

def test_empty_value_removes(tmp_path, monkeypatch):
    envf = tmp_path / ".env"
    envf.write_text("OPENAI_API_KEY=old\n")
    monkeypatch.setattr(settings, "env_path", lambda: envf)
    monkeypatch.setenv("OPENAI_API_KEY", "old")
    settings.save({"OPENAI_API_KEY": ""})
    assert "OPENAI_API_KEY" not in envf.read_text()
    assert os.environ.get("OPENAI_API_KEY") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_settings.py -v`
Expected: FAIL — `AttributeError: module 'settings' has no attribute 'env_path'` (or ModuleNotFound)

- [ ] **Step 3: Write minimal implementation**

```python
# settings.py
from __future__ import annotations
import os
from pathlib import Path

KNOWN_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OCRSPACE_API_KEY",
    "AZURE_VISION_KEY", "AZURE_VISION_ENDPOINT", "OPENAI_MODEL", "HF_TROCR_CKPT",
)


def env_path() -> Path:
    return Path(__file__).with_name(".env")


def _read_lines() -> list[str]:
    p = env_path()
    return p.read_text(encoding="utf-8").splitlines() if p.exists() else []


def save(values: dict[str, str]) -> list[str]:
    lines = _read_lines()
    written: list[str] = []
    for key in KNOWN_KEYS:
        if key not in values:
            continue
        val = (values[key] or "").strip()
        idx = next((i for i, ln in enumerate(lines)
                    if ln.strip().startswith(f"{key}=")), None)
        if val == "":
            if idx is not None:
                lines.pop(idx)
            os.environ.pop(key, None)
        else:
            new_line = f"{key}={val}"
            if idx is None:
                lines.append(new_line)
            else:
                lines[idx] = new_line
            os.environ[key] = val
        written.append(key)
    env_path().write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return written


def status() -> dict[str, bool]:
    return {k: bool(os.environ.get(k)) for k in KNOWN_KEYS}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_settings.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/settings.py urdu-free-toolkit/tests/test_settings.py
git commit -m "feat: settings module for reading/writing API keys to .env"
```

---

### Task 8: Flask app over the provider pipeline

**Files:**
- Modify (rewrite): `urdu-free-toolkit/app.py`
- Test: `urdu-free-toolkit/tests/test_app.py`
- Delete: `urdu-free-toolkit/ocr.py`, `urdu-free-toolkit/imgedit.py` (logic now lives in providers)

**Interfaces:**
- Consumes: `providers.registry`, `runner`, `settings`, `providers.base`.
- Produces these routes:
  - `GET /` → `render_template("index.html")`.
  - `GET /api/providers` → `{"ocr":[...], "translit":[...], "translate":[...], "render":[...]}`
    each list from `registry.for_ui(Capability.X)`.
  - `POST /api/ocr` (multipart: `image`, `providers` = comma list) → **SSE**
    (`Content-Type: text/event-stream`). For each `runner.stream` result emit
    `data: {json}\n\n` with `{provider_id, ok, error, ms, text, notes, devanagari, roman}`
    (last two from `meta` when present). Final event `data: {"done": true}\n\n`.
    Missing image → 400 JSON. Empty `providers` → 400 JSON.
  - `POST /api/transliterate` (json: `text`, `providers`, `roman_style`, `targets`)
    → `{"results": [ {provider_id, ok, error, ms, devanagari, roman} ]}` via `runner.run`.
  - `POST /api/render` (multipart: `image`, `provider`, `lines` = newline string)
    → `image/png` bytes on success, else 500 JSON `{error}`.
  - `POST /api/settings` (json: key→value) → `{"saved": [names]}`; then
    `registry.reset_cache()`.
  - `GET /api/settings` → `settings.status()`.
- `app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024`.
- `/api/translate` route is added but returns `{"results": []}` until Phase 3 adds
  translate providers (keeps the frontend contract stable).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app.py
import json
import io
import pytest
from providers import registry

@pytest.fixture
def client(monkeypatch):
    registry.reset_cache()
    registry.discover(package="tests.fakes")  # deterministic provider set
    import app as appmod
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

def test_providers_lists_fake(client):
    data = client.get("/api/providers").get_json()
    ids = [r["id"] for r in data["ocr"]]
    assert "fake_ok" in ids

def test_ocr_requires_image(client):
    r = client.post("/api/ocr", data={"providers": "fake_ok"})
    assert r.status_code == 400

def test_ocr_streams_results(client):
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000154a24f5f0000000049454e44ae426082")
    r = client.post("/api/ocr", data={
        "image": (io.BytesIO(png), "x.png"), "providers": "fake_ok"},
        content_type="multipart/form-data")
    body = r.get_data(as_text=True)
    assert "text/event-stream" in r.content_type
    assert "salaam" in body
    assert '"done": true' in body

def test_settings_roundtrip(client, monkeypatch, tmp_path):
    import settings
    monkeypatch.setattr(settings, "env_path", lambda: tmp_path / ".env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/api/settings", json={"OPENAI_API_KEY": "sk-xyz"})
    assert r.get_json()["saved"] == ["OPENAI_API_KEY"]
```

> The fake package needs a translit provider too for `/api/transliterate` coverage —
> add `tests/fakes/translit_fake.py` mirroring `ocr_fake_ok.py` with
> `capability=Capability.TRANSLIT` and a `translit(self, text, opts)` returning
> `TranslitResult(provider_id="tr_fake", devanagari="देव", roman="dev")`, plus a
> test asserting `/api/transliterate` returns that row.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_app.py -v`
Expected: FAIL — app still imports the old `ocr` module / routes missing.

- [ ] **Step 3: Write minimal implementation**

Rewrite `app.py`:

```python
# -*- coding: utf-8 -*-
"""Flask app: pick engines, run them in parallel, compare. Providers live in
providers/<capability>/. Free/offline providers work with no API key; API
providers light up when their key is set in Settings."""
from __future__ import annotations
import json

from flask import Flask, request, render_template, jsonify, Response

from providers import registry
from providers.base import Capability, TranslitOpts
import runner
import settings

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/providers")
def api_providers():
    return jsonify({c.value: registry.for_ui(c) for c in Capability})


@app.post("/api/ocr")
def api_ocr():
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "No image uploaded."}), 400
    ids = [s for s in (request.form.get("providers") or "").split(",") if s]
    if not ids:
        return jsonify({"error": "Pick at least one OCR engine."}), 400
    image = request.files["image"].read()
    if not image:
        return jsonify({"error": "Uploaded file is empty."}), 400

    def gen():
        for res in runner.stream(Capability.OCR, ids, lambda p: p.ocr(image), timeout_s=120):
            row = {"provider_id": res.provider_id, "ok": res.ok, "error": res.error,
                   "ms": res.ms, "text": getattr(res, "text", ""),
                   "notes": getattr(res, "notes", ""),
                   "devanagari": res.meta.get("devanagari", ""),
                   "roman": res.meta.get("roman", "")}
            yield f"data: {json.dumps(row, ensure_ascii=False)}\n\n"
        yield 'data: {"done": true}\n\n'

    return Response(gen(), mimetype="text/event-stream")


@app.post("/api/transliterate")
def api_transliterate():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    ids = data.get("providers") or []
    if not text:
        return jsonify({"error": "No text provided."}), 400
    opts = TranslitOpts(roman_style=data.get("roman_style", "natural"),
                        targets=tuple(data.get("targets") or ("devanagari", "roman")))
    results = runner.run(Capability.TRANSLIT, ids, lambda p: p.translit(text, opts), timeout_s=120)
    return jsonify({"results": [
        {"provider_id": r.provider_id, "ok": r.ok, "error": r.error, "ms": r.ms,
         "devanagari": getattr(r, "devanagari", ""), "roman": getattr(r, "roman", "")}
        for r in results]})


@app.post("/api/translate")
def api_translate():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    ids = data.get("providers") or []
    if not text:
        return jsonify({"error": "No text provided."}), 400
    from providers.base import TranslateOpts
    opts = TranslateOpts(targets=tuple(data.get("targets") or ("english",)))
    results = runner.run(Capability.TRANSLATE, ids, lambda p: p.translate(text, opts), timeout_s=180)
    return jsonify({"results": [
        {"provider_id": r.provider_id, "ok": r.ok, "error": r.error, "ms": r.ms,
         "english": getattr(r, "english", ""), "hindi": getattr(r, "hindi", "")}
        for r in results]})


@app.post("/api/render")
def api_render():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    pid = request.form.get("provider") or ""
    lines = [ln for ln in (request.form.get("lines") or "").split("\n") if ln.strip()]
    image = request.files["image"].read()
    if not lines:
        return jsonify({"error": "No transliteration lines given."}), 400
    try:
        prov = registry.get(pid)
    except KeyError:
        return jsonify({"error": f"Unknown render provider: {pid}"}), 400
    res = prov.render(image, lines)
    if not res.ok:
        return jsonify({"error": res.error}), 500
    return Response(res.png, mimetype="image/png")


@app.get("/api/settings")
def api_settings_get():
    return jsonify(settings.status())


@app.post("/api/settings")
def api_settings_post():
    data = request.get_json(force=True) or {}
    saved = settings.save({k: v for k, v in data.items() if isinstance(v, str)})
    registry.reset_cache()
    return jsonify({"saved": saved})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
```

Then `git rm ocr.py imgedit.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_app.py -v`
Expected: PASS. Also run the whole suite: `python -m pytest -q` → all green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: Flask app over provider registry + runner; drop ocr.py/imgedit.py"
```

---

### Task 9: Single-image compare UI

**Files:**
- Modify (rewrite): `urdu-free-toolkit/templates/index.html`
- Create: `urdu-free-toolkit/static/app.css`
- Create: `urdu-free-toolkit/static/app.js`
- Test: `urdu-free-toolkit/tests/test_frontend_static.py` (lightweight asserts only)

**Interfaces:**
- Consumes: `/api/providers`, `/api/ocr` (SSE), `/api/transliterate`, `/api/render`,
  `/api/settings`.
- Produces: a page with — tabs (Single image / Paste text; "Batch" tab present but
  shows "coming in phase 5"), an OCR engine checkbox list (populated from
  `/api/providers`, greyed + tooltip when `available` is false, badge shown),
  "All offline" / "Recommended" buttons, upload dropzone, **Run OCR** → results
  render as columns (one per provider) as SSE events arrive: label, ms, editable
  `<textarea>` with the text, **Use this** button, notes line, error line.
  "Use this" copies into the transliteration input. Translit engine checkboxes +
  **Run transliteration** → columns with Devanagari + Roman + copy buttons.
  Render block: provider dropdown (render-capable only) + Roman/Devanagari radio +
  **Generate** → shows `<img>` + Download PNG. Gear icon → Settings modal with a
  field per `settings.KNOWN_KEYS`, **Save** → `POST /api/settings` → refetch
  `/api/providers`.
- Flask serves `static/` by default at `/static/...`; reference
  `{{ url_for('static', filename='app.css') }}` / `app.js`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frontend_static.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_static_files_exist():
    assert (ROOT / "static/app.css").is_file()
    assert (ROOT / "static/app.js").is_file()

def test_index_references_static_and_no_cdn():
    html = (ROOT / "templates/index.html").read_text(encoding="utf-8")
    assert "app.js" in html and "app.css" in html
    assert "http://" not in html and "https://" not in html  # no CDN

def test_app_js_hits_endpoints():
    js = (ROOT / "static/app.js").read_text(encoding="utf-8")
    for ep in ("/api/providers", "/api/ocr", "/api/transliterate", "/api/render", "/api/settings"):
        assert ep in js
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_frontend_static.py -v`
Expected: FAIL — `static/app.css` missing.

- [ ] **Step 3: Write minimal implementation**

Build `templates/index.html` (structure + `url_for` links, keep the existing
colour theme vars), `static/app.css` (move the `<style>` block out, add
`.columns{display:flex;gap:12px;overflow-x:auto}` and `.col{flex:0 0 280px}`),
`static/app.js`:
- `loadProviders()` → `GET /api/providers`, render checkbox lists into
  `#ocr-engines`, `#translit-engines`, `#render-provider`.
- `runOcr()` → `fetch('/api/ocr', {method:'POST', body: FormData})`, read the
  response body as a stream (`res.body.getReader()`), split on `\n\n`, `JSON.parse`
  each `data:` line, append/update a `.col` per `provider_id`; on `{done:true}` stop.
- `useThis(pid)` → copy that column's textarea value into `#urdu-input`.
- `runTranslit()` → `POST /api/transliterate` JSON, render columns.
- `runRender()` → `POST /api/render` FormData, set `#rendered` img src to blob URL.
- Settings modal wiring.

(Implementer: write real, working JS here — no placeholders. Keep it dependency-free.)

- [ ] **Step 4: Run test to verify it passes + manual check**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_frontend_static.py -v` → PASS
Manual: `python app.py`, open `http://localhost:5000`:
- OCR list shows `rule`? (no — rule is translit) — OCR list shows `gpt` greyed
  ("OPENAI_API_KEY not set"); translit list shows `rule` (available) + `gpt` (greyed).
- Paste `میں ٹھیک ہوں`, tick `rule`, Run transliteration → a column with Devanagari + Roman.
- Open Settings, paste an `OPENAI_API_KEY`, Save → reload providers → `gpt` now enabled.
- With the key set: upload an Urdu image, tick `gpt`, Run OCR → a column fills with
  the Urdu text + notes; **Use this** → text lands in the transliteration box.

- [ ] **Step 5: Commit**

```bash
git add urdu-free-toolkit/templates/index.html urdu-free-toolkit/static/ urdu-free-toolkit/tests/test_frontend_static.py
git commit -m "feat: single-image compare UI (engine pickers, SSE columns, settings modal)"
```

---

### Task 10: Dependencies, gitignore, README, full-suite gate

**Files:**
- Modify: `urdu-free-toolkit/requirements.txt`
- Modify: `urdu-free-toolkit/.gitignore`
- Modify: `urdu-free-toolkit/README.md`
- Create: `urdu-free-toolkit/tests/conftest.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: Phase-1 `requirements.txt` (no heavy ML yet — those arrive with their
  providers in later phases), `.gitignore` covering `jobs/`, `__pycache__/`,
  `.pytest_cache/`, model caches; README "Modes / providers" intro + Settings note;
  `conftest.py` adding the repo dir to `sys.path` and a `RUN_HEAVY` skip marker for
  later phases.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_frontend_static.py (or new tests/test_packaging.py)
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_requirements_has_phase1_core():
    req = (ROOT / "requirements.txt").read_text().lower()
    for pkg in ("flask", "openai", "python-dotenv", "pillow", "pytest"):
        assert pkg in req

def test_gitignore_covers_jobs_and_caches():
    gi = (ROOT / ".gitignore").read_text()
    for pat in ("jobs/", "__pycache__/", ".pytest_cache/"):
        assert pat in gi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd urdu-free-toolkit && python -m pytest tests/test_packaging.py -v`
Expected: FAIL — `pytest` / `jobs/` not present yet.

- [ ] **Step 3: Write minimal implementation**

`requirements.txt`:

```
# --- Phase 1 core ---
Flask==3.0.3
python-dotenv==1.0.1
Pillow==10.4.0
openai>=1.40.0
pytest>=8.0
# Heavy OCR / transliteration / translation / inpainting deps are added in
# phases 2-6 alongside the providers that need them. See docs/superpowers/plans/.
```

`.gitignore` — append:

```
__pycache__/
.pytest_cache/
jobs/
*.pyc
.env
```

`tests/conftest.py`:

```python
import os, sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_HEAVY"):
        return
    skip_heavy = pytest.mark.skip(reason="heavy provider; set RUN_HEAVY=1 to run")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
```

README: add a short "What this is now" paragraph (free/offline-first, pick engines,
compare), and a "Settings" note (keys entered in the UI are written to `.env`).
Keep the honest limitations section.

- [ ] **Step 4: Run the whole suite**

Run: `cd urdu-free-toolkit && python -m pytest -q`
Expected: all green (test_base, test_registry, test_runner, test_translit_rule,
test_provider_gpt, test_provider_gpt_image, test_settings, test_app,
test_frontend_static, test_packaging).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: phase-1 requirements, gitignore, README intro, heavy-test gate"
```

---

## Self-Review

**1. Spec coverage (Phase 1 slice):**
- Provider registry + `available()` gate + UI metadata → Tasks 1, 2. ✓
- Parallel runner + per-provider timeout + failure isolation + SSE stream → Task 3. ✓
- Wrap existing: rule translit → Task 4; GPT OCR/translit → Task 5; GPT render → Task 6. ✓
- Settings panel writing `.env`, keys never returned → Task 7 + Task 8 (`/api/settings` echoes names only). ✓
- Endpoints `/api/providers`, `/api/ocr` (SSE), `/api/transliterate`, `/api/render`,
  `/api/settings`; `MAX_CONTENT_LENGTH` 32 MB → Task 8. `/api/translate` stubbed
  (providers land Phase 3) — noted, contract stable. ✓
- Single-image compare UI: pickers with badges, SSE columns, "Use this", translit
  columns, render block, settings modal → Task 9. ✓
- `static/` split from `index.html`; no CDN → Task 9 + test. ✓
- Deferred to later phases (explicitly, per spec rollout order): offline OCR
  providers (Phase 2), offline translit + translate capability (Phase 3), render
  providers pil/opencv/lama (Phase 4), batch mode (Phase 5), remaining API
  providers + warmup (Phase 6). Not gaps — separate plans.

**2. Placeholder scan:** Task 6 Step 3 and Task 9 Step 3 describe a port/build
rather than pasting full code. Task 6 is a mechanical port of the existing,
already-tested `imgedit.py` (kwargs unchanged) with a test pinning the contract —
acceptable. Task 9 is UI glue with a manual verification script and static-asset
tests; the exact JS/CSS is left to the implementer but every function it must
contain and every endpoint it must call is enumerated. No `TBD`/`TODO` remain.

**3. Type consistency:** `Result`/`OcrResult`/`TranslitResult`/`RenderResult`,
`TranslitOpts(roman_style, targets)`, `ProviderInfo(id,label,capability,kind,needs,note)`,
`registry.get`/`for_ui`/`reset_cache`/`discover(package=)`,
`runner.run(capability, provider_ids, call, timeout_s)` / `runner.stream(...)`,
`settings.save`/`status`/`env_path`/`KNOWN_KEYS`, `PROVIDER` module global,
provider methods `ocr(image)` / `translit(text, opts)` / `translate(text, opts)` /
`render(image, lines)` — consistent across Tasks 1-10. `providers/_openai_common`
exports `get_client`, `have_key`, `prepare_image`, `parse_json`, `VISION_SYSTEM`,
`TEXT_SYSTEM`, `MODEL` — used identically in Tasks 5 and 6.
