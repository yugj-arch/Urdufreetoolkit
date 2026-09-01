# -*- coding: utf-8 -*-
"""Run several providers of one capability at once.

``run`` returns every result in the order the ids were given; ``stream`` yields
each result the moment it lands (completion order) for the SSE endpoint. One
provider raising or hanging never blocks the others — its row comes back with
``ok=False`` and an ``error`` string, and a timed-out worker is abandoned
(``cancel_futures``) rather than awaited.

Offline engines (PaddleOCR, PyTorch/EasyOCR, Surya, …) each bundle their own
native runtime — OpenMP, BLAS, threadpools. Two of them running inference in
different worker threads at the same time deadlocks the process: the wedged
native threads never drop the GIL, so *every* Python thread in the request
starves and unrelated API engines in the same batch time out too. So offline
engines are serialized against each other via ``_OFFLINE_LOCK`` while API
engines (pure network I/O) still run fully in parallel — and one offline engine
still overlaps the API ones.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures import TimeoutError as FutTimeout

from providers import registry
from providers.base import BaseProvider, Capability, Result

_OFFLINE_LOCK = threading.Lock()


def _resolve(capability: Capability, pid: str) -> BaseProvider | None:
    try:
        return registry.get(capability, pid)
    except KeyError:
        return None


def _invoke(prov: BaseProvider, call: Callable[[BaseProvider], Result]) -> Result:
    """Run ``call(prov)``, holding ``_OFFLINE_LOCK`` for offline engines so no two
    native inference runs overlap. API engines are unaffected."""
    if getattr(prov.info, "kind", "") == "offline":
        with _OFFLINE_LOCK:
            return call(prov)
    return call(prov)


def run(capability: Capability, provider_ids: list[str],
        call: Callable[[BaseProvider], Result], timeout_s: float = 60.0) -> list[Result]:
    by_id: dict[str, Result] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, len(provider_ids)))
    futures: dict = {}
    try:
        for pid in provider_ids:
            prov = _resolve(capability, pid)
            if prov is None:
                by_id[pid] = Result(provider_id=pid, ok=False, error="unknown provider")
                continue
            futures[pool.submit(_invoke, prov, call)] = pid
        for fut, pid in futures.items():
            try:
                by_id[pid] = fut.result(timeout=timeout_s)
            except FutTimeout:
                by_id[pid] = Result(provider_id=pid, ok=False,
                                    error=f"timed out after {timeout_s:.0f}s")
            except Exception as e:  # noqa: BLE001 - failures are data
                by_id[pid] = Result(provider_id=pid, ok=False,
                                    error=f"{type(e).__name__}: {e}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return [by_id[pid] for pid in provider_ids]


def stream(capability: Capability, provider_ids: list[str],
           call: Callable[[BaseProvider], Result], timeout_s: float = 60.0) -> Iterator[Result]:
    pool = ThreadPoolExecutor(max_workers=max(1, len(provider_ids)))
    futures: dict = {}
    try:
        for pid in provider_ids:
            prov = _resolve(capability, pid)
            if prov is None:
                yield Result(provider_id=pid, ok=False, error="unknown provider")
                continue
            futures[pool.submit(_invoke, prov, call)] = pid

        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=timeout_s, return_when=FIRST_COMPLETED)
            if not done:  # nothing finished within the window — give up on the rest
                for f in pending:
                    yield Result(provider_id=futures[f], ok=False,
                                 error=f"timed out after {timeout_s:.0f}s")
                return
            for f in done:
                pid = futures[f]
                try:
                    yield f.result(timeout=0)
                except Exception as e:  # noqa: BLE001
                    yield Result(provider_id=pid, ok=False, error=f"{type(e).__name__}: {e}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
