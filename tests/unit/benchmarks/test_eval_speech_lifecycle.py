"""Lifecycle checks for the benchmark speech look-ahead worker."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from threading import Event, Thread
from typing import cast

import pytest

from mindbridge import AssetRef, MemoryComposition, Modality
from mindbridge.benchmarks.eval import _BackendPool, _BorrowedSpeechBackend
from mindbridge.benchmarks.eval_adapters import MemoryItem
from mindbridge.models.base import SpeechAnalysis


def test_backend_pool_drains_speech_prefetch_before_closing_its_model(tmp_path: Path) -> None:
    """Pool shutdown waits for running work, cancels queued work, and rejects later work."""
    analysis_started = Event()
    release_analysis = Event()
    analysis_finished = Event()
    model_closed = Event()
    close_returned = Event()
    queued_cancelled = Event()
    calls: list[tuple[str, ...]] = []
    lifecycle: list[str] = []

    class Backend:
        transcription_capabilities = frozenset({Modality.AUDIO})

        def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
            calls.append(tuple(asset.id for asset in assets))
            analysis_started.set()
            if not release_analysis.wait(timeout=2):
                raise RuntimeError("test did not release the running speech analysis")
            analysis_finished.set()
            lifecycle.append("analysis finished")
            return tuple(SpeechAnalysis((), ()) for _asset in assets)

        def close(self) -> None:
            lifecycle.append("model closed")
            model_closed.set()

    class ResolvedConfig:
        def __init__(self, backend: Backend) -> None:
            self._backend = backend

        def close(self) -> None:
            self._backend.close()

    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    first_id = sha256(first.read_bytes()).hexdigest()
    second_id = sha256(second.read_bytes()).hexdigest()

    backend = Backend()
    borrowed = _BorrowedSpeechBackend(backend)
    pool = object.__new__(_BackendPool)
    pool._description_cache = None
    pool._resolved_config = cast(MemoryComposition, ResolvedConfig(backend))
    pool._transcriber = borrowed

    borrowed.prefetch((MemoryItem("first", (first,)),))
    assert analysis_started.wait(timeout=1)
    borrowed.prefetch((MemoryItem("second", (second,)),))
    queued = borrowed._ahead[second_id]

    def note_cancellation() -> None:
        if queued.cancelled():
            queued_cancelled.set()

    def close_pool() -> None:
        pool.close()
        close_returned.set()

    queued.add_done_callback(lambda _future: note_cancellation())
    closing = Thread(target=close_pool)
    closing.start()
    try:
        assert not model_closed.wait(timeout=0.1)
        assert queued_cancelled.wait(timeout=1)
        assert queued.cancelled()
    finally:
        release_analysis.set()
        closing.join(timeout=2)
        borrowed._ahead_worker.shutdown(wait=True, cancel_futures=True)

    assert not closing.is_alive()
    assert analysis_finished.is_set()
    assert model_closed.is_set()
    assert close_returned.is_set()
    assert calls == [(first_id,)]
    assert lifecycle == ["analysis finished", "model closed"]
    with pytest.raises(RuntimeError):
        borrowed.prefetch((MemoryItem("after-close", (first,)),))
