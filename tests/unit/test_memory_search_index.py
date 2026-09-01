"""`Memory` driven against the real Zvec index rather than a stand-in.

The API tests install a fake index so they can steer retrieval, which means they cannot see any
behaviour the real index actually implements. One such rule cost a release: a memory's full-text
document is written on its ``object_part == 0`` row alone, so a memory stored without that row is
reachable by the dense route only. A test asserting the memory was "still searchable" passed
against the fake while the real index could not match it at all.

These tests therefore construct `Memory` with no index substitution. Only the models are faked;
the store, the outbox, and the index are the real ones. `zvec` is a base dependency, so this runs
wherever the suite runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from mindbridge import Memory
from mindbridge.exceptions import ModelError
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.types import Blob, Modality

_ALL_INPUT_MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})


class _Embedder:
    """A deterministic embedder that can refuse a named asset the way a provider limit does."""

    embedding_model = "fake-real-index"
    embedding_space = "fake-real-index:2:test"
    embedding_dimension = 2

    def __init__(self) -> None:
        self.embedding_capabilities = _ALL_INPUT_MODALITIES
        self.oversized_assets: frozenset[str] = frozenset()

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        batch = tuple(inputs)
        if any(asset.id in self.oversized_assets for value in batch for asset in value.assets):
            raise ModelError(
                "encoded inline model media item exceeds the limit",
                reason="payload_too_large",
            )
        # One vector for everything: dense relevance is deliberately uninformative so that a
        # lexical assertion cannot pass on the strength of the dense route.
        return tuple((1.0, 0.0) for _ in batch)

    def close(self) -> None:
        return None


def _lexical_matches(memory: Memory, query: str) -> set[str]:
    """Memory ids the full-text route matched, ignoring stale-index candidates with no parent."""
    traced = memory.search_with_trace(query)
    return {
        candidate.memory_id
        for candidate in traced.trace.candidates
        if candidate.lexical_match and candidate.memory_id is not None
    }


def test_the_real_index_matches_a_memory_on_its_own_words(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_Embedder()) as memory:
        kitchen = memory.add("the kitchen at dusk")
        garden = memory.add("the garden at noon")

        # Establishes that these tests exercise real BM25: the dense route returns one vector for
        # every memory, so only the full-text route can tell these two apart.
        matched = _lexical_matches(memory, "kitchen")
        assert kitchen.id in matched
        assert garden.id not in matched


def test_a_memory_whose_aggregate_key_was_elided_keeps_its_full_text_document(
    tmp_path: Path,
) -> None:
    embedder = _Embedder()
    with Memory(tmp_path, embedder=embedder) as memory:
        probe = memory.add(("the kitchen at dusk", Blob(b"oversized-clip", "video/mp4")))
        oversized = probe.assets[0].id
        memory.delete(probe.id)
        embedder.oversized_assets = frozenset({oversized})

        elided = memory.add(("the kitchen at dusk", Blob(b"oversized-clip", "video/mp4")))
        # A second document, so the index scores against a corpus instead of degenerately
        # matching its only entry.
        intact = memory.add("the kitchen at dawn")

        # The retrieval key holding the media is also the aggregate key, which is part 0 -- the
        # only row the index writes a full-text document on. Dropping it without renumbering the
        # survivors leaves this memory with an empty document, and the query its own text answers
        # cannot reach it.
        matched = _lexical_matches(memory, "kitchen")
        assert intact.id in matched
        assert elided.id in matched

        # The memory kept its media and the write did not fail; only the key was degraded.
        assert memory.get(elided.id).assets[0].id == oversized


def test_a_memory_with_no_carriable_key_never_reaches_the_index(tmp_path: Path) -> None:
    embedder = _Embedder()
    with Memory(tmp_path, embedder=embedder) as memory:
        # Two assets and no text, so the memory has several retrieval keys and every one of them
        # carries refused media. A single-asset memory would have one key and take the earlier
        # path that re-raises before any degradation is attempted, leaving the guard that decides
        # a memory is unreachable untested.
        clips = (Blob(b"first-clip", "video/mp4"), Blob(b"second-clip", "video/mp4"))
        probe = memory.add(clips)
        assert len(probe.assets) == 2
        embedder.oversized_assets = frozenset(asset.id for asset in probe.assets)
        memory.delete(probe.id)

        # Degrading every key would store a memory no query could reach, so the write fails
        # rather than leaving an unreachable row behind.
        with pytest.raises(ModelError) as failure:
            memory.add(clips)
        assert failure.value.reason == "payload_too_large"

        assert memory.list().items == ()
