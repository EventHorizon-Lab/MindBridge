"""High-selectivity scope predicates are answered by the index, not by widening a window.

`identity_id` and `place_id` are static per document, so before this the kernel retrieved an
unscoped candidate window and dropped everything outside the scope in SQLite afterwards. That
post-filter widens the window when too few candidates survive, but the widening is bounded, and
for a predicate that matches two records in a library the bound is reached long before the matches
are: the answer silently came back short. These tests pin the two halves of the fix -- the filter
reaches Zvec, and the collection carrying it is detected as stale and rebuilt from SQLite without
re-embedding a thing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from mindbridge import (
    EmbedTask,
    IndexQuantization,
    Memory,
    Modality,
    ModelInput,
    ObservationContext,
    RetrievalScope,
)
from mindbridge.infrastructure.local import IndexDocument, StoredEmbedding
from mindbridge.infrastructure.local.zvec_index import ZvecIndex
from mindbridge.memory import _index_recipe

_NOW = datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc)
_SPACE = "pushdown-probe:2"
_TASK = "retrieval.document"
_CROWD = 200


def _document(
    memory_id: str,
    values: tuple[float, ...],
    *,
    place_id: str | None = None,
    identity_ids: tuple[str, ...] = (),
) -> IndexDocument:
    return IndexDocument(
        embedding=StoredEmbedding(
            embedding_id=memory_id,
            memory_id=memory_id,
            values=values,
            model_id="pushdown-probe",
            space_id=_SPACE,
            task=_TASK,
            created_at=_NOW,
        ),
        content=f"kettle {memory_id}",
        metadata_json="{}",
        occurred_at=_NOW,
        place_id=place_id,
        identity_ids=identity_ids,
    )


def _off_axis(index: int) -> tuple[float, ...]:
    """A vector that is close to the query, but never as close as its own neighbours."""
    angle = (index + 1) * 0.001
    return (math.cos(angle), math.sin(angle))


def test_identity_and_place_scopes_reach_the_matches_a_window_would_lose(
    tmp_path: Path,
) -> None:
    query = (1.0, 0.0)
    # Two hundred near-identical neighbours, and four records the query matches worst of all:
    # exactly the shape that defeats a post-filter over a fixed candidate window.
    documents = [_document(f"crowd{index:03d}", _off_axis(index)) for index in range(_CROWD)]
    far = (math.cos(1.5), math.sin(1.5))
    documents += [
        _document("aliceone", far, identity_ids=("identity-alice",)),
        _document("alicetwo", far, identity_ids=("identity-alice", "identity-bob")),
        _document("kitchenone", far, place_id="kitchen"),
        _document("kitchentwo", far, place_id="kitchen"),
    ]

    with ZvecIndex(tmp_path / "zvec", dimension=2) as index:
        index.upsert(documents)
        index.flush()

        # The pre-fix path: an unscoped window, post-filtered afterwards. Nothing in scope is in
        # it, so scoping by either predicate returned nothing at all.
        unscoped = index.search(query, limit=5, space_id=_SPACE, task=_TASK)
        assert len(unscoped) == 5
        assert all(hit.id.startswith("crowd") for hit in unscoped)

        by_person = index.search(
            query,
            limit=5,
            space_id=_SPACE,
            task=_TASK,
            identity_id="identity-alice",
        )
        by_place = index.search(query, limit=5, space_id=_SPACE, task=_TASK, place_id="kitchen")
        lexical = index.lexical_search(
            "kettle",
            limit=5,
            space_id=_SPACE,
            task=_TASK,
            identity_id="identity-bob",
        )

    assert {hit.id for hit in by_person} == {"aliceone", "alicetwo"}
    assert {hit.id for hit in by_place} == {"kitchenone", "kitchentwo"}
    assert {hit.id for hit in lexical} == {"alicetwo"}


def test_a_place_label_with_an_apostrophe_indexes_and_scopes(tmp_path: Path) -> None:
    """A room called "Ana's room" is a legal `place_id`, so the filter literal escapes it.

    The projection used to be checked against the filter grammar on the write path, which
    refused an apostrophe outright -- after SQLite had committed the memory and queued its
    outbox row. The row could then never be acknowledged, so every later add, search or settle
    on that directory re-raised the same rejection.
    """
    with Memory(tmp_path, embedder=_CountingEmbedder(), minimum_relevance=0) as memory:
        record = memory.add(
            "a kettle on a worktop",
            context=ObservationContext(place_id="Ana's room"),
        )
        scoped = memory.search("kettle", scope=RetrievalScope(place_id="Ana's room"))
        assert [hit.id for hit in scoped] == [record.id]
        # The outbox is not poisoned: the next write drains it and commits.
        assert memory.add("a second kettle").id != record.id


def test_a_place_label_the_grammar_cannot_spell_still_scopes(tmp_path: Path) -> None:
    """Non-empty and trimmed is the whole `place_id` contract, so `attic\\` is storable.

    The literal cannot escape a trailing backslash or a control character, and raising for one
    turned a legal stored value into `IndexUnavailableError` -- a 503 over REST. The predicate
    is dropped from the pushdown instead, and SQLite hydration still enforces the scope.
    """
    with Memory(tmp_path, embedder=_CountingEmbedder(), minimum_relevance=0) as memory:
        for place in ("attic\\", "Ana\tRoom"):
            record = memory.add(
                "a kettle on a worktop",
                context=ObservationContext(place_id=place),
            )
            scoped = memory.search("kettle", limit=10, scope=RetrievalScope(place_id=place))
            assert [hit.id for hit in scoped] == [record.id]


class _CountingEmbedder:
    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "pushdown-probe"
    embedding_space = "pushdown-probe:2"
    embedding_dimension = 2

    def __init__(self) -> None:
        self.calls = 0

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        del task
        self.calls += 1
        return tuple((1.0, 0.0) for _value in inputs)

    def close(self) -> None:
        pass


def test_a_collection_built_before_the_filter_fields_rebuilds_without_re_embedding(
    tmp_path: Path,
) -> None:
    previous = (
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-stemmed-plus-bigram:grouped-range:"
        f"context-keys-v10:quantization-{IndexQuantization.NONE.value}"
    )
    assert previous != _index_recipe(IndexQuantization.NONE)

    with Memory(tmp_path, embedder=_CountingEmbedder(), minimum_relevance=0) as memory:
        record = memory.add(
            "a kettle on a worktop",
            context=ObservationContext(place_id="kitchen"),
        )
        # Stamp the store as one whose collection was built by the previous schema.
        memory._store.set_metadata("index.recipe", previous)
    assert (tmp_path / "zvec").exists()

    reopened = _CountingEmbedder()
    with Memory(tmp_path, embedder=reopened, minimum_relevance=0) as memory:
        # The stale collection was removed and replayed from SQLite: no vector was recomputed,
        # and the new filter field is populated from authoritative rows.
        assert reopened.calls == 0
        scoped = memory.search("kettle", scope=RetrievalScope(place_id="kitchen"))
    assert [hit.id for hit in scoped] == [record.id]
