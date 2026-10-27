"""Temporal lexical discovery must not enumerate the global corpus to find a rare window."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindbridge import (
    EmbedTask,
    IndexQuantization,
    Memory,
    MemoryType,
    Modality,
    ModelInput,
    RetrievalMode,
    RetrievalScope,
)
from mindbridge.infrastructure.local import IndexDocument, StoredEmbedding
from mindbridge.infrastructure.local.zvec_index import IndexHit, ZvecIndex

_REFERENCE = datetime(2026, 10, 27, tzinfo=timezone.utc)
_TARGET_TIME = datetime(2024, 6, 1, tzinfo=timezone.utc)


class _OrthogonalEmbedder:
    embedding_model = "temporal-lexical-fixture-v1"
    embedding_space = "temporal-lexical-fixture-v1:64"
    embedding_dimension = 64
    embedding_capabilities = frozenset({Modality.TEXT})

    def __init__(self) -> None:
        self.query_calls = 0

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        # Semantic retrieval cannot rescue a missing lexical route in these fixtures.
        if task is EmbedTask.QUERY:
            self.query_calls += 1
        vector = (1.0, 0.0) if task is EmbedTask.QUERY else (0.0, 1.0)
        return tuple((*vector, *((0.0,) * 62)) for _ in inputs)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("quantization", tuple(IndexQuantization))
@pytest.mark.parametrize("mode", (RetrievalMode.HYBRID, RetrievalMode.LEXICAL))
def test_temporal_lexical_discovery_keeps_topk_bounded_beyond_global_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quantization: IndexQuantization,
    mode: RetrievalMode,
) -> None:
    route_depth = 100 if quantization is IndexQuantization.NONE else 200
    query = "common witness in 2024"
    embedder = _OrthogonalEmbedder()
    with Memory(
        tmp_path,
        embedder=embedder,
        retrieval_mode=mode,
        index_quantization=quantization,
        minimum_relevance=0.4,
    ) as memory:
        memory.add_many(
            tuple(f"common witness {number}" for number in range(384)),
            occurred_at=(_REFERENCE,) * 384,
        )
        target = memory.add("common witness target extra", occurred_at=_TARGET_TIME)
        global_hits = memory._index.lexical_search(query, limit=route_depth)
        global_documents = memory._store.index.read_index_candidates(
            tuple(hit.id for hit in global_hits)
        )
        assert len(global_hits) == route_depth
        assert target.id not in {document.memory_id for document in global_documents}

        requested_depths: list[int] = []
        original = memory._index.lexical_search

        def measured(
            text: str,
            *,
            limit: int = 10,
            space_id: str | None = None,
            task: str | None = None,
            memory_type: str | None = None,
            occurred_from: datetime | None = None,
            occurred_until: datetime | None = None,
            place_id: str | None = None,
            identity_id: str | None = None,
        ) -> tuple[IndexHit, ...]:
            requested_depths.append(limit)
            return original(
                text,
                limit=limit,
                space_id=space_id,
                task=task,
                memory_type=memory_type,
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                place_id=place_id,
                identity_id=identity_id,
            )

        monkeypatch.setattr(memory._index, "lexical_search", measured)
        outcome = memory.search_with_trace(query, limit=5, reference_at=_REFERENCE)

        assert target.id in {
            candidate.memory_id for candidate in outcome.trace.candidates if candidate.lexical_match
        }
        if mode is RetrievalMode.HYBRID:
            assert target.id in {hit.id for hit in outcome.hits}
        assert max(requested_depths) <= route_depth
        assert len(requested_depths) == 2
        if mode is RetrievalMode.LEXICAL:
            assert embedder.query_calls == 0


@pytest.mark.parametrize("query", ("common witness", "共同见证", "Alice 共同见证"))
def test_native_time_fts_survives_dense_queries_optimization_and_reopening(
    tmp_path: Path,
    query: str,
) -> None:
    start = _TARGET_TIME
    until = start + timedelta(days=1)
    base = IndexDocument(
        embedding=StoredEmbedding(
            embedding_id="point-from",
            memory_id="point-from",
            values=(0.0, 1.0),
            model_id="time-boundary-fixture-v1",
            space_id="time-boundary-fixture-v1:2",
            task="retrieval.document",
            created_at=_REFERENCE,
        ),
        content="common witness Alice 共同见证",
        metadata_json="{}",
        occurred_at=start,
        place_id="kitchen",
        identity_ids=("alice",),
    )
    documents = tuple(
        replace(
            base,
            embedding=replace(base.embedding, embedding_id=name, memory_id=name),
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            memory_type=memory_type,
        )
        for name, occurred_at, occurred_end, memory_type in (
            ("point-from", start, None, "semantic"),
            ("point-until", until, None, "semantic"),
            ("span-ending-at-from", start - timedelta(hours=2), start, "semantic"),
            (
                "span-crossing-from",
                start - timedelta(hours=1),
                start + timedelta(hours=1),
                "semantic",
            ),
            ("episodic-point", start + timedelta(hours=2), None, "episodic"),
            ("unknown-time", None, None, "semantic"),
        )
    )
    path = tmp_path / "zvec"
    for phase in ("fresh", "optimized", "reopened"):
        with ZvecIndex(path, dimension=2) as index:
            if phase == "fresh":
                index.upsert(documents)
                index.flush()
            if phase == "optimized":
                index.optimize()
            for _ in range(3):
                # Exercise the exact native combination the former workaround avoided.
                with ThreadPoolExecutor(max_workers=2) as executor:
                    dense = executor.submit(index.search, (1.0, 0.0), limit=100)
                    lexical = executor.submit(
                        index.lexical_search,
                        query,
                        limit=100,
                        occurred_from=start.astimezone(timezone(timedelta(hours=8))),
                        occurred_until=until,
                        place_id="kitchen",
                        identity_id="alice",
                    )
                    dense.result()
                    assert {hit.id for hit in lexical.result()} == {
                        "point-from",
                        "span-crossing-from",
                        "episodic-point",
                    }
                assert {
                    hit.id
                    for hit in index.lexical_search(
                        query,
                        limit=100,
                        occurred_from=start,
                        occurred_until=until,
                        memory_type="episodic",
                    )
                } == {"episodic-point"}
                assert (
                    index.lexical_search(
                        query,
                        occurred_from=start,
                        occurred_until=until,
                        identity_id="absent",
                    )
                    == ()
                )


@pytest.mark.parametrize("mode", (RetrievalMode.HYBRID, RetrievalMode.LEXICAL))
def test_temporal_lexical_routes_preserve_hard_filters_and_global_fallback(
    tmp_path: Path,
    mode: RetrievalMode,
) -> None:
    with Memory(
        tmp_path,
        embedder=_OrthogonalEmbedder(),
        retrieval_mode=mode,
        minimum_relevance=0.4,
    ) as memory:
        target = memory.add("common witness", occurred_at=_TARGET_TIME)
        outside = memory.add("global fallback in 2024", occurred_at=_REFERENCE)
        episodic = memory.add(
            "common witness episodic",
            occurred_at=_TARGET_TIME,
            memory_type=MemoryType.EPISODIC,
        )
        # A preferred date is soft: a disjoint explicit hard window wins, and its fallback
        # must still run even though the preferred intersection is empty.
        disjoint = memory.search(
            "global fallback in 2024",
            occurred_from=_REFERENCE,
            occurred_until=_REFERENCE + timedelta(days=1),
            reference_at=_REFERENCE,
        )
        assert [hit.id for hit in disjoint] == [outside.id]
        assert {hit.id for hit in memory.search("common witness in 2024")} >= {
            target.id,
            episodic.id,
        }
        assert [
            hit.id
            for hit in memory.search("common witness in 2024", memory_type=MemoryType.EPISODIC)
        ] == [episodic.id]
        assert (
            memory.search("common witness in 2024", scope=RetrievalScope(place_id="missing")) == ()
        )
        assert memory.search("unmatchedxyz") == ()


def test_temporal_lexical_scores_do_not_depend_on_requested_result_count(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_OrthogonalEmbedder(), minimum_relevance=0.4) as memory:
        memory.add_many(
            tuple(f"common witness outside {number}" for number in range(140)),
            occurred_at=(_REFERENCE,) * 140,
        )
        memory.add_many(
            tuple(f"common witness target {number}" for number in range(12)),
            occurred_at=(_TARGET_TIME,) * 12,
        )
        narrow = memory.search("common witness in 2024", limit=5, reference_at=_REFERENCE)
        wide = memory.search("common witness in 2024", limit=10, reference_at=_REFERENCE)
        assert len(narrow) == 5
        assert narrow == wide[:5]


@pytest.mark.parametrize("crowd", (0, 120))
@pytest.mark.parametrize("suffix", ("", " 望远镜"))
def test_weak_temporal_lexical_match_does_not_displace_exact_global_match(
    tmp_path: Path,
    crowd: int,
    suffix: str,
) -> None:
    query = f"Alice bought telescope in 2024{suffix}"
    with Memory(
        tmp_path,
        embedder=_OrthogonalEmbedder(),
        retrieval_mode=RetrievalMode.LEXICAL,
        minimum_relevance=0.4,
    ) as memory:
        exact = memory.add(query, occurred_at=_REFERENCE)
        memory.add("Alice coffee", occurred_at=_TARGET_TIME)
        if crowd:
            memory.add_many(
                tuple(f"Alice bought telescope sample {number}" for number in range(crowd)),
                occurred_at=(_REFERENCE,) * crowd,
            )
        assert [hit.id for hit in memory.search(query, limit=1, reference_at=_REFERENCE)] == [
            exact.id
        ]
