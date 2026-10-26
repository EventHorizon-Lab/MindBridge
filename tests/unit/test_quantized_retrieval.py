"""Authoritative FP32 scores must decide quantized candidate admission and ranking."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from mindbridge import IndexQuantization, Memory, Modality, RetrievalMode
from mindbridge.infrastructure.local.zvec_index import IndexHit
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.types import RetrievalRejection


class _Embedder:
    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "quantized-retrieval-fixture-v1"
    embedding_space = "quantized-retrieval-fixture-v1:64"
    embedding_dimension = 64

    def __init__(self) -> None:
        self.query_calls = 0

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        if task is EmbedTask.QUERY:
            self.query_calls += 1
        return tuple(
            (
                (1.0, 0.0)
                if task is EmbedTask.QUERY
                else (0.9, math.sqrt(0.19))
                if value.text == "precise target"
                else (0.8, 0.6)
            )
            + (0.0,) * 62
            for value in inputs
        )

    def close(self) -> None:
        return None


@pytest.mark.parametrize(
    "quantization", (IndexQuantization.FP16, IndexQuantization.INT8, IndexQuantization.RABITQ)
)
def test_quantized_candidates_use_fp32_before_relevance_gating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quantization: IndexQuantization
) -> None:
    embedder = _Embedder()
    with Memory(
        tmp_path,
        embedder=embedder,
        index_quantization=quantization,
        retrieval_mode=RetrievalMode.DENSE,
        minimum_relevance=0.85,
        ambiguity_margin=0,
    ) as memory:
        target = memory.add("precise target")
        decoy = memory.add("approximate neighbor")
        monkeypatch.setattr(
            memory._index,
            "search",
            lambda *_args, **_kwargs: (
                IndexHit(id=decoy.id, relevance=0.95, confidence=0.975),
                IndexHit(id=target.id, relevance=0.2, confidence=0.6),
            ),
        )
        result = memory.search_with_trace("query", limit=2)

    assert [hit.id for hit in result.hits] == [target.id]
    by_id = {candidate.memory_id: candidate for candidate in result.trace.candidates}
    assert by_id[target.id].dense_relevance == pytest.approx(0.9, abs=1e-6)
    assert by_id[target.id].dense_confidence == pytest.approx(0.95, abs=1e-6)
    assert by_id[decoy.id].dense_relevance == pytest.approx(0.8, abs=1e-6)
    assert embedder.query_calls == 1


def test_quantized_retrieval_can_recover_a_parent_beyond_the_approximate_top_hundred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        index_quantization=IndexQuantization.INT8,
        retrieval_mode=RetrievalMode.DENSE,
        minimum_relevance=0,
        ambiguity_margin=0,
    ) as memory:
        decoys = tuple(memory.add(f"approximate neighbor {number}") for number in range(100))
        target = memory.add("precise target")
        approximate_order = (*decoys, target)

        def search(*_args: object, **kwargs: object) -> tuple[IndexHit, ...]:
            limit = kwargs["limit"]
            assert isinstance(limit, int)
            return tuple(
                IndexHit(id=record.id, relevance=0.99, confidence=0.995)
                for record in approximate_order[:limit]
            )

        monkeypatch.setattr(memory._index, "search", search)
        assert memory.search("query", limit=1)[0].id == target.id
        assert memory.search("query", limit=10)[0].id == target.id


def test_traced_candidates_report_final_rank_and_rejection(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        retrieval_mode=RetrievalMode.DENSE,
        minimum_relevance=0.85,
        ambiguity_margin=0,
    ) as memory:
        target = memory.add("precise target")
        decoy = memory.add("approximate neighbor")
        result = memory.search_with_trace("query", limit=1)

    assert [hit.id for hit in result.hits] == [target.id]
    assert len(result.trace.candidates) == 2
    assert {
        (candidate.memory_id, candidate.rank, candidate.rejected_by)
        for candidate in result.trace.candidates
    } == {
        (target.id, 1, None),
        (decoy.id, None, RetrievalRejection.MINIMUM_RELEVANCE),
    }
