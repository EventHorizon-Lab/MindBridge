"""Exact source-media context uses existing scoped corpus neighbors, never benchmark labels."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindbridge import (
    AnswerPolicy,
    AnswerResult,
    Blob,
    EmbedTask,
    Memory,
    MemoryRecord,
    MemoryType,
    Modality,
    ModelInput,
    ObservationContext,
    RetrievalMode,
    RetrievalScope,
    SearchHit,
)
from mindbridge.context import evidence_cost

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)
CODE = Blob(b"original code pixels", "image/png")
PLOT = Blob(b"plot result pixels", "image/png")
QUERY = ("What did this code produce?", CODE)


class _Embedder:
    embedding_model = embedding_space = "query-media-context-test-2"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        angles = {
            "code source": 0.0,
            "tail noise": 0.1,
            "second source": 0.2,
            "ranked C": 0.3,
            "linked D": math.pi,
            "plot result": 1.0,
        }
        return tuple(
            (math.cos(angle), math.sin(angle))
            for value in inputs
            for angle in (
                0.0
                if task is EmbedTask.QUERY
                else next((angle for text, angle in angles.items() if text in value.text), 1.2),
            )
        )

    def close(self) -> None:
        pass


class _Answerer:
    generation_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})

    def __init__(self, *, sequence: bool = False, semantic: bool = False) -> None:
        self.calls: list[tuple[ModelInput, tuple[SearchHit, ...]]] = []
        self.sequence, self.semantic = sequence, semantic

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        self.calls.append((question, tuple(hits)))
        return AnswerResult(answer="fixture", hits=tuple(hits))

    def plan_recall(
        self, question: str, *, reference_at: datetime, corpus_digest: str, attempted: str = ""
    ) -> str:
        return json.dumps(
            {
                "shape": "sequence" if self.sequence else "point",
                "steps": [
                    {"op": "similar", "k": 1, "query": "tail noise" if self.semantic else question}
                ],
            }
        )

    def close(self) -> None:
        pass


class _SetAnswerer(_Answerer):
    def plan_recall(
        self, question: str, *, reference_at: datetime, corpus_digest: str, attempted: str = ""
    ) -> str:
        return '{"shape":"set","steps":[{"op":"match","terms":["matched"]}]}'


def _memory(
    path: Path,
    answerer: _Answerer,
    *,
    enabled: bool = True,
    budget: int | None = None,
    max_rows: int = 24,
    planning: bool = False,
    semantic: bool = False,
    set_max_rows: int = 200,
    set_budget: int = 30000,
    minimum_relevance: float = 0.0,
) -> Memory:
    return Memory(
        path,
        embedder=_Embedder(),
        answerer=answerer,
        retrieval_mode=RetrievalMode.DENSE,
        minimum_relevance=minimum_relevance,
        ambiguity_margin=0.0,
        reinforce_on_answer=False,
        query_vision=False,
        recall_planning=planning,
        recall_query_expansion=semantic,
        recall_set_max_rows=set_max_rows,
        recall_set_budget_chars=set_budget,
        evidence_expansion=enabled,
        evidence_expansion_max_rows=max_rows,
        evidence_budget_chars=budget,
    )


def _add(memory: Memory) -> tuple[MemoryRecord, MemoryRecord, MemoryRecord]:
    code = memory.add(("code source", CODE), occurred_at=NOW)
    plot = memory.add(("plot result", PLOT), occurred_at=NOW)
    tail = memory.add(("tail noise", Blob(b"noise pixels", "image/png")), occurred_at=NOW)
    return code, plot, tail


def test_exact_query_source_recovers_next_row_at_equal_event_time(tmp_path: Path) -> None:
    answerer = _Answerer()
    with _memory(tmp_path, answerer) as memory:
        code, plot, _ = _add(memory)
        memory.ask(QUERY, limit=1)
        question, hits = answerer.calls[-1]
        assert tuple(hit.id for hit in hits) == (code.id, plot.id)
        assert hits[0].assets == code.assets and hits[1].assets == plot.assets
        assert question.assets == code.assets
        assert "adjacency does not establish a shared event" in question.text
        assert "do not state a total" in question.text


def test_exact_source_in_ranked_tail_is_delivered_with_its_context(tmp_path: Path) -> None:
    answerer = _Answerer()
    with _memory(tmp_path, answerer) as memory:
        floor = memory.add(("code source", Blob(b"other pixels", "image/png")), occurred_at=NOW)
        source = memory.add(("second source", CODE), occurred_at=NOW)
        plot = memory.add(("plot result", PLOT), occurred_at=NOW)
        memory.add(("tail noise", Blob(b"tail pixels", "image/png")), occurred_at=NOW)
        memory.ask(QUERY, limit=1)
        assert tuple(hit.id for hit in answerer.calls[-1][1]) == (floor.id, source.id, plot.id)


def test_renamed_native_file_matches_by_bytes_not_filename(tmp_path: Path) -> None:
    renamed = tmp_path / "renamed-copy.png"
    renamed.write_bytes(b"original code pixels")
    answerer = _Answerer()
    with _memory(tmp_path / "store", answerer) as memory:
        code, plot, _ = _add(memory)
        memory.ask((QUERY[0], renamed), limit=1)
        assert {hit.id for hit in answerer.calls[-1][1]} == {code.id, plot.id}


@pytest.mark.parametrize("semantic", (False, True))
def test_native_context_uses_tight_budget_before_replaceable_rank_tail(
    tmp_path: Path, semantic: bool
) -> None:
    answerer = _Answerer(semantic=semantic)
    with _memory(tmp_path, answerer) as memory:
        code, plot, tail = _add(memory)
        ranked = {hit.id: hit for hit in memory.search(QUERY, limit=100)}
        budget = evidence_cost(ranked[code.id]) + evidence_cost(ranked[plot.id])
    with _memory(tmp_path, answerer, budget=budget, planning=semantic, semantic=semantic) as memory:
        memory.ask(QUERY, limit=1)
        ids = tuple(hit.id for hit in answerer.calls[-1][1])
        assert code.id in ids and plot.id in ids and tail.id not in ids
        assert sum(evidence_cost(hit) for hit in answerer.calls[-1][1]) <= budget


@pytest.mark.parametrize("enabled", (False, True))
def test_other_pixels_do_not_trigger_context_and_disabled_keeps_original_window(
    tmp_path: Path, enabled: bool
) -> None:
    answerer = _Answerer()
    with _memory(tmp_path, answerer, enabled=enabled) as memory:
        code, plot, _ = _add(memory)
        different = (QUERY[0], Blob(b"visually similar but different bytes", "image/png"))
        memory.ask(different if enabled else QUERY, limit=1)
        assert tuple(hit.id for hit in answerer.calls[-1][1]) == (code.id,)
        assert plot.id not in {hit.id for hit in answerer.calls[-1][1]}
        assert "exact query-media" not in answerer.calls[-1][0].text


def test_scope_filters_both_exact_anchors_and_neighbor_hydration(tmp_path: Path) -> None:
    answerer = _Answerer()
    with _memory(tmp_path, answerer) as memory:
        code = memory.add(
            ("code source", CODE), occurred_at=NOW, context=ObservationContext(place_id="office")
        )
        hidden = memory.add(
            ("plot result", PLOT), occurred_at=NOW, context=ObservationContext(place_id="private")
        )
        memory.ask(QUERY, limit=1, scope=RetrievalScope(place_id="office"))
        ids = {hit.id for hit in answerer.calls[-1][1]}
        assert ids == {code.id} and hidden.id not in ids
        # The only equal-asset row is outside this scope, so it cannot become an anchor.
        safe = memory.add(
            "tail noise", occurred_at=NOW, context=ObservationContext(place_id="hall")
        )
        memory.ask(QUERY, limit=1, scope=RetrievalScope(place_id="hall"))
        assert {hit.id for hit in answerer.calls[-1][1]} == {safe.id}
        assert "exact query-media" not in answerer.calls[-1][0].text


def test_neighbor_rows_do_not_escape_requested_memory_type(tmp_path: Path) -> None:
    answerer = _Answerer()
    with _memory(tmp_path, answerer) as memory:
        code = memory.add(("code source", CODE), occurred_at=NOW, memory_type=MemoryType.EPISODIC)
        plot = memory.add(("plot result", PLOT), occurred_at=NOW, memory_type=MemoryType.SEMANTIC)
        memory.ask(QUERY, limit=1, memory_type=MemoryType.EPISODIC)
        assert {hit.id for hit in answerer.calls[-1][1]} == {code.id}
        assert plot.id not in {hit.id for hit in answerer.calls[-1][1]}


def test_neighbor_hydration_keeps_valid_time_scope(tmp_path: Path) -> None:
    answerer = _Answerer()
    with _memory(tmp_path, answerer) as memory:
        code = memory.add(
            ("code source", CODE), occurred_at=NOW, context=ObservationContext(valid_from=NOW)
        )
        expired = memory.add(
            ("plot result", PLOT),
            occurred_at=NOW,
            context=ObservationContext(valid_from=NOW, valid_until=NOW + timedelta(days=1)),
        )
        memory.ask(QUERY, limit=1, scope=RetrievalScope(valid_at=NOW + timedelta(days=2)))
        assert {hit.id for hit in answerer.calls[-1][1]} == {code.id}
        assert expired.id not in {hit.id for hit in answerer.calls[-1][1]}


def test_repeated_assets_and_multiple_source_records_use_one_joint_added_row_cap(
    tmp_path: Path,
) -> None:
    answerer = _Answerer()
    with _memory(tmp_path, answerer, max_rows=1) as memory:
        code = memory.add(("code source", CODE, CODE), occurred_at=NOW)
        plot = memory.add(("plot result", PLOT), occurred_at=NOW)
        memory.add(("second source", CODE), occurred_at=NOW + timedelta(days=1))
        memory.add(
            ("another result", Blob(b"another plot", "image/png")),
            occurred_at=NOW + timedelta(days=1),
        )
        memory.ask((QUERY[0], CODE, CODE), limit=2)
        hits = answerer.calls[-1][1]
        assert len({hit.id for hit in hits}) == len(hits)
        floor = {hit.id for hit in memory.search(QUERY, limit=2)}
        assert floor <= {hit.id for hit in hits}
        assert len({hit.id for hit in hits} - floor) <= 1
        assert code.id in {hit.id for hit in hits}
        assert plot.id in {hit.id for hit in hits}


def test_same_asset_in_another_physical_directory_cannot_supply_context(tmp_path: Path) -> None:
    first, second = _Answerer(), _Answerer()
    with _memory(tmp_path / "first", first) as one, _memory(tmp_path / "second", second) as two:
        code, plot, _ = _add(one)
        other = two.add("tail noise", occurred_at=NOW)
        one.ask(QUERY, limit=1)
        two.ask(QUERY, limit=1)
        assert {code.id, plot.id} <= {hit.id for hit in first.calls[-1][1]}
        assert {hit.id for hit in second.calls[-1][1]} == {other.id}


def test_sequence_final_order_covers_new_source_context(tmp_path: Path) -> None:
    answerer = _Answerer(sequence=True)
    with _memory(tmp_path, answerer, planning=True) as memory:
        before = memory.add("before the code", occurred_at=NOW)
        code, plot, _ = _add(memory)
        memory.ask(QUERY, limit=1)
        question, hits = answerer.calls[-1]
        assert tuple(hit.id for hit in hits) == (before.id, code.id, plot.id)
        assert "insertion order when event times tie" in question.text


def test_native_source_cannot_reintroduce_predicate_row_omitted_by_row_cap(tmp_path: Path) -> None:
    answerer = _SetAnswerer()
    with _memory(tmp_path, answerer, planning=True, set_max_rows=1) as memory:
        floor = memory.add(("code source", Blob(b"floor pixels", "image/png")), occurred_at=NOW)
        admitted = memory.add(("matched plot result", PLOT), occurred_at=NOW)
        omitted = memory.add(("matched second source", CODE), occurred_at=NOW)
        memory.add("tail noise", occurred_at=NOW)
        memory.add("another unrelated row", occurred_at=NOW)
        memory.ask(QUERY, limit=1)
        question, hits = answerer.calls[-1]
        assert {floor.id, admitted.id} <= {hit.id for hit in hits}
        assert omitted.id not in {hit.id for hit in hits}
        assert "not a complete set" in question.text


@pytest.mark.parametrize("enabled", (False, True))
def test_predicate_budget_reserves_required_floor_before_first_supplemental_row(
    tmp_path: Path, enabled: bool
) -> None:
    answerer = _SetAnswerer()
    with _memory(tmp_path, answerer) as memory:
        floor = memory.add(("code source", Blob(b"floor pixels", "image/png")), occurred_at=NOW)
        matched = memory.add(("matched plot result", PLOT), occurred_at=NOW)
        memory.add(("matched second source", CODE), occurred_at=NOW)
        for index in range(3):
            memory.add(f"unrelated row {index}", occurred_at=NOW)
        hits = {hit.id: hit for hit in memory.search(QUERY, limit=100)}
        budget = evidence_cost(hits[floor.id]) + evidence_cost(hits[matched.id]) - 1
    with _memory(
        tmp_path, answerer, enabled=enabled, budget=budget, planning=True, set_budget=budget
    ) as memory:
        memory.ask(QUERY, limit=1)
        delivered = answerer.calls[-1][1]
        assert floor.id in {hit.id for hit in delivered}
        assert matched.id not in {hit.id for hit in delivered}
        assert sum(evidence_cost(hit) for hit in delivered) <= budget


def test_native_and_structural_context_share_one_added_row_cap(tmp_path: Path) -> None:
    answerer = _Answerer()
    with _memory(tmp_path, answerer, max_rows=1) as memory:
        code = memory.add(
            ("code source", CODE), occurred_at=NOW, context=ObservationContext(source_id="capture")
        )
        plot = memory.add(("plot result", PLOT), occurred_at=NOW)
        memory.add("tail noise", occurred_at=NOW)
        linked = memory.add(
            "capture partner", occurred_at=NOW, context=ObservationContext(source_id="capture")
        )
        memory.ask(QUERY, limit=1)
        assert {hit.id for hit in answerer.calls[-1][1]} == {code.id, plot.id}
        assert linked.id not in {hit.id for hit in answerer.calls[-1][1]}


def test_no_novel_native_context_keeps_original_structural_seeds(tmp_path: Path) -> None:
    answerer = _Answerer()
    with _memory(tmp_path, answerer, budget=20000, minimum_relevance=0.2) as memory:
        before = memory.add("tail noise B", occurred_at=NOW)
        anchor = memory.add(("code source", CODE), occurred_at=NOW)
        after = memory.add("second source E", occurred_at=NOW)
        tail = memory.add(
            "ranked C", occurred_at=NOW, context=ObservationContext(source_id="other capture")
        )
        linked = memory.add(
            "linked D", occurred_at=NOW, context=ObservationContext(source_id="other capture")
        )
        ranked = memory.search(QUERY, limit=100)
        assert [hit.id for hit in ranked[:3]] == [anchor.id, before.id, after.id]
        assert linked.id not in {hit.id for hit in ranked}
        memory.ask(QUERY, limit=3)
        question, hits = answerer.calls[-1]
        assert tuple(hit.id for hit in hits) == (anchor.id, before.id, after.id, tail.id)
        assert linked.id not in {hit.id for hit in hits}
        assert "exact query-media sources" not in question.text
