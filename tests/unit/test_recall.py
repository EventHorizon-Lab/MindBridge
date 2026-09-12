from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

from mindbridge.infrastructure.local.store import RECALL_MAX_TERM_CHARS
from mindbridge.recall import (
    _MAX_NEIGHBOR_ANCHORS,
    DEFAULT_MAX_ROWS,
    RecallPlan,
    RecallRows,
    RecallStep,
    RecallStepResult,
    execute,
    fallback_plan,
    parse_recall_plan,
    recall_note,
)
from mindbridge.types import MemoryType, Modality, SearchHit

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def _hit(identifier: str, *, minutes: int = 0, score: float = 0.5) -> SearchHit:
    return SearchHit(
        id=identifier,
        content=f"record {identifier}",
        score=score,
        created_at=NOW,
        occurred_at=NOW + timedelta(minutes=minutes),
    )


class _Reader:
    """Records every read a plan makes and replays a scripted answer for each op.

    `selected` scripts a read whose predicate selected more records than it returned rows for,
    which is what the caller's own scope does to a hydrated read; by default a read selects
    exactly the rows it answers with.
    """

    def __init__(
        self,
        *,
        selected: Mapping[str, int] | None = None,
        **rows: tuple[SearchHit, ...],
    ) -> None:
        self.rows = rows
        self.selected = dict(selected or {})
        self.calls: list[tuple[str, object]] = []

    def _answer(self, op: str) -> RecallRows:
        rows = self.rows.get(op, ())
        return RecallRows(rows, self.selected.get(op, len(rows)))

    def similar(self, query: str, *, k: int) -> RecallRows:
        self.calls.append(("similar", (query, k)))
        return self._answer("similar")

    def match(
        self,
        terms: Sequence[str],
        *,
        any_of: bool,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        modality: Modality | None,
        memory_type: MemoryType | None,
        max_rows: int,
    ) -> RecallRows:
        self.calls.append(("match", (tuple(terms), any_of, modality, memory_type, max_rows)))
        return self._answer("match")

    def window(
        self,
        *,
        occurred_from: datetime,
        occurred_until: datetime,
        modality: Modality | None,
        memory_type: MemoryType | None,
        max_rows: int,
    ) -> RecallRows:
        self.calls.append(("window", (occurred_from, occurred_until, modality, max_rows)))
        return self._answer("window")

    def neighbors(
        self,
        memory_ids: Sequence[str],
        *,
        before: int,
        after: int,
        max_rows: int,
    ) -> RecallRows:
        self.calls.append(("neighbors", (tuple(memory_ids), before, after, max_rows)))
        return self._answer("neighbors")

    def entity(self, name: str, *, max_rows: int) -> RecallRows:
        self.calls.append(("entity", (name, max_rows)))
        return self._answer("entity")


def test_the_fallback_plan_is_todays_behaviour() -> None:
    """Every failure lands here, so this plan has to be one plain similarity search."""
    plan = fallback_plan("who signed the lease?", k=7)

    assert plan == RecallPlan(
        shape="point",
        steps=(RecallStep(op="similar", query="who signed the lease?", k=7),),
    )
    assert plan.exhaustive is False


def test_a_full_plan_parses_every_op_with_its_own_fields() -> None:
    payload = json.dumps(
        {
            "shape": "composite",
            "steps": [
                {"op": "similar", "query": "the Cairo trip", "k": 8, "time": None},
                {
                    "op": "match",
                    "terms": ["Cairo", "flight", "Cairo"],
                    "any_of": False,
                    "time": ["2024-09-01", "2024-10-01T06:30:00Z"],
                    "max_rows": 40,
                },
                {"op": "window", "time": ["2024-09-20", "2024-09-30"], "modality": "image"},
                {"op": "neighbors", "of": "step:1", "before": 1, "after": 3},
                {"op": "entity", "name": " Lily "},
            ],
        }
    )

    plan = parse_recall_plan(payload, reference_at=NOW)

    assert plan is not None
    assert plan.shape == "composite" and plan.exhaustive is True
    assert [step.op for step in plan.steps] == [
        "similar",
        "match",
        "window",
        "neighbors",
        "entity",
    ]
    assert plan.steps[1].terms == ("Cairo", "flight")
    assert plan.steps[1].any_of is False
    assert plan.steps[1].occurred_from == datetime(2024, 9, 1, tzinfo=timezone.utc)
    assert plan.steps[1].occurred_until == datetime(2024, 10, 1, 6, 30, tzinfo=timezone.utc)
    assert plan.steps[2].modality is Modality.IMAGE
    assert plan.steps[3].of == 1 and plan.steps[3].before == 1 and plan.steps[3].after == 3
    assert plan.steps[4].name == "Lily"


@pytest.mark.parametrize(
    "document",
    [
        {"shape": "hunch", "steps": [{"op": "similar", "query": "q"}]},
        {"shape": "set", "steps": [{"op": "grep", "terms": ["a"]}]},
        {"shape": "set", "steps": [{"op": "match", "terms": ["a"], "rerank": True}]},
        {"shape": "set", "steps": [{"op": "match", "terms": ["a"], "k": 5, "query": "q"}]},
        {"shape": "set", "steps": [{"op": "window", "time": ["2024-09-01"]}]},
        {"shape": "set", "steps": [{"op": "window", "time": ["2024-10-01", "2024-09-01"]}]},
        {"shape": "set", "steps": [{"op": "window", "time": ["not a date", None]}]},
        {"shape": "set", "steps": [{"op": "match", "terms": ["a"], "modality": "hologram"}]},
        {"shape": "set", "steps": [{"op": "match", "terms": "Cairo"}]},
        {"shape": "set", "steps": [{"op": "match", "terms": ["a"], "any_of": "yes"}]},
        {"shape": "sequence", "steps": [{"op": "neighbors", "of": "step:0"}]},
        {"shape": "sequence", "steps": [{"op": "neighbors", "of": "step:9"}]},
        {"shape": "sequence", "steps": [{"op": "neighbors", "of": 0}]},
        {"shape": "point", "steps": [{"op": "match", "terms": ["Cairo"]}]},
        {"shape": "set", "steps": []},
        {"shape": "set", "steps": [{"op": "similar", "query": "  "}]},
        {"shape": "set", "steps": [{"op": "entity", "name": None}]},
        {"shape": "set", "steps": [{"op": "similar", "query": "q"}], "notes": "hi"},
        {"shape": "set", "steps": [{"op": "similar", "query": "q"}] * 7},
        {"shape": "set", "steps": [{"op": "match", "terms": ["x" * 201]}]},
        {"shape": "entity", "steps": [{"op": "entity", "name": "L" * 201}]},
    ],
    ids=[
        "unknown-shape",
        "unknown-op",
        "unknown-field",
        "field-the-op-cannot-honour",
        "one-sided-window",
        "reversed-window",
        "unparseable-date",
        "unknown-modality",
        "terms-as-a-string",
        "any-of-not-a-boolean",
        "self-reference",
        "forward-reference",
        "anchor-not-a-step-reference",
        "point-plan-asking-for-a-set",
        "no-steps",
        "blank-query",
        "entity-without-a-name",
        "unknown-top-level-key",
        "too-many-steps",
        "term-longer-than-the-store-accepts",
        "name-longer-than-the-store-accepts",
    ],
)
def test_an_unrunnable_plan_falls_back_instead_of_running_something_else(
    document: dict[str, object],
) -> None:
    """A plan MindBridge cannot run exactly as written is not run at all.

    The alternative -- dropping the part that did not parse -- would answer a set question from
    a silently smaller set and still report it as complete.
    """
    assert parse_recall_plan(json.dumps(document), reference_at=NOW) is None


@pytest.mark.parametrize(
    "payload",
    ["", "   ", "not json", "[]", '"a string"', "null", '{"shape": "set"}'],
)
def test_output_that_is_not_a_plan_document_falls_back(payload: str) -> None:
    assert parse_recall_plan(payload, reference_at=NOW) is None


def test_a_null_filter_is_accepted_where_the_op_cannot_honour_it() -> None:
    """The prompt spells the keys out, so a model that echoes a null must keep its plan."""
    payload = json.dumps(
        {
            "shape": "point",
            "steps": [
                {"op": "similar", "query": "q", "k": 5, "time": None, "modality": None},
            ],
        }
    )

    plan = parse_recall_plan(payload, reference_at=NOW)

    assert plan is not None and plan.steps[0].modality is None


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("k", 900, 100),
        ("k", 0, 1),
        ("k", "twelve", 12),
        ("k", True, 12),
        ("max_rows", 10_000, 500),
        ("max_rows", -3, 1),
        ("before", 99, 10),
        ("after", 99, 10),
    ],
)
def test_a_magnitude_is_clamped_rather_than_refused(
    field: str, value: object, expected: int
) -> None:
    """A number out of range is a planner being loose, not a planner being wrong."""
    step: dict[str, object] = {"op": "similar", "query": "q"}
    if field in {"max_rows", "before", "after"}:
        step = {"op": "neighbors", "of": "step:0"}
    payload = json.dumps(
        {
            "shape": "composite",
            "steps": [{"op": "similar", "query": "anchor"}, {**step, field: value}],
        }
    )

    plan = parse_recall_plan(payload, reference_at=NOW)

    assert plan is not None
    assert getattr(plan.steps[1], field) == expected


def test_a_naive_date_is_read_in_the_reference_clocks_zone() -> None:
    tokyo = timezone(timedelta(hours=9))
    payload = json.dumps(
        {
            "shape": "set",
            "steps": [{"op": "window", "time": ["2024-09-01", "2024-09-02T00:00:00"]}],
        }
    )

    plan = parse_recall_plan(payload, reference_at=NOW.astimezone(tokyo))

    assert plan is not None
    assert plan.steps[0].occurred_from == datetime(2024, 9, 1, tzinfo=tokyo)
    assert plan.steps[0].occurred_until == datetime(2024, 9, 2, tzinfo=tokyo)


def test_execute_unions_the_ops_with_exhaustive_rows_in_time_order_first() -> None:
    """No weighted fusion: a union with ID dedup, and neither order recomputed.

    The high-scoring similarity hit stays behind every exhaustive row even though its score is
    the largest in the set, because a set answer is read as a timeline and its completeness is
    the property the prompt states.
    """
    reader = _Reader(
        match=(_hit("m-late", minutes=30), _hit("m-early", minutes=-30)),
        window=(_hit("w-mid", minutes=0), _hit("m-early", minutes=-30)),
        similar=(_hit("s-top", minutes=90, score=0.99), _hit("m-late", minutes=30, score=0.9)),
    )
    plan = parse_recall_plan(
        json.dumps(
            {
                "shape": "set",
                "steps": [
                    {"op": "match", "terms": ["a"]},
                    {"op": "window", "time": ["2026-09-01", "2026-09-30"]},
                    {"op": "similar", "query": "a"},
                ],
            }
        ),
        reference_at=NOW,
    )

    assert plan is not None
    result = execute(plan, reader, limit=12, active_records=1_000)

    assert [hit.id for hit in result.exhaustive] == ["m-early", "w-mid", "m-late"]
    assert [hit.id for hit in result.ranked] == ["s-top"]
    assert [hit.id for hit in result.hits] == ["m-early", "w-mid", "m-late", "s-top"]
    assert [(step.op, step.rows) for step in result.executed] == [
        ("match", 2),
        ("window", 2),
        ("similar", 2),
    ]
    assert result.complete is True


def test_a_read_that_returned_its_whole_bound_is_not_complete() -> None:
    """Completeness is what licenses a count, so a truncated read has to say so."""
    reader = _Reader(match=tuple(_hit(f"m-{index}", minutes=index) for index in range(3)))
    plan = parse_recall_plan(
        json.dumps({"shape": "set", "steps": [{"op": "match", "terms": ["a"], "max_rows": 3}]}),
        reference_at=NOW,
    )

    assert plan is not None
    result = execute(plan, reader, limit=12, active_records=1_000)

    assert result.executed[0].bounded is True
    assert result.complete is False


def test_a_bounded_similarity_read_does_not_make_a_ranking_incomplete() -> None:
    reader = _Reader(similar=tuple(_hit(f"s-{index}") for index in range(4)))

    result = execute(fallback_plan("q", k=4), reader, limit=4, active_records=1_000)

    assert result.executed[0].bounded is True
    assert result.complete is True
    assert [hit.id for hit in result.hits] == ["s-0", "s-1", "s-2", "s-3"]


def test_neighbors_anchor_on_the_rows_the_named_step_returned() -> None:
    reader = _Reader(
        match=(_hit("m-1", minutes=0), _hit("m-2", minutes=10)),
        neighbors=(_hit("n-1", minutes=-5),),
    )
    plan = parse_recall_plan(
        json.dumps(
            {
                "shape": "sequence",
                "steps": [
                    {"op": "match", "terms": ["mentioning X"]},
                    {"op": "neighbors", "of": "step:0", "before": 2, "after": 0},
                ],
            }
        ),
        reference_at=NOW,
    )

    assert plan is not None
    result = execute(plan, reader, limit=12, active_records=1_000)

    assert reader.calls[1] == ("neighbors", (("m-1", "m-2"), 2, 0, DEFAULT_MAX_ROWS))
    assert [hit.id for hit in result.hits] == ["n-1", "m-1", "m-2"]


def test_a_step_whose_anchor_read_nothing_reads_nothing_itself() -> None:
    reader = _Reader(neighbors=(_hit("n-1"),))
    plan = parse_recall_plan(
        json.dumps(
            {
                "shape": "sequence",
                "steps": [
                    {"op": "match", "terms": ["nothing matches this"]},
                    {"op": "neighbors", "of": "step:0"},
                ],
            }
        ),
        reference_at=NOW,
    )

    assert plan is not None
    result = execute(plan, reader, limit=12, active_records=1_000)

    assert [name for name, _arguments in reader.calls] == ["match"]
    assert result.hits == ()


def test_a_term_the_store_accepts_at_its_limit_still_plans() -> None:
    """The cap is the store's own, so the longest term it takes is still a runnable plan."""
    payload = json.dumps(
        {"shape": "set", "steps": [{"op": "match", "terms": ["x" * RECALL_MAX_TERM_CHARS]}]}
    )

    plan = parse_recall_plan(payload, reference_at=NOW)

    assert plan is not None
    assert plan.steps[0].terms == ("x" * RECALL_MAX_TERM_CHARS,)


def test_a_primitive_that_refuses_its_arguments_is_a_read_that_did_not_happen() -> None:
    """A store `ValueError` must not leave `ask()` raising on model-authored plan text.

    The step reads nothing and is reported as bounded: the rows its predicate matched were never
    read, so the other steps' rows are not a complete set and no count is licensed over them.
    """

    class Refusing(_Reader):
        def match(self, terms: Sequence[str], **arguments: object) -> RecallRows:
            del terms, arguments
            raise ValueError("a term must be at most 200 characters")

    reader = Refusing(similar=(_hit("s-1"),))
    plan = parse_recall_plan(
        json.dumps(
            {
                "shape": "composite",
                "steps": [
                    {"op": "match", "terms": ["cairo"]},
                    {"op": "similar", "query": "cairo"},
                ],
            }
        ),
        reference_at=NOW,
    )

    assert plan is not None
    result = execute(plan, reader, limit=12, active_records=1_000)

    assert result.executed[0] == RecallStepResult(op="match", rows=0, bounded=True)
    assert result.complete is False
    assert [hit.id for hit in result.hits] == ["s-1"]


@pytest.mark.parametrize(
    ("selected", "active", "flooded"),
    [(300, 600, True), (120, 600, False), (13, 10, True), (12, 10, False)],
    ids=(
        "over-the-corpus-share",
        "at-the-corpus-share",
        "over-the-limit-floor-on-a-small-corpus",
        "at-the-limit-floor-on-a-small-corpus",
    ),
)
def test_a_predicate_that_selected_most_of_the_corpus_contributes_no_rows(
    selected: int, active: int, flooded: bool
) -> None:
    """Completeness over a non-selective predicate is not evidence, it is the corpus.

    Measured on LoCoMo dev: the planner chose `entity` on 229 of 525 questions, the step
    degraded to matching the name as text with no identity registry behind it, and the name
    selected roughly half a corpus whose every turn reads "[date] Caroline said: ...". Those
    rows flooded the reader -- accuracy 0.721 -> 0.528 on exactly those questions, abstention
    18 -> 57 -- so the read contributes nothing and says so instead.
    """
    reader = _Reader(
        match=tuple(_hit(f"m-{index}", minutes=index) for index in range(30)),
        selected={"match": selected},
    )
    plan = parse_recall_plan(
        json.dumps({"shape": "set", "steps": [{"op": "match", "terms": ["caroline"]}]}),
        reference_at=NOW,
    )

    assert plan is not None
    result = execute(plan, reader, limit=3, active_records=active)

    assert len(result.hits) == (0 if flooded else 30)
    assert result.executed[0].non_selective == (selected if flooded else 0)
    assert result.non_selective_steps == int(flooded)
    assert result.complete is not flooded


def test_a_non_selective_step_tells_the_reader_what_it_holds_instead() -> None:
    """The note may not read as "nothing matched": what the reader holds is the ranking."""
    reader = _Reader(
        entity=tuple(_hit(f"m-{index}", minutes=index) for index in range(30)),
        selected={"entity": 300},
    )
    plan = parse_recall_plan(
        json.dumps({"shape": "entity", "steps": [{"op": "entity", "name": "Caroline"}]}),
        reference_at=NOW,
    )

    assert plan is not None
    note = recall_note(execute(plan, reader, limit=3, active_records=600))

    assert "records about Caroline matched too many records to enumerate (300 of 600)" in note
    assert "only the top-ranked records for the question are shown" in note
    assert "every record those reads matched" not in note


def test_a_neighbors_step_anchors_on_a_bounded_head_of_what_it_follows() -> None:
    """The store scans corpus order twice per anchor, so the anchor list cannot be a whole set.

    Measured: 500 anchors at 10 before and 10 after spent 4.3 seconds in index-less scans. The
    question is about what sits next to what was found, and the deepest anchors of a large match
    buy nothing the first ones do not.
    """
    reader = _Reader(
        match=tuple(_hit(f"m-{index}", minutes=index) for index in range(50)),
        neighbors=(_hit("n-1", minutes=-5),),
    )
    plan = parse_recall_plan(
        json.dumps(
            {
                "shape": "sequence",
                "steps": [
                    {"op": "match", "terms": ["wrench"]},
                    {"op": "neighbors", "of": "step:0", "before": 10, "after": 10},
                ],
            }
        ),
        reference_at=NOW,
    )

    assert plan is not None
    execute(plan, reader, limit=12, active_records=1_000)

    anchors = cast(tuple[tuple[str, ...], int, int, int], reader.calls[1][1])[0]
    assert anchors == tuple(f"m-{index}" for index in range(_MAX_NEIGHBOR_ANCHORS))


def test_a_read_the_bound_truncated_stays_bounded_when_a_scope_drops_a_row() -> None:
    """Completeness follows the selection, not the rows the caller was allowed to see.

    The bound is applied while the IDs are selected and the caller's bitemporal, spatial and
    metric scope is applied while they are hydrated, so a truncated read can come back shorter
    than its own bound. Reading that as "everything the predicate matched" is what lets a count
    be stated over a set the answer does not hold.
    """
    reader = _Reader(
        match=(_hit("m-1", minutes=0),),
        selected={"match": 3},
    )
    plan = parse_recall_plan(
        json.dumps({"shape": "set", "steps": [{"op": "match", "terms": ["a"], "max_rows": 3}]}),
        reference_at=NOW,
    )

    assert plan is not None
    result = execute(plan, reader, limit=12, active_records=1_000)

    assert result.executed[0] == RecallStepResult(op="match", rows=1, bounded=True)
    assert result.complete is False
