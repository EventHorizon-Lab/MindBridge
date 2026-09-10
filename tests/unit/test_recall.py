from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import pytest

from mindbridge.recall import (
    DEFAULT_MAX_ROWS,
    RecallPlan,
    RecallStep,
    execute,
    fallback_plan,
    parse_recall_plan,
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
    """Records every read a plan makes and replays a scripted answer for each op."""

    def __init__(self, **rows: tuple[SearchHit, ...]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, object]] = []

    def similar(
        self,
        query: str,
        *,
        k: int,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        memory_type: MemoryType | None,
    ) -> tuple[SearchHit, ...]:
        self.calls.append(("similar", (query, k, occurred_from, occurred_until, memory_type)))
        return self.rows.get("similar", ())

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
    ) -> tuple[SearchHit, ...]:
        self.calls.append(("match", (tuple(terms), any_of, modality, memory_type, max_rows)))
        return self.rows.get("match", ())

    def window(
        self,
        *,
        occurred_from: datetime,
        occurred_until: datetime,
        modality: Modality | None,
        memory_type: MemoryType | None,
        max_rows: int,
    ) -> tuple[SearchHit, ...]:
        self.calls.append(("window", (occurred_from, occurred_until, modality, max_rows)))
        return self.rows.get("window", ())

    def neighbors(
        self,
        memory_ids: Sequence[str],
        *,
        before: int,
        after: int,
        max_rows: int,
    ) -> tuple[SearchHit, ...]:
        self.calls.append(("neighbors", (tuple(memory_ids), before, after, max_rows)))
        return self.rows.get("neighbors", ())

    def entity(self, name: str, *, max_rows: int) -> tuple[SearchHit, ...]:
        self.calls.append(("entity", (name, max_rows)))
        return self.rows.get("entity", ())


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
    result = execute(plan, reader)

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
    result = execute(plan, reader)

    assert result.executed[0].bounded is True
    assert result.complete is False


def test_a_bounded_similarity_read_does_not_make_a_ranking_incomplete() -> None:
    reader = _Reader(similar=tuple(_hit(f"s-{index}") for index in range(4)))

    result = execute(fallback_plan("q", k=4), reader)

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
    result = execute(plan, reader)

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
    result = execute(plan, reader)

    assert [name for name, _arguments in reader.calls] == ["match"]
    assert result.hits == ()
