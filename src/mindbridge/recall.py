"""Recall programs: the retrieval semantics a question's shape asks for.

Top-k similarity answers "what is most like this". A set question ("how many times", "list
all"), a sequence question ("what did I say just before that") and an entity question ("what do
I know about her") each ask for something similarity cannot state: completeness. A recall plan
names the shape and the bounded reads that answer it, and this module validates, executes and
merges them.

The merge is a union with ID dedup, never a weighted fusion of ranks and scores: measured
locally, reciprocal-rank fusion cost 8.7 points of R@1, and the literature's one reproducible
result on combining an exhaustive set with a ranked one is that the union neither helps nor
hurts while replacement loses. So exhaustive rows keep their chronological order, similarity
rows keep their rank, and no score is ever recomputed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Literal, Protocol, TypeVar, cast

# The store's own bound on one primitive read. Imported rather than restated so a plan can never
# ask for more rows than the read will return.
from mindbridge.infrastructure.local.store import _RECALL_MAX_ROWS as RECALL_MAX_ROWS
from mindbridge.types import MemoryType, Modality, SearchHit

RecallShape = Literal["point", "set", "sequence", "entity", "composite"]
RecallOp = Literal["similar", "match", "window", "neighbors", "entity"]

_SHAPES: frozenset[str] = frozenset({"point", "set", "sequence", "entity", "composite"})
_OPS: frozenset[str] = frozenset({"similar", "match", "window", "neighbors", "entity"})
# Only `similar` is a ranking; the rest answer by predicate and can therefore be complete.
_EXHAUSTIVE_OPS: frozenset[str] = frozenset({"match", "window", "neighbors", "entity"})
# Every field any op accepts, and the fields each op actually honours. A field an op cannot
# honour is accepted only as null: the planner prompt shows steps with keys spelled out, and a
# model that echoes one must not lose its plan -- but a filter that was asked for and silently
# dropped would make the evidence set a lie about what it contains.
_FIELDS: frozenset[str] = frozenset(
    {
        "op",
        "query",
        "k",
        "time",
        "modality",
        "memory_type",
        "terms",
        "any_of",
        "max_rows",
        "of",
        "before",
        "after",
        "name",
    }
)
_HONOURED: Mapping[str, frozenset[str]] = {
    # `similar` carries no filters: it ranks the caller's own question, which already carries the
    # temporal window and scope the caller asked with, and `k` is only how deep to rank it.
    "similar": frozenset({"query", "k"}),
    "match": frozenset({"terms", "any_of", "time", "modality", "memory_type", "max_rows"}),
    "window": frozenset({"time", "modality", "memory_type", "max_rows"}),
    "neighbors": frozenset({"of", "before", "after", "max_rows"}),
    "entity": frozenset({"name", "max_rows"}),
}
_MAX_STEPS = 6
_MAX_TERMS = 8
_MAX_NEIGHBOURS = 10
DEFAULT_SIMILAR_K = 12
DEFAULT_MAX_ROWS = 200

_EnumT = TypeVar("_EnumT", Modality, MemoryType)


@dataclass(frozen=True, slots=True)
class RecallStep:
    """One bounded read. `op` decides which fields mean anything."""

    op: RecallOp
    query: str | None = None
    k: int = DEFAULT_SIMILAR_K
    occurred_from: datetime | None = None
    occurred_until: datetime | None = None
    modality: Modality | None = None
    memory_type: MemoryType | None = None
    terms: tuple[str, ...] = ()
    any_of: bool = True
    max_rows: int = DEFAULT_MAX_ROWS
    # The index of an earlier step whose rows are this step's anchors.
    of: int | None = None
    before: int = 2
    after: int = 2
    name: str | None = None


@dataclass(frozen=True, slots=True)
class RecallPlan:
    """A question's shape and the reads that answer it."""

    shape: RecallShape
    steps: tuple[RecallStep, ...]

    @property
    def exhaustive(self) -> bool:
        """Whether this plan claims a set rather than a ranking."""
        return any(step.op in _EXHAUSTIVE_OPS for step in self.steps)


@dataclass(frozen=True, slots=True)
class RecallStepResult:
    """What one executed step read, for the trace and the grounding prompt."""

    op: RecallOp
    rows: int
    # True when the read returned exactly its bound, so rows beyond it were never read.
    bounded: bool


@dataclass(frozen=True, slots=True)
class RecallResult:
    """The merged evidence set one plan produced."""

    plan: RecallPlan
    executed: tuple[RecallStepResult, ...]
    # Chronological, from the exhaustive ops.
    exhaustive: tuple[SearchHit, ...] = ()
    # By rank, from `similar`, with anything already exhaustive removed.
    ranked: tuple[SearchHit, ...] = ()

    @property
    def hits(self) -> tuple[SearchHit, ...]:
        """The union: exhaustive rows in time order, then similarity rows by rank."""
        return self.exhaustive + self.ranked

    @property
    def complete(self) -> bool:
        """Whether every exhaustive read returned everything its predicate matched."""
        return not any(step.bounded for step in self.executed if step.op in _EXHAUSTIVE_OPS)


class RecallReader(Protocol):
    """The reads a recall program may make. `Memory` implements it over its own store."""

    def similar(self, query: str, *, k: int) -> tuple[SearchHit, ...]: ...

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
    ) -> tuple[SearchHit, ...]: ...

    def window(
        self,
        *,
        occurred_from: datetime,
        occurred_until: datetime,
        modality: Modality | None,
        memory_type: MemoryType | None,
        max_rows: int,
    ) -> tuple[SearchHit, ...]: ...

    def neighbors(
        self,
        memory_ids: Sequence[str],
        *,
        before: int,
        after: int,
        max_rows: int,
    ) -> tuple[SearchHit, ...]: ...

    def entity(self, name: str, *, max_rows: int) -> tuple[SearchHit, ...]: ...


def fallback_plan(question: str, *, k: int = DEFAULT_SIMILAR_K) -> RecallPlan:
    """The plan that is today's behaviour: one similarity search over the raw question.

    Every failure resolves here -- no planner, an unparseable plan, a plan that asks for an op
    that does not exist -- so a broken planner costs a call and changes no answer.
    """
    return RecallPlan(shape="point", steps=(RecallStep(op="similar", query=question, k=k),))


def parse_recall_plan(payload: str, *, reference_at: datetime) -> RecallPlan | None:
    """Validate a planner's JSON into a plan, or return None to fall back.

    Strict on meaning and forgiving on magnitude: an unknown op, an unknown field, a filter the
    op cannot honour or a step referring forward is a plan MindBridge will not run, while a `k`
    of 900 or 0 is clamped. Model output is data, never an instruction, and nothing here raises:
    the caller's next line is the fallback plan either way.
    """
    document = _document(payload)
    if document is None:
        return None
    shape = document.get("shape")
    raw_steps = document.get("steps")
    if shape not in _SHAPES or not isinstance(raw_steps, list) or not raw_steps:
        return None
    if len(raw_steps) > _MAX_STEPS or set(document) - {"shape", "steps"}:
        return None
    steps: list[RecallStep] = []
    for index, raw_step in enumerate(raw_steps):
        step = _step(raw_step, index=index, reference_at=reference_at)
        if step is None:
            return None
        steps.append(step)
    # A point question is a ranking question: widening k on one is a measured null, and running
    # set ops over it only adds noise. A planner that says "point" and then asks for a set has
    # contradicted itself, and the fallback is what a point plan would have been anyway.
    if shape == "point" and any(step.op in _EXHAUSTIVE_OPS for step in steps):
        return None
    return RecallPlan(shape=cast(RecallShape, shape), steps=tuple(steps))


def execute(plan: RecallPlan, reader: RecallReader) -> RecallResult:
    """Run every step and merge the rows into one evidence set.

    Exhaustive rows come first in chronological order, then the similarity rows by rank with
    anything already present removed. IDs are the only thing deduplicated and no score decides
    the order, so the set a caller reports as complete is the set the predicate defined.
    """
    executed: list[RecallStepResult] = []
    exhaustive: dict[str, SearchHit] = {}
    ranked: dict[str, SearchHit] = {}
    by_step: list[tuple[SearchHit, ...]] = []
    for step in plan.steps:
        rows = _run(step, reader, by_step)
        by_step.append(rows)
        bound = step.k if step.op == "similar" else step.max_rows
        executed.append(RecallStepResult(op=step.op, rows=len(rows), bounded=len(rows) >= bound))
        target = ranked if step.op == "similar" else exhaustive
        for hit in rows:
            target.setdefault(hit.id, hit)
    return RecallResult(
        plan=plan,
        executed=tuple(executed),
        exhaustive=tuple(sorted(exhaustive.values(), key=_chronological)),
        ranked=tuple(hit for hit in ranked.values() if hit.id not in exhaustive),
    )


def recall_note(result: RecallResult, *, omitted: int = 0) -> str:
    """State the program and whether its evidence is a complete set.

    A count or a list is only licensed by completeness, and the reader cannot see the predicate
    that produced its evidence. So the reads are named, and either the set is declared complete
    or the shortfall is: measured, an answerer given an incomplete set with no warning states a
    total for it anyway.
    """
    reads = "; ".join(
        _step_note(step, outcome)
        for step, outcome in zip(result.plan.steps, result.executed, strict=True)
    )
    if result.complete and not omitted:
        completeness = (
            "The evidence below is every record those reads matched, in time order, so a count "
            "or a list over it is complete."
        )
    else:
        shortfall = (
            f"{omitted} further matched records are not shown"
            if omitted
            else "some matching records were not read"
        )
        completeness = (
            f"{shortfall}, so the evidence is not a complete set: answer from what is shown and "
            "do not state a total."
        )
    return f"Recall program ({result.plan.shape}): {reads}. {completeness}"


def _step_note(step: RecallStep, outcome: RecallStepResult) -> str:
    if step.op == "match":
        detail = f"records containing {', '.join(step.terms)}"
    elif step.op == "window":
        detail = "records in the time span"
    elif step.op == "neighbors":
        detail = f"the {step.before} before and {step.after} after each of them"
    elif step.op == "entity":
        detail = f"records about {step.name}"
    else:
        detail = "the top-ranked records for the question"
    return f"{detail}{_span_note(step)} ({outcome.rows})"


def _span_note(step: RecallStep) -> str:
    start, until = step.occurred_from, step.occurred_until
    if start is not None and until is not None:
        return f" between {start.isoformat()} and {until.isoformat()}"
    if start is not None:
        return f" since {start.isoformat()}"
    if until is not None:
        return f" before {until.isoformat()}"
    return ""


def _run(
    step: RecallStep,
    reader: RecallReader,
    by_step: Sequence[tuple[SearchHit, ...]],
) -> tuple[SearchHit, ...]:
    """Dispatch one step, or read nothing when its own inputs are missing."""
    if step.op == "similar":
        return reader.similar(step.query or "", k=step.k)
    if step.op == "match":
        return reader.match(
            step.terms,
            any_of=step.any_of,
            occurred_from=step.occurred_from,
            occurred_until=step.occurred_until,
            modality=step.modality,
            memory_type=step.memory_type,
            max_rows=step.max_rows,
        )
    if step.op == "window":
        if step.occurred_from is None or step.occurred_until is None:
            return ()
        return reader.window(
            occurred_from=step.occurred_from,
            occurred_until=step.occurred_until,
            modality=step.modality,
            memory_type=step.memory_type,
            max_rows=step.max_rows,
        )
    if step.op == "neighbors":
        anchors = () if step.of is None else tuple(hit.id for hit in by_step[step.of])
        if not anchors:
            return ()
        return reader.neighbors(
            anchors,
            before=step.before,
            after=step.after,
            max_rows=step.max_rows,
        )
    return () if step.name is None else reader.entity(step.name, max_rows=step.max_rows)


def _chronological(hit: SearchHit) -> tuple[datetime, str]:
    """Order exhaustive rows the way a timeline reads, with the ID as the stable tie-break."""
    return (hit.occurred_at or hit.created_at, hit.id)


def _document(payload: str) -> Mapping[str, object] | None:
    if not isinstance(payload, str) or not payload.strip():
        return None
    try:
        document = json.loads(payload)
    except (TypeError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _fields(raw_step: object) -> Mapping[str, object] | None:
    """Prove one step is a mapping this op may carry, honoured filters included."""
    if not isinstance(raw_step, dict):
        return None
    op = raw_step.get("op")
    if not isinstance(op, str) or op not in _OPS or set(raw_step) - _FIELDS:
        return None
    honoured = _HONOURED[op]
    for key, value in raw_step.items():
        if key != "op" and key not in honoured and value is not None:
            return None
    return raw_step


def _step(raw_step: object, *, index: int, reference_at: datetime) -> RecallStep | None:
    values = _fields(raw_step)
    if values is None:
        return None
    window = _window(values.get("time"), reference_at=reference_at)
    modality = _enum(values.get("modality"), Modality)
    memory_type = _enum(values.get("memory_type"), MemoryType)
    terms = _terms(values.get("terms"))
    anchor = _anchor(values.get("of"), index=index)
    any_of = values.get("any_of", True)
    if window is None or modality is False or memory_type is False:
        return None
    if terms is None or anchor is False or not isinstance(any_of, bool):
        return None
    return _usable(
        RecallStep(
            op=str(values["op"]),  # type: ignore[arg-type]
            query=_text(values.get("query")),
            k=_clamped(values.get("k"), default=DEFAULT_SIMILAR_K, high=100),
            occurred_from=window[0],
            occurred_until=window[1],
            modality=modality,
            memory_type=memory_type,
            terms=terms,
            any_of=any_of,
            max_rows=_clamped(
                values.get("max_rows"),
                default=DEFAULT_MAX_ROWS,
                high=RECALL_MAX_ROWS,
            ),
            of=anchor,
            before=_clamped(values.get("before"), default=2, low=0, high=_MAX_NEIGHBOURS),
            after=_clamped(values.get("after"), default=2, low=0, high=_MAX_NEIGHBOURS),
            name=_text(values.get("name")),
        )
    )


def _usable(step: RecallStep) -> RecallStep | None:
    """Reject a step whose own required input is missing, which no clamp can repair."""
    if step.op == "similar" and not step.query:
        return None
    if step.op == "match" and not step.terms:
        return None
    if step.op == "window" and (step.occurred_from is None or step.occurred_until is None):
        return None
    if step.op == "neighbors" and (step.of is None or step.before + step.after == 0):
        return None
    if step.op == "entity" and not step.name:
        return None
    return step


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _terms(value: object) -> tuple[str, ...] | None:
    if value is None:
        return ()
    if not isinstance(value, list):
        return None
    terms = tuple(
        dict.fromkeys(term.strip() for term in value if isinstance(term, str) and term.strip())
    )
    return terms[:_MAX_TERMS]


def _anchor(value: object, *, index: int) -> int | Literal[False] | None:
    """Resolve `"step:N"` to an earlier step's index; a forward reference is not a plan."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("step:"):
        return False
    try:
        anchor = int(value.removeprefix("step:"))
    except ValueError:
        return False
    return anchor if 0 <= anchor < index else False


def _clamped(value: object, *, default: int, high: int, low: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(low, min(high, value))


def _enum(value: object, kind: type[_EnumT]) -> _EnumT | Literal[False] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return False
    try:
        return kind(value)
    except ValueError:
        return False


def _window(
    value: object,
    *,
    reference_at: datetime,
) -> tuple[datetime | None, datetime | None] | None:
    """Read `["from", "until"]`, either half nullable, and reject anything else."""
    if value is None:
        return (None, None)
    if not isinstance(value, list) or len(value) != 2:
        return None
    bounds: list[datetime | None] = []
    for bound in value:
        if bound is None:
            bounds.append(None)
            continue
        moment = _moment(bound, reference_at=reference_at)
        if moment is None:
            return None
        bounds.append(moment)
    start, until = bounds
    if start is not None and until is not None and until <= start:
        return None
    return (start, until)


def _moment(value: object, *, reference_at: datetime) -> datetime | None:
    """Parse one ISO date or timestamp, placed in the reference clock's zone when it has none."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return _date_only(text, reference_at=reference_at)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=reference_at.tzinfo or timezone.utc)
    return parsed


def _date_only(text: str, *, reference_at: datetime) -> datetime | None:
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return None
    return datetime.combine(parsed, time.min, tzinfo=reference_at.tzinfo or timezone.utc)
