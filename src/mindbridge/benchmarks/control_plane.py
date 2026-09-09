"""Behaviour benchmark for the slow loop: `deliberate()`, `record_outcome()`, `rollback()`.

Everything here drives the public SDK -- `Memory.add_many`, `Memory.apply`, `Memory.search`,
`Memory.deliberate`, `Memory.operations`, `Memory.record_outcome`, `Memory.rollback`,
`Memory.get` -- against a
deterministic synthetic long run whose ground truth the scenario builder holds. That ground truth
is what makes the quality metrics `docs/context-os.md` names computable at all: without knowing
which observations were duplicates of each other, which assertion superseded which, and which
records were gold, an operation log says only that operations happened.

The scenario is sized so one deliberation window covers all of it. That is deliberate: the
measurement is of the loop and of the consolidation backend, not of retrieval, and a candidate
whose window happened to miss half a duplicate group would score the embedder instead.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from mindbridge.configuration import MindBridgeConfig, resolve_memory_config
from mindbridge.memory import Memory
from mindbridge.models.base import ConsolidationBackend, EmbeddingBackend, FormationInput
from mindbridge.types import (
    EvidenceBasis,
    FormationProposal,
    MemoryIntent,
    MemoryKind,
    MemoryOperationRecord,
    MemoryOutcome,
    Modality,
)

# The scenario's own clock. Fixed, so two runs of one seed ingest byte-identical occurrence times.
_EPOCH = datetime(2026, 1, 6, 9, tzinfo=timezone.utc)
_NAMES = ("Ana", "Bruno", "Chen", "Dara", "Eli", "Fatima", "Goran", "Hana")
# After the last person's day, whatever `persons` is: the window the failing recalls ask about.
_HORIZON = _EPOCH + timedelta(days=len(_NAMES))
_JOBS = ("radiographer", "luthier", "cartographer", "beekeeper")
_DRINKS = ("oat flat white", "genmaicha", "cold brew", "barley tea")
_PREFERENCES = (("aisle seats", "window seats"), ("early meetings", "late meetings"))


@dataclass(frozen=True, slots=True)
class Scenario:
    """What was injected, which is the ground truth every metric below is scored against."""

    # Each group is one fact observed twice: a correct CONSOLIDATE cites exactly one of these.
    duplicates: tuple[frozenset[str], ...] = ()
    # `(superseded, current)`: the two host-derived claims of one preference flip, oldest first.
    # A recovered contradiction leaves the second in force and the first out of recall.
    contradictions: tuple[tuple[str, str], ...] = ()
    # Standing facts nothing in this scenario contradicts or duplicates. Retiring one is a false
    # retirement whatever else the operation got right.
    gold: frozenset[str] = frozenset()
    ingested: frozenset[str] = frozenset()
    failing_queries: tuple[str, ...] = ()


@dataclass(slots=True)
class _Ingest:
    duplicates: list[frozenset[str]] = field(default_factory=list)
    contradictions: list[tuple[str, str]] = field(default_factory=list)
    gold: set[str] = field(default_factory=set)
    ingested: set[str] = field(default_factory=set)
    queries: list[str] = field(default_factory=list)


class _PreferenceFormer:
    """Make each injected first-person preference a user-stated functional claim."""

    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "control-plane-ground-truth"
    formation_space = "control-plane-ground-truth:v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        proposals: list[tuple[FormationProposal, ...]] = []
        for item in inputs:
            content = item.content.text.rstrip(".")
            marker = (
                " now says their only seat preference is "
                if " now says their only seat preference is " in content
                else " says their only seat preference is "
            )
            if marker not in content:
                proposals.append(())
                continue
            subject, value = content.split(marker, 1)
            proposals.append(
                (
                    FormationProposal(
                        kind=MemoryKind.TRAIT,
                        basis=EvidenceBasis.USER_STATEMENT,
                        content=item.content.text,
                        subject=subject,
                        predicate="only-seat-preference",
                        value=value,
                        confidence=0.9,
                    ),
                )
            )
        return tuple(proposals)

    def close(self) -> None:
        pass


def build_scenario(memory: Memory, *, persons: int, seed: int) -> Scenario:
    """Ingest one seeded synthetic long run and return its ground truth.

    Each exclusive preference flip is formed as one user-stated trait per side in the same
    `add_many` transaction. Both claims therefore stand for the loop to weigh, while the raw
    observations remain the benchmark's source evidence.

    Order matters and is part of the design. The failing recalls come last, because a candidate
    is dropped while nothing about it has changed since an operation last weighed it -- and the
    claims above are derived from the very records the failing query is nearest to. They are still
    genuinely empty recalls: the window they ask about is after everything this scenario ever
    recorded, which is exactly the shape of a real miss.
    """
    if persons < 1 or persons > len(_NAMES):
        raise ValueError(f"persons must be between 1 and {len(_NAMES)}")
    rng = random.Random(seed)
    names = _NAMES[:persons]
    built = _Ingest()
    for index, name in enumerate(names):
        job = _JOBS[rng.randrange(len(_JOBS))]
        drink = _DRINKS[rng.randrange(len(_DRINKS))]
        stale, current = _PREFERENCES[rng.randrange(len(_PREFERENCES))]
        base = _EPOCH + timedelta(days=index)
        # One transaction per person rather than five: the ordering the ground truth needs is the
        # declared `occurred_at`, not the order the index happened to flush in.
        gold, first, second, older, newer = memory.add_many(
            (
                f"{name} works as a {job}.",
                f"{name} drinks a {drink} every morning.",
                f"{name} has a {drink} every single morning without fail.",
                f"{name} says their only seat preference is {stale}.",
                f"{name} now says their only seat preference is {current}.",
            ),
            occurred_at=[base + timedelta(hours=hour) for hour in range(5)],
        )
        built.gold.add(gold.id)
        built.duplicates.append(frozenset({first.id, second.id}))
        stale_claim = _claim(memory, name, stale)
        current_claim = _claim(memory, name, current)
        built.contradictions.append((stale_claim, current_claim))
        built.ingested.update(
            {gold.id, first.id, second.id, older.id, newer.id, stale_claim, current_claim}
        )
    for name in names:
        query = f"what did {name} say about the archive keys"
        built.queries.append(query)
        # Two near-equal empty recalls are the whole of the trigger; one is not a signal.
        for _attempt in range(2):
            if memory.search(query, limit=10, occurred_from=_HORIZON):
                raise RuntimeError("the control-plane scenario requires these recalls to fail")
    return Scenario(
        duplicates=tuple(built.duplicates),
        contradictions=tuple(built.contradictions),
        gold=frozenset(built.gold),
        ingested=frozenset(built.ingested),
        failing_queries=tuple(built.queries),
    )


def _claim(memory: Memory, subject: str, value: str) -> str:
    """Find one typed preference claim the deterministic scenario former just committed."""
    for hit in memory.search(f"{subject} only seat preference {value}", limit=100):
        context = hit.context
        if (
            context is not None
            and context.kind is MemoryKind.TRAIT
            and context.basis is EvidenceBasis.USER_STATEMENT
            and context.subject == subject
            and context.predicate == "only-seat-preference"
            and context.value == value
        ):
            return hit.id
    raise RuntimeError(f"the control-plane scenario did not form {subject!r} preference {value!r}")


def run_benchmark(
    data_dir: str | Path,
    *,
    embedder: EmbeddingBackend,
    consolidator: ConsolidationBackend,
    persons: int = 4,
    seed: int = 42,
    limit: int = 100,
    max_rounds: int = 4,
) -> dict[str, object]:
    """Run one seeded control-plane scenario and return its JSON-ready quality metrics."""
    with Memory(
        Path(data_dir),
        embedder=embedder,
        former=_PreferenceFormer(),
        consolidator=consolidator,
        # The deliberation window must cover the whole scenario, and reinforcement must not make
        # one query's retrieval depend on which earlier one ran. Both are measurement policy.
        minimum_relevance=0,
        reinforce_on_answer=False,
    ) as memory:
        scenario = build_scenario(memory, persons=persons, seed=seed)
        before = {record.operation_id for record in memory.operations(limit=limit)}
        report = memory.deliberate(limit=limit, max_rounds=max_rounds)
        applied = tuple(
            record for record in memory.operations(limit=limit) if record.operation_id not in before
        )
        metrics = _metrics(memory, scenario, applied)
        # P5's only consumer: the log now carries what later evidence -- here, the ground truth --
        # said about each operation, which is what makes precision derivable from the log alone.
        for record in applied:
            memory.record_outcome(record.operation_id, _verdict(scenario, record))
        judged = tuple(
            record.outcome
            for record in memory.operations(limit=limit)
            if record.operation_id not in before
        )
        metrics.update(_rollback(memory, scenario, applied))
        return {
            "scenario": {
                "seed": seed,
                "persons": persons,
                "records": len(scenario.ingested),
                "duplicate_groups": len(scenario.duplicates),
                "contradictions": len(scenario.contradictions),
                "gold_records": len(scenario.gold),
                "failing_queries": len(scenario.failing_queries),
            },
            "deliberation": {
                "rounds": report.rounds,
                "weighed": report.weighed,
                "skipped": report.skipped,
                "applied": report.applied,
                "rejected": report.rejected,
                # The loop's whole cost proxy: `ConsolidationBackend` reports no token or currency
                # cost, so counting round trips is the honest measure rather than an invented one.
                "model_calls": report.model_calls,
            },
            "outcomes": {
                "confirmed": sum(1 for value in judged if value is MemoryOutcome.CONFIRMED),
                "refuted": sum(1 for value in judged if value is MemoryOutcome.REFUTED),
            },
            "metrics": metrics,
        }


def _metrics(
    memory: Memory,
    scenario: Scenario,
    applied: Sequence[MemoryOperationRecord],
) -> dict[str, object]:
    """Score one deliberation against the ground truth, before anything is rolled back."""
    consolidations = tuple(
        record for record in applied if record.operation.intent is MemoryIntent.CONSOLIDATE
    )
    matched = sum(
        1
        for record in consolidations
        if frozenset(record.operation.evidence_ids) in set(scenario.duplicates)
    )
    retired = {
        memory_id
        for record in applied
        for memory_id in (*record.forgotten_ids, *(name for name, _version in record.superseded))
    }
    recovered = sum(
        1
        for stale, current in scenario.contradictions
        if _retired(memory, stale) and not _retired(memory, current)
    )
    return {
        # Applied CONSOLIDATE operations whose cited evidence is exactly one injected duplicate
        # group, over every applied CONSOLIDATE. `None` when the loop proposed none: an
        # undefined rate is not a perfect one.
        "consolidation_precision": _ratio(matched, len(consolidations)),
        # Injected preference flips where the newer derived claim is the one left in force.
        "contradiction_recovery": _ratio(recovered, len(scenario.contradictions)),
        # Retired records that were gold-standing, over every record this run retired.
        "false_retirement": _ratio(
            len(retired & scenario.gold),
            len(retired),
        ),
        "applied_operations": len(applied),
        "consolidate_operations": len(consolidations),
        "retired_records": len(retired),
    }


def _rollback(
    memory: Memory,
    scenario: Scenario,
    applied: Sequence[MemoryOperationRecord],
) -> dict[str, object]:
    """Reverse everything this run applied, newest first, and say whether the store came back."""
    reversed_count = sum(
        1
        for record in sorted(applied, key=lambda item: item.operation_id, reverse=True)
        if memory.rollback(record.operation_id)
    )
    restored = not any(_retired(memory, memory_id) for memory_id in scenario.ingested)
    return {
        "rollback_success": _ratio(reversed_count, len(applied)),
        # Reversing every operation is not the same as restoring the state they changed; this is
        # the second half of the claim.
        "state_restored": restored,
    }


def _verdict(scenario: Scenario, record: MemoryOperationRecord) -> MemoryOutcome:
    """What the ground truth says about one applied operation."""
    retired = {*record.forgotten_ids, *(name for name, _version in record.superseded)}
    if retired & scenario.gold:
        return MemoryOutcome.REFUTED
    if record.operation.intent is MemoryIntent.CONSOLIDATE:
        cited = frozenset(record.operation.evidence_ids)
        return (
            MemoryOutcome.CONFIRMED if cited in set(scenario.duplicates) else MemoryOutcome.REFUTED
        )
    if record.operation.intent is MemoryIntent.CORRECT and set(record.operation.target_ids) & {
        current for _stale, current in scenario.contradictions
    }:
        # Retiring the claim that is actually in force is the wrong half of the flip, whatever
        # else the operation cited.
        return MemoryOutcome.REFUTED
    return MemoryOutcome.CONFIRMED


def _retired(memory: Memory, memory_id: str) -> bool:
    """Whether one record is out of active recall -- forgotten, or its claim version retired.

    Both halves matter: `FORGET` and consolidation forgetting set `forgotten_at`, while the
    `CORRECT` that resolves a contradiction retires the version instead.
    """
    record = memory.get(memory_id)
    return record.forgotten_at is not None or (
        record.context is not None and record.context.retired_at is not None
    )


def _ratio(part: int, whole: int) -> float | None:
    return None if whole == 0 else part / whole


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    """Parse CLI arguments, run the scenario against a configured backend, print one document."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "MindBridge configuration file declaring `embedding` and `consolidation`; the "
            "`benchmark` section an eval config carries is ignored"
        ),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--persons", type=_positive_int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args(argv)
    composition = resolve_memory_config(_config(arguments.config))
    try:
        consolidator = composition.plugins.consolidator
        if consolidator is None:
            parser.error(f"{arguments.config} declares no `consolidation` section")
        result = run_benchmark(
            arguments.data_dir,
            embedder=composition.plugins.embedder,
            consolidator=consolidator,
            persons=arguments.persons,
            seed=arguments.seed,
        )
    finally:
        composition.close()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _config(path: Path) -> MindBridgeConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"configuration {path} must be a mapping")
    values = {name: value for name, value in document.items() if name != "benchmark"}
    return MindBridgeConfig.model_validate(values)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
