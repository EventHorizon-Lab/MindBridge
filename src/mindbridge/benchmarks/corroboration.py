"""Paired public-SDK experiment on fixed evidence graphs, without an answer model.

These are structural simulations with scripted assessments and a constant embedder, including
media transport payloads. They do not measure perception, embedding quality or natural language
formation. Variants of one graph family are not independent natural-distribution observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from mindbridge import (
    Blob,
    ContentInput,
    ContextBudget,
    EmbedTask,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryKind,
    MemoryNotFoundError,
    MemoryOperation,
    MemoryRecord,
    MemoryTrigger,
    Modality,
    ModelInput,
    ObservationContext,
)

_REFERENCE = datetime(2026, 9, 1, tzinfo=timezone.utc)
_BUDGET = ContextBudget(max_items=64, max_chars=256_000)
_ROUTES = ("text", "image", "audio", "video", "omni")


@dataclass(frozen=True)
class Step:
    name: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    family: str
    split: str
    groups: tuple[str, ...]
    steps: tuple[Step, ...]
    expected_visible: bool
    withdraw: str | None = None
    expected_after_withdrawal: bool | None = None


def scenarios() -> tuple[Scenario, ...]:
    """Predeclared graph families; raw observations are named r0, r1, ... ."""
    return (
        Scenario(
            "joint_independent",
            "dev",
            ("a", "b", "c", "d"),
            (Step("target", ("r0", "r1")), Step("target", ("r2", "r3"))),
            True,
        ),
        Scenario("joint_single", "dev", ("a", "b"), (Step("target", ("r0", "r1")),), False),
        Scenario(
            "capture_duplicates",
            "dev",
            ("a", "a"),
            (Step("target", ("r0",)), Step("target", ("r1",))),
            False,
        ),
        Scenario(
            "singleton_independent",
            "dev",
            ("a", "b"),
            (Step("target", ("r0",)), Step("target", ("r1",))),
            True,
        ),
        Scenario(
            "nested_overlap",
            "dev",
            ("a", "b", "c"),
            (
                Step("x", ("r0", "r1")),
                Step("y", ("r1", "r2")),
                Step("target", ("x",)),
                Step("target", ("y",)),
            ),
            False,
        ),
        Scenario(
            "or_alternative",
            "dev",
            ("a", "b"),
            (
                Step("x", ("r0",)),
                Step("x", ("r1",)),
                Step("target", ("x",)),
                Step("target", ("r1",)),
            ),
            True,
        ),
        Scenario(
            "shared_intermediate",
            "dev",
            ("a", "b"),
            (
                Step("x", ("r0",)),
                Step("x", ("r1",)),
                Step("y", ("x",)),
                Step("z", ("x",)),
                Step("target", ("y",)),
                Step("target", ("z",)),
            ),
            False,
        ),
        Scenario(
            "or_inside_one_assessment",
            "holdout",
            ("a", "b"),
            (Step("x", ("r0",)), Step("x", ("r1",)), Step("target", ("x",))),
            False,
        ),
        Scenario(
            "shared_deep_intermediate",
            "holdout",
            ("a", "b", "c", "d", "e", "f"),
            (
                Step("m", ("r0", "r1")),
                Step("m", ("r2", "r3")),
                Step("x", ("m", "r4")),
                Step("y", ("m", "r5")),
                Step("target", ("x",)),
                Step("target", ("y",)),
            ),
            False,
        ),
        Scenario(
            "and_or_crossproduct",
            "holdout",
            ("a", "b", "c", "d"),
            (
                Step("x", ("r0",)),
                Step("x", ("r1",)),
                Step("y", ("r0",)),
                Step("y", ("r2",)),
                Step("target", ("x", "y")),
                Step("target", ("r0", "r3")),
            ),
            True,
            "r1",
            False,
        ),
        Scenario(
            "aliased_deep_sources",
            "holdout",
            ("a", "a", "b", "c"),
            (
                Step("x", ("r0", "r2")),
                Step("y", ("r1", "r3")),
                Step("target", ("x",)),
                Step("target", ("y",)),
            ),
            False,
        ),
        Scenario(
            "surviving_ancestry",
            "holdout",
            ("a", "b", "c", "d", "e"),
            (
                Step("x", ("r0", "r1")),
                Step("x", ("r2", "r4")),
                Step("y", ("r2", "r3")),
                Step("target", ("x",)),
                Step("target", ("y",)),
            ),
            True,
            "r1",
            False,
        ),
        Scenario(
            "bypass_shared_summary",
            "holdout",
            ("a", "b", "c", "d", "e", "f"),
            (
                Step("m", ("r0", "r1")),
                Step("m", ("r2", "r3")),
                Step("x", ("m", "r4")),
                Step("y", ("m", "r5")),
                Step("y", ("r2", "r3", "r5")),
                Step("target", ("x",)),
                Step("target", ("y",)),
            ),
            True,
        ),
    )


class _Embedder:
    embedding_capabilities = frozenset(value for value in Modality if value is not Modality.OMNI)
    embedding_model = "corroboration-constant"
    embedding_space = "corroboration-constant:4:v1"
    embedding_dimension = 4

    def __init__(self) -> None:
        self.items = 0

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        del task
        self.items += len(inputs)
        return tuple((1.0, 0.0, 0.0, 0.0) for _ in inputs)

    def close(self) -> None:
        pass


class _Consolidator:
    consolidation_model = "corroboration-fixed-proposals"
    consolidation_recipe = "corroboration-fixed-proposals:v1"

    def __init__(self) -> None:
        self.pending: tuple[MemoryOperation, ...] = ()
        self.calls = 0
        self.input_records = 0
        self.input_chars = 0

    def consolidate(
        self, evidence: Sequence[MemoryRecord], *, trigger: MemoryTrigger
    ) -> tuple[MemoryOperation, ...]:
        del trigger
        self.calls += 1
        self.input_records += len(evidence)
        self.input_chars += sum(len(record.content) for record in evidence)
        pending, self.pending = self.pending, ()
        return pending

    def close(self) -> None:
        pass


def _content(label: str, route: str) -> ContentInput:
    # No decoder or recognizer is configured. Distinct bytes exercise durable asset identity;
    # intentionally not advertised as valid pictures, audio waveforms or video scenes.
    payload = label.encode()
    if route == "text":
        return label
    if route == "omni":
        return (
            label,
            Blob(b"image:" + payload, media_type="image/png"),
            Blob(b"audio:" + payload, media_type="audio/wav"),
        )
    media_type = {"image": "image/png", "audio": "audio/wav", "video": "video/mp4"}[route]
    return Blob(payload, media_type=media_type)


def run_case(scenario: Scenario, route: str, seed: int, arm: str, data_dir: Path) -> dict[str, Any]:
    """Run identical captures/proposals under one policy in an exclusive physical store."""
    rng = random.Random(seed)
    actor = f"person-{rng.getrandbits(48):012x}"
    embedder, consolidator = _Embedder(), _Consolidator()
    started = perf_counter()
    with Memory(
        data_dir,
        embedder=embedder,
        consolidator=consolidator,
        independent_evidence=arm == "B",
        minimum_relevance=0.0,
        ambiguity_margin=0.0,
        reinforce_on_answer=False,
    ) as memory:
        ids: dict[str, str] = {}
        root_order = list(range(len(scenario.groups)))
        rng.shuffle(root_order)
        for index in root_order:
            record = memory.add(
                _content(f"{actor} observation {index}", route),
                occurred_at=_REFERENCE + timedelta(seconds=index),
                context=ObservationContext(source_id=f"{actor}:{scenario.groups[index]}"),
            )
            ids[f"r{index}"] = record.id
        for step in scenario.steps:
            sources = tuple(ids[source] for source in step.sources)
            consolidator.pending = (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=sources,
                    proposal=FormationProposal(
                        kind=MemoryKind.TRAIT if step.name == "target" else MemoryKind.RELATION,
                        content=f"{actor} has pattern {step.name}",
                        subject=actor,
                        predicate=f"pattern_{step.name}",
                        value=step.name,
                        confidence=0.6,
                    ),
                ),
            )
            report = memory.consolidate(evidence_ids=sources)
            if report.rejected or not report.operations:
                raise RuntimeError(f"{scenario.family}/{step.name}: proposal not applied: {report}")
            result = report.operations[0]
            derived_ids = result.created_ids or result.changed_ids
            if not derived_ids:
                raise RuntimeError(f"{scenario.family}/{step.name}: no derived record")
            ids[step.name] = derived_ids[0]
        target = memory.get(ids["target"])
        assert target.context is not None
        write_seconds = perf_counter() - started
        compiled = memory.compile(
            f"{actor} has pattern target",
            budget=_BUDGET,
            reference_at=_REFERENCE + timedelta(days=1),
        )
        selected = {hit.id for hit in compiled.hits}
        result_row: dict[str, Any] = {
            "family": scenario.family,
            "split": scenario.split,
            "route": route,
            "seed": seed,
            "arm": arm,
            "expected_visible": scenario.expected_visible,
            "visible": target.context.visible,
            "confidence": target.context.confidence,
            "compiled": target.id in selected,
            "compile_chars": compiled.chars,
            "budget_chars": _BUDGET.max_chars,
            "budget_items": _BUDGET.max_items,
            "budget_ok": compiled.chars <= _BUDGET.max_chars
            and len(compiled.hits) <= _BUDGET.max_items,
            "consolidation_calls": consolidator.calls,
            "input_records": consolidator.input_records,
            "input_chars": consolidator.input_chars,
            "embedding_items": embedder.items,
            "write_seconds": write_seconds,
            "compile_ms": compiled.elapsed_ms,
            "target_evidence_count": len(target.context.evidence_ids),
        }
        if scenario.withdraw is not None:
            memory.delete(ids[scenario.withdraw])
            try:
                after = memory.get(target.id)
                assert after.context is not None
                after_visible = after.context.visible
            except MemoryNotFoundError:
                after_visible = False
            result_row["post_visible"] = after_visible
            result_row["withdrawal_ok"] = after_visible == scenario.expected_after_withdrawal
    result_row["total_seconds"] = perf_counter() - started
    result_row["store_bytes"] = sum(
        path.stat().st_size for path in data_dir.rglob("*") if path.is_file()
    )
    return result_row


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["expected_visible"]]
    negative = [row for row in rows if not row["expected_visible"]]
    withdrawals = [row for row in rows if "withdrawal_ok" in row]
    return {
        "n": len(rows),
        "withdrawal_accuracy": None
        if not withdrawals
        else sum(row["withdrawal_ok"] for row in withdrawals) / len(withdrawals),
        "learning_rate": None
        if not positive
        else sum(row["visible"] for row in positive) / len(positive),
        "false_corroboration_rate": None
        if not negative
        else sum(row["visible"] for row in negative) / len(negative),
        "accuracy": None
        if not rows
        else sum(row["visible"] == row["expected_visible"] for row in rows) / len(rows),
        "compiled_accuracy": None
        if not rows
        else sum(row["compiled"] == row["expected_visible"] for row in rows) / len(rows),
    }


def _ablation(scenario: Scenario, *, union_alternatives: bool) -> bool:
    """Deliberately weakened offline controls, not product policies or quality scorers."""
    clauses = {frozenset(step.sources) for step in scenario.steps if step.name == "target"}
    if not union_alternatives:
        return len(clauses) >= 2
    roots: dict[str, set[str]] = {f"r{i}": {group} for i, group in enumerate(scenario.groups)}
    parents: dict[str, set[str]] = {}
    for step in scenario.steps:
        roots.setdefault(step.name, {f"derived:{step.name}"})
        parents.setdefault(step.name, set()).update(step.sources)
    for _ in range(len(roots)):
        changed = False
        for name, sources in parents.items():
            union = set().union(*(roots[source] for source in sources))
            changed |= not union <= roots[name]
            roots[name].update(union)
        if not changed:
            break
    target = [set().union(*(roots[source] for source in sources)) for sources in clauses]
    return any(not first & second for i, first in enumerate(target) for second in target[i + 1 :])


def _paired(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    def key(row: Mapping[str, Any]) -> tuple[str, str, int]:
        return row["family"], row["route"], row["seed"]

    baseline = {key(row): row for row in rows if row["arm"] == "A"}
    pairs = [(baseline[key(row)], row) for row in rows if row["arm"] == arm]
    wins = losses = 0
    for first, second in pairs:
        delta = int(second["visible"] == second["expected_visible"]) - int(
            first["visible"] == first["expected_visible"]
        )
        wins += delta > 0
        losses += delta < 0
    costs = [second["write_seconds"] / first["write_seconds"] for first, second in pairs]
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(pairs) - wins - losses,
        "median_write_ratio": statistics.median(costs) if costs else None,
        "same_inputs": all(
            (a["consolidation_calls"], a["input_records"], a["input_chars"])
            == (b["consolidation_calls"], b["input_records"], b["input_chars"])
            for a, b in pairs
        ),
        "same_embedding_items": all(a["embedding_items"] == b["embedding_items"] for a, b in pairs),
        "same_projections": all(
            (a["visible"], a["confidence"], a["compiled"], a.get("post_visible"))
            == (b["visible"], b["confidence"], b["compiled"], b.get("post_visible"))
            for a, b in pairs
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="all")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--routes", nargs="+", choices=_ROUTES, default=list(_ROUTES))
    args = parser.parse_args(argv)
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    cases = [case for case in scenarios() if args.split == "all" or case.split == args.split]
    if not cases:
        parser.error("selected split has no scenarios")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.dumps(
        [asdict(case) for case in cases], ensure_ascii=False, sort_keys=True, indent=2
    )
    (args.output / "cases.json").write_text(manifest + "\n")
    source_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted(source_root.rglob("*.py")):
        digest.update(str(source.relative_to(source_root)).encode())
        digest.update(source.read_bytes())
    meta = {
        "base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": digest.hexdigest(),
        "cases_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "routes": args.routes,
        "replicates": args.replicates,
        "external_model_calls": 0,
        "limitations": "Fixed proposals, constant embeddings, synthetic media bytes; no perception or natural-distribution quality claims.",
    }
    (args.output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    rows: list[dict[str, Any]] = []
    with (
        TemporaryDirectory(prefix="mindbridge-corroboration-") as temporary,
        (args.output / "samples.jsonl").open("w") as output,
    ):
        for case in cases:
            for route in args.routes:
                for replicate in range(args.replicates):
                    seed = (7100 if case.split == "dev" else 9100) + replicate
                    arms = ["A", "A2", "B"]
                    random.Random(f"{case.family}:{route}:{seed}").shuffle(arms)
                    for arm in arms:
                        path = Path(temporary) / f"{case.family}-{route}-{seed}-{arm}"
                        row = run_case(case, route, seed, arm, path)
                        rows.append(row)
                        output.write(json.dumps(row, sort_keys=True) + "\n")
                        output.flush()
            print(f"completed {case.split}/{case.family}", flush=True)
    report: dict[str, Any] = {
        "arms": {
            arm: summarize([row for row in rows if row["arm"] == arm]) for arm in ("A", "A2", "B")
        },
        "paired": {arm: _paired(rows, arm) for arm in ("A2", "B")},
        "families": {
            case.family: {
                arm: summarize(
                    [row for row in rows if row["family"] == case.family and row["arm"] == arm]
                )
                for arm in ("A", "B")
            }
            for case in cases
        },
        "offline_ablations": {
            case.family: {
                "expected": case.expected_visible,
                "C_no_provenance": _ablation(case, union_alternatives=False),
                "D_union_ancestors": _ablation(case, union_alternatives=True),
            }
            for case in cases
        },
        "all_budgets_ok": all(row["budget_ok"] for row in rows),
        "splits": {
            split: {
                arm: summarize([row for row in rows if row["split"] == split and row["arm"] == arm])
                for arm in ("A", "B")
            }
            for split in sorted({case.split for case in cases})
        },
        "resources": {
            arm: {
                metric: statistics.median([row[metric] for row in rows if row["arm"] == arm])
                for metric in (
                    "write_seconds",
                    "compile_ms",
                    "total_seconds",
                    "store_bytes",
                    "consolidation_calls",
                    "embedding_items",
                    "input_chars",
                )
            }
            for arm in ("A", "A2", "B")
        },
        "inference": "Descriptive paired structural cases. Repeated routes/seeds are correlated variants, not independent samples for a significance test.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["arms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
