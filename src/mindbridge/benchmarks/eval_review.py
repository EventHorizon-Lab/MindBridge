"""Export a reproducible stratified sample for human evidence annotation."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from mindbridge.benchmarks.eval_artifacts import _jsonl_bytes
from mindbridge.benchmarks.eval_diagnostics import LAYERS
from mindbridge.benchmarks.eval_results import EVAL_SCHEMA_VERSION


def _positive(value: str) -> int:
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return count


def _sample_row(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not isinstance(value.get("sample_id"), str):
        raise ValueError("each sample must be an object with a sample_id")
    if value.get("schema_version") != EVAL_SCHEMA_VERSION:
        raise ValueError("review sample schema version is unsupported")
    if any(not isinstance(value.get(name), str) or not value[name] for name in ("task", "arm")):
        raise ValueError("review samples require task and arm names")
    return value


def _stratum(row: dict[str, object]) -> tuple[str, ...]:
    metadata = row.get("metadata")
    category = (
        metadata.get(
            "question_type", metadata.get("qtype", metadata.get("category", "unspecified"))
        )
        if isinstance(metadata, dict)
        else "unspecified"
    )
    diagnostic = row.get("quality_diagnostics")
    labelled = isinstance(diagnostic, dict) and diagnostic.get("gold_source_key") is not None
    outcome = (
        "error"
        if row.get("error_code") is not None
        else "abstained"
        if row.get("abstained") is True
        else "unscored"
        if row.get("score") is None
        else "zero_score"
        if row.get("score") == 0
        else "positive_score"
    )
    return (
        str(row["task"]),
        str(row["arm"]),
        str(category),
        outcome,
        "source_labelled" if labelled else "source_unlabelled",
    )


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=_positive, default=10)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    if arguments.samples.resolve() == arguments.output.resolve():
        parser.error("--output must differ from --samples")
    strata: dict[tuple[str, ...], list[tuple[int, str, dict[str, object]]]] = defaultdict(list)
    populations: Counter[tuple[str, ...]] = Counter()
    seen: set[str] = set()
    with arguments.samples.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = _sample_row(json.loads(line))
            sample_id = cast(str, row["sample_id"])
            if sample_id in seen:
                raise ValueError("duplicate sample_id")
            seen.add(sample_id)
            stratum = _stratum(row)
            populations[stratum] += 1
            priority = int.from_bytes(
                hashlib.sha256(
                    json.dumps(
                        [arguments.seed, row.get("dataset_sha256"), sample_id],
                        ensure_ascii=False,
                    ).encode()
                ).digest(),
                "big",
            )
            # Retain only k full rows per stratum; negative priorities evict the largest hash.
            heapq.heappush(strata[stratum], (-priority, sample_id, row))
            if len(strata[stratum]) > arguments.per_stratum:
                heapq.heappop(strata[stratum])
    selected: list[dict[str, object]] = []
    for stratum, rows in sorted(strata.items()):
        for _, _, row in sorted(rows, key=lambda entry: (-entry[0], entry[1])):
            selected.append(
                {
                    **row,
                    "review_sampling": {
                        "seed": arguments.seed,
                        "stratum": stratum,
                        "population_count": populations[stratum],
                        "selected_count": len(rows),
                    },
                    "annotation": {
                        "status": "pending",
                        "annotator": None,
                        "source_spans": None,
                        "layers": dict.fromkeys(LAYERS),
                        "rationale": None,
                    },
                }
            )
    payload = _jsonl_bytes(selected)
    arguments.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _publish_review(arguments.output, payload)
    return 0


def _publish_review(output: Path, payload: bytes) -> None:
    """Publish exclusively so even a concurrent human annotation cannot be replaced."""
    with NamedTemporaryFile(dir=output.parent, prefix=".review.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(str(temporary), output)
    except FileExistsError:
        if output.read_bytes() != payload:
            raise FileExistsError(
                "review output exists; choose a new path to preserve annotations"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
