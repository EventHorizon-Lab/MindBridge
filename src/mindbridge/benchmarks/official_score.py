"""Bind an external official scorer's numbers to the exact run that produced them.

Most benchmarks here are scored outside MindBridge: LoCoMo-Refined by
`mem-eval-suite/LoCoMo_refined`, MM-Lifelong by its released scorer, EgoMemReason by a held-out
leaderboard. A run manifest is written before any of them execute, so it can only pin inputs —
never results. This module
writes the missing half as a separate sidecar that refuses to attach numbers to predictions the
manifest did not produce, and records which judge and answer model stood behind them.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from mindbridge.benchmarks.artifacts import write_text_atomically
from mindbridge.benchmarks.cli import parser as build_parser
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file

OFFICIAL_SCORE_VERSION = "official_score_v1"


class OfficialScore(ContractModel):
    """One official scorer result, bound to the run and the protocol behind it."""

    score_version: NonEmptyString = OFFICIAL_SCORE_VERSION
    benchmark: NonEmptyString
    run_id: Identifier
    predictions_sha256: Sha256Hex
    scorer_repository: NonEmptyString
    scorer_command: NonEmptyString
    judge_model: NonEmptyString | None = None
    answer_backbone: NonEmptyString | None = None
    scored_question_count: int = Field(gt=0)
    metrics: dict[str, float] = Field(min_length=1)
    scorer_output_sha256: Sha256Hex
    recorded_at: AwareDatetime


class _ScoredRun(BaseModel):
    """The three manifest fields every benchmark run shares."""

    model_config = ConfigDict(extra="ignore")

    benchmark: str
    run_id: str
    predictions_sha256: str


def score_sidecar_path(predictions_path: Path) -> Path:
    """Return the stable score path paired with a prediction artifact."""
    return predictions_path.with_suffix(predictions_path.suffix + ".score.json")


def parse_metric_assignment(assignment: str) -> tuple[str, float]:
    """Parse one `name=value` metric exactly as the official scorer printed it."""
    name, separator, value = assignment.partition("=")
    if not separator or not name.strip():
        raise ValueError(f"metric must be given as name=value: {assignment}")
    return name.strip(), float(value)


def parse_metric_assignments(assignments: Iterable[str]) -> dict[str, float]:
    """Collect `name=value` metrics, refusing a repeated name rather than keeping the last.

    One scorer can legitimately produce the same metric name twice under different protocols --
    LoCoMo-Refined's `run_eval.sh` scores `llm` under either its refined judge or the original
    LoCoMo one -- so a repeated name means two different numbers were measured. Silently keeping
    one of them would put an unrecoverable figure in the artifact that exists to be audited.
    """
    metrics: dict[str, float] = {}
    for assignment in assignments:
        name, value = parse_metric_assignment(assignment)
        if name in metrics:
            raise ValueError(f"metric {name} was given more than once; give each name once")
        metrics[name] = value
    return metrics


def build_official_score(
    *,
    manifest_path: Path,
    predictions_path: Path,
    scorer_output_path: Path,
    scorer_repository: str,
    scorer_command: str,
    judge_model: str | None,
    answer_backbone: str | None,
    scored_question_count: int,
    metrics: dict[str, float],
) -> OfficialScore:
    """Attach scorer numbers to a run, refusing predictions that run did not emit."""
    run = _ScoredRun.model_validate_json(manifest_path.read_bytes())
    predictions_sha256 = sha256_file(predictions_path)
    if predictions_sha256 != run.predictions_sha256:
        raise ValueError(
            "predictions do not match the run manifest; score the artifact the run emitted"
        )
    return OfficialScore(
        benchmark=run.benchmark,
        run_id=run.run_id,
        predictions_sha256=predictions_sha256,
        scorer_repository=scorer_repository,
        scorer_command=scorer_command,
        judge_model=judge_model,
        answer_backbone=answer_backbone,
        scored_question_count=scored_question_count,
        metrics=metrics,
        scorer_output_sha256=sha256_file(scorer_output_path),
        recorded_at=datetime.now(timezone.utc),
    )


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Record one official scorer result beside the predictions it scored."""
    parsed = _parse_arguments(argv, prog)
    score = build_official_score(
        manifest_path=parsed.manifest,
        predictions_path=parsed.predictions,
        scorer_output_path=parsed.scorer_output,
        scorer_repository=parsed.scorer_repository,
        scorer_command=parsed.scorer_command,
        judge_model=parsed.judge_model,
        answer_backbone=parsed.answer_backbone,
        scored_question_count=parsed.scored_question_count,
        metrics=parse_metric_assignments(parsed.metric),
    )
    output_path = score_sidecar_path(parsed.predictions)
    if output_path.exists() and not parsed.overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    write_text_atomically(output_path, score.model_dump_json(indent=2) + "\n")


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> argparse.Namespace:
    parser = build_parser(
        prog=prog,
        description="Bind an external official scorer's numbers to the run that produced them.",
    )
    parser.add_argument(
        "--predictions", type=Path, required=True, help="predictions file the scorer read"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="run manifest written beside those predictions"
    )
    parser.add_argument(
        "--scorer-output", type=Path, required=True, help="raw output the official scorer emitted"
    )
    parser.add_argument(
        "--scorer-repository", required=True, help="repository the official scorer came from"
    )
    parser.add_argument(
        "--scorer-command", required=True, help="exact command line that produced the output"
    )
    parser.add_argument("--judge-model", help="judge model the scorer used, when it uses one")
    parser.add_argument(
        "--answer-backbone", help="answer model the run used, when the score depends on it"
    )
    parser.add_argument(
        "--scored-question-count",
        type=int,
        required=True,
        help="questions the scorer actually scored",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        required=True,
        help="one official metric; repeatable",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing score sidecar"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
