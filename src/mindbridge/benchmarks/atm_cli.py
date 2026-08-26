"""Reproducible ATM-Bench runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from mindbridge.benchmarks.artifacts import (
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
)
from mindbridge.benchmarks.atm_bench import (
    ATM_BENCH_ADAPTER_VERSION,
    AtmBenchQuestion,
    AtmEmail,
    AtmSgmRecord,
    load_atm_bench,
    load_atm_emails,
    load_atm_sgm,
)
from mindbridge.benchmarks.atm_bench_runner import (
    AtmMediaSource,
    AtmPreparedArchive,
    AtmQuestionResult,
    answer_atm_question,
    ingest_atm_archive,
    load_prepared_atm,
    validate_prepared_atm,
)
from mindbridge.benchmarks.cli_common import (
    MediaArguments,
    MediaBenchmarkRunManifest,
    add_media_arguments,
    connected_memory,
    core_parser,
    media_arguments,
    media_manifest,
    report,
    report_unit,
    select_by_id,
    write_run_artifacts,
)
from mindbridge.benchmarks.prompts import ATM_BENCH_QUERY_PROMPT
from mindbridge.benchmarks.runtime import benchmark_tenant_id
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.core import MediaKind
from mindbridge.file_integrity import sha256_file
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT

ATM_BENCH_RUNNER_VERSION = "atm_bench_production_api_v1"
AtmSplit = Literal["main", "hard"]


class AtmRunManifest(MediaBenchmarkRunManifest):
    """Immutable source, protocol, deployment, model, and prediction identity."""

    benchmark: Literal["ATM-Bench"] = "ATM-Bench"
    split: AtmSplit
    media_source: AtmMediaSource
    dataset_repository: NonEmptyString
    evaluator_repository: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex | None = None
    emails_sha256: Sha256Hex
    sgm_image_sha256: Sha256Hex | None = None
    sgm_video_sha256: Sha256Hex | None = None
    perception_prompt_version: NonEmptyString | None = None
    query_prompt_version: NonEmptyString
    question_ids: tuple[Identifier, ...] = Field(min_length=1)
    media_item_count: int = Field(ge=0)
    email_count: int = Field(gt=0)
    ingest_failure_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    emails_path: Path
    prepared_media_path: Path | None
    sgm_image_path: Path | None
    sgm_video_path: Path | None
    split: AtmSplit
    media_source: AtmMediaSource
    question_ids: tuple[str, ...]


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Ingest the archive once, then answer the selected questions against it."""
    arguments = _parse_arguments(argv, prog)
    questions = select_by_id(
        load_atm_bench(arguments.dataset_path),
        arguments.question_ids,
        key=lambda question: question.question_id,
        label="selected ATM-Bench question IDs",
        limit=arguments.limit,
    )
    if not questions:
        raise ValueError("ATM-Bench selection must not be empty")
    prepared = (
        load_prepared_atm(arguments.prepared_media_path)
        if arguments.prepared_media_path is not None
        else None
    )
    if arguments.media_source == "raw" and prepared is None:
        raise ValueError("raw ATM-Bench runs require prepared media")
    if arguments.media_source == "sgm" and arguments.sgm_image_path is None:
        raise ValueError("sgm ATM-Bench runs require batch results")
    sgm_records = tuple(
        record
        for path in (arguments.sgm_image_path, arguments.sgm_video_path)
        if path is not None
        for record in load_atm_sgm(path)
    )
    validate_prepared_atm(questions, prepared, sgm_records, media_source=arguments.media_source)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=arguments.media_source == "raw",
    )
    emails = load_atm_emails(arguments.emails_path)
    _require_one_clock(prepared, sgm_records, media_source=arguments.media_source)
    report(f"running {len(questions)} questions", quiet=arguments.quiet)
    failures, results = asyncio.run(_run(arguments, questions, prepared, sgm_records, emails))
    _write_artifacts(
        arguments, questions, prepared, results, deployment, failures, emails, sgm_records
    )
    report(f"wrote {arguments.output_path}", quiet=arguments.quiet)


async def _run(
    arguments: _Arguments,
    questions: tuple[AtmBenchQuestion, ...],
    prepared: AtmPreparedArchive | None,
    sgm_records: tuple[AtmSgmRecord, ...],
    emails: tuple[AtmEmail, ...],
) -> tuple[int, tuple[AtmQuestionResult, ...]]:
    tenant_id = benchmark_tenant_id(arguments.tenant_prefix, "archive", arguments.run_id)
    async with connected_memory(arguments) as memory:
        failures = await ingest_atm_archive(
            memory,
            tenant_id=tenant_id,
            device_id=arguments.device_id,
            media_source=arguments.media_source,
            prepared=prepared,
            sgm_records=sgm_records,
            emails=emails,
            request_concurrency=arguments.request_concurrency,
            poll_interval_seconds=arguments.poll_interval_seconds,
            processing_timeout_seconds=arguments.processing_timeout_seconds,
        )
        results = []
        for index, question in enumerate(questions, start=1):
            report_unit(
                f"question {question.question_id}",
                index=index,
                total=len(questions),
                quiet=arguments.quiet,
            )
            results.append(
                await answer_atm_question(
                    memory,
                    question,
                    tenant_id=tenant_id,
                    recall_limit=arguments.recall_limit,
                    ingest_failure_count=failures,
                )
            )
    return failures, tuple(results)


def _write_artifacts(
    arguments: _Arguments,
    questions: tuple[AtmBenchQuestion, ...],
    prepared: AtmPreparedArchive | None,
    results: tuple[AtmQuestionResult, ...],
    deployment: LoadedDeployment,
    failures: int,
    emails: tuple[AtmEmail, ...],
    sgm_records: tuple[AtmSgmRecord, ...],
) -> None:
    if tuple(result.question_id for result in results) != tuple(
        question.question_id for question in questions
    ):
        raise ValueError("ATM-Bench predictions must match annotation question order")
    # The official evaluator reads a list of {id, question, answer, prediction} objects.
    predictions = (
        json.dumps(
            [
                {
                    "id": result.question_id,
                    "question": result.question,
                    "qtype": result.qtype,
                    "answer": result.reference_answer,
                    "prediction": result.prediction,
                    "evidence_ids": list(result.evidence_ids),
                    "retrieved_evidence_ids": list(result.mindbridge_retrieved_evidence_ids),
                    "retrieved_gold_evidence_count": result.retrieved_gold_evidence_count,
                    "mindbridge_confidence": result.mindbridge_confidence,
                    "mindbridge_memory_ids": list(result.mindbridge_memory_ids),
                    "mindbridge_trace_id": result.mindbridge_trace_id,
                    "mindbridge_failure_reason": result.mindbridge_failure_reason,
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    manifest = media_manifest(
        AtmRunManifest,
        arguments,
        deployment,
        runner_version=ATM_BENCH_RUNNER_VERSION,
        adapter_version=ATM_BENCH_ADAPTER_VERSION,
        annotation_sha256=sha256_file(arguments.dataset_path),
        predictions=predictions,
        split=arguments.split,
        media_source=arguments.media_source,
        dataset_repository="Jingbiao/ATM-Bench",
        evaluator_repository="JingbiaoMei/ATM-Bench",
        prepared_media_manifest_sha256=(
            sha256_file(arguments.prepared_media_path)
            if arguments.prepared_media_path is not None
            else None
        ),
        emails_sha256=sha256_file(arguments.emails_path),
        sgm_image_sha256=(
            sha256_file(arguments.sgm_image_path) if arguments.sgm_image_path is not None else None
        ),
        sgm_video_sha256=(
            sha256_file(arguments.sgm_video_path) if arguments.sgm_video_path is not None else None
        ),
        perception_prompt_version=(
            PERCEIVE_EVENTS_PROMPT.version if arguments.media_source == "raw" else None
        ),
        query_prompt_version=ATM_BENCH_QUERY_PROMPT.version,
        question_ids=tuple(question.question_id for question in questions),
        # The raw arm ingests `prepared.media`; the sgm arm ingests `sgm_records`. Counting
        # whichever tuple this run did not actually write would pin a number nothing here
        # produced -- e.g. 0 for a raw run staged without `--sgm-image`, when thousands of
        # images had just been ingested.
        media_item_count=(
            len(prepared.media)
            if arguments.media_source == "raw" and prepared is not None
            else len(sgm_records)
        ),
        email_count=len(emails),
        ingest_failure_count=failures,
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _require_one_clock(
    prepared: AtmPreparedArchive | None,
    sgm_records: tuple[AtmSgmRecord, ...],
    *,
    media_source: AtmMediaSource,
) -> None:
    """Refuse a run whose two arms would disagree about when the archive happened.

    The `raw` arm takes capture time from the prepared manifest and the `sgm` arm from the
    release's own record. Both are supposed to originate in the same filename stem, but
    staging is an operator step and nothing enforces that for a manifest this repository
    did not produce. This is the one place both are in hand, so the disagreement is caught
    here rather than read out of a finished score as a retrieval result.

    The same pass refuses a raw run that stages a video no record gives a duration for: the
    observation would otherwise declare a zero-length span for a clip that runs for seconds.
    """
    if prepared is None:
        return
    by_id = {record.media_id: record for record in sgm_records}
    skewed = sorted(
        item.media_id
        for item in prepared.media
        if item.media_id in by_id
        and item.media_object.created_at != by_id[item.media_id].occurred_at
    )
    if skewed:
        raise ValueError(
            "prepared ATM-Bench media disagree with the release about capture time: "
            + ", ".join(skewed)
        )
    if media_source != "raw":
        return
    undated = sorted(
        item.media_id
        for item in prepared.media
        if item.media_object.kind is MediaKind.VIDEO and item.media_id not in by_id
    )
    if undated:
        raise ValueError(
            "raw ATM-Bench runs need --sgm-video for the duration of staged videos: "
            + ", ".join(undated)
        )


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = add_media_arguments(
        core_parser(tenant_prefix="benchmark_atm", prog=prog, description=__doc__),
        device_id="atm_archive",
    )
    parser.add_argument("--emails", type=Path, required=True, help="official emails.json to ingest")
    parser.add_argument(
        "--prepared-media", type=Path, help="manifest of staged archive media; required for raw"
    )
    parser.add_argument(
        "--sgm-image", type=Path, help="official image_batch_results.json; required for sgm"
    )
    parser.add_argument("--sgm-video", type=Path, help="official video_batch_results.json")
    parser.add_argument(
        "--split",
        choices=("main", "hard"),
        required=True,
        help="official split this dataset file is, recorded in the manifest",
    )
    parser.add_argument(
        "--media-source",
        choices=("raw", "sgm"),
        default="raw",
        help="ingest the archive's own bytes, or the official schema-guided text",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="official question to run; repeatable, default the whole split",
    )
    parsed = parser.parse_args(argv)
    return media_arguments(
        _Arguments,
        parsed,
        emails_path=parsed.emails,
        prepared_media_path=parsed.prepared_media,
        sgm_image_path=parsed.sgm_image,
        sgm_video_path=parsed.sgm_video,
        split=cast(AtmSplit, parsed.split),
        media_source=cast(AtmMediaSource, parsed.media_source),
        question_ids=tuple(parsed.question_id),
    )


if __name__ == "__main__":
    main()
