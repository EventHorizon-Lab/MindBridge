"""Argument and artifact checks for the ATM-Bench CLI."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from benchmark_deployment import write_deployment_snapshot

from mindbridge.benchmarks import atm_cli
from mindbridge.benchmarks.atm_bench import AtmSgmRecord
from mindbridge.benchmarks.atm_bench_runner import AtmPreparedArchive


def _write_release(directory: Path) -> Path:
    dataset_path = directory / "atm-bench.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "question_01",
                    "question": "How much did I pay?",
                    "answer": "£799.74",
                    "notes": "",
                    "evidence_ids": ["email202411160004"],
                    "qtype": "number",
                }
            ]
        ),
        encoding="utf-8",
    )
    return dataset_path


def test_raw_run_requires_a_prepared_media_manifest(tmp_path: Path) -> None:
    dataset_path = _write_release(tmp_path)
    deployment = write_deployment_snapshot(tmp_path)
    emails = tmp_path / "emails.json"
    emails.write_text(
        json.dumps(
            [
                {
                    "id": "email202411160004",
                    "timestamp": "2024-11-16 09:12:00",
                    "short_summary": "Hotel",
                    "detail": "Total £799.74.",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw ATM-Bench runs require prepared media"):
        atm_cli.main(
            [
                "--dataset",
                str(dataset_path),
                "--emails",
                str(emails),
                "--output",
                str(tmp_path / "predictions.json"),
                "--api-base-url",
                "http://localhost:8000",
                "--deployment-config",
                str(deployment),
                "--run-id",
                "run1",
                "--split",
                "main",
                "--media-source",
                "raw",
            ],
            prog="mindbridge-bench atm",
        )


def test_sgm_run_requires_the_official_batch_results(tmp_path: Path) -> None:
    dataset_path = _write_release(tmp_path)
    deployment = write_deployment_snapshot(tmp_path, worker=False)
    emails = tmp_path / "emails.json"
    emails.write_text(
        json.dumps(
            [
                {
                    "id": "email202411160004",
                    "timestamp": "2024-11-16 09:12:00",
                    "short_summary": "Hotel",
                    "detail": "Total £799.74.",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sgm ATM-Bench runs require batch results"):
        atm_cli.main(
            [
                "--dataset",
                str(dataset_path),
                "--emails",
                str(emails),
                "--output",
                str(tmp_path / "predictions.json"),
                "--api-base-url",
                "http://localhost:8000",
                "--deployment-config",
                str(deployment),
                "--run-id",
                "run1",
                "--split",
                "main",
                "--media-source",
                "sgm",
            ],
            prog="mindbridge-bench atm",
        )


def test_cli_table_dispatches_atm() -> None:
    from mindbridge.benchmarks.cli import RUNNERS

    assert RUNNERS["atm"].module == "mindbridge.benchmarks.atm_cli"
    assert RUNNERS["atm"].extra is None


def _prepared_media(**overrides: object) -> "AtmPreparedArchive":
    from mindbridge.benchmarks.atm_bench_runner import AtmPreparedArchive, AtmPreparedMedia
    from mindbridge.contracts import MediaObjectInput
    from mindbridge.core import MediaKind

    fields: dict[str, object] = {
        "media_object_id": "20250223_130249",
        "kind": MediaKind.IMAGE,
        "uri": "s3://mindbridge-media/atm-bench/20250223_130249.jpg",
        "sha256": "a" * 64,
        "size_bytes": 1,
        "created_at": datetime(2025, 2, 23, 13, 2, 49, tzinfo=timezone.utc),
    }
    return AtmPreparedArchive(
        media=(
            AtmPreparedMedia(
                media_id=str(fields["media_object_id"]),
                media_object=MediaObjectInput.model_validate(fields | overrides),
            ),
        )
    )


def _sgm_record(**overrides: object) -> "AtmSgmRecord":
    from mindbridge.benchmarks.atm_bench import AtmSgmRecord
    from mindbridge.core import MediaKind

    fields: dict[str, object] = {
        "media_id": "20250223_130249",
        "media_kind": MediaKind.IMAGE,
        "occurred_at": datetime(2025, 2, 23, 13, 2, 49, tzinfo=timezone.utc),
        "raw_timestamp": "2025-02-23 13:02:49",
        "location_name": "Porto, Portugal",
        "city": "Porto, Portugal",
        "short_caption": "A steel bridge.",
        "caption": "A wide steel arch bridge.",
        "ocr_text": "",
        "tags": ("bridge",),
        "size_bytes": 1,
    }
    return AtmSgmRecord.model_validate(fields | overrides)


def test_a_manifest_whose_capture_time_disagrees_with_the_release_is_refused() -> None:
    skewed = _prepared_media(created_at=datetime(2025, 2, 23, 12, 2, 49, tzinfo=timezone.utc))

    with pytest.raises(ValueError, match="disagree with the release about capture time"):
        atm_cli._require_one_clock((skewed), (_sgm_record(),), media_source="raw")

    # The same manifest is fine once both sides name the same instant.
    atm_cli._require_one_clock(_prepared_media(), (_sgm_record(),), media_source="raw")


def test_a_raw_run_staging_a_video_needs_the_record_that_dates_it() -> None:
    from mindbridge.core import MediaKind

    video = _prepared_media(
        kind=MediaKind.VIDEO,
        uri="s3://mindbridge-media/atm-bench/20250223_130249.mp4",
    )

    with pytest.raises(ValueError, match="need --sgm-video"):
        atm_cli._require_one_clock(video, (), media_source="raw")

    # An sgm run never observes the bytes, so it needs no duration for them.
    atm_cli._require_one_clock(video, (), media_source="sgm")


def test_manifest_media_item_count_matches_the_arm_that_actually_ran(tmp_path: Path) -> None:
    """`media_item_count` must name what this run wrote, not a tuple it merely had in hand.

    The `raw` arm ingests `prepared.media`; the `sgm` arm ingests `sgm_records`. Two prepared
    items and one sgm record makes the two arms disagree on the count, so a regression back
    to always counting `sgm_records` is caught here rather than read out of a manifest that
    quietly claims a raw run touched one media item when it touched two.
    """
    from mindbridge.benchmarks.artifacts import load_deployment_snapshot
    from mindbridge.benchmarks.atm_bench import AtmBenchQuestion, AtmEmail
    from mindbridge.benchmarks.atm_bench_runner import (
        AtmMediaSource,
        AtmPreparedMedia,
        AtmQuestionResult,
    )
    from mindbridge.contracts import MediaObjectInput
    from mindbridge.core import MediaKind

    def _image(stem: str) -> AtmPreparedMedia:
        return AtmPreparedMedia(
            media_id=stem,
            media_object=MediaObjectInput(
                media_object_id=stem,
                kind=MediaKind.IMAGE,
                uri=f"s3://mindbridge-media/atm-bench/{stem}.jpg",
                sha256="a" * 64,
                size_bytes=1,
                created_at=datetime(2025, 2, 23, 13, 2, 49, tzinfo=timezone.utc),
            ),
        )

    prepared = AtmPreparedArchive(media=(_image("20250223_130249"), _image("20250223_140000")))
    sgm_records = (_sgm_record(),)
    question = AtmBenchQuestion(
        question_id="question_01",
        question="How much did I pay?",
        reference_answer="£799.74",
        qtype="number",
        evidence_ids=("email202411160004",),
    )
    result = AtmQuestionResult(
        question_id="question_01",
        question="How much did I pay?",
        qtype="number",
        reference_answer="£799.74",
        prediction="£799.74",
        evidence_ids=("email202411160004",),
        mindbridge_confidence=0.9,
        mindbridge_memory_ids=("memory_01",),
        mindbridge_media_object_ids=(),
        mindbridge_trace_id="trace_01",
        retrieved_gold_evidence_count=0,
    )
    email = AtmEmail(
        email_id="email202411160004",
        occurred_at=datetime(2024, 11, 16, 9, 12, tzinfo=timezone.utc),
        summary="Hotel",
        body="Total £799.74.",
    )
    dataset_path = _write_release(tmp_path)
    deployment_path = write_deployment_snapshot(tmp_path)
    deployment = load_deployment_snapshot(deployment_path, require_worker=True)
    (tmp_path / "emails.json").write_text("official-emails", encoding="utf-8")
    (tmp_path / "prepared.json").write_text("staged-archive-media", encoding="utf-8")
    (tmp_path / "sgm.json").write_text("official-sgm-records", encoding="utf-8")

    def _arguments(media_source: AtmMediaSource, output_name: str) -> atm_cli._Arguments:
        return atm_cli._Arguments(
            dataset_path=dataset_path,
            output_path=tmp_path / output_name,
            api_base_url="https://memory.example.test",
            deployment_config_path=deployment_path,
            run_id="run_01",
            tenant_prefix="benchmark_atm",
            recall_limit=20,
            request_concurrency=4,
            request_timeout_seconds=1_800.0,
            overwrite=False,
            quiet=True,
            device_id="atm_archive",
            poll_interval_seconds=1.0,
            processing_timeout_seconds=1_800.0,
            emails_path=tmp_path / "emails.json",
            prepared_media_path=tmp_path / "prepared.json",
            sgm_image_path=tmp_path / "sgm.json",
            sgm_video_path=None,
            split="main",
            media_source=media_source,
            question_ids=(),
        )

    atm_cli._write_artifacts(
        _arguments("raw", "raw-predictions.json"),
        (question,),
        prepared,
        (result,),
        deployment,
        0,
        (email,),
        sgm_records,
    )
    raw_manifest = json.loads(
        (tmp_path / "raw-predictions.json.manifest.json").read_text(encoding="utf-8")
    )
    assert raw_manifest["media_item_count"] == 2

    atm_cli._write_artifacts(
        _arguments("sgm", "sgm-predictions.json"),
        (question,),
        None,
        (result,),
        deployment,
        0,
        (email,),
        sgm_records,
    )
    sgm_manifest = json.loads(
        (tmp_path / "sgm-predictions.json.manifest.json").read_text(encoding="utf-8")
    )
    assert sgm_manifest["media_item_count"] == 1
