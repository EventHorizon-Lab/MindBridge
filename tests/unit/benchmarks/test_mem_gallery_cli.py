"""Argument and artifact checks for the Mem-Gallery CLI."""

import json
from pathlib import Path

import pytest
from benchmark_deployment import write_deployment_snapshot

from mindbridge.benchmarks import mem_gallery_cli


def _write_dialog_directory(root: Path) -> Path:
    directory = root / "dialog"
    directory.mkdir()
    (directory / "Baking.json").write_text(
        json.dumps(
            {
                "character_profile": {
                    "name": "Maya",
                    "persona_summary": "A librarian who bakes.",
                    "traits": ["curious"],
                    "conversation_style": "Earnest.",
                },
                "multi_session_dialogues": [
                    {
                        "session_id": "D1",
                        "date": "2024-06-24",
                        "dialogues": [
                            {
                                "round": "D1:1",
                                "user": "Basics?",
                                "assistant": "A 30 litre oven.",
                            }
                        ],
                    }
                ],
                "human-annotated QAs": [
                    {
                        "point": "FR",
                        "question": "What oven size?",
                        "answer": "30 litres.",
                        "session_id": ["D1"],
                        "clue": ["D1:1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_run_requires_a_prepared_images_manifest(tmp_path: Path) -> None:
    directory = _write_dialog_directory(tmp_path)
    deployment = write_deployment_snapshot(tmp_path)

    with pytest.raises(SystemExit):
        mem_gallery_cli.main(
            [
                "--dataset",
                str(directory),
                "--output",
                str(tmp_path / "predictions.json"),
                "--api-base-url",
                "http://localhost:8000",
                "--deployment-config",
                str(deployment),
                "--run-id",
                "run1",
            ],
            prog="mindbridge-bench mem-gallery",
        )


def test_unknown_topic_selection_is_refused(tmp_path: Path) -> None:
    directory = _write_dialog_directory(tmp_path)
    deployment = write_deployment_snapshot(tmp_path)
    prepared = tmp_path / "prepared.json"
    prepared.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_key": "../image/Baking/D1_IMG_001.jpg",
                        "media_object": {
                            "media_object_id": "D1:IMG_001",
                            "kind": "image",
                            "uri": "s3://mindbridge-media/mem-gallery/D1_IMG_001.jpg",
                            "sha256": "b" * 64,
                            "size_bytes": 1,
                            "created_at": "2024-06-24T00:00:00Z",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown selected Mem-Gallery topics"):
        mem_gallery_cli.main(
            [
                "--dataset",
                str(directory),
                "--prepared-images",
                str(prepared),
                "--output",
                str(tmp_path / "predictions.json"),
                "--api-base-url",
                "http://localhost:8000",
                "--deployment-config",
                str(deployment),
                "--run-id",
                "run1",
                "--topic",
                "Nope",
            ],
            prog="mindbridge-bench mem-gallery",
        )


def test_cli_table_dispatches_mem_gallery() -> None:
    from mindbridge.benchmarks.cli import RUNNERS

    assert RUNNERS["mem-gallery"].module == "mindbridge.benchmarks.mem_gallery_cli"
    assert RUNNERS["mem-gallery"].extra is None


def _write_dialog_directory_with_photo(root: Path) -> Path:
    """Like `_write_dialog_directory`, but with one round image that a prepared-images
    manifest can be missing.

    The image-free fixture above cannot exercise `validate_mem_gallery_images`: a topic
    with no image reference has nothing for a prepared-images manifest to be missing.
    """
    directory = root / "dialog"
    directory.mkdir()
    (directory / "Baking.json").write_text(
        json.dumps(
            {
                "character_profile": {
                    "name": "Maya",
                    "persona_summary": "A librarian who bakes.",
                    "traits": ["curious"],
                    "conversation_style": "Earnest.",
                },
                "multi_session_dialogues": [
                    {
                        "session_id": "D1",
                        "date": "2024-06-24",
                        "dialogues": [
                            {
                                "round": "D1:1",
                                "user": "What does it look like?",
                                "assistant": "Golden brown.",
                                "image_id": ["D1:IMG_001"],
                                "input_image": ["Baking/D1_IMG_001.jpg"],
                                "image_caption": ["A golden loaf."],
                            }
                        ],
                    }
                ],
                "human-annotated QAs": [
                    {
                        "point": "FR",
                        "question": "What oven size?",
                        "answer": "30 litres.",
                        "session_id": ["D1"],
                        "clue": ["D1:1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_main_validates_images_before_running_any_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`validate_mem_gallery_images` must run inside `main()` before `_run` ever starts.

    `run_mem_gallery_topic` (Task 7) re-validates images per topic on its own, so a
    single-topic run would raise the very same ValueError even with `main()`'s pre-check
    deleted -- just one call deeper, after a live connection had already been opened and
    ingestion for this topic had already begun. Failing the test if `_run` is ever invoked
    is what makes that missing call site visible here, rather than only in the direct
    tests Task 7 already has for `validate_mem_gallery_images` itself.
    """
    directory = _write_dialog_directory_with_photo(tmp_path)
    deployment = write_deployment_snapshot(tmp_path)
    prepared = tmp_path / "prepared.json"
    prepared.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_key": "Baking/UNRELATED.jpg",
                        "media_object": {
                            "media_object_id": "D1:IMG_999",
                            "kind": "image",
                            "uri": "s3://mindbridge-media/mem-gallery/UNRELATED.jpg",
                            "sha256": "c" * 64,
                            "size_bytes": 1,
                            "created_at": "2024-06-24T00:00:00Z",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    async def fail_if_run(*_args: object) -> None:
        raise AssertionError("main() started a run before validating prepared images")

    monkeypatch.setattr(mem_gallery_cli, "_run", fail_if_run)

    with pytest.raises(ValueError, match="missing prepared Mem-Gallery images"):
        mem_gallery_cli.main(
            [
                "--dataset",
                str(directory),
                "--prepared-images",
                str(prepared),
                "--output",
                str(tmp_path / "predictions.json"),
                "--api-base-url",
                "http://localhost:8000",
                "--deployment-config",
                str(deployment),
                "--run-id",
                "run1",
            ],
            prog="mindbridge-bench mem-gallery",
        )


def test_manifest_counts_match_the_topics_this_run_actually_answered(tmp_path: Path) -> None:
    """Every manifest count must total the *selected* topics, not the whole release.

    Two topics with deliberately different session/round/image shapes make a bug that
    always reports one topic's counts, or the release's full count regardless of
    selection, visible: the totals asserted below only come out right if `_write_artifacts`
    sums over the `topics` tuple it was actually given for this run.
    """
    from mindbridge.benchmarks.artifacts import load_deployment_snapshot
    from mindbridge.benchmarks.mem_gallery import load_mem_gallery
    from mindbridge.benchmarks.mem_gallery_runner import MemGalleryQuestionResult

    directory = tmp_path / "dialog"
    directory.mkdir()
    (directory / "TopicA.json").write_text(
        json.dumps(
            {
                "character_profile": {
                    "name": "Ana",
                    "persona_summary": "An engineer.",
                    "traits": [],
                    "conversation_style": "Direct.",
                },
                "multi_session_dialogues": [
                    {
                        "session_id": "S1",
                        "date": "2024-01-01",
                        "dialogues": [
                            {"round": "S1:1", "user": "u1", "assistant": "a1"},
                            {
                                "round": "S1:2",
                                "user": "u2",
                                "assistant": "a2",
                                "image_id": ["TopicA:IMG_1"],
                                "input_image": ["TopicA/IMG_1.jpg"],
                                "image_caption": ["A photo."],
                            },
                        ],
                    },
                    {
                        "session_id": "S2",
                        "date": "2024-01-02",
                        "dialogues": [{"round": "S2:1", "user": "u3", "assistant": "a3"}],
                    },
                ],
                "human-annotated QAs": [
                    {
                        "point": "FR",
                        "question": "q1?",
                        "answer": "ans1",
                        "session_id": ["S1"],
                        "clue": ["S1:1"],
                        "question_image": "TopicA/Q_IMG.jpg",
                        "image_caption": "A question photo.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (directory / "TopicB.json").write_text(
        json.dumps(
            {
                "character_profile": {
                    "name": "Ben",
                    "persona_summary": "A gardener.",
                    "traits": [],
                    "conversation_style": "Calm.",
                },
                "multi_session_dialogues": [
                    {
                        "session_id": "S1",
                        "date": "2024-02-01",
                        "dialogues": [{"round": "S1:1", "user": "u1", "assistant": "a1"}],
                    }
                ],
                "human-annotated QAs": [
                    {
                        "point": "MR",
                        "question": "q2?",
                        "answer": "ans2",
                        "session_id": ["S1"],
                        "clue": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    topic_a, topic_b = load_mem_gallery(directory)

    result_a = MemGalleryQuestionResult(
        question_id="TopicA:1",
        topic="TopicA",
        point="FR",
        question="q1?",
        reference_answer="ans1",
        prediction="ans1",
        clue_round_ids=("S1:1",),
        mindbridge_confidence=0.9,
        mindbridge_memory_ids=("memory_01",),
        mindbridge_round_ids=("S1:1",),
        mindbridge_media_object_ids=(),
        mindbridge_trace_id="trace_01",
        retrieved_clue_round_count=1,
    )
    result_b = MemGalleryQuestionResult(
        question_id="TopicB:1",
        topic="TopicB",
        point="MR",
        question="q2?",
        reference_answer="ans2",
        prediction="ans2",
        clue_round_ids=(),
        mindbridge_confidence=0.5,
        mindbridge_memory_ids=(),
        mindbridge_round_ids=(),
        mindbridge_media_object_ids=(),
        mindbridge_trace_id="trace_02",
        retrieved_clue_round_count=0,
    )

    deployment_path = write_deployment_snapshot(tmp_path)
    deployment = load_deployment_snapshot(deployment_path, require_worker=True)
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("staged-images", encoding="utf-8")

    def _arguments(output_name: str) -> mem_gallery_cli._Arguments:
        return mem_gallery_cli._Arguments(
            dataset_path=directory,
            output_path=tmp_path / output_name,
            api_base_url="https://memory.example.test",
            deployment_config_path=deployment_path,
            run_id="run_01",
            tenant_prefix="benchmark_mem_gallery",
            recall_limit=20,
            request_concurrency=4,
            request_timeout_seconds=1_800.0,
            overwrite=False,
            quiet=True,
            device_id="mem_gallery_conversation",
            poll_interval_seconds=1.0,
            processing_timeout_seconds=1_800.0,
            prepared_images_path=prepared_path,
            topics=(),
        )

    mem_gallery_cli._write_artifacts(
        _arguments("both-predictions.json"),
        (topic_a, topic_b),
        (result_a, result_b),
        deployment,
    )
    both_manifest = json.loads(
        (tmp_path / "both-predictions.json.manifest.json").read_text(encoding="utf-8")
    )
    assert both_manifest["session_count"] == 3
    assert both_manifest["round_count"] == 4
    assert both_manifest["image_reference_count"] == 1
    assert both_manifest["question_image_count"] == 1

    mem_gallery_cli._write_artifacts(
        _arguments("one-predictions.json"),
        (topic_a,),
        (result_a,),
        deployment,
    )
    one_manifest = json.loads(
        (tmp_path / "one-predictions.json.manifest.json").read_text(encoding="utf-8")
    )
    assert one_manifest["session_count"] == 2
    assert one_manifest["round_count"] == 3
    assert one_manifest["image_reference_count"] == 1
    assert one_manifest["question_image_count"] == 1
