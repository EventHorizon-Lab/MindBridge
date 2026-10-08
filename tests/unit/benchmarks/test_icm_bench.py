from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from mindbridge.benchmarks._official.icm_prompt import SEMANTIC_EQUIVALENCE_PROMPT
from mindbridge.benchmarks.download import _extract_tar, acquire_media
from mindbridge.benchmarks.eval import _prefix_end
from mindbridge.benchmarks.eval_adapters import load_task
from mindbridge.benchmarks.icm_bench import load_icm_bench
from mindbridge.benchmarks.official_scorers import JudgeMessage, judge_plan, parse_judge_response
from mindbridge.benchmarks.task_catalog import TASKS


def _question(**overrides: object) -> dict[str, object]:
    return {
        "question_id": "q1",
        "question": "Who picked up the cup?",
        "reference_answer": "SECRET ANSWER",
        "category": "Identity Recall",
        "target_character_ids": ["SECRET ID"],
        "evidence_video_ids": ["clip_001"],
        "before_clip": "clip_001",
        **overrides,
    }


def _fixture(root: Path) -> Path:
    spec = TASKS["icm-bench"]
    path = spec.dataset_path(root)
    path.parent.mkdir(parents=True)
    rows = [
        _question(),
        _question(
            question_id="profile", category="Long-Term Identity Profile Inference", before_clip=None
        ),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    media = root / "icm-bench/videos"
    media.mkdir()
    clips = [f"clip_{i:03d}" for i in range(839)]
    (media / "metadata.jsonl").write_text("\n".join(json.dumps({"video_id": c}) for c in clips))
    for clip in clips:
        (media / f"{clip}.mp4").write_bytes(b"fixture video")
    transcripts = root / "icm-bench/resources/asr_transcripts"
    transcripts.mkdir(parents=True)
    (transcripts / "clip_001.srt").write_text("Observed speech, no speaker label")
    (root / "icm-bench/characters.json").write_text("FORBIDDEN CHARACTERS")
    return path


def test_icm_causal_prefix_calibration_and_no_label_leakage(tmp_path: Path) -> None:
    _fixture(tmp_path)
    unit = load_task(TASKS["icm-bench"], root=tmp_path, verify_digest=False).units[0]
    assert unit.memories[0].source_id == "clip_000"
    assert _prefix_end(unit.memories, unit.questions[0].cutoff_seconds) == 2
    assert _prefix_end(unit.memories, unit.questions[1].cutoff_seconds) == 839
    inputs = repr([m.content for m in unit.memories]) + repr([q.content for q in unit.questions])
    assert "Observed speech" in inputs
    assert "SECRET" not in inputs and "FORBIDDEN" not in inputs
    assert unit.questions[0].metadata["evidence_ids"] == ("clip_001",)


@pytest.mark.parametrize(
    "override",
    [
        {"before_clip": "clip_000"},
        {"before_clip": "../clip_002"},
        {"before_clip": None},
        {"evidence_video_ids": []},
    ],
)
def test_icm_rejects_invalid_temporal_annotations(
    tmp_path: Path, override: dict[str, object]
) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text(json.dumps(_question(**override)))
    with pytest.raises(ValueError):
        load_icm_bench(path)


def test_icm_scorer_uses_exact_yes_verdict() -> None:
    plan = judge_plan(
        "icm-bench", question="Who?", references=("Li",), prediction="Li.", metadata={}
    )
    assert plan is not None
    assert plan.calls == (
        (
            JudgeMessage(
                "user",
                SEMANTIC_EQUIVALENCE_PROMPT.format(
                    question="Who?", ground_truth_answer="Li", agent_answer="Li."
                ),
            ),
        ),
    )
    assert plan.protocol == "icm_semantic_equivalence_10f02babe3c7_user_only_v2"
    assert plan.max_tokens == 8192
    assert parse_judge_response(plan, " Yes. ") == {"accuracy": 1.0}
    assert parse_judge_response(plan, "not yes") == {"accuracy": 0.0}


@pytest.mark.parametrize("name,link", [("../outside", False), ("videos/link", True)])
def test_tar_rejects_unsafe_members_before_extracting(
    tmp_path: Path, name: str, link: bool
) -> None:
    archive = tmp_path / "videos.tar"
    with tarfile.open(archive, "w") as volume:
        safe = tarfile.TarInfo("videos/safe.mp4")
        safe.size = 4
        volume.addfile(safe, io.BytesIO(b"safe"))
        bad = tarfile.TarInfo(name)
        if link:
            bad.type = tarfile.SYMTYPE
            bad.linkname = "../../outside"
        volume.addfile(bad)
    with pytest.raises(ValueError, match="unsafe"):
        _extract_tar(archive, announce=None)
    assert not (tmp_path / "videos/safe.mp4").exists()


def test_tar_extraction_is_resumable(tmp_path: Path) -> None:
    archive = tmp_path / "videos.tar"
    with tarfile.open(archive, "w") as volume:
        item = tarfile.TarInfo("videos/clip_000.mp4")
        item.size = 4
        volume.addfile(item, io.BytesIO(b"data"))
    _extract_tar(archive, announce=None)
    _extract_tar(archive, announce=None)
    assert (tmp_path / "videos/clip_000.mp4").read_bytes() == b"data"


def test_icm_offline_media_does_not_require_retaining_tar(tmp_path: Path) -> None:
    _fixture(tmp_path)
    release = tmp_path / "icm-bench"
    archive = release / "videos.tar"
    with tarfile.open(archive, "w") as volume:
        item = tarfile.TarInfo("videos/clip_000.mp4")
        item.size = 4
        volume.addfile(item, io.BytesIO(b"data"))
    assert acquire_media(TASKS["icm-bench"], tmp_path, download=False) == release
    archive.unlink()
    assert acquire_media(TASKS["icm-bench"], tmp_path, download=False) == release
    unit = load_task(TASKS["icm-bench"], root=tmp_path, verify_digest=False).units[0]
    assert len(unit.memories) == 839


@pytest.mark.parametrize("mutation", ["truncated", "gap", "duplicate", "extra", "calibration"])
def test_icm_rejects_incomplete_or_invalid_timelines(tmp_path: Path, mutation: str) -> None:
    _fixture(tmp_path)
    metadata = tmp_path / "icm-bench/videos/metadata.jsonl"
    rows = metadata.read_text().splitlines()
    if mutation == "truncated":
        rows.pop()
    elif mutation == "gap":
        rows.pop(400)
    elif mutation == "duplicate":
        rows.append(rows[-1])
    elif mutation == "extra":
        rows.append(json.dumps({"video_id": "clip_839"}))
    else:
        rows.pop(0)
    metadata.write_text("\n".join(rows))
    with pytest.raises(ValueError, match="exactly clip_000 through clip_838"):
        load_task(TASKS["icm-bench"], root=tmp_path, verify_digest=False, limit=1)


def test_icm_offline_media_still_requires_non_archive_inputs(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "icm-bench/videos/metadata.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match=r"metadata\.jsonl"):
        acquire_media(
            TASKS["icm-bench"],
            tmp_path,
            patterns=("videos.tar", "videos/metadata.jsonl"),
            download=False,
        )


def test_icm_download_still_requires_requested_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path)

    def empty_download(*args: object) -> None:
        pass

    monkeypatch.setattr("mindbridge.benchmarks.download._snapshot", empty_download)
    with pytest.raises(FileNotFoundError, match=r"download did not produce: videos\.tar"):
        acquire_media(TASKS["icm-bench"], tmp_path)
