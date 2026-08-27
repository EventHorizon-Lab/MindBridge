"""Checks for the MM-Lifelong prepared-media producer.

The one thing worth testing here is the clock. `start_seconds` is on the split's own
`total_intervals` timeline, and every way of getting it wrong -- taking the source video's local
seconds, taking a Week day's offset from the wrong end, counting EgoLife's partial first slice --
produces a manifest that loads, validates, ingests, and localizes every answer to the wrong hour.

So the four splits get four different fixtures rather than four copies of one, each with its own
shape of clue span (`clue_intervals` bare, `clue_intervals` grouped, `clue_interval` grouped) and
its own expected segment starts, taken from offsets the released annotations really imply. A test
that built all four from one shape and asserted a field derived from the path would pass against
a producer that had only ever implemented one of them.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import pytest

from mindbridge.benchmarks import prepare_lifelong
from mindbridge.benchmarks.prepare_lifelong import global_offsets, prepare_mm_lifelong
from mindbridge.benchmarks.staging import PrepareRequest, Staging

pytest.importorskip("av", reason="prepared media is cut with the media extra's decoders")

DAY1_OFFSET = 0.0
DAY5_OFFSET = 105_826.63
"""What `week/test.json` really implies for EgoLife's fifth day.

Its first question gives `day5` seconds 13-19 as global 105839.63-105845.63, and every other
question that cites `day5` gives the same difference. Written out so this test fails if the
derivation changes, rather than agreeing with whatever it produces.
"""
MONTH_3_OFFSET = 49_713.0
MONTH_11_OFFSET = 119_832.0
MONTH_14_OFFSET = 157_278.0


class _RecordingClient:
    """An S3 double that keeps what it was asked to write."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[f"{Bucket}/{Key}"] = Body


class _Fetches:
    """What `ensure_media` was asked for, which is what a scoped run must not over-ask."""

    def __init__(self) -> None:
        self.only: list[tuple[str, ...]] = []

    def __call__(
        self, release: str, *, root: Path, only: tuple[str, ...] = (), **_: object
    ) -> Path:
        self.only.append(tuple(only))
        return root / release


@pytest.fixture
def fetches(monkeypatch: pytest.MonkeyPatch) -> _Fetches:
    """`ensure_media` reaches the Hub; the fixtures are already on disk."""
    from mindbridge.benchmarks import releases

    recorder = _Fetches()
    monkeypatch.setattr(releases, "ensure_media", recorder)
    return recorder


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> _RecordingClient:
    """Stage into a double rather than the deployment's bucket."""
    recording = _RecordingClient()
    monkeypatch.setattr(prepare_lifelong, "staging", lambda: Staging("bucket", recording))
    return recording


@pytest.mark.parametrize(
    ("split", "expected"),
    [
        (
            # One video, and clue spans already on the global clock -- so a producer that used
            # local seconds passes this split and only this split.
            "day_test",
            (
                ("video_0_00000", 0.0, 30_000),
                ("video_0_00001", 30.0, 30_000),
                ("video_0_00002", 60.0, 5_000),
            ),
        ),
        (
            # Two days of pre-cut slices, the first of which opens with a partial that is not on
            # the clock, and a second day whose offset is five days in.
            "week_test",
            (
                ("video_day1_00000", DAY1_OFFSET, 30_000),
                ("video_day1_00001", DAY1_OFFSET + 30.0, 30_000),
                ("video_day5_00000", DAY5_OFFSET, 30_000),
            ),
        ),
        (
            # `clue_interval`, singular, and one video that starts thirteen hours in.
            "month_train",
            (
                ("video_3_00000", MONTH_3_OFFSET, 30_000),
                ("video_3_00001", MONTH_3_OFFSET + 30.0, 30_000),
                ("video_3_00002", MONTH_3_OFFSET + 60.0, 5_000),
            ),
        ),
        (
            # Two videos, one question citing both, so the pairing of clue spans to global ones
            # has to be positional across the groups rather than per group.
            "month_val",
            (
                ("video_11_00000", MONTH_11_OFFSET, 30_000),
                ("video_11_00001", MONTH_11_OFFSET + 30.0, 5_000),
                ("video_14_00000", MONTH_14_OFFSET, 30_000),
                ("video_14_00001", MONTH_14_OFFSET + 30.0, 5_000),
            ),
        ),
    ],
)
def test_each_split_lands_on_its_own_global_clock(
    tmp_path: Path,
    client: _RecordingClient,
    fetches: _Fetches,
    split: str,
    expected: tuple[tuple[str, float, int], ...],
) -> None:
    """The whole point of the producer: a segment's start is where the split says it happened."""
    from mindbridge.benchmarks.mm_lifelong_runner import load_prepared_mm_lifelong

    _release(tmp_path, split)
    manifest = tmp_path / "prepared.json"

    prepare_mm_lifelong(
        PrepareRequest(argv=_argv(tmp_path, split, manifest), benchmarks_root=tmp_path, quiet=True)
    )

    prepared = load_prepared_mm_lifelong(manifest)
    assert prepared.split == split
    # Written out rather than read back from the module: a wall-clock origin is still constant
    # within one process, so only a literal catches it -- and the run manifest pins this file's
    # digest across runs, which is where that would churn.
    assert prepared.timeline_origin == datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert (
        tuple(
            (segment.segment_id, segment.start_seconds, segment.duration_ms)
            for segment in prepared.segments
        )
        == expected
    )
    assert len(client.objects) == len(expected)


def test_a_week_days_partial_opening_slice_is_not_on_the_clock(
    tmp_path: Path,
    client: _RecordingClient,
    fetches: _Fetches,
) -> None:
    """EgoLife's day 1 opens with an 18-second remnant the official offsets do not count.

    Counted, every question about that day localizes 18 seconds late and the day's own segments
    are cut from the wrong slices. This is the check that the remnant's own duration never
    reaches the timeline.
    """
    from mindbridge.benchmarks.mm_lifelong_runner import load_prepared_mm_lifelong

    _release(tmp_path, "week_test")
    manifest = tmp_path / "prepared.json"

    prepare_mm_lifelong(
        PrepareRequest(
            argv=_argv(tmp_path, "week_test", manifest), benchmarks_root=tmp_path, quiet=True
        )
    )

    day1 = [
        segment
        for segment in load_prepared_mm_lifelong(manifest).segments
        if segment.segment_id.startswith("video_day1_")
    ]
    assert [segment.duration_ms for segment in day1] == [30_000, 30_000]
    assert [segment.start_seconds for segment in day1] == [0.0, 30.0]


@pytest.mark.parametrize(("split", "expected"), [("day_test", True), ("month_train", False)])
def test_a_clip_carries_audio_exactly_when_its_source_did(
    tmp_path: Path,
    client: _RecordingClient,
    fetches: _Fetches,
    split: str,
    expected: bool,
) -> None:
    """A lifelong corpus is mostly speech, and a video-only clip deletes it from memory.

    Not hypothetical: derived clips were produced video-only here once while the source had
    audio, and every audio-dependent score went with it. Both directions are checked because
    only the pair says the probe discriminates: a check that a track is present passes against
    a producer that always writes one, and against a probe that always answers yes.
    """
    import av

    _release(tmp_path, split)

    prepare_mm_lifelong(
        PrepareRequest(
            argv=_argv(tmp_path, split, tmp_path / "prepared.json"),
            benchmarks_root=tmp_path,
            quiet=True,
        )
    )

    assert client.objects
    for content in client.objects.values():
        probe = cast(Any, av.open(io.BytesIO(content)))
        with probe:
            assert bool(probe.streams.audio) is expected


def test_a_split_whose_spans_imply_two_offsets_for_one_video_is_refused(tmp_path: Path) -> None:
    """The agreement between the two clocks is the only evidence the derivation is right."""
    _release(tmp_path, "month_val")
    dataset = tmp_path / "mm-lifelong" / "month" / "val.json"
    questions = json.loads(dataset.read_text(encoding="utf-8"))
    questions[0]["total_intervals"][0] = [999_999, 1_000_001]
    dataset.write_text(json.dumps(questions), encoding="utf-8")

    with pytest.raises(ValueError, match="more than one global offset for 11"):
        global_offsets(dataset, "month_val")


def test_a_video_id_that_would_leave_the_corpus_is_refused(tmp_path: Path) -> None:
    """Video IDs come out of release content and are joined into a path and an object key."""
    _release(tmp_path, "month_val")
    dataset = tmp_path / "mm-lifelong" / "month" / "val.json"
    questions = json.loads(dataset.read_text(encoding="utf-8"))
    questions[0]["clue_interval"][0]["video_id"] = "../../../etc"
    dataset.write_text(json.dumps(questions), encoding="utf-8")

    with pytest.raises(ValueError, match="one object-key component"):
        global_offsets(dataset, "month_val")


def test_a_limited_run_prepares_only_what_its_questions_reach(
    tmp_path: Path,
    client: _RecordingClient,
    fetches: _Fetches,
) -> None:
    """`--limit` bounds the download and the encode, not only the questions answered.

    The offsets are still derived from the whole split -- a video the selected questions do not
    cite still fixes where the ones they do cite sit -- but nothing else about it is paid for.
    """
    from mindbridge.benchmarks.mm_lifelong_runner import load_prepared_mm_lifelong

    _release(tmp_path, "month_val")
    manifest = tmp_path / "prepared.json"

    prepare_mm_lifelong(
        PrepareRequest(
            argv=(*_argv(tmp_path, "month_val", manifest), "--limit", "1"),
            benchmarks_root=tmp_path,
            quiet=True,
        )
    )

    prepared = load_prepared_mm_lifelong(manifest)
    # Question 0 cites video 11 seconds 10-12 only; question 1, which cites video 14, is not
    # selected. One segment covers that span, and video 14 is neither fetched nor cut.
    assert [segment.segment_id for segment in prepared.segments] == ["video_11_00000"]
    assert len(client.objects) == 1


def test_only_the_videos_a_run_reads_are_fetched(
    tmp_path: Path,
    client: _RecordingClient,
    fetches: _Fetches,
) -> None:
    """A scoped run must not pull the release's 189 GB to answer one question.

    Fetching is asked for on absence, so this is checked by taking the media away: the video the
    selected question cites is asked for by name, and the one only an unselected question cites
    is not asked for at all even though it is missing too.
    """
    _release(tmp_path, "month_val")
    for video in ("11", "14"):
        (tmp_path / "mm-lifelong" / "videos" / "month" / f"{video}.mp4").unlink()

    with pytest.raises(FileNotFoundError, match=r"month/11\.mp4"):
        prepare_mm_lifelong(
            PrepareRequest(
                argv=(*_argv(tmp_path, "month_val", tmp_path / "prepared.json"), "--limit", "1"),
                benchmarks_root=tmp_path,
                quiet=True,
            )
        )
    assert fetches.only == [("videos/month/11.mp4",)]
    assert client.objects == {}


def test_media_already_on_disk_is_not_fetched_at_all(
    tmp_path: Path,
    client: _RecordingClient,
    fetches: _Fetches,
) -> None:
    """`ensure_media` refuses an unobtainable set before it looks at the disk.

    So an operator who placed a release by hand has to be able to prepare from it without the
    fetch they do not need being consulted.
    """
    _release(tmp_path, "month_train")

    prepare_mm_lifelong(
        PrepareRequest(
            argv=_argv(tmp_path, "month_train", tmp_path / "prepared.json"),
            benchmarks_root=tmp_path,
            quiet=True,
        )
    )

    assert fetches.only == []


def test_a_run_whose_questions_reach_past_the_media_is_refused_before_cutting(
    tmp_path: Path,
    client: _RecordingClient,
    fetches: _Fetches,
) -> None:
    """`run_mm_lifelong` refuses a timeline shorter than its questions, after the whole encode."""
    _release(tmp_path, "month_train")
    dataset = tmp_path / "mm-lifelong" / "month" / "train.json"
    questions = json.loads(dataset.read_text(encoding="utf-8"))
    questions[0]["clue_interval"][0]["intervals"] = [[10, 900]]
    questions[0]["total_intervals"] = [[MONTH_3_OFFSET + 10, MONTH_3_OFFSET + 900]]
    dataset.write_text(json.dumps(questions), encoding="utf-8")

    with pytest.raises(ValueError, match="shorter than its own annotations"):
        prepare_mm_lifelong(
            PrepareRequest(
                argv=_argv(tmp_path, "month_train", tmp_path / "prepared.json"),
                benchmarks_root=tmp_path,
                quiet=True,
            )
        )
    assert client.objects == {}


def test_two_preparations_of_one_split_write_the_same_bytes(
    tmp_path: Path,
    client: _RecordingClient,
    fetches: _Fetches,
) -> None:
    """The run manifest pins this file's digest, so a wall clock anywhere in it would churn."""
    _release(tmp_path, "week_test")

    first, second = tmp_path / "one.json", tmp_path / "two.json"
    for manifest in (first, second):
        prepare_mm_lifelong(
            PrepareRequest(
                argv=_argv(tmp_path, "week_test", manifest), benchmarks_root=tmp_path, quiet=True
            )
        )

    assert first.read_bytes() == second.read_bytes()


_DATASETS = {
    "day_test": "day/test.json",
    "week_test": "week/test.json",
    "month_train": "month/train.json",
    "month_val": "month/val.json",
}


def _argv(root: Path, split: str, manifest: Path) -> tuple[str, ...]:
    return (
        "--dataset",
        str(root / "mm-lifelong" / _DATASETS[split]),
        "--prepared-media",
        str(manifest),
        "--split",
        split,
        "--output",
        str(root / "out.jsonl"),
        "--api-base-url",
        "http://localhost:8000",
        "--deployment-config",
        str(root / "deployment.json"),
        "--run-id",
        "prep-01",
    )


def _release(root: Path, split: str) -> None:
    """Write one split's annotations and the media it names, in the release's own layout."""
    dataset = root / "mm-lifelong" / _DATASETS[split]
    dataset.parent.mkdir(parents=True, exist_ok=True)
    videos = root / "mm-lifelong" / "videos"
    if split == "day_test":
        # The only fixture with an audio track: it is the one the audio check reads, and every
        # other source is silent so the suite is not four minutes of AAC.
        _video(videos / "day" / "0.mp4", seconds=65, audio=True)
    elif split == "week_test":
        _video(videos / "week" / "day1" / "DAY1_A1_JAKE_11094208.mp4", seconds=18)
        _video(videos / "week" / "day1" / "DAY1_A1_JAKE_11100000.mp4", seconds=30)
        _video(videos / "week" / "day1" / "DAY1_A1_JAKE_11103000.mp4", seconds=30)
        _video(videos / "week" / "day5" / "DAY5_A1_JAKE_11100000.mp4", seconds=30)
    elif split == "month_train":
        _video(videos / "month" / "3.mp4", seconds=65)
    else:
        _video(videos / "month" / "11.mp4", seconds=35)
        _video(videos / "month" / "14.mp4", seconds=35)
    dataset.write_text(json.dumps(_questions(split)), encoding="utf-8")


def _questions(split: str) -> list[dict[str, Any]]:
    """Each split's own annotation shape, which is three different shapes across the four."""
    if split == "day_test":
        return [
            _question(0, "Counting", clue_intervals=[[10, 12]], total=[[10, 12]]),
            _question(1, "Temporal Reasoning", clue_intervals=[[40, 42]], total=[[40, 42]]),
        ]
    if split == "week_test":
        return [
            _question(
                0,
                "Counting",
                clue_intervals=[{"video_id": "day5", "intervals": [[13, 19]]}],
                total=[[DAY5_OFFSET + 13, DAY5_OFFSET + 19]],
            ),
            _question(
                1,
                "Event Recall",
                clue_intervals=[{"video_id": "day1", "intervals": [[5, 9]]}],
                total=[[DAY1_OFFSET + 5, DAY1_OFFSET + 9]],
            ),
        ]
    if split == "month_train":
        return [
            _question(
                6,
                "Hallucination Detection",
                clue_interval=[{"video_id": "3", "intervals": [[10, 12]]}],
                total=[[MONTH_3_OFFSET + 10, MONTH_3_OFFSET + 12]],
            )
        ]
    return [
        _question(
            0,
            "Temporal Reasoning",
            clue_interval=[{"video_id": "11", "intervals": [[10, 12]]}],
            total=[[MONTH_11_OFFSET + 10, MONTH_11_OFFSET + 12]],
        ),
        _question(
            1,
            "Counting",
            clue_interval=[
                {"video_id": "11", "intervals": [[20, 25]]},
                {"video_id": "14", "intervals": [[6, 9]]},
            ],
            total=[
                [MONTH_11_OFFSET + 20, MONTH_11_OFFSET + 25],
                [MONTH_14_OFFSET + 6, MONTH_14_OFFSET + 9],
            ],
        ),
    ]


def _question(
    index: int,
    question_type: str,
    *,
    total: list[list[float]],
    **clue: object,
) -> dict[str, Any]:
    return {
        "index": index,
        "question": f"What happened during moment {index}?",
        "answer": "something",
        "question_type": question_type,
        "temporal_certificate": "Short",
        "total_intervals": total,
        **clue,
    }


def _video(path: Path, *, seconds: int, audio: bool = False, rate: int = 5) -> Path:
    """Encode a source of a known length, so a segment's declared start is checkable."""
    import av
    import numpy

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        video = container.add_stream("libx264", rate=rate)
        video.width, video.height, video.pix_fmt = 128, 96, "yuv420p"
        video.thread_count, video.thread_type = 1, "NONE"
        sound = container.add_stream("aac", rate=16_000) if audio else None
        if sound is not None:
            sound.layout = "mono"
        for index in range(rate * seconds):
            array = numpy.full((96, 128, 3), index % 255, dtype=numpy.uint8)
            container.mux(video.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(video.encode())
        if sound is not None:
            _mux_tone(container, sound, seconds=seconds)
    return path


def _mux_tone(container: Any, sound: Any, *, seconds: int) -> None:  # noqa: ANN401
    import av
    import numpy

    samples = numpy.sin(numpy.arange(16_000 * seconds) * 0.05) * 8_000
    frame = av.AudioFrame.from_ndarray(
        samples.astype(numpy.int16).reshape(1, -1), format="s16", layout="mono"
    )
    frame.sample_rate = 16_000
    frame.pts = 0
    frame.time_base = Fraction(1, 16_000)
    for packet in sound.encode(frame):
        container.mux(packet)
    container.mux(sound.encode())
