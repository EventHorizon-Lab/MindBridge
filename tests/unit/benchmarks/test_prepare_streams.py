"""Checks for the producers whose media sits on an absolute clock rather than a video-local one.

Staging reaches an object store, fetching reaches the Hub, and cutting reaches a video encoder,
so all three are exercised against doubles or a synthetic release rather than the real corpus.
What is checked is what the two shapes have to satisfy and what no other producer has to get
right: that a clip lands at the instant its own file name claims on EgoLife's seven-day clock,
that nothing is prepared past the last question a run may causally reach, and that every selected
SuperMemory question ends exactly on a prepared segment boundary -- the check `run_supermemory_vqa`
makes before it ingests anything.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from mindbridge.benchmarks.prepare_streams import (
    prepare_egolife,
    prepare_egomem,
    prepare_supermemory,
)
from mindbridge.benchmarks.staging import STAGED_AT, PrepareRequest, Staging
from mindbridge.benchmarks.supermemory_runner import (
    SuperMemoryPreparedSubject,
    SuperMemoryPreparedVideo,
)
from mindbridge.benchmarks.supermemory_vqa import SuperMemoryQuestion

pytest.importorskip("av", reason="prepared media is cut with the media extra's decoders")

SUPERMEMORY_EPOCH = 1_773_000_000
"""An arbitrary Unix second one synthetic recording starts at; the release supplies real ones."""


class _Recorder:
    """Everything a producer asked the world for: what it uploaded, and what it asked to fetch."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fetched: list[tuple[str, tuple[str, ...]]] = []
        self.may_download: list[bool] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[f"{Bucket}/{Key}"] = Body

    def ensure_media(
        self,
        release: str,
        *,
        root: Path,
        only: tuple[str, ...] = (),
        announce: object = None,
        download: bool = True,
    ) -> Path:
        self.fetched.append((release, tuple(only)))
        self.may_download.append(download)
        return root / release


@pytest.fixture
def staged(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Point every producer at a bucket double, and stop `ensure_media` reaching the Hub.

    The fetch is recorded rather than absent: these tests write the release onto disk themselves,
    so what is left to check about it is which files a producer asked for -- narrowing that is
    the whole download bound, and a fetch that is merely wasteful leaves the manifest correct.
    """
    from mindbridge.benchmarks import prepare_streams

    recorder = _Recorder()
    monkeypatch.setattr(prepare_streams, "staging", lambda: Staging("bucket", recorder))
    monkeypatch.setattr(prepare_streams, "ensure_media", recorder.ensure_media)
    return recorder


def test_an_egolife_clip_lands_where_its_own_file_name_says_on_the_seven_day_clock(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """`DAY2` at `11100010` is 126,600,500 ms into the release, and that is derived by hand.

    `(2 - 1) * 86400` seconds for the day, plus `11 * 3600 + 10 * 60` for the timecode, is
    126,600 seconds; the trailing `10` is ten frames at the release's 20 FPS, which is 500 ms.
    Asserted through `egolife_runner._clip_start_ms`, which is what orders and withholds clips,
    so a producer that put the day in the wrong field or read the timecode off the wrong part of
    the file name lands somewhere else on the same clock and this says so.
    """
    from mindbridge.benchmarks.egolife_runner import _clip_start_ms, load_prepared_egolife

    _egolife_release(tmp_path, "A1_JAKE", day=2, timecodes=("11100010",), seconds=2)
    dataset = _egolife_questions(tmp_path, [("DAY2", "11100500")])
    manifest = tmp_path / "prepared.json"

    prepare_egolife(_request(_egolife_argv(dataset, manifest, tmp_path), tmp_path))

    stream = load_prepared_egolife(manifest)
    assert stream.subject_id == "A1_JAKE"
    assert [(clip.day, clip.start_timecode) for clip in stream.clips] == [(2, "11100010")]
    assert _clip_start_ms(stream.clips[0]) == 126_600_500


def test_an_egolife_clip_is_declared_no_longer_than_the_gap_to_the_one_after_it(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """The released containers run past the grid their names sit on, and overlap is fatal.

    A clip named `11100000` followed by one a second later decodes to more than that second,
    because the last frame's own duration is inside the container. Declaring the measured length
    makes the two overlap, which `EgoLifePreparedStream` refuses outright, so the whole manifest
    is lost. Here the second file is two seconds of video one second after the first.
    """
    from mindbridge.benchmarks.egolife_runner import load_prepared_egolife

    _egolife_release(tmp_path, "A1_JAKE", day=1, timecodes=("11100000", "11100100"), seconds=2)
    dataset = _egolife_questions(tmp_path, [("DAY1", "11101000")])
    manifest = tmp_path / "prepared.json"

    prepare_egolife(_request(_egolife_argv(dataset, manifest, tmp_path), tmp_path))

    clips = load_prepared_egolife(manifest).clips
    assert [clip.start_timecode for clip in clips] == ["11100000", "11100100"]
    assert [clip.media_object.duration_ms for clip in clips if clip.media_object] == [1_000, 2_000]


def test_egolife_prepares_only_the_clips_a_selected_question_can_causally_reach(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """A clip crossing the query time is withheld and one after it is never read at all.

    That bound is also the download bound, which is what makes `--limit` affordable against a
    103 GB wearer. The question here is asked at `11:10:03`, so the two-second clip starting at
    `11:10:02` crosses it and the one at `11:10:04` is past it; only the first may be staged.
    """
    from mindbridge.benchmarks.egolife_runner import load_prepared_egolife

    _egolife_release(
        tmp_path, "A1_JAKE", day=1, timecodes=("11100000", "11100200", "11100400"), seconds=2
    )
    dataset = _egolife_questions(tmp_path, [("DAY1", "11100300"), ("DAY1", "11101000")])
    manifest = tmp_path / "prepared.json"

    prepare_egolife(
        _request((*_egolife_argv(dataset, manifest, tmp_path), "--limit", "1"), tmp_path)
    )

    assert [clip.start_timecode for clip in load_prepared_egolife(manifest).clips] == ["11100000"]
    assert [key.rsplit("/", 1)[-1] for key in staged.objects] == ["11100000.mp4"]
    assert staged.fetched == [("egolife", ("A1_JAKE/DAY1/*.mp4",))]


def test_two_preparations_of_one_egolife_stream_produce_the_same_manifest(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """A run manifest pins this file's digest, so a wall clock anywhere in it would churn."""
    _egolife_release(tmp_path, "A1_JAKE", day=1, timecodes=("11100000", "11100200"), seconds=2)
    dataset = _egolife_questions(tmp_path, [("DAY1", "11101000")])
    first, second = tmp_path / "one.json", tmp_path / "two.json"

    prepare_egolife(_request(_egolife_argv(dataset, first, tmp_path), tmp_path))
    prepare_egolife(_request(_egolife_argv(dataset, second, tmp_path), tmp_path))

    assert first.read_bytes() == second.read_bytes()
    assert b"2000-01-01T00:00:00Z" in first.read_bytes()


def test_egomem_bounds_each_identity_by_its_own_last_question_not_by_the_latest_one(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """`run_egomem_reason` replays one wearer at a time and refuses another's stream.

    So a wearer whose last question is early gains nothing from a later wearer's: clips past its
    own query time are uploads no run ingests. The later wearer is asked first on purpose, and
    that ordering buys exactly one bug shape: an implementation seeding each identity from the
    running maximum instead of from its own questions. Asked second, that leak returns the right
    answer and nothing notices. One horizon shared by everyone is caught in either order --
    measured rather than assumed, because an early sweep reported this as a survivor for an
    unrelated reason and the ordering got the credit.
    """
    from mindbridge.benchmarks.egolife_runner import load_prepared_egomem

    for subject in ("A1_JAKE", "A2_ALICE"):
        _egolife_release(
            tmp_path, subject, day=1, timecodes=("11100000", "11100200", "11100400"), seconds=2
        )
    dataset = _egomem_questions(
        tmp_path,
        [("A2_ALICE", "DAY1, 11:10:07"), ("A1_JAKE", "DAY1, 11:10:03")],
    )
    manifest = tmp_path / "prepared.json"

    prepare_egomem(_request(_egolife_argv(dataset, manifest, tmp_path), tmp_path))

    streams = load_prepared_egomem(manifest)
    assert [stream.subject_id for stream in streams] == ["A2_ALICE", "A1_JAKE"]
    assert [len(stream.clips) for stream in streams] == [3, 1]


def test_every_selected_supermemory_question_ends_exactly_on_a_prepared_boundary(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """`run_supermemory_vqa` rejects a manifest that misses one, before it ingests anything.

    Checked with the runner's own `_segment_bounds` against the runner's own selection, because
    the 30-second grid alone satisfies this only by accident: the question below ends 35 seconds
    in, which no multiple of the grid reaches.
    """
    from mindbridge.benchmarks.supermemory_runner import _segment_bounds, load_prepared_supermemory

    dataset = _supermemory_release(tmp_path)
    manifest = tmp_path / "prepared.json"

    prepare_supermemory(_request(_supermemory_argv(dataset, manifest, tmp_path), tmp_path))

    prepared = load_prepared_supermemory(manifest)
    boundaries = {
        (video.video_id, _segment_bounds(video, segment)[1])
        for video in prepared.videos
        for segment in video.segments
    } | {(video.video_id, video.started_at) for video in prepared.videos}
    for question in _selected_supermemory_questions(dataset):
        assert (question.question_video_id, question.question_ended_at) in boundaries


def test_a_supermemory_segment_carries_its_transcript_as_text_and_as_timed_voice_spans(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """The released MP4s have no audio track, so the aligned transcript is the only speech.

    Perception is told to name people only when a name is seen or heard; handed silent video and
    no spans, it can never name anyone on a corpus whose questions are about what B said.

    Both sound events below belong in the segment's text and in no span, and they are there to
    test two different guards. The unattributed one needs only "an identity has to be somebody".
    The one attributed to B is filtered by `kind` alone, and without that check it becomes a
    voice span whose transcript is `[B laughs]` -- perception handed a bracketed stage direction
    as something B said. A fixture with only the first sound event cannot tell the two guards
    apart, and dropping `kind` stayed green until this line existed.

    Today's bytes do not contain that shape: in the one real transcript on disk, all 21 sound
    events have a null person and all 183 speech lines have one, so either guard alone suffices
    against the release as pinned. It is kept because the release's own README makes Gemini
    authoritative for person labels *and* for bracketed sound events, so the two can meet in one
    line without any schema change -- and `kind` is the field that says which it is.
    """
    from mindbridge.benchmarks.supermemory_runner import load_prepared_supermemory

    dataset = _supermemory_release(tmp_path)
    manifest = tmp_path / "prepared.json"

    prepare_supermemory(_request(_supermemory_argv(dataset, manifest, tmp_path), tmp_path))

    filmed = _by_id(load_prepared_supermemory(manifest))[_SUPERMEMORY_FILMED]
    opening = filmed.segments[0]
    assert opening.transcript is not None
    assert "B: He cooks beef." in opening.transcript
    assert "[Papers rustle]" in opening.transcript
    assert "B: [B laughs]" in opening.transcript
    # The last line of the recording's first 30 seconds runs to 33 s and the one after it starts
    # at 31 s, so this pair is what puts a span on the clip's own timeline rather than the
    # recording's: the crossing line is clipped to the media end, and the later one is absent.
    spans = opening.identity_observations
    assert [(span.identity_id, span.start_ms, span.end_ms) for span in spans] == [
        ("B", 1_000, 3_000),
        ("B", 9_000, 9_000),
        ("User", 28_000, 30_000),
    ]
    assert spans[0].transcript == "He cooks beef."
    assert spans[0].confidence == 1.0
    assert all(span.end_ms <= (opening.media_objects[0].duration_ms or 0) for span in spans)

    # And the segment after it: the crossing line resumes at zero rather than at -2000.
    following = _by_id(load_prepared_supermemory(manifest))[_SUPERMEMORY_FILMED].segments[1]
    assert [
        (span.identity_id, span.start_ms, span.end_ms) for span in following.identity_observations
    ] == [
        ("User", 0, 3_000),
        ("B", 1_000, 3_000),
    ]


def test_two_preparations_of_one_supermemory_subject_produce_the_same_manifest(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """Unlike EgoLife's, these clips are re-encoded, so the digests in here are the encoder's.

    The manifest records each clip's SHA-256 and a run manifest pins the manifest's own digest,
    so a split that is stable but an encode that is not would still churn the pin.
    """
    dataset = _supermemory_release(tmp_path)
    first, second = tmp_path / "one.json", tmp_path / "two.json"

    prepare_supermemory(_request(_supermemory_argv(dataset, first, tmp_path), tmp_path))
    prepare_supermemory(_request(_supermemory_argv(dataset, second, tmp_path), tmp_path))

    assert first.read_bytes() == second.read_bytes()
    # Named rather than left to byte equality alone: two runs a millisecond apart would agree
    # on a wall clock too, and `STAGED_AT` is what makes them agree a week apart.
    assert b'"created_at": "1970-01-01T00:00:00Z"' in first.read_bytes()


def test_repeated_transcript_timings_still_produce_an_acceptable_observation(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """The release gives several lines the same speaker and the same instant, and that is fatal.

    `ObserveRequest` refuses two identity spans sharing `(kind, identity_id, start_ms, end_ms,
    model_id)`, so one collision rejects the whole observation and that segment's media never
    reaches memory -- silently, since the manifest itself has no such rule and validates fine.
    Not hypothetical: the one real transcript on disk has three such groups in 204 lines, all of
    them zero-length spans carrying different text, one of them three lines deep. The fixture
    copies that shape. Every line still reaches the segment's own transcript; only the spans are
    collapsed, which is the field with the constraint.
    """
    from mindbridge.benchmarks.supermemory_runner import load_prepared_supermemory
    from mindbridge.contracts import ObserveRequest
    from mindbridge.core import SensorKind

    dataset = _supermemory_release(tmp_path)
    manifest = tmp_path / "prepared.json"

    prepare_supermemory(_request(_supermemory_argv(dataset, manifest, tmp_path), tmp_path))

    opening = _by_id(load_prepared_supermemory(manifest))[_SUPERMEMORY_FILMED].segments[0]
    assert "B: Yeah." in (opening.transcript or "")
    assert "B: Uh, yeah, it is done." in (opening.transcript or "")
    # The dict is what makes the spans unique; the `if identity in spans` beside it only decides
    # which of the colliding texts survives. Pinned so that cannot flip unnoticed: the release
    # lists "Yeah." first, so "Yeah." is what the span carries.
    collided = next(span for span in opening.identity_observations if span.start_ms == 9_000)
    assert collided.transcript == "Yeah."
    # The real validator rather than a restatement of its rule, built as `_ingest_segment` does.
    ObserveRequest(
        tenant_id="benchmark_supermemory_1_prep-01",
        device_id="supermemory_glasses",
        boot_id="supermemory_vqa_official_v3",
        sequence=0,
        sensor=SensorKind.CAMERA,
        media_objects=opening.media_objects,
        occurred_at=STAGED_AT,
        ended_at=STAGED_AT + timedelta(milliseconds=opening.duration_ms),
        observed_at=STAGED_AT + timedelta(milliseconds=opening.duration_ms),
        identity_observations=opening.identity_observations,
    )


def test_a_supermemory_recording_with_no_released_mp4_is_prepared_from_its_transcript(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """One of the release's 83 sessions is VRS-only, and losing its speech is worse than a gap."""
    from mindbridge.benchmarks.supermemory_runner import load_prepared_supermemory

    dataset = _supermemory_release(tmp_path)
    manifest = tmp_path / "prepared.json"

    prepare_supermemory(_request(_supermemory_argv(dataset, manifest, tmp_path), tmp_path))

    silent = _by_id(load_prepared_supermemory(manifest))[_SUPERMEMORY_SILENT]
    assert all(segment.media_objects == () for segment in silent.segments)
    assert silent.segments[0].transcript == "User: Nothing was filmed here."
    assert silent.segments[0].identity_observations == ()


def test_no_download_reaches_the_fetch_rather_than_stopping_at_the_annotations(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """`--no-download` is documented as refusing to fetch, and preparation is where media is got.

    A producer that drops the flag on the floor leaves it governing annotations only, so
    `--no-download` on these two benchmarks still pulls the largest media sets in the table --
    477 GiB for EgoLife, about 51 GB for one SuperMemory participant. Only the propagation is
    checked here; refusing is `ensure_media`'s own job and its own test.
    """
    _egolife_release(tmp_path, "A1_JAKE", day=1, timecodes=("11100000",), seconds=2)
    dataset = _egolife_questions(tmp_path, [("DAY1", "11101000")])
    argv = _egolife_argv(dataset, tmp_path / "prepared.json", tmp_path)

    prepare_egolife(PrepareRequest(argv=argv, benchmarks_root=tmp_path, quiet=True, download=False))

    assert staged.may_download == [False]


def test_a_supermemory_recording_starting_after_the_last_question_is_not_prepared(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """A run never ingests it, so fetching and cutting it is 2.5 GB spent on nothing.

    `--limit 1` selects the earlier question, which is what moves the horizon in front of the
    third recording; without the bound the same invocation would prepare all three.
    """
    from mindbridge.benchmarks.supermemory_runner import _segment_bounds, load_prepared_supermemory

    dataset = _supermemory_release(tmp_path)
    manifest = tmp_path / "prepared.json"

    prepare_supermemory(
        _request((*_supermemory_argv(dataset, manifest, tmp_path), "--limit", "1"), tmp_path)
    )

    prepared = load_prepared_supermemory(manifest)
    assert [video.video_id for video in prepared.videos] == [
        _SUPERMEMORY_SILENT,
        _SUPERMEMORY_FILMED,
    ]
    # The bound that matters is the download: an unprepared recording still costs 2.5 GB if the
    # producer asked for it, and the manifest looks identical either way.
    assert staged.fetched == [
        (
            "supermemory-vqa",
            (
                f"data/video/Person_1/{_SUPERMEMORY_SILENT}.mp4",
                f"data/video/Person_1/{_SUPERMEMORY_FILMED}.mp4",
            ),
        )
    ]
    for video in prepared.videos:
        for segment in video.segments:
            for media in segment.media_objects:
                assert media.duration_ms is not None
                assert 0 < media.duration_ms <= segment.duration_ms
    # And the recording that IS prepared stops at the question rather than at the end of the
    # file. The filmed session runs 40 s and the selected question ends at 35 s, so without the
    # horizon the split would add a segment of media recorded after the question was asked --
    # the future entering memory, which is the whole point of the causal protocol.
    filmed = _by_id(prepared)[_SUPERMEMORY_FILMED]
    assert (
        _segment_bounds(filmed, filmed.segments[-1])[1]
        == _selected_supermemory_questions(dataset, limit=1)[0].question_ended_at
    )


def test_a_supermemory_segment_holds_exactly_the_span_its_manifest_declares(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """`cut_clips` keeps the frame at the end of its span, which is wrong for a split.

    Left closed, the first 30-second segment encodes 31 seconds and shares its last second with
    the segment after it, while still declaring 30 -- so a benchmark that withholds media
    recorded after a question's time would have had a second of the future inside the clip
    before it. Probed rather than trusted, because the declared number does not move.
    """
    import io

    import av

    from mindbridge.benchmarks.supermemory_runner import load_prepared_supermemory

    dataset = _supermemory_release(tmp_path)
    manifest = tmp_path / "prepared.json"

    prepare_supermemory(_request(_supermemory_argv(dataset, manifest, tmp_path), tmp_path))

    opening = _by_id(load_prepared_supermemory(manifest))[_SUPERMEMORY_FILMED].segments[0]
    media = opening.media_objects[0]
    probe = cast(Any, av.open(io.BytesIO(staged.objects[media.uri.removeprefix("s3://")])))
    with probe:
        assert int(probe.duration / 1_000) == media.duration_ms == 30_000


def test_a_question_boundary_past_the_container_keeps_only_the_media_that_exists(
    tmp_path: Path,
    staged: _Recorder,
) -> None:
    """The protocol's boundary wins and the media stops where the file does.

    A segment may hold media shorter than itself -- that is how a question asked after the
    recording ended still lands on a boundary -- but never longer, which is what the shape
    rejects. The filmed recording here is 40 seconds with a question ending at 50.
    """
    from mindbridge.benchmarks.supermemory_runner import load_prepared_supermemory

    dataset = _supermemory_release(tmp_path)
    manifest = tmp_path / "prepared.json"

    prepare_supermemory(_request(_supermemory_argv(dataset, manifest, tmp_path), tmp_path))

    tail = _by_id(load_prepared_supermemory(manifest))[_SUPERMEMORY_FILMED].segments[-1]
    assert tail.duration_ms == 15_000
    assert tail.media_objects[0].duration_ms is not None
    assert 0 < tail.media_objects[0].duration_ms < tail.duration_ms


def _request(argv: tuple[str, ...], root: Path) -> PrepareRequest:
    return PrepareRequest(argv=argv, benchmarks_root=root, quiet=True)


def _egolife_argv(dataset: Path, manifest: Path, root: Path) -> tuple[str, ...]:
    return (
        "--dataset",
        str(dataset),
        "--prepared-media",
        str(manifest),
        *_common_argv(root),
    )


def _supermemory_argv(dataset: Path, manifest: Path, root: Path) -> tuple[str, ...]:
    return (*_egolife_argv(dataset, manifest, root), "--subject", "1")


def _common_argv(root: Path) -> tuple[str, ...]:
    return (
        "--output",
        str(root / "out.json"),
        "--api-base-url",
        "http://localhost:8000",
        "--deployment-config",
        str(root / "deployment.json"),
        "--run-id",
        "prep-01",
    )


def _video(path: Path, *, seconds: int, fps: int = 5) -> None:
    """Encode a small source of a known length, so a declared duration is checkable."""
    import av
    import numpy

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 128, 96, "yuv420p"
        stream.thread_count, stream.thread_type = 1, "NONE"
        for index in range(fps * seconds):
            array = numpy.full((96, 128, 3), index % 255, dtype=numpy.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(stream.encode())


def _egolife_release(
    root: Path,
    subject: str,
    *,
    day: int,
    timecodes: tuple[str, ...],
    seconds: int,
) -> None:
    """Write the release's own layout: identity directory, day-leading file names.

    Every earlier day gets an empty directory, because the release has all seven and a producer
    that reaches a day the corpus does not hold is looking at a broken download, not an empty one.
    """
    for earlier in range(1, day + 1):
        (root / "egolife" / subject / f"DAY{earlier}").mkdir(parents=True, exist_ok=True)
    for timecode in timecodes:
        _video(
            root / "egolife" / subject / f"DAY{day}" / f"DAY{day}_{subject}_{timecode}.mp4",
            seconds=seconds,
        )


def _egolife_questions(root: Path, moments: list[tuple[str, str]]) -> Path:
    """The official EgoLifeQA shape, which carries a `DAYn` date and an `HHMMSSFF` time."""
    path = root / "egolife" / "EgoLifeQA" / "EgoLifeQA_A1_JAKE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "ID": str(index),
                    "query_time": {"date": date, "time": time},
                    "type": "EntityLog",
                    "need_audio": False,
                    "need_name": True,
                    "last_time": False,
                    "question": "Who used the screwdriver first?",
                    "choice_a": "Tasha",
                    "choice_b": "Alice",
                    "choice_c": "Shure",
                    "choice_d": "Lucia",
                    "answer": "B",
                }
                for index, (date, time) in enumerate(moments, start=1)
            ]
        ),
        encoding="utf-8",
    )
    return path


def _egomem_questions(root: Path, moments: list[tuple[str, str]]) -> Path:
    """The official EgoMemReason JSONL, whose `query_time` is `DAYn, HH:MM:SS`."""
    path = root / "egomem-reason" / "annotations_public.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "example_id": index,
                    "p_id": f"{identity}_q{index}",
                    "identity": identity,
                    "query_time": query_time,
                    "question": "What do I most often eat for breakfast?",
                    "options": {"A": "Rice", "B": "Dumplings", "C": "Burger", "D": "Pancake"},
                    "query_type": "Activity Pattern",
                }
            )
            for index, (identity, query_time) in enumerate(moments, start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _supermemory_release(root: Path) -> Path:
    """Three recordings: one VRS-only, one with media, and one past every selected question.

    The filmed one is 40 seconds long and carries a question ending at 50, which is the released
    case of a boundary running past the container's own tail.
    """
    video = root / "supermemory-vqa" / "data" / "video" / "Person_1"
    _video(video / f"{_SUPERMEMORY_FILMED}.mp4", seconds=40, fps=2)
    _supermemory_transcript(
        root,
        _SUPERMEMORY_FILMED,
        [
            (1.0, 3.0, "B", "He cooks beef.", "speech", "high"),
            (4.0, 5.0, None, "[Papers rustle]", "sound", "low"),
            (6.0, 7.0, "B", "[B laughs]", "sound", "high"),
            (9.0, 9.0, "B", "Yeah.", "speech", "low"),
            (9.0, 9.0, "B", "Uh, yeah, it is done.", "speech", "low"),
            (28.0, 33.0, "User", "This line runs past the split.", "speech", "medium"),
            (31.0, 33.0, "B", "This one starts after it.", "speech", "low"),
        ],
    )
    _supermemory_transcript(
        root,
        _SUPERMEMORY_SILENT,
        [(2.0, 20.0, "User", "Nothing was filmed here.", "speech", "medium")],
    )
    _supermemory_transcript(
        root,
        _SUPERMEMORY_LATER,
        [(1.0, 15.0, "User", "This is after every selected question.", "speech", "low")],
    )
    path = root / "supermemory-vqa" / "data" / "json" / "all_qa.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                _supermemory_question(1, _SUPERMEMORY_FILMED, SUPERMEMORY_EPOCH, 35),
                _supermemory_question(2, _SUPERMEMORY_SILENT, SUPERMEMORY_EPOCH - 10_000, 10),
                _supermemory_question(3, _SUPERMEMORY_LATER, SUPERMEMORY_EPOCH + 100_000, 20),
                _supermemory_question(4, _SUPERMEMORY_FILMED, SUPERMEMORY_EPOCH, 50),
            ]
        ),
        encoding="utf-8",
    )
    return path


_SUPERMEMORY_FILMED = "Person_1_session_8_03102026_glasses_1264"
_SUPERMEMORY_SILENT = "Person_1_session_2_01312026_glasses_vrs_only"
_SUPERMEMORY_LATER = "Person_1_session_9_03102026_glasses_1266"


def _supermemory_transcript(
    root: Path,
    video_id: str,
    lines: list[tuple[float, float, str | None, str, str, str]],
) -> None:
    """The release's aligned sidecar, whose directory lower-cases what the video's capitalises."""
    path = (
        root
        / "supermemory-vqa"
        / "data"
        / "transcripts"
        / "person_1"
        / f"{video_id.lower()}_gemini_aligned_transcript.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {"source": "gemini_caption_transcript_aligned_to_whisper"},
                "transcript": [
                    {
                        "start": start,
                        "end": end,
                        "person": person,
                        "text": text,
                        "kind": kind,
                        "alignment_confidence": confidence,
                    }
                    for start, end, person, text, kind, confidence in lines
                ],
            }
        ),
        encoding="utf-8",
    )


def _supermemory_question(
    question_id: int,
    video_id: str,
    started_unix: int,
    ended_seconds: int,
) -> dict[str, Any]:
    """The release's own key names, including the origin its adapter drops."""
    return {
        "question_id": question_id,
        "question": "What did B say he cooks?",
        "choices": ["This question can not be answered.", "Beef", "Chicken", "Meat"],
        "correct_answer": "Beef",
        "correct_option_index": 1,
        "choice_types": ["incorrect", "correct", "incorrect", "vague"],
        "subject": 1,
        "metadata": {"skill": "conversational_memory", "primary_video_id": video_id},
        "video_ids": [video_id],
        "start_time": started_unix,
        "question_evidence": {
            "time_spans": [
                {
                    "start_time": 0,
                    "end_time": ended_seconds,
                    "video_id": video_id,
                    "video_start_time_unix": started_unix,
                }
            ]
        },
        "is_answerable": True,
        "answer_evidence": {"text": "SECRET ANSWER EVIDENCE"},
    }


def _by_id(prepared: SuperMemoryPreparedSubject) -> dict[str, SuperMemoryPreparedVideo]:
    return {video.video_id: video for video in prepared.videos}


def _selected_supermemory_questions(
    dataset: Path, *, limit: int | None = None
) -> tuple[SuperMemoryQuestion, ...]:
    """The runner's own selection for subject 1, which is what the producer prepared for."""
    from mindbridge.benchmarks.supermemory_cli import _select_questions
    from mindbridge.benchmarks.supermemory_vqa import load_supermemory_vqa

    return _select_questions(load_supermemory_vqa(dataset), 1, (), limit)


def test_a_supermemory_video_start_is_the_release_own_unix_origin(tmp_path: Path) -> None:
    """The adapter folds that origin into one absolute end time and drops it, so it is re-read.

    A prepared video has to declare the instant its own clock starts at, and the only place the
    release states it is the raw annotation's `video_start_time_unix`.
    """
    from mindbridge.benchmarks.prepare_streams import _supermemory_video_starts

    dataset = _supermemory_release(tmp_path)

    starts = _supermemory_video_starts(dataset, 1)

    assert starts[_SUPERMEMORY_FILMED] == datetime(2026, 3, 8, 20, 0, tzinfo=timezone.utc)
    assert starts[_SUPERMEMORY_SILENT] == starts[_SUPERMEMORY_FILMED] - timedelta(seconds=10_000)
    assert set(starts) == {_SUPERMEMORY_FILMED, _SUPERMEMORY_SILENT, _SUPERMEMORY_LATER}
