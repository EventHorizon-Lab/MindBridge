"""Checks for the acquisition that turns a signed Ego4D agreement into EgoTempo's clips.

The two halves this reaches are an external CLI and an AWS credential chain, and both are
exercised rather than replaced. The `ego4d` executable is a real script on `PATH` that records the
argv it was given and publishes the source videos a test grants access to, so what is asserted is
the command actually issued and the layout actually read -- not a stub of this module's own
internals, which would pass just as happily with the wrong flags. Credentials resolve through
botocore itself from an `AWS_SHARED_CREDENTIALS_FILE` in `tmp_path`, which is offline and is the
same resolution the CLI performs.

What is checked is what each failure costs. Losing `only` downloads 222 Ego4D videos instead of
one; cutting from the wrong offset produces a clip that opens, prepares, and scores against the
wrong seconds of video; dropping the audio track produces a corpus that is silently mute, which
this project has shipped before and measured as audio-visual benchmarks scoring below their own
blind baselines. So the span is asserted from the pixels -- the synthetic source encodes each
frame's index as its brightness -- and the audio is asserted in seconds of decoded samples.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from mindbridge.benchmarks import acquire_ego4d

pytest.importorskip("av", reason="acquisition cuts clips with the media extra's decoders")

VIDEO_A = "aaaa1111-2222-3333-4444-555566667777"
VIDEO_B = "bbbb1111-2222-3333-4444-555566667777"
CLIP_A_FIRST = f"{VIDEO_A}_10.0_25.0"
CLIP_A_SECOND = f"{VIDEO_A}_30.0_36.0"
CLIP_B = f"{VIDEO_B}_5.0_12.0"
SOURCE_FRAMES_PER_SECOND = 5
"""The synthetic source's own rate, which is what turns a brightness back into an offset."""

_FAKE_CLI = '''#!{python}
"""A stand-in for the official ego4d CLI, driven by the plan file baked in below."""
import json
import shutil
import sys
from pathlib import Path

plan_path = Path({plan!r})
plan = json.loads(plan_path.read_text())
argv = sys.argv[1:]
plan.setdefault("calls", []).append(argv)
plan_path.write_text(json.dumps(plan))

output = Path(argv[argv.index("--output_directory") + 1])
requested = argv[argv.index("--video_uids") + 1 :]
directory = output / "v2" / "full_scale"
for video_uid in requested:
    if video_uid in plan["published"]:
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(plan["source"], directory / (video_uid + ".mp4"))
sys.exit(plan["exit_code"])
'''


@dataclass(frozen=True, slots=True)
class _FakeEgo4d:
    """The plan the fake CLI obeys, and the record of how it was called."""

    plan: Path

    def publish(self, *video_uids: str, exit_code: int = 0) -> None:
        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        plan["published"] = list(video_uids)
        plan["exit_code"] = exit_code
        self.plan.write_text(json.dumps(plan), encoding="utf-8")

    @property
    def calls(self) -> list[list[str]]:
        return list(json.loads(self.plan.read_text(encoding="utf-8")).get("calls", []))


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """One synthetic Ego4D recording, long enough for every span the fixtures name."""
    return _synthetic_video(tmp_path / "source.mp4", seconds=40)


@pytest.fixture
def ego4d(tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeEgo4d:
    """Put a fake official CLI on `PATH`, publishing both source videos by default."""
    executable = tmp_path / "bin" / "ego4d"
    executable.parent.mkdir(parents=True, exist_ok=True)
    plan = tmp_path / "ego4d-plan.json"
    plan.write_text(
        json.dumps({"published": [VIDEO_A, VIDEO_B], "exit_code": 0, "source": str(source)}),
        encoding="utf-8",
    )
    executable.write_text(_FAKE_CLI.format(python=sys.executable, plan=str(plan)), encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent), prepend=os.pathsep)
    return _FakeEgo4d(plan)


@pytest.fixture
def credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Resolve credentials the way the CLI's own session does: a profile in a file, offline.

    Environment keys are removed rather than set, because they would not answer for this: naming
    a profile explicitly is what botocore drops the environment provider for, and the CLI names
    one. The values are the AWS documentation's own examples and authorise nothing.
    """
    for name in (
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "aws-credentials"
    path.write_text(
        "[default]\n"
        "aws_access_key_id = AKIAIOSFODNN7EXAMPLE\n"
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(path))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws-config-that-does-not-exist"))
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A corpus holding EgoTempo's annotation and no media, which is the released state."""
    corpus = tmp_path / "corpus"
    annotation = corpus / "egotempo" / "egotempo_openQA.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps(
            {
                "info": {"release date": "19.03.2025", "version": "1.0"},
                "annotations": [
                    {
                        "question_id": question_id,
                        "clip_id": clip_id,
                        "question_type": "temporal event ordering",
                        "question": "What happened first?",
                        "answer": "the kettle boiled",
                    }
                    for question_id, clip_id in (
                        # The suffixed form is the release's own: `question_id` is `clip_id` plus
                        # an optional ordinal, which is why acquisition is addressed by file name.
                        (f"{CLIP_A_FIRST}_0", CLIP_A_FIRST),
                        (f"{CLIP_A_FIRST}_1", CLIP_A_FIRST),
                        (f"{CLIP_A_SECOND}_0", CLIP_A_SECOND),
                        (f"{CLIP_B}_0", CLIP_B),
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    return corpus


def test_only_narrows_the_download_to_the_source_video_a_limited_run_reads(
    root: Path, ego4d: _FakeEgo4d, credentials: Path
) -> None:
    """The whole point of `only`: 367 clips span 222 Ego4D videos of gigabytes each.

    A `--limit 1` run reads one clip, so the download has to be one source video. Asserted on the
    argv the CLI was handed and on the absence of the other video's file, because an acquisition
    that fetched both would produce exactly the same clip and pass every other check here.
    """
    acquire_ego4d.acquire(root=root, only=(f"videos/{CLIP_A_FIRST}.mp4",))

    assert len(ego4d.calls) == 1
    command = ego4d.calls[0]
    assert command[command.index("--video_uids") + 1 :] == [VIDEO_A]
    downloads = root / "egotempo" / "ego4d" / "v2" / "full_scale"
    assert (downloads / f"{VIDEO_A}.mp4").exists()
    assert not (downloads / f"{VIDEO_B}.mp4").exists()
    clips = root / "egotempo" / "videos"
    assert sorted(path.name for path in clips.iterdir()) == [f"{CLIP_A_FIRST}.mp4"]
    # The pinned coordinates, which decide what a task name means: the release is `v2_1`, whose
    # files land under `v2/`, and the prompt the CLI would otherwise stop on is answered.
    assert command[command.index("--datasets") + 1] == "full_scale"
    assert command[command.index("--version") + 1] == "v2_1"
    assert command[command.index("--aws_profile_name") + 1] == "default"
    assert command[command.index("--output_directory") + 1] == str(root / "egotempo" / "ego4d")
    assert "--yes" in command
    assert "--no-metadata" in command


def test_the_cut_clip_holds_the_span_its_name_spells_and_keeps_the_audio(
    root: Path, ego4d: _FakeEgo4d, credentials: Path
) -> None:
    """Two silent failures, checked on the bytes rather than on the file existing.

    A clip cut from the wrong offset opens, prepares, and is scored against the wrong seconds of
    video with nothing to notice; so the synthetic source encodes each frame's index as its
    brightness, and 10.0 s in at 5 fps is frame 50, which is brightness 50 rather than 0.

    And a clip cut video-only is a corpus that is silently mute -- the defect that put this
    project's audio-visual benchmarks below their own blind baselines. A 1 fps video track is
    sparse, which is the case where PyAV cannot interleave an audio track, so this is measured:
    the span's audio has to come through in full.
    """
    acquire_ego4d.acquire(root=root, only=(f"videos/{CLIP_A_FIRST}.mp4",))

    clip = root / "egotempo" / "videos" / f"{CLIP_A_FIRST}.mp4"
    brightness, video_frames, audio_seconds = _describe(clip)
    assert abs(brightness - 10.0 * SOURCE_FRAMES_PER_SECOND) <= 6
    # 15 s sampled at the product's own 1 fps, on a span closed at both ends.
    assert video_frames == 16
    assert audio_seconds == pytest.approx(15.0, abs=0.2)


def test_a_clip_already_on_disk_is_left_alone_and_asks_for_no_agreement(
    root: Path, ego4d: _FakeEgo4d, credentials: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running a prepared corpus must not demand the signature the first run used.

    The second acquisition runs with the downloaded Ego4D tree deleted, the CLI off `PATH` and
    the credentials file gone -- which is an operator who reclaimed the disk the sources took.
    Anything checked or fetched before what is already present is subtracted raises here.

    The clip's mtime is the other assertion, and it is the one that has teeth: the encode is
    deterministic, so a re-cut produces identical bytes and byte equality alone cannot tell a
    skipped clip from one re-cut at the cost of an hour.
    """
    acquire_ego4d.acquire(root=root, only=(f"videos/{CLIP_A_FIRST}.mp4",))
    clip = root / "egotempo" / "videos" / f"{CLIP_A_FIRST}.mp4"
    cut, written_at = clip.read_bytes(), clip.stat().st_mtime_ns
    shutil.rmtree(root / "egotempo" / "ego4d")

    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(root / "credentials-that-are-gone"))
    acquire_ego4d.acquire(root=root, only=(f"videos/{CLIP_A_FIRST}.mp4",))

    assert len(ego4d.calls) == 1
    assert clip.read_bytes() == cut
    assert clip.stat().st_mtime_ns == written_at


def test_a_second_clip_of_a_downloaded_video_needs_no_second_download(
    root: Path, ego4d: _FakeEgo4d, credentials: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common case at scale: 367 clips come out of 222 videos, so most are not the first.

    The second acquisition runs with the CLI off `PATH` and the credentials gone, which is what
    makes this an assertion rather than a preference -- a download narrowed per clip rather than
    per source video would ask the CLI about a file it already has, and an operator whose corpus
    already holds the video would be told to go and sign an agreement for it again.
    """
    acquire_ego4d.acquire(root=root, only=(f"videos/{CLIP_A_FIRST}.mp4",))

    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(root / "credentials-that-are-gone"))
    acquire_ego4d.acquire(root=root, only=(f"videos/{CLIP_A_SECOND}.mp4",))

    assert len(ego4d.calls) == 1
    # Cut from the retained source, at its own offset rather than the first clip's.
    brightness, _, _ = _describe(root / "egotempo" / "videos" / f"{CLIP_A_SECOND}.mp4")
    assert abs(brightness - 30.0 * SOURCE_FRAMES_PER_SECOND) <= 6


def test_an_absent_cli_names_what_to_install(
    root: Path, credentials: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prerequisite an operator hits first, and the one nothing else can substitute for."""
    monkeypatch.setenv("PATH", "")

    with pytest.raises(ImportError, match=r"uv pip install ego4d"):
        acquire_ego4d.acquire(root=root, only=(f"videos/{CLIP_A_FIRST}.mp4",))

    assert not (root / "egotempo" / "videos").exists()


@pytest.mark.parametrize("profile", [None, "[default]\n"])
def test_absent_credentials_are_refused_before_anything_is_downloaded(
    root: Path,
    ego4d: _FakeEgo4d,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: str | None,
) -> None:
    """An S3 403 inside somebody else's progress bar is not an instruction; this is.

    Two arrangements, because botocore answers them differently and only one instruction is
    right for both: no AWS configuration at all raises `ProfileNotFound` on the way in, while a
    profile that exists with no keys in it resolves to nothing and returns. An operator who
    created the profile and has not pasted the key in yet is the second one, and it is the case
    that reaches the emptier of the two branches.

    That the CLI was never invoked is the other assertion: the check exists to name the agreement
    before the download starts rather than after it has failed object by object.
    """
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    credentials_file = tmp_path / "credentials-for-this-case"
    if profile is not None:
        credentials_file.write_text(profile, encoding="utf-8")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_file))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config-here"))
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(PermissionError, match=r"ego4d-data\.org"):
        acquire_ego4d.acquire(root=root, only=(f"videos/{CLIP_A_FIRST}.mp4",))

    assert ego4d.calls == []


@pytest.mark.parametrize(
    ("exit_code", "expected", "message"),
    [
        # Matched on what only this guard says. `ego4d` alone passed with the guard deleted: the
        # absent file then reached `read_bytes`, whose own FileNotFoundError names a path with
        # `ego4d` in it, so the assertion held while the diagnosis was gone.
        (0, FileNotFoundError, r"ABORT: All S3 Objects Invalid"),
        (1, RuntimeError, r"CLI exited 1"),
    ],
)
def test_a_cli_that_produced_no_video_is_never_treated_as_success(
    root: Path,
    ego4d: _FakeEgo4d,
    credentials: Path,
    exit_code: int,
    expected: type[Exception],
    message: str,
) -> None:
    """The failure this whole module is arranged around.

    An Ego4D agreement that is unsigned or not yet approved does not make the CLI fail: S3
    refuses each object with a 403, the CLI prints `ABORT: All S3 Objects Invalid`, and it exits
    0. So a successful exit with no file written has to raise, or a run continues into
    preparation and reports the media as missing work rather than as an unsigned agreement.
    """
    ego4d.publish(exit_code=exit_code)

    with pytest.raises(expected, match=message):
        acquire_ego4d.acquire(root=root, only=(f"videos/{CLIP_A_FIRST}.mp4",))

    assert not (root / "egotempo" / "videos").exists()


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (f"videos/{VIDEO_A}_99.0_120.0.mp4", r"unknown EgoTempo clip IDs"),
        # A question ID, which is a clip ID plus an ordinal. The file-name channel makes this
        # merely wrong instead of ambiguous, and wrong has to be refused: 2 of the release's 500
        # question IDs are identical to a clip ID, so an ID-shaped argument could not tell them
        # apart at all.
        (f"videos/{CLIP_A_FIRST}_0.mp4", r"unknown EgoTempo clip IDs"),
        # The last two are refused for their shape, and the message is the reason the shape is
        # checked at all: dropping the check leaves both of them refused anyway, as an unknown
        # clip ID with a decimal quietly missing from it, which reads as a corpus problem rather
        # than as the wiring mistake it is.
        (CLIP_A_FIRST, r"does not name an EgoTempo clip"),
        ("videos/../../etc/passwd.mp4", r"does not name an EgoTempo clip"),
    ],
)
def test_a_request_that_names_no_clip_is_refused_rather_than_guessed(
    root: Path, ego4d: _FakeEgo4d, credentials: Path, entry: str, message: str
) -> None:
    """Every one of these is a wiring mistake, and each would otherwise cost a whole download."""
    with pytest.raises(ValueError, match=message):
        acquire_ego4d.acquire(root=root, only=(entry,))

    assert ego4d.calls == []


def test_every_clip_is_acquired_when_nothing_narrows_the_selection(
    root: Path, ego4d: _FakeEgo4d, credentials: Path
) -> None:
    """Empty `only` is the whole benchmark, which is the other half of the narrowing contract.

    Both source videos in one CLI invocation: it fetches its manifest once and transfers in
    parallel, so one call per video would be the slower shape as well as the noisier one.
    """
    acquire_ego4d.acquire(root=root)

    assert len(ego4d.calls) == 1
    command = ego4d.calls[0]
    assert sorted(command[command.index("--video_uids") + 1 :]) == sorted([VIDEO_A, VIDEO_B])
    clips = root / "egotempo" / "videos"
    assert sorted(path.name for path in clips.iterdir()) == sorted(
        [f"{CLIP_A_FIRST}.mp4", f"{CLIP_A_SECOND}.mp4", f"{CLIP_B}.mp4"]
    )
    # The second clip of video A is the one a per-clip download would have re-fetched, and the
    # one a grouping keyed on the clip rather than the video would have cut from the wrong read.
    brightness, _, _ = _describe(clips / f"{CLIP_A_SECOND}.mp4")
    assert abs(brightness - 30.0 * SOURCE_FRAMES_PER_SECOND) <= 6


def test_an_annotation_that_never_arrived_is_named_rather_than_guessed_at(
    root: Path, ego4d: _FakeEgo4d, credentials: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one branch nothing upstream exercises, kept because it catches a mis-wired path.

    The caller fetches this annotation before dispatching here, so in a real run it is on disk.
    A fetch standing in for one that produced nothing -- the table naming a path this release
    does not carry -- has to say which file is missing, because without it the clip list is
    unknowable and every later message would be about media instead.
    """
    (root / "egotempo" / "egotempo_openQA.json").unlink()
    monkeypatch.setattr(acquire_ego4d, "fetch", lambda *_, **__: ())

    with pytest.raises(FileNotFoundError, match=r"could not be fetched"):
        acquire_ego4d.acquire(root=root, only=(f"videos/{CLIP_A_FIRST}.mp4",))

    assert ego4d.calls == []


def _describe(path: Path) -> tuple[int, int, float]:
    """The first frame's brightness, the video frame count, and the seconds of audio."""
    import av

    with av.open(str(path)) as container:
        frames = list(container.decode(container.streams.video[0]))
        brightness = int(frames[0].to_ndarray(format="rgb24")[0][0][0])
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return brightness, len(frames), 0.0
        stream = container.streams.audio[0]
        samples = sum(frame.samples for frame in container.decode(stream))
        return brightness, len(frames), samples / stream.rate


def _synthetic_video(path: Path, *, seconds: int) -> Path:
    """Encode a source whose frames say what time it is, with an audio track to lose.

    Brightness is the frame index, so a decoded pixel names the offset the clip was cut from.
    """
    import av
    import numpy

    sample_rate, block = 48_000, 1_024
    with av.open(str(path), mode="w") as container:
        video = container.add_stream("libx264", rate=SOURCE_FRAMES_PER_SECOND)
        video.width, video.height, video.pix_fmt = 128, 96, "yuv420p"
        video.thread_count, video.thread_type = 1, "NONE"
        audio = container.add_stream("aac", rate=sample_rate)
        audio.layout = "mono"
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for index in range(SOURCE_FRAMES_PER_SECOND * seconds):
            array = numpy.full((96, 128, 3), index % 255, dtype=numpy.uint8)
            container.mux(video.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        for index in range(int(seconds * sample_rate / block)):
            container.mux(audio.encode(_tone(av, numpy, index, sample_rate, block, resampler)))
        container.mux(video.encode())
        container.mux(audio.encode())
    return path


def _tone(
    av: Any,  # noqa: ANN401 - the media extra is untyped, as mindbridge.media.clipping notes
    numpy: Any,  # noqa: ANN401
    index: int,
    sample_rate: int,
    block: int,
    resampler: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """One block of a 440 Hz tone, resampled into the encoder's own format."""
    offsets = numpy.arange(block) + index * block
    wave = numpy.sin(2 * numpy.pi * 440.0 * offsets / sample_rate) * 0.3
    frame = av.AudioFrame.from_ndarray(
        wave.astype("float32").reshape(1, -1), format="fltp", layout="mono"
    )
    frame.sample_rate = sample_rate
    frame.pts = index * block
    frame.time_base = Fraction(1, sample_rate)
    return resampler.resample(frame)[0]
