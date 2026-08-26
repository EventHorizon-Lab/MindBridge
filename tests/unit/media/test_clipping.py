"""Checks that an evidence clip really covers its span and nothing else."""

import io
from fractions import Fraction

import pytest

from mindbridge.core import DomainInvariantError, MediaKind
from mindbridge.media.clipping import (
    AUDIO_WINDOW_MS,
    DEFAULT_VIDEO_MAX_PIXELS,
    MINIMUM_CLIP_MS,
    ClipRequest,
    _sample_video_frames,
    _stream_rate,
    audio_windows,
    cut_clips,
    cut_generation_proxy,
    scaled_size,
)

# Real decoding needs the cloud-models extra; without it this module skips rather than
# failing collection, the same way the PostgreSQL tests skip without a database. Importing
# mindbridge.media.clipping stays safe: it loads its decoders lazily.
numpy = pytest.importorskip("numpy")
soundfile = pytest.importorskip("soundfile")
pytest.importorskip("av")
pytest.importorskip("PIL")


SAMPLE_RATE = 16_000


def test_audio_windows_split_a_long_span_without_losing_its_tail() -> None:
    """A 70s span must reach the encoder as three windows, not a truncated one."""
    windows = audio_windows(0, 70_000)

    assert windows == ((0, 30_000), (30_000, 60_000), (60_000, 70_000))
    assert windows[-1][1] == 70_000
    assert all(end - start <= AUDIO_WINDOW_MS for start, end in windows)


def test_audio_windows_keep_a_short_span_whole() -> None:
    assert audio_windows(1_000, 4_000) == ((1_000, 4_000),)


def test_scaled_size_respects_the_pixel_budget_and_stays_even() -> None:
    width, height = scaled_size(1_920, 1_080, 200_704)

    assert width * height <= 200_704
    assert width % 2 == 0 and height % 2 == 0
    assert width / height == pytest.approx(1_920 / 1_080, rel=0.02)
    assert scaled_size(64, 48, 200_704) == (64, 48)


def test_audio_clip_covers_only_its_span() -> None:
    """The cut must carry the span's own samples, not the head of the file."""
    source = _audio_bytes(seconds=10.0)

    clips = cut_clips(source, ClipRequest(kind=MediaKind.AUDIO, start_ms=4_000, end_ms=6_000))

    assert len(clips) == 1
    samples, rate = soundfile.read(io.BytesIO(clips[0].content), dtype="float32")
    assert rate == SAMPLE_RATE
    assert len(samples) == pytest.approx(2 * SAMPLE_RATE, rel=0.01)
    # Marker tone lives only in seconds 4-6 of the source.
    assert float(numpy.abs(samples).mean()) > 0.1


def test_audio_clip_beyond_the_encoder_window_becomes_several_clips() -> None:
    """This is the regression for silently dropping everything past 30s."""
    source = _audio_bytes(seconds=70.0, marker_start=0.0, marker_end=70.0)

    clips = cut_clips(source, ClipRequest(kind=MediaKind.AUDIO, start_ms=0, end_ms=70_000))

    assert [(clip.start_ms, clip.end_ms) for clip in clips] == [
        (0, 30_000),
        (30_000, 60_000),
        (60_000, 70_000),
    ]
    for clip in clips:
        samples, _rate = soundfile.read(io.BytesIO(clip.content), dtype="float32")
        assert len(samples) / SAMPLE_RATE <= AUDIO_WINDOW_MS / 1_000 + 0.01
        assert float(numpy.abs(samples).mean()) > 0.1


def test_video_clip_applies_the_requested_frame_rate_and_pixel_budget() -> None:
    """Frame sampling must be the deployment's choice, not the encoder default."""
    import av

    source = _video_bytes(seconds=6.0, fps=10, width=320, height=240)

    clips = cut_clips(
        source,
        ClipRequest(
            kind=MediaKind.VIDEO,
            start_ms=2_000,
            end_ms=5_000,
            frames_per_second=2.0,
            max_pixels=10_000,
        ),
    )

    assert len(clips) == 1
    with av.open(io.BytesIO(clips[0].content), mode="r") as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(stream))
    # A 3s span at 2 fps is about 6 frames, far fewer than the 60 in the source.
    assert 5 <= len(decoded) <= 8
    assert stream.codec_context.width * stream.codec_context.height <= 10_000


def test_stored_clip_keeps_the_speech_memory_is_supposed_to_hold() -> None:
    """The stored clip is what the media embedder encodes and the read path attaches, so a
    video-only clip deletes speech from memory itself rather than from one model call. Measured
    on a nine-benchmark evaluation: every derived clip was h264-only while its source carried
    aac, and no question that depended on hearing could be answered from memory."""
    import av

    source = _audiovisual_bytes(seconds=6.0, fps=10, width=320, height=240)

    clips = cut_clips(
        source,
        ClipRequest(kind=MediaKind.VIDEO, start_ms=2_000, end_ms=5_000, frames_per_second=2.0),
    )

    with av.open(io.BytesIO(clips[0].content), mode="r") as container:
        assert len(container.streams.audio) == 1
        samples = sum(frame.samples for frame in container.decode(container.streams.audio[0]))
    assert samples > SAMPLE_RATE  # over a second of audio survived the re-encode


def test_stored_clip_rebases_speech_onto_its_own_clock() -> None:
    """Unlike the generation proxy, a stored clip restarts at zero. Audio left on the source's
    clock would sit `start_ms` past a clip that claims to be three seconds long, and the muxer
    drops those packets silently -- speech would be present in the encoder and absent in the
    file."""
    import av

    source = _audiovisual_bytes(seconds=6.0, fps=10, width=320, height=240)

    clips = cut_clips(
        source,
        ClipRequest(kind=MediaKind.VIDEO, start_ms=2_000, end_ms=5_000, frames_per_second=2.0),
    )

    with av.open(io.BytesIO(clips[0].content), mode="r") as container:
        video = [frame.time for frame in container.decode(container.streams.video[0])]
    with av.open(io.BytesIO(clips[0].content), mode="r") as container:
        audio = [frame.time for frame in container.decode(container.streams.audio[0])]
    assert video[0] == pytest.approx(0.0, abs=0.2)
    assert audio[0] == pytest.approx(video[0], abs=0.2)
    assert audio[-1] == pytest.approx(video[-1], abs=0.5)


def test_stored_clip_of_a_silent_source_stays_video_only() -> None:
    import av

    source = _video_bytes(seconds=3.0, fps=10, width=160, height=120)

    clips = cut_clips(
        source,
        ClipRequest(kind=MediaKind.VIDEO, start_ms=0, end_ms=3_000, frames_per_second=1.0),
    )

    with av.open(io.BytesIO(clips[0].content), mode="r") as container:
        assert not container.streams.audio


def test_sampled_video_frames_are_resized_before_they_are_retained() -> None:
    """Peak memory must follow the output budget, not every sampled source frame's pixels."""
    source = _video_bytes(seconds=3.0, fps=10, width=640, height=480)
    request = ClipRequest(
        kind=MediaKind.VIDEO,
        start_ms=0,
        end_ms=3_000,
        frames_per_second=2.0,
        max_pixels=10_000,
    )

    sampled = _sample_video_frames(source, request)

    assert len(sampled) >= 5
    assert all(frame.width * frame.height <= request.max_pixels for frame in sampled)


def test_generation_proxy_keeps_the_speech_the_model_has_to_hear() -> None:
    """Perception reads this copy instead of the source, so dropping its audio track would
    silently take speech away from every question that depends on what was said."""
    import av

    source = _audiovisual_bytes(seconds=4.0, fps=10, width=320, height=240)

    proxy = cut_generation_proxy(
        source,
        ClipRequest(
            kind=MediaKind.VIDEO,
            start_ms=0,
            end_ms=4_000,
            frames_per_second=1.0,
            max_pixels=10_000,
        ),
    )

    with av.open(io.BytesIO(proxy.content), mode="r") as container:
        assert len(container.streams.audio) == 1
        samples = sum(frame.samples for frame in container.decode(container.streams.audio[0]))
    assert samples > SAMPLE_RATE  # over a second of audio survived the re-encode


def test_generation_proxy_applies_the_requested_frame_rate_and_pixel_budget() -> None:
    """The point of the proxy is that it carries the sampling the model was going to apply."""
    import av

    source = _audiovisual_bytes(seconds=6.0, fps=10, width=320, height=240)

    proxy = cut_generation_proxy(
        source,
        ClipRequest(
            kind=MediaKind.VIDEO,
            start_ms=2_000,
            end_ms=5_000,
            frames_per_second=2.0,
            max_pixels=10_000,
        ),
    )

    assert len(proxy.content) < len(source)
    with av.open(io.BytesIO(proxy.content), mode="r") as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(stream))
    assert 5 <= len(decoded) <= 8
    assert stream.codec_context.width * stream.codec_context.height <= 10_000


def test_generation_proxy_covers_the_span_length_observations_actually_use() -> None:
    """The span length every ingest path in the repo actually produces, at the default rate.

    Kept as its own case now that `test_stored_clip_has_no_frame_ceiling` covers the long
    spans: this is the shape `SEGMENT_SECONDS` yields, and it is the one every ingest path
    depends on whatever else moves around it.
    """
    import av

    source = _audiovisual_bytes(seconds=30.0, fps=10, width=160, height=120)

    proxy = cut_generation_proxy(
        source,
        ClipRequest(kind=MediaKind.VIDEO, start_ms=0, end_ms=30_000, frames_per_second=1.0),
    )

    assert len(proxy.content) < len(source)
    with av.open(io.BytesIO(proxy.content), mode="r") as container:
        assert {stream.type for stream in container.streams} == {"video", "audio"}
        assert len(list(container.decode(container.streams.video[0]))) >= 25


def test_generation_proxy_keeps_picture_and_speech_on_one_timeline() -> None:
    """Video was rebased to zero while audio kept the source timeline, so the model heard speech
    against the wrong frames by exactly the span offset."""
    import av

    source = _audiovisual_bytes(seconds=6.0, fps=10, width=320, height=240)

    proxy = cut_generation_proxy(
        source,
        ClipRequest(kind=MediaKind.VIDEO, start_ms=2_000, end_ms=5_000, frames_per_second=2.0),
    )

    with av.open(io.BytesIO(proxy.content), mode="r") as container:
        video = [frame.time for frame in container.decode(container.streams.video[0])]
    with av.open(io.BytesIO(proxy.content), mode="r") as container:
        audio = [frame.time for frame in container.decode(container.streams.audio[0])]
    assert video[0] == pytest.approx(audio[0], abs=0.2)
    assert video[-1] == pytest.approx(audio[-1], abs=0.2)
    # Perception is told event times are milliseconds from observation start, so the copy it
    # reads has to stay on the source's clock rather than restarting at zero.
    assert video[0] == pytest.approx(2.0, abs=0.2)


def test_generation_proxy_of_a_silent_source_stays_video_only() -> None:
    import av

    source = _video_bytes(seconds=3.0, fps=10, width=160, height=120)

    proxy = cut_generation_proxy(
        source,
        ClipRequest(kind=MediaKind.VIDEO, start_ms=0, end_ms=3_000, frames_per_second=1.0),
    )

    with av.open(io.BytesIO(proxy.content), mode="r") as container:
        assert not container.streams.audio


def test_generation_proxy_drops_audio_when_the_generator_cannot_hear() -> None:
    """Measured against the evaluation's endpoint, prompt_tokens was identical with and without
    the track, so a deployment whose generator ignores audio pays an encode and a transfer for
    bytes nothing reads."""
    import av

    source = _audiovisual_bytes(seconds=10.0, fps=10, width=160, height=120)

    heard = cut_generation_proxy(
        source,
        ClipRequest(kind=MediaKind.VIDEO, start_ms=0, end_ms=10_000, frames_per_second=1.0),
    )
    deaf = cut_generation_proxy(
        source,
        ClipRequest(
            kind=MediaKind.VIDEO,
            start_ms=0,
            end_ms=10_000,
            frames_per_second=1.0,
            include_audio=False,
        ),
    )

    with av.open(io.BytesIO(deaf.content), mode="r") as container:
        assert not container.streams.audio
        # The picture is untouched: this drops a track, it does not re-sample the video.
        assert len(list(container.decode(container.streams.video[0]))) >= 8
    with av.open(io.BytesIO(heard.content), mode="r") as container:
        assert container.streams.audio
    assert len(deaf.content) < len(heard.content)


def test_stored_clip_has_no_frame_ceiling() -> None:
    """A span of 43 or more sampled frames used to fail, and the frame count was the only thing
    that decided it: 42 frames encoded, 43 raised `[Errno 22] Invalid argument` out of the flush
    that drains the encoder, whatever the span length, the offset or the frame rate that
    produced them.

    It was read as a limit of the encoder or the MP4 muxer. It was neither. `stream.time_base`
    is a proposal libavformat overwrites with the container's timescale when it writes the
    header, the header goes out on the first `mux`, and libx264 holds `rc_lookahead` frames --
    40 by default -- before it emits a first packet. A loop that re-read that attribute per
    frame therefore stamped the first ~40 frames in milliseconds and the rest in 1/16000ths,
    handed x264 timestamps that no longer rose, and got back a packet whose pts sat behind its
    dts. Stamping from the constant instead removes the bound rather than documenting it.

    `cut_clips` is where this mattered most. The generation-proxy path at least had a guard
    -- a frame budget that skipped the span, since removed with the defect -- while the
    stored-clip path every observation and every benchmark producer goes through had none, and
    only stayed clear of the failure because `SEGMENT_SECONDS` is 30.
    """
    import av

    source = _audiovisual_bytes(seconds=60.0, fps=10, width=160, height=120)

    for start_ms, end_ms, frames_per_second in (
        # One frame either side of where it used to break, then far enough past it that a
        # ceiling merely moved rather than removed would still be caught.
        (0, 42_000, 1.0),
        (0, 43_000, 1.0),
        (10_000, 53_000, 1.0),
        (0, 60_000, 0.75),
        (0, 60_000, 1.0),
        (0, 60_000, 4.0),
    ):
        request = ClipRequest(
            kind=MediaKind.VIDEO,
            start_ms=start_ms,
            end_ms=end_ms,
            frames_per_second=frames_per_second,
        )
        sampled = _sample_video_frames(source, request)
        # The sampler snaps to real source frames, so the expected timeline is the one those
        # frames carry, rebased to zero -- not the nominal 1/frames_per_second grid.
        first = sampled[0].time or 0.0
        expected = [(frame.time or 0.0) - first for frame in sampled]
        assert len(expected) >= 43

        clips = cut_clips(source, request)

        with av.open(io.BytesIO(clips[0].content), mode="r") as container:
            decoded = list(container.decode(container.streams.video[0]))
        # Not just "it did not raise". Half the frames landing on a 16x wrong timeline is the
        # same defect one step short of the muxer noticing it, so every frame has to come back
        # on the offset it went in on, in order. The tolerance is the clip's own tick: H.264
        # takes an integral rate, so a 0.75 fps clip cannot express 1.4 s any finer than 1 s --
        # a separate limit, and one 16x too small a timestamp clears by a wide margin.
        tick_seconds = 1.0 / _stream_rate(frames_per_second)
        assert len(decoded) == len(expected)
        assert [frame.time for frame in decoded] == sorted(frame.time for frame in decoded), (
            "frames came back out of order"
        )
        for rendered, offset_seconds in zip(decoded, expected, strict=True):
            assert rendered.time == pytest.approx(offset_seconds, abs=tick_seconds)


def test_generation_proxy_has_no_frame_ceiling_with_or_without_audio() -> None:
    """The same defect, on the path that carried the guard.

    Both cases are kept because the ceiling was once documented as the MP4 muxer refusing to
    interleave a sparse video track with continuous audio, and a deployment acting on that
    would turn `proxy_audio` off to buy longer spans. Audio was never the cause: a silent
    source cut with `include_audio=False` failed at the same frame count, and now neither
    fails. The proxy keeps the source's absolute clock, so its frames start at the span.
    """
    import av

    audiovisual = _audiovisual_bytes(seconds=60.0, fps=10, width=160, height=120)
    silent = _video_bytes(seconds=60.0, fps=10, width=160, height=120)

    for source, include_audio in ((audiovisual, True), (silent, False)):
        request = ClipRequest(
            kind=MediaKind.VIDEO,
            start_ms=0,
            end_ms=60_000,
            frames_per_second=1.0,
            include_audio=include_audio,
        )
        sampled = _sample_video_frames(source, request)
        expected = [frame.time or 0.0 for frame in sampled]
        assert len(expected) > 42

        proxy = cut_generation_proxy(source, request)

        with av.open(io.BytesIO(proxy.content), mode="r") as container:
            assert bool(container.streams.audio) is include_audio
            decoded = list(container.decode(container.streams.video[0]))
        assert len(decoded) == len(expected)
        for rendered, offset_seconds in zip(decoded, expected, strict=True):
            assert rendered.time == pytest.approx(offset_seconds, abs=1.0 / _stream_rate(1.0))


def test_cut_rejects_empty_source_and_backwards_spans() -> None:
    with pytest.raises(DomainInvariantError, match="source media must not be empty"):
        cut_clips(b"", ClipRequest(kind=MediaKind.AUDIO, start_ms=0, end_ms=1_000))
    with pytest.raises(DomainInvariantError, match="non-negative forward span"):
        ClipRequest(kind=MediaKind.AUDIO, start_ms=900, end_ms=100)


def _audio_bytes(
    *,
    seconds: float,
    marker_start: float = 4.0,
    marker_end: float = 6.0,
) -> bytes:
    """Silence everywhere except a loud tone, so a wrong cut is audible in the mean."""
    times = numpy.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    samples = numpy.zeros_like(times, dtype="float32")
    marker = (times >= marker_start) & (times < marker_end)
    samples[marker] = 0.8 * numpy.sin(2 * numpy.pi * 440 * times[marker])
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _video_bytes(*, seconds: float, fps: int, width: int, height: int) -> bytes:
    import av

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index in range(int(seconds * fps)):
            array = numpy.full((height, width, 3), index % 256, dtype="uint8")
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = index
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return buffer.getvalue()


def _audiovisual_bytes(*, seconds: float, fps: int, width: int, height: int) -> bytes:
    """One MP4 carrying both a video track and a real audio track."""
    import av

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        audio = container.add_stream("aac", rate=SAMPLE_RATE)
        audio.layout = "mono"
        for index in range(int(seconds * fps)):
            array = numpy.full((height, width, 3), index % 256, dtype="uint8")
            video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            video_frame.pts = index
            container.mux(video.encode(video_frame))
        samples_per_frame = SAMPLE_RATE // fps
        for index in range(int(seconds * fps)):
            tone = numpy.sin(
                2
                * numpy.pi
                * 440
                * numpy.arange(index * samples_per_frame, (index + 1) * samples_per_frame)
                / SAMPLE_RATE
            )
            audio_frame = av.AudioFrame.from_ndarray(
                (tone * 16_000).astype("int16").reshape(1, -1), format="s16", layout="mono"
            )
            audio_frame.sample_rate = SAMPLE_RATE
            audio_frame.pts = index * samples_per_frame
            container.mux(audio.encode(audio_frame))
        container.mux(video.encode())
        container.mux(audio.encode())
    return buffer.getvalue()


def _late_video_audiovisual_bytes(
    *, seconds: float, video_start: float, fps: int, width: int, height: int
) -> bytes:
    """One MP4 whose audio runs from zero but whose video track only starts at `video_start`.

    Not a contrived shape: a capture whose camera warms up after its microphone produces it, and
    so does any remux that concatenates a silent lead-in. It is the case that separates rebasing
    the audio from clamping it, because everything before the first video frame has no timestamp
    left on the clip's own clock.
    """
    import av

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        video.time_base = Fraction(1, 1_000)
        audio = container.add_stream("aac", rate=SAMPLE_RATE)
        audio.layout = "mono"
        for index in range(int((seconds - video_start) * fps)):
            array = numpy.full((height, width, 3), index % 256, dtype="uint8")
            video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            video_frame.pts = round((video_start + index / fps) * 1_000)
            video_frame.time_base = video.time_base
            container.mux(video.encode(video_frame))
        samples_per_frame = SAMPLE_RATE // fps
        for index in range(int(seconds * fps)):
            tone = numpy.sin(
                2
                * numpy.pi
                * 440
                * numpy.arange(index * samples_per_frame, (index + 1) * samples_per_frame)
                / SAMPLE_RATE
            )
            audio_frame = av.AudioFrame.from_ndarray(
                (tone * 16_000).astype("int16").reshape(1, -1), format="s16", layout="mono"
            )
            audio_frame.sample_rate = SAMPLE_RATE
            audio_frame.pts = index * samples_per_frame
            container.mux(audio.encode(audio_frame))
        container.mux(video.encode())
        container.mux(audio.encode())
    return buffer.getvalue()


def test_stored_clip_of_a_late_starting_video_keeps_one_timestamp_per_audio_frame() -> None:
    """Audio ahead of the rebase point is dropped, never pulled back onto timestamp zero.

    Clamping it looks harmless and is not: every frame before the clip's first video frame
    lands on the same timestamp, and a run of identical timestamps is a non-monotonic MP4.
    Measured on this shape before the fix, 93 audio frames shared pts 0 -- a file whose own
    writer accepted it and a stricter demuxer need not.
    """
    import av

    source = _late_video_audiovisual_bytes(
        seconds=5.0, video_start=2.0, fps=10, width=160, height=120
    )

    # Sampled at the source's own rate on purpose. Below it, the sampler catches up on a
    # late-starting track by taking consecutive frames, which the encoder then quantises onto
    # one tick -- a separate, pre-existing interaction that has nothing to do with the audio.
    clips = cut_clips(
        source,
        ClipRequest(kind=MediaKind.VIDEO, start_ms=0, end_ms=5_000, frames_per_second=10.0),
    )

    with av.open(io.BytesIO(clips[0].content), mode="r") as container:
        stamps = [
            frame.pts
            for frame in container.decode(container.streams.audio[0])
            if frame.pts is not None
        ]
    assert stamps, "the clip carried no audio at all"
    assert len(set(stamps)) == len(stamps), "audio frames collapsed onto a shared timestamp"
    assert stamps == sorted(stamps), "audio timestamps are not monotonic"


def test_point_audio_span_widens_instead_of_failing_the_job() -> None:
    """A zero-width grounded span used to raise and fail the observation forever."""
    source = _audio_bytes(seconds=10.0, marker_start=0.0, marker_end=10.0)

    clips = cut_clips(source, ClipRequest(kind=MediaKind.AUDIO, start_ms=4_000, end_ms=4_000))

    assert len(clips) == 1
    samples, rate = soundfile.read(io.BytesIO(clips[0].content), dtype="float32")
    assert len(samples) == pytest.approx(MINIMUM_CLIP_MS * rate / 1_000, rel=0.05)


def test_point_video_span_keeps_the_frame_it_points_at() -> None:
    source = _video_bytes(seconds=4.0, fps=10, width=160, height=120)

    clips = cut_clips(source, ClipRequest(kind=MediaKind.VIDEO, start_ms=2_000, end_ms=2_000))

    assert len(clips) == 1
    assert clips[0].content


def test_fractional_frame_rate_keeps_real_playback_length() -> None:
    """Rounding the stream rate used to halve a 0.5 fps clip's real duration."""
    import av

    source = _video_bytes(seconds=10.0, fps=10, width=160, height=120)

    clips = cut_clips(
        source,
        ClipRequest(
            kind=MediaKind.VIDEO,
            start_ms=0,
            end_ms=8_000,
            frames_per_second=0.5,
            max_pixels=200_704,
        ),
    )

    with av.open(io.BytesIO(clips[0].content), mode="r") as container:
        stream = container.streams.video[0]
        times = [frame.time for frame in container.decode(stream) if frame.time is not None]
    # Five samples two seconds apart must still span eight seconds, not four.
    assert len(times) == 5
    assert times[-1] == pytest.approx(8.0, abs=0.25)


def test_image_uses_its_own_budget_and_honours_a_region() -> None:
    """Images were silently shrunk to the video budget with the region discarded."""
    from PIL import Image

    source = _image_bytes(width=1_000, height=800)

    whole = cut_clips(source, ClipRequest(kind=MediaKind.IMAGE, start_ms=0, end_ms=0))
    cropped = cut_clips(
        source,
        ClipRequest(kind=MediaKind.IMAGE, start_ms=0, end_ms=0, region=(0, 0, 300, 300)),
    )

    with Image.open(io.BytesIO(whole[0].content)) as rendered:
        # 1000x800 fits the image budget untouched; the video budget would shrink it.
        assert (rendered.width, rendered.height) == (1_000, 800)
        assert rendered.width * rendered.height > DEFAULT_VIDEO_MAX_PIXELS
    with Image.open(io.BytesIO(cropped[0].content)) as rendered:
        assert rendered.width / rendered.height == pytest.approx(1.0, rel=0.02)
        assert rendered.width <= 300


def test_region_larger_than_the_frame_is_clamped_to_it() -> None:
    """An out-of-bounds region must be clamped per axis, not transposed or dropped."""
    from PIL import Image

    source = _image_bytes(width=1_000, height=800)

    # 100x100 survives only if x clamps to width and y clamps to height; swapping the two
    # would clamp x at 799 and stretch y to 1000, and PIL would refuse the crop outright.
    clipped = cut_clips(
        source,
        ClipRequest(kind=MediaKind.IMAGE, start_ms=0, end_ms=0, region=(900, 700, 2_000, 2_000)),
    )

    with Image.open(io.BytesIO(clipped[0].content)) as rendered:
        assert (rendered.width, rendered.height) == (100, 100)


def _image_bytes(*, width: int, height: int) -> bytes:
    from PIL import Image

    array = numpy.zeros((height, width, 3), dtype="uint8")
    array[:300, :300] = 255
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_a_one_frame_video_clip_is_repeated_rather_than_left_unembeddable() -> None:
    """`_sampled_window` widens the window asked for, not the frames a source can deliver.

    Two sources still reach the encoder with one frame: one shorter than the widened window,
    and one whose own frame rate is below the requested sampling. A one-frame video raises
    `t:1 must be larger than temporal_factor:2` at the embed call and takes the whole
    observation down with it, so the frame is repeated instead.
    """
    import av

    shorter_than_its_window = _video_bytes(seconds=0.5, fps=10, width=160, height=120)
    slower_than_the_sampling = _video_bytes(seconds=4.0, fps=1, width=160, height=120)

    for source, request in (
        (
            shorter_than_its_window,
            ClipRequest(kind=MediaKind.VIDEO, start_ms=0, end_ms=2_000, frames_per_second=1.0),
        ),
        (
            slower_than_the_sampling,
            ClipRequest(kind=MediaKind.VIDEO, start_ms=0, end_ms=500, frames_per_second=4.0),
        ),
    ):
        clips = cut_clips(source, request)

        with av.open(io.BytesIO(clips[0].content), mode="r") as container:
            decoded = list(container.decode(container.streams.video[0]))
        assert len(decoded) >= 2
        assert decoded[0].time != decoded[1].time
