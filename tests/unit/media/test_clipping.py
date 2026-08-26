"""Checks that an evidence clip really covers its span and nothing else."""

import io

import pytest

from mindbridge.core import DomainInvariantError, MediaKind
from mindbridge.media.clipping import (
    AUDIO_WINDOW_MS,
    DEFAULT_VIDEO_MAX_PIXELS,
    MINIMUM_CLIP_MS,
    ClipRequest,
    _sample_video_frames,
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


def test_video_evidence_clip_keeps_speech_aligned_with_its_frames() -> None:
    """Jina auto-fuses a video's audio only when the derived MP4 still carries the track."""
    import av

    source = _audiovisual_bytes(seconds=6.0, fps=10, width=320, height=240)

    clip = cut_clips(
        source,
        ClipRequest(
            kind=MediaKind.VIDEO,
            start_ms=2_000,
            end_ms=5_000,
            frames_per_second=2.0,
            max_pixels=10_000,
        ),
    )[0]

    with av.open(io.BytesIO(clip.content), mode="r") as container:
        assert {stream.type for stream in container.streams} == {"video", "audio"}
        video = [frame.time for frame in container.decode(container.streams.video[0])]
    with av.open(io.BytesIO(clip.content), mode="r") as container:
        audio = [frame.time for frame in container.decode(container.streams.audio[0])]
    assert video[0] == pytest.approx(audio[0], abs=0.2)
    assert video[-1] == pytest.approx(audio[-1], abs=0.2)
    assert video[0] == pytest.approx(0.0, abs=0.2)


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


def test_generation_proxy_refuses_a_span_past_its_frame_ceiling() -> None:
    """Past roughly forty sampled frames the encode fails on the flush that drains the encoder.
    It has to raise rather than hand back a file whose audio or video is silently truncated,
    because the caller's fallback is the source itself.

    The cause is not the audio interleave this test used to name -- see
    `test_dropping_audio_does_not_lift_the_frame_ceiling`, which fails the same way with no
    audio anywhere."""
    source = _audiovisual_bytes(seconds=60.0, fps=10, width=160, height=120)

    with pytest.raises(Exception, match=r"Invalid argument|monotonic"):
        cut_generation_proxy(
            source,
            ClipRequest(kind=MediaKind.VIDEO, start_ms=0, end_ms=60_000, frames_per_second=1.0),
        )


def test_generation_proxy_covers_the_span_length_observations_actually_use() -> None:
    """Every ingest path in the repo segments video well inside the ceiling above."""
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


def test_dropping_audio_does_not_lift_the_frame_ceiling() -> None:
    """The ceiling was documented as the MP4 muxer refusing to interleave a sparse video track
    with continuous audio, which would make audio the thing to give up to cut a longer span.
    It is not: a silent source cut with `include_audio=False` -- no audio anywhere in the
    pipeline -- fails at the same frame count an audiovisual one does.

    Pinned as a test because the wrong explanation is the kind that gets acted on: it invites a
    deployment to disable proxy audio expecting longer spans to start working, and they do not.
    """
    silent_source = _video_bytes(seconds=60.0, fps=10, width=160, height=120)

    with pytest.raises(Exception, match=r"Invalid argument|monotonic") as failure:
        cut_generation_proxy(
            silent_source,
            ClipRequest(
                kind=MediaKind.VIDEO,
                start_ms=0,
                end_ms=60_000,
                frames_per_second=1.0,
                include_audio=False,
            ),
        )

    assert type(failure.value).__module__.startswith("av")
    # The same silent source inside the ceiling cuts fine, so what binds is the frame count --
    # not the span length, and not the audio track that is absent from both calls.
    inside = cut_generation_proxy(
        silent_source,
        ClipRequest(
            kind=MediaKind.VIDEO,
            start_ms=0,
            end_ms=30_000,
            frames_per_second=1.0,
            include_audio=False,
        ),
    )
    assert inside.content


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
