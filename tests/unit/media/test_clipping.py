"""Checks that an evidence clip really covers its span and nothing else."""

import io

import pytest

from mindbridge.core import DomainInvariantError, MediaKind
from mindbridge.media.clipping import (
    AUDIO_WINDOW_MS,
    DEFAULT_VIDEO_MAX_PIXELS,
    MINIMUM_CLIP_MS,
    ClipRequest,
    audio_windows,
    cut_clips,
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
