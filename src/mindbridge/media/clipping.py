"""Cut one evidence span out of source media before it is embedded.

Embedding a whole recording produces a single vector whose temporal resolution
is the entire file, and the audio tower only ever sees the first Whisper
window of it. Cutting the span first makes the stored vector mean what its
EvidenceSpan claims, and lets the deployment choose the video frame rate
instead of inheriting whatever the encoder defaults to.

Media libraries live in the optional `media` extra, so every import is deferred to the
call that needs it. Deferring them is what lets a `server`-only install start and pass an
import probe while being unable to cut a single clip, so the error below has to name the
extra that actually carries them.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from fractions import Fraction
from types import ModuleType
from typing import Any, cast

from mindbridge.core import DomainInvariantError, MediaKind, ModelUnavailableError

DEFAULT_VIDEO_FRAMES_PER_SECOND = 1.0
DEFAULT_VIDEO_MAX_PIXELS = 200_704
# Jina's audio tower pads or truncates every clip to Whisper's 30 second
# window, so a longer span is split into windows rather than losing its tail.
AUDIO_WINDOW_MS = 30_000
# Grounding an event against a source span can produce a zero-width
# intersection. Point evidence is legitimate, so it is widened to the
# shortest clip the encoders can still read instead of failing the job.
MINIMUM_CLIP_MS = 200
# Millisecond ticks keep sampled frame offsets exact in the container.
_MILLISECOND_TIME_BASE = Fraction(1, 1_000)
DEFAULT_IMAGE_MAX_PIXELS = 1_003_520


@dataclass(frozen=True, slots=True)
class ClipRequest:
    """One evidence span plus the sampling the deployment wants applied."""

    kind: MediaKind
    start_ms: int
    end_ms: int
    frames_per_second: float = DEFAULT_VIDEO_FRAMES_PER_SECOND
    max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS
    image_max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS
    region: tuple[int, int, int, int] | None = None
    # Read by both the generation proxy and the stored clip. A generator that ignores the track
    # pays for it twice, in the encode and in the transfer -- but the stored clip is what the
    # embedder encodes and the read path attaches, so dropping it there deletes speech from
    # memory rather than merely from one model call.
    include_audio: bool = True

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise DomainInvariantError("clip range must be a non-negative forward span")
        if not math.isfinite(self.frames_per_second) or self.frames_per_second <= 0:
            raise DomainInvariantError("clip frame rate must be finite and positive")
        if self.max_pixels <= 0 or self.image_max_pixels <= 0:
            raise DomainInvariantError("clip pixel limit must be positive")
        if self.region is not None:
            x_min, y_min, x_max, y_max = self.region
            if x_max <= x_min or y_max <= y_min or x_min < 0 or y_min < 0:
                raise DomainInvariantError("clip region must be a positive in-bounds box")


@dataclass(frozen=True, slots=True)
class MediaClip:
    """Encoded bytes covering exactly one window of one evidence span."""

    content: bytes
    suffix: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if not self.content:
            raise DomainInvariantError("clip content must not be empty")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise DomainInvariantError("clip range must be a non-negative forward span")


def cut_clips(source: bytes, request: ClipRequest) -> tuple[MediaClip, ...]:
    """Return every window of one span, already sampled for the encoder."""
    if not source:
        raise DomainInvariantError("source media must not be empty")
    if request.kind is MediaKind.IMAGE:
        return (_cut_image(source, request),)
    if request.kind is MediaKind.AUDIO:
        return _cut_audio(source, request)
    return (_cut_video(source, request),)


def cut_generation_proxy(source: bytes, request: ClipRequest) -> MediaClip:
    """Re-encode one video span at the sampling a generator was going to apply anyway.

    This is the copy a model reads instead of the source, so two things have to hold. It keeps
    the source's audio by default, because a video-only proxy would silently take speech away
    from every question that depends on what was said. And both tracks stay on the source's
    clock, because perception is told event times are milliseconds from the start of the
    observation and is handed each span's absolute offsets alongside this media.

    `request.include_audio` exists because that default is wrong for a generator that cannot
    hear. Measured against the evaluation's endpoint on one 15 s clip: `prompt_tokens` was 1009
    whether or not the file carried its audio track, so the track was never ingested, while the
    file itself was 336 KiB with audio against 212 KiB without. A deployment whose generator
    ignores audio is paying an encode and a transfer for nothing.

    ponytail: the ceiling is sampled frame count, not span length, and dropping the audio does
    not lift it -- a silent source cut with `include_audio=False` fails at the same 45 frames.
    See `MAX_PROXY_SAMPLED_FRAMES` for what has and has not been ruled out. This raises rather
    than returning a truncated file, and callers treat the proxy as best-effort and fall back
    to the source.
    """
    if not source:
        raise DomainInvariantError("source media must not be empty")
    if request.kind is not MediaKind.VIDEO:
        raise DomainInvariantError("a generation proxy is only defined for video")
    av = cast(Any, _import("av"))
    sampled = _sample_video_frames(source, request)
    if not sampled:
        raise DomainInvariantError("video span selected no frames from its source")
    width, height = scaled_size(sampled[0].width, sampled[0].height, request.max_pixels)
    buffer = io.BytesIO()
    with (
        av.open(io.BytesIO(source)) as container,
        av.open(buffer, mode="w", format="mp4") as output,
    ):
        source_audio = (
            container.streams.audio[0]
            if request.include_audio and container.streams.audio
            else None
        )
        # Both tracks are declared before the first packet: libavformat writes the container
        # header on that packet, and a stream added afterwards is rejected.
        video = output.add_stream("libx264", rate=_stream_rate(request.frames_per_second))
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        # Single-threaded for the same reason _cut_video's encoder is.
        video.thread_count = 1
        video.thread_type = "NONE"
        video.time_base = _MILLISECOND_TIME_BASE
        audio, resampler = _proxy_audio_stream(av, output, source_audio)
        for frame in sampled:
            resized = frame.reformat(width=width, height=height, format="yuv420p")
            # Absolute, unlike _cut_video: this copy stands in for the whole source in a call
            # whose prompt counts milliseconds from the observation's start.
            resized.pts = round((frame.time or 0.0) * 1_000)
            resized.time_base = video.time_base
            output.mux(video.encode(resized))
        output.mux(video.encode())
        if source_audio is not None:
            _copy_span_audio(
                container, output, request, source=source_audio, audio=audio, resampler=resampler
            )
    return MediaClip(
        content=buffer.getvalue(),
        suffix=".mp4",
        start_ms=request.start_ms,
        end_ms=request.end_ms,
    )


def _proxy_audio_stream(av: Any, output: Any, source_audio: Any) -> tuple[Any, Any]:  # noqa: ANN401
    """Declare the proxy's audio track up front, keeping the source's rate and channel layout."""
    if source_audio is None:
        return None, None
    audio = output.add_stream("aac", rate=source_audio.rate)
    audio.layout = source_audio.layout
    # AAC only accepts planar float, and a source in any other sample format would raise
    # mid-encode rather than at the boundary.
    resampler = av.AudioResampler(format="fltp", layout=source_audio.layout, rate=source_audio.rate)
    return audio, resampler


def _copy_span_audio(
    container: Any,  # noqa: ANN401
    output: Any,  # noqa: ANN401
    request: ClipRequest,
    *,
    source: Any,  # noqa: ANN401
    audio: Any,  # noqa: ANN401
    resampler: Any,  # noqa: ANN401
    offset_seconds: float = 0.0,
) -> None:
    """Copy the span's audio so speech lines up with the frames above it.

    `offset_seconds` is what the video track already subtracted from its own timestamps. A
    generation proxy keeps the source's clock and passes 0; a stored clip restarts at zero and
    passes its first sampled frame's time, because audio left on the source clock would sit
    minutes past a clip that claims to be thirty seconds long, and every player and decoder
    would then read the speech as silence.
    """
    start_seconds, end_seconds = request.start_ms / 1_000, request.end_ms / 1_000
    if start_seconds > 0:
        container.seek(int(start_seconds * 1_000_000), backward=True)
    for frame in container.decode(source):
        if frame.time is None or frame.time < start_seconds:
            continue
        if frame.time > end_seconds:
            break
        for resampled in resampler.resample(frame):
            _shift_audio_frame(resampled, offset_seconds)
            output.mux(audio.encode(resampled))
    for resampled in resampler.resample(None):
        _shift_audio_frame(resampled, offset_seconds)
        output.mux(audio.encode(resampled))
    output.mux(audio.encode())


def _shift_audio_frame(frame: Any, offset_seconds: float) -> None:  # noqa: ANN401
    """Rebase one resampled frame onto a clip that starts at zero."""
    if not offset_seconds or frame.pts is None:
        return
    time_base = frame.time_base
    if time_base:
        ticks_per_second = 1.0 / float(time_base)
    elif frame.sample_rate:
        ticks_per_second = float(frame.sample_rate)
    else:
        # Nothing to rebase against; leaving the timestamp alone is better than guessing a rate
        # and shifting the track by an unknown amount.
        return
    # Clamp rather than emit a negative pts: the seek above lands on a keyframe at or before the
    # span, so the first decoded frames can precede it, and a negative timestamp is dropped by
    # the muxer with no error, which would silently truncate the opening words.
    frame.pts = max(0, frame.pts - round(offset_seconds * ticks_per_second))


def audio_windows(start_ms: int, end_ms: int) -> tuple[tuple[int, int], ...]:
    """Split a span into encoder-sized windows without dropping the tail."""
    if end_ms <= start_ms:
        return ((start_ms, end_ms),)
    return tuple(
        (window_start, min(window_start + AUDIO_WINDOW_MS, end_ms))
        for window_start in range(start_ms, end_ms, AUDIO_WINDOW_MS)
    )


def scaled_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Shrink to the pixel budget, keeping aspect ratio and even dimensions."""
    if width <= 0 or height <= 0:
        raise DomainInvariantError("media dimensions must be positive")
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    return (
        max(2, int(width * scale) // 2 * 2),
        max(2, int(height * scale) // 2 * 2),
    )


def _cut_audio(source: bytes, request: ClipRequest) -> tuple[MediaClip, ...]:
    soundfile = cast(Any, _import("soundfile"))
    samples, sample_rate = soundfile.read(io.BytesIO(source), dtype="float32", always_2d=True)
    clips = []
    for window_start, window_end in audio_windows(request.start_ms, request.end_ms):
        first = min(len(samples), int(window_start * sample_rate / 1_000))
        last = min(len(samples), int(window_end * sample_rate / 1_000))
        if last <= first:
            last = min(len(samples), first + int(MINIMUM_CLIP_MS * sample_rate / 1_000))
        if last <= first:
            continue
        buffer = io.BytesIO()
        soundfile.write(buffer, samples[first:last], sample_rate, format="WAV", subtype="PCM_16")
        clips.append(
            MediaClip(
                content=buffer.getvalue(),
                suffix=".wav",
                start_ms=window_start,
                end_ms=window_end,
            )
        )
    if not clips:
        raise DomainInvariantError("audio span selected no samples from its source")
    return tuple(clips)


def _cut_image(source: bytes, request: ClipRequest) -> MediaClip:
    image_module = cast(Any, _import("PIL.Image"))
    with image_module.open(io.BytesIO(source)) as image:
        converted = image.convert("RGB")
        if request.region is not None:
            converted = converted.crop(
                _clamped_region(request.region, converted.width, converted.height)
            )
        width, height = scaled_size(converted.width, converted.height, request.image_max_pixels)
        resized = converted.resize((width, height))
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
    return MediaClip(
        content=buffer.getvalue(),
        suffix=".png",
        start_ms=request.start_ms,
        end_ms=request.end_ms,
    )


def _cut_video(source: bytes, request: ClipRequest) -> MediaClip:
    """Cut one stored evidence clip, carrying the source's audio track when it has one.

    The clip keeps its audio because this is the object the read path attaches and the media
    embedder encodes. Cutting it video-only silently removed speech from memory itself: an
    omni-modal embedder was handed a silent file, so nothing a person said could ever influence
    retrieval, and an audio-capable reader was handed the same silence. The generation proxy has
    always kept its track; the stored clip is the copy that outlives the model call, so it is the
    one that decides what the memory contains.
    """
    av = cast(Any, _import("av"))
    sampled = _sample_video_frames(source, request)
    if not sampled:
        raise DomainInvariantError("video span selected no frames from its source")
    width, height = scaled_size(sampled[0].width, sampled[0].height, request.max_pixels)
    buffer = io.BytesIO()
    first_seconds = sampled[0].time or 0.0
    with (
        av.open(io.BytesIO(source)) as origin,
        av.open(buffer, mode="w", format="mp4") as container,
    ):
        source_audio = (
            origin.streams.audio[0] if request.include_audio and origin.streams.audio else None
        )
        stream = container.add_stream("libx264", rate=_stream_rate(request.frames_per_second))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        # Single-threaded for the same reason the decode side is, and set here because the
        # decode setting does not reach it: libx264 carries AV_CODEC_CAP_OTHER_THREADS, so
        # libavcodec leaves the pool to x264 itself and `thread_type` never touches it.
        # Measured on PyAV 16.1.0: this loop took the process from 1 OS thread to 7, in the
        # same Worker that loads torchvision's OpenMP runtime. Whether that pool was the one
        # that deadlocked is not established -- the reproduction was one run in three and
        # nobody caught it in the act -- so this closes the remaining way to start native
        # threads here rather than claiming the earlier fix was aimed wrong.
        stream.thread_count = 1
        stream.thread_type = "NONE"
        # Stamp real offsets on a millisecond timeline: rounding the rate to an
        # integer would make a fractional-fps clip play back at the wrong speed
        # and disagree with the duration recorded for it.
        stream.time_base = _MILLISECOND_TIME_BASE
        # Declared before the first packet: libavformat writes the container header on that
        # packet and rejects a stream added afterwards.
        audio, resampler = _proxy_audio_stream(av, container, source_audio)
        for frame in sampled:
            resized = frame.reformat(width=width, height=height, format="yuv420p")
            resized.pts = round(((frame.time or 0.0) - first_seconds) * 1_000)
            resized.time_base = stream.time_base
            container.mux(stream.encode(resized))
        if len(sampled) == 1:
            # `_sampled_window` widens the window a span is cut from, which cannot help a source
            # that is shorter than the widened window or whose own frame rate is below the
            # sampling. Those still arrive here with one frame, and a one-frame video is not a
            # video to a Qwen3-VL encoder: `t:1 must be larger than temporal_factor:2` fails the
            # embed and the observation behind it. A repeated frame is a still, which is what a
            # span this sparse holds anyway, and it is embeddable.
            repeated = sampled[0].reformat(width=width, height=height, format="yuv420p")
            # Two sampling intervals on, because whatever reads this clip samples it at the same
            # rate: one interval is the instant a frame served late already misses.
            repeated.pts = round(2_000 / request.frames_per_second)
            repeated.time_base = stream.time_base
            container.mux(stream.encode(repeated))
        container.mux(stream.encode())
        if source_audio is not None:
            # Rebased onto the clip's own clock, because the video above starts at zero.
            _copy_span_audio(
                origin,
                container,
                request,
                source=source_audio,
                audio=audio,
                resampler=resampler,
                offset_seconds=first_seconds,
            )
    return MediaClip(
        content=buffer.getvalue(),
        suffix=".mp4",
        start_ms=request.start_ms,
        end_ms=request.end_ms,
    )


def _sample_video_frames(source: bytes, request: ClipRequest) -> list[Any]:
    """Keep one frame per requested interval inside the span, decoding once."""
    av = cast(Any, _import("av"))
    interval_seconds = 1.0 / request.frames_per_second
    start_seconds, end_seconds = request.start_ms / 1_000, request.end_ms / 1_000
    frames: list[Any] = []
    next_sample_seconds = start_seconds
    with av.open(io.BytesIO(source)) as container:
        stream = container.streams.video[0]
        # Single-threaded on purpose. FFmpeg frame threading intermittently deadlocks against
        # the OpenMP runtime torchvision loads, and the Worker cuts clips in the same process as
        # the local embedder, so AUTO hung raw-media jobs until their Celery time limit. It
        # reproduced roughly one run in three, which is why the cheap fix is to not start the
        # pool at all rather than to order the imports.
        # This is not free: a 30s 720p source decodes in 1.7s here against 0.35s with AUTO, and
        # the gap widens with resolution.
        stream.thread_type = "NONE"
        # Which is why this seeks rather than decoding from byte zero. `cut_clips` runs once per
        # span, so without a seek a source costs sum(span end offset) rather than
        # sum(span length): a 30 min recording with 32 spans decoded ~28,800 source-seconds,
        # ran past `task_soft_time_limit`, and `SoftTimeLimitExceeded` being in `autoretry_for`
        # then re-ran the identical doomed decode up to `max_retries` times. Measured on a 60s
        # 720p source with 8 five-second spans: 7.57s without the seek against 1.38s with it.
        # `backward=True` is load-bearing -- it lands on the keyframe at or before the target,
        # so the `frame.time < start_seconds` filter below discards the pre-roll instead of the
        # span losing its opening frames. The offset is in microseconds because no stream is
        # passed, which keeps the untyped `stream.time_base` out of the arithmetic.
        if start_seconds > 0:
            container.seek(int(start_seconds * 1_000_000), backward=True)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            if frame.time < start_seconds:
                continue
            if frame.time > end_seconds:
                # A zero-width span still deserves the frame it points at.
                if not frames:
                    frames.append(frame)
                break
            if frame.time + 1e-9 >= next_sample_seconds:
                frames.append(frame)
                next_sample_seconds += interval_seconds
    return frames


def _clamped_region(
    region: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Keep a region of interest inside the frame it was measured against."""
    x_min, y_min, x_max, y_max = region
    return (min(x_min, width - 1), min(y_min, height - 1), min(x_max, width), min(y_max, height))


def _stream_rate(frames_per_second: float) -> int:
    """H.264 needs an integral rate; the sampling interval already did the work."""
    return max(1, round(frames_per_second))


# ponytail: av/soundfile/PIL sit in mypy's ignore_missing_imports list, so a Protocol per
# module would only re-describe libraries mypy already treats as untyped. Callers cast the
# handle at the use site because ANN401 bans `Any` in a signature.
def _import(module_name: str) -> ModuleType:
    from importlib import import_module

    try:
        return import_module(module_name)
    except ImportError as error:
        raise ModelUnavailableError(
            "install MindBridge with the media extra to cut evidence clips"
        ) from error
