"""Cut one evidence span out of source media before it is embedded.

Embedding a whole recording produces a single vector whose temporal resolution
is the entire file, and the audio tower only ever sees the first Whisper
window of it. Cutting the span first makes the stored vector mean what its
EvidenceSpan claims, and lets the deployment choose the video frame rate
instead of inheriting whatever the encoder defaults to.

Media libraries live in the optional cloud-models extra, so every import is
deferred to the call that needs it.
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, cast

from mindbridge.core import DomainInvariantError, MediaKind, ModelUnavailableError


class _AudioSamples(Protocol):
    """The slice-and-measure surface this module needs from a sample array."""

    def __len__(self) -> int: ...
    def __getitem__(self, item: slice) -> _AudioSamples: ...


class _SoundFileModule(Protocol):
    def read(
        self, file: io.BytesIO, *, dtype: str, always_2d: bool
    ) -> tuple[_AudioSamples, int]: ...
    def write(
        self,
        file: io.BytesIO,
        data: _AudioSamples,
        samplerate: int,
        *,
        format: str,
        subtype: str,
    ) -> None: ...


class _VideoFrame(Protocol):
    time: float | None
    pts: int | None
    time_base: object
    width: int
    height: int

    def reformat(self, *, width: int, height: int, format: str) -> _VideoFrame: ...


class _VideoStream(Protocol):
    thread_type: str
    width: int
    height: int
    pix_fmt: str
    time_base: object

    def encode(self, frame: _VideoFrame | None = None) -> list[object]: ...


class _Container(Protocol):
    streams: _Streams

    def decode(self, stream: _VideoStream) -> Iterator[_VideoFrame]: ...
    def add_stream(self, codec: str, *, rate: int) -> _VideoStream: ...
    def mux(self, packets: list[object]) -> None: ...
    def __enter__(self) -> _Container: ...
    def __exit__(self, *arguments: object) -> None: ...


class _Streams(Protocol):
    video: list[_VideoStream]


class _AvModule(Protocol):
    def open(self, target: io.BytesIO, mode: str = ..., format: str = ...) -> _Container: ...


class _Image(Protocol):
    width: int
    height: int

    def convert(self, mode: str) -> _Image: ...
    def crop(self, box: tuple[int, int, int, int]) -> _Image: ...
    def resize(self, size: tuple[int, int]) -> _Image: ...
    def save(self, file: io.BytesIO, *, format: str) -> None: ...
    def __enter__(self) -> _Image: ...
    def __exit__(self, *arguments: object) -> None: ...


class _ImageModule(Protocol):
    def open(self, file: io.BytesIO) -> _Image: ...


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
    soundfile = cast(_SoundFileModule, _import("soundfile"))
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
    image_module = cast(_ImageModule, _import("PIL.Image"))
    with image_module.open(io.BytesIO(source)) as image:
        converted = image.convert("RGB")
        if request.region is not None:
            converted = converted.crop(_clamped_region(request.region, converted))
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
    av = cast(_AvModule, _import("av"))
    sampled = _sample_video_frames(av, source, request)
    if not sampled:
        raise DomainInvariantError("video span selected no frames from its source")
    width, height = scaled_size(sampled[0].width, sampled[0].height, request.max_pixels)
    buffer = io.BytesIO()
    first_seconds = sampled[0].time or 0.0
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=_stream_rate(request.frames_per_second))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        # Stamp real offsets on a millisecond timeline: rounding the rate to an
        # integer would make a fractional-fps clip play back at the wrong speed
        # and disagree with the duration recorded for it.
        stream.time_base = _MILLISECOND_TIME_BASE
        for frame in sampled:
            resized = frame.reformat(width=width, height=height, format="yuv420p")
            resized.pts = round(((frame.time or 0.0) - first_seconds) * 1_000)
            resized.time_base = stream.time_base
            container.mux(stream.encode(resized))
        container.mux(stream.encode())
    return MediaClip(
        content=buffer.getvalue(),
        suffix=".mp4",
        start_ms=request.start_ms,
        end_ms=request.end_ms,
    )


def _sample_video_frames(
    av: _AvModule,
    source: bytes,
    request: ClipRequest,
) -> list[_VideoFrame]:
    """Keep one frame per requested interval inside the span, decoding once."""
    interval_seconds = 1.0 / request.frames_per_second
    start_seconds, end_seconds = request.start_ms / 1_000, request.end_ms / 1_000
    frames: list[_VideoFrame] = []
    next_sample_seconds = start_seconds
    with av.open(io.BytesIO(source)) as container:
        stream = container.streams.video[0]
        # Single-threaded on purpose. FFmpeg frame threading intermittently deadlocks against
        # the OpenMP runtime torchvision loads, and the Worker cuts clips in the same process as
        # the local embedder, so AUTO hung raw-media jobs until their Celery time limit. It
        # reproduced roughly one run in three, which is why the cheap fix is to not start the
        # pool at all rather than to order the imports.
        # This is not free: a 30s 720p source decodes in 1.7s here against 0.35s with AUTO, and
        # the gap widens with resolution. ponytail: the ceiling is that this loop has no seek,
        # so cost is (span end offset x span count) and a long source with many spans can reach
        # the 900s task limit — seek to start_seconds before that becomes real.
        stream.thread_type = "NONE"
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
    image: _Image,
) -> tuple[int, int, int, int]:
    """Keep a region of interest inside the frame it was measured against."""
    x_min, y_min, x_max, y_max = region
    return (
        min(x_min, image.width - 1),
        min(y_min, image.height - 1),
        min(x_max, image.width),
        min(y_max, image.height),
    )


def _stream_rate(frames_per_second: float) -> int:
    """H.264 needs an integral rate; the sampling interval already did the work."""
    return max(1, round(frames_per_second))


def _import(module_name: str) -> object:
    from importlib import import_module

    try:
        return import_module(module_name)
    except ImportError as error:
        raise ModelUnavailableError(
            "install MindBridge with the cloud-models extra to cut evidence clips"
        ) from error
