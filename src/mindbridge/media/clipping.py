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


class _ProxyContainer(Protocol):
    """The same PyAV container seen by the proxy cutter, which drives both media kinds.

    PyAV's own `decode` and `add_stream` serve video and audio through one name each; these
    overloads keep the frame and stream types apart so no call site has to widen a type it
    already knows.
    """

    streams: _Streams

    def decode(self, stream: _VideoStream) -> Iterator[_VideoFrame]: ...
    def add_stream(self, codec: str, *, rate: int) -> _VideoStream: ...
    def mux(self, packets: list[object]) -> None: ...
    def seek(self, offset: int) -> None: ...
    def __enter__(self) -> _ProxyContainer: ...
    def __exit__(self, *arguments: object) -> None: ...


class _AudioReader(Protocol):
    """The same container seen through its audio decode, which PyAV spells with one name."""

    def decode(self, stream: _AudioStream) -> Iterator[_AudioFrame]: ...


class _Streams(Protocol):
    video: list[_VideoStream]
    audio: list[_AudioStream]


class _AudioFrame(Protocol):
    time: float | None
    pts: int | None
    samples: int


class _AudioStream(Protocol):
    rate: int
    layout: object
    time_base: object

    def encode(self, frame: _AudioFrame | None = None) -> list[object]: ...


class _AudioResampler(Protocol):
    def resample(self, frame: _AudioFrame | None) -> list[_AudioFrame]: ...


class _AvModule(Protocol):
    def open(self, target: io.BytesIO, mode: str = ..., format: str = ...) -> _Container: ...
    def AudioResampler(
        self,
        *,
        format: str,
        layout: object,
        rate: int,
    ) -> _AudioResampler: ...


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


def cut_generation_proxy(source: bytes, request: ClipRequest) -> MediaClip:
    """Re-encode one video span at the sampling a generator was going to apply anyway.

    This is the copy a model reads instead of the source, so two things have to hold. It keeps
    the source's audio, because a video-only proxy would silently take speech away from every
    question that depends on what was said. And both tracks stay on the source's clock, because
    perception is told event times are milliseconds from the start of the observation and is
    handed each span's absolute offsets alongside this media.

    ponytail: the ceiling is span length. A sampled video track is sparse next to continuous
    audio, and past roughly forty sampled frames the MP4 muxer refuses the interleave, so this
    raises rather than returning a broken file. Callers treat the proxy as best-effort and fall
    back to the source; writing the two tracks through a real interleaving buffer is the upgrade
    path if long single-span observations become common.
    """
    if not source:
        raise DomainInvariantError("source media must not be empty")
    if request.kind is not MediaKind.VIDEO:
        raise DomainInvariantError("a generation proxy is only defined for video")
    av = cast(_AvModule, _import("av"))
    buffer = io.BytesIO()
    with cast(_ProxyContainer, av.open(io.BytesIO(source))) as container:
        if not container.streams.video:
            raise DomainInvariantError("video source has no decodable video stream")
        source_video = container.streams.video[0]
        width, height = scaled_size(source_video.width, source_video.height, request.max_pixels)
        source_audio = container.streams.audio[0] if container.streams.audio else None
        with cast(_ProxyContainer, av.open(buffer, mode="w", format="mp4")) as output:
            # Both tracks are declared before the first packet: libavformat writes the container
            # header on that packet, and a stream added afterwards is rejected.
            video = output.add_stream("libx264", rate=_stream_rate(request.frames_per_second))
            video.width, video.height, video.pix_fmt = width, height, "yuv420p"
            video.time_base = _MILLISECOND_TIME_BASE
            audio, resampler = _proxy_audio_stream(av, output, source_audio)
            frames = 0
            for frame in _sampled_span_frames(source_video, container, request):
                resized = frame.reformat(width=width, height=height, format="yuv420p")
                resized.pts = round((frame.time or 0.0) * 1_000)
                resized.time_base = video.time_base
                output.mux(video.encode(resized))
                frames += 1
            if not frames:
                raise DomainInvariantError("video span selected no frames from its source")
            output.mux(video.encode())
            if source_audio is not None and audio is not None and resampler is not None:
                _copy_span_audio(
                    container,
                    output,
                    request,
                    source=source_audio,
                    audio=audio,
                    resampler=resampler,
                )
    return MediaClip(
        content=buffer.getvalue(),
        suffix=".mp4",
        start_ms=request.start_ms,
        end_ms=request.end_ms,
    )


def _proxy_audio_stream(
    av: _AvModule,
    output: _ProxyContainer,
    source_audio: _AudioStream | None,
) -> tuple[_AudioStream | None, _AudioResampler | None]:
    """Declare the proxy's audio track up front, keeping the source's rate and channel layout."""
    if source_audio is None:
        return None, None
    audio = cast(_AudioStream, output.add_stream("aac", rate=source_audio.rate))
    audio.layout = source_audio.layout
    # AAC only accepts planar float, and a source in any other sample format would raise
    # mid-encode rather than at the boundary.
    resampler = av.AudioResampler(format="fltp", layout=source_audio.layout, rate=source_audio.rate)
    return audio, resampler


def _copy_span_audio(
    container: _ProxyContainer,
    output: _ProxyContainer,
    request: ClipRequest,
    *,
    source: _AudioStream,
    audio: _AudioStream,
    resampler: _AudioResampler,
) -> None:
    """Pass the span's audio through unshifted, so speech lines up with the frames above it."""
    # The video pass left the demuxer wherever the span ended and a decoder cannot read
    # backwards; without this rewind the audio track comes out empty.
    container.seek(0)
    start_seconds, end_seconds = request.start_ms / 1_000, request.end_ms / 1_000
    for frame in cast(_AudioReader, container).decode(source):
        if frame.time is None or frame.time < start_seconds:
            continue
        if frame.time > end_seconds:
            break
        for resampled in resampler.resample(frame):
            output.mux(audio.encode(resampled))
    for resampled in resampler.resample(None):
        output.mux(audio.encode(resampled))
    output.mux(audio.encode())


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
    with av.open(io.BytesIO(source)) as container:
        return list(
            _sampled_span_frames(
                container.streams.video[0], cast(_ProxyContainer, container), request
            )
        )


def _sampled_span_frames(
    stream: _VideoStream,
    container: _ProxyContainer,
    request: ClipRequest,
) -> Iterator[_VideoFrame]:
    """Yield one frame per requested interval inside the span, from an already-open container."""
    interval_seconds = 1.0 / request.frames_per_second
    start_seconds, end_seconds = request.start_ms / 1_000, request.end_ms / 1_000
    next_sample_seconds = start_seconds
    yielded = False
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
        if frame.time is None or frame.time < start_seconds:
            continue
        if frame.time > end_seconds:
            # A zero-width span still deserves the frame it points at.
            if not yielded:
                yield frame
            break
        if frame.time + 1e-9 >= next_sample_seconds:
            next_sample_seconds += interval_seconds
            yielded = True
            yield frame


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
