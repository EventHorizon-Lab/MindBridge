"""Small shared helpers for reading local media metadata."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Literal, Protocol


class _Container(Protocol):
    duration: int | None


class _Stream(Protocol):
    duration: int | None
    time_base: Fraction | None


def media_duration_seconds(path: Path, *, stream_kind: Literal["audio", "video"]) -> float | None:
    """Return a positive stream/container duration when an installed decoder can read it."""
    try:
        import av

        with av.open(str(path)) as container:
            streams = container.streams.audio if stream_kind == "audio" else container.streams.video
            duration = container_duration_seconds(container, streams[0]) if streams else None
            if duration is not None:
                return duration
    except Exception:
        pass
    if stream_kind == "audio":
        try:
            import soundfile  # type: ignore[import-untyped]

            duration = float(soundfile.info(str(path)).duration)
            return duration if math.isfinite(duration) and duration > 0 else None
        except Exception:
            pass
    return None


def container_duration_seconds(
    container: _Container, stream: _Stream | None = None
) -> float | None:
    """Read duration from an already-open PyAV container without decoding it."""
    if stream is not None and stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
        if math.isfinite(duration) and duration > 0:
            return duration
    if container.duration is not None:
        import av

        duration = float(container.duration / av.time_base)
        if math.isfinite(duration) and duration > 0:
            return duration
    return None
