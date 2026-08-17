"""Small runtime helpers shared by production-path benchmark runners."""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from pydantic import AwareDatetime, Field, TypeAdapter, model_validator

from mindbridge.contracts import (
    ContractModel,
    Identifier,
    MediaObjectInput,
    NonEmptyString,
    ObservationProcessingJobView,
    ObserveRequest,
    RememberRequest,
)
from mindbridge.core import JobState, MediaKind, MemoryType, SensorKind
from mindbridge.sdk import MindBridge

OPTION_LABELS = tuple("ABCDEFGHIJ")
_ALLOWED_RESPONSE_WORDS = re.compile(
    r"\b(?:answer|best|choice|choices|from|is|option|options|order|rank|ranking|to|worst)\b",
    re.IGNORECASE,
)


class PreparedVideoSegment(ContractModel):
    """One time-aligned segment prepared outside MindBridge."""

    segment_id: Identifier
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    duration_ms: int = Field(gt=0)
    media_objects: tuple[MediaObjectInput, ...] = Field(default=(), max_length=8)
    transcript: NonEmptyString | None = None

    @model_validator(mode="after")
    def require_aligned_content(self) -> PreparedVideoSegment:
        if not self.media_objects and self.transcript is None:
            raise ValueError("prepared video segments require media or a transcript")
        media_ids = tuple(item.media_object_id for item in self.media_objects)
        if len(set(media_ids)) != len(media_ids):
            raise ValueError("prepared segment media_object_ids must be unique")
        if any(
            item.kind in {MediaKind.VIDEO, MediaKind.AUDIO}
            and (item.duration_ms is None or not 0 < item.duration_ms <= self.duration_ms)
            for item in self.media_objects
        ):
            raise ValueError("timed media must fit its prepared segment")
        if any(
            item.duration_ms is not None and item.duration_ms > self.duration_ms
            for item in self.media_objects
        ):
            raise ValueError("media duration must not exceed its prepared segment")
        return self


class PreparedVideo(ContractModel):
    """Prepared media and a deterministic clock for one source video."""

    video_id: Identifier
    timeline_origin: AwareDatetime
    segments: tuple[PreparedVideoSegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_ordered_non_overlapping_segments(self) -> PreparedVideo:
        starts = tuple(segment.start_seconds for segment in self.segments)
        if starts != tuple(sorted(starts)) or len(set(starts)) != len(starts):
            raise ValueError("prepared segments must have unique chronological starts")
        segment_ids = tuple(segment.segment_id for segment in self.segments)
        media_ids = tuple(
            media.media_object_id for segment in self.segments for media in segment.media_objects
        )
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("prepared segment IDs must be unique")
        if len(set(media_ids)) != len(media_ids):
            raise ValueError("prepared media_object_ids must be globally unique per video")
        previous_end = Decimal()
        for segment in self.segments:
            start = Decimal(str(segment.start_seconds))
            if start < previous_end:
                raise ValueError("prepared video segments must not overlap")
            previous_end = start + Decimal(segment.duration_ms) / 1_000
        return self


def load_prepared_videos(path: Path) -> tuple[PreparedVideo, ...]:
    """Load metadata for already addressable benchmark media."""
    videos = TypeAdapter(tuple[PreparedVideo, ...]).validate_json(path.read_bytes())
    if not videos:
        raise ValueError("prepared video manifest must not be empty")
    video_ids = tuple(video.video_id for video in videos)
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("prepared video manifest contains duplicate video IDs")
    return videos


def prepared_video_end(video: PreparedVideo) -> AwareDatetime:
    """Return the exclusive end of the final prepared segment."""
    segment = video.segments[-1]
    return video.timeline_origin + timedelta(
        seconds=segment.start_seconds,
        milliseconds=segment.duration_ms,
    )


async def ingest_prepared_video(
    memory: MindBridge,
    tenant_id: str,
    device_id: str,
    video: PreparedVideo,
    *,
    adapter_version: str,
    request_concurrency: int,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
) -> None:
    """Ingest one prepared video through the public observation contracts."""
    if request_concurrency <= 0:
        raise ValueError("request_concurrency must be positive")
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")
    TypeAdapter(Identifier).validate_python(adapter_version)
    semaphore = asyncio.Semaphore(request_concurrency)
    for offset in range(0, len(video.segments), request_concurrency):
        await asyncio.gather(
            *(
                _ingest_prepared_segment(
                    memory,
                    tenant_id,
                    device_id,
                    video,
                    segment,
                    offset + index,
                    adapter_version,
                    poll_interval_seconds,
                    processing_timeout_seconds,
                    semaphore,
                )
                for index, segment in enumerate(
                    video.segments[offset : offset + request_concurrency]
                )
            )
        )


async def _ingest_prepared_segment(
    memory: MindBridge,
    tenant_id: str,
    device_id: str,
    video: PreparedVideo,
    segment: PreparedVideoSegment,
    sequence: int,
    adapter_version: str,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
    semaphore: asyncio.Semaphore,
) -> None:
    occurred_at = video.timeline_origin + timedelta(seconds=segment.start_seconds)
    ended_at = occurred_at + timedelta(milliseconds=segment.duration_ms)
    async with semaphore:
        evidence_ids: tuple[str, ...] = ()
        if segment.media_objects:
            receipt = await memory.observe(
                ObserveRequest(
                    tenant_id=tenant_id,
                    device_id=device_id,
                    boot_id=adapter_version,
                    sequence=sequence,
                    sensor=(
                        SensorKind.MICROPHONE
                        if all(item.kind is MediaKind.AUDIO for item in segment.media_objects)
                        else SensorKind.CAMERA
                    ),
                    media_objects=segment.media_objects,
                    occurred_at=occurred_at,
                    ended_at=ended_at,
                    observed_at=ended_at,
                    idempotency_key=_prepared_idempotency_key(
                        adapter_version, video.video_id, segment.segment_id, "media"
                    ),
                )
            )
            evidence_ids = receipt.evidence_ids
            await wait_for_observation_job(
                memory,
                tenant_id,
                receipt.processing_job_id,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=processing_timeout_seconds,
            )
        if segment.transcript is not None:
            await memory.remember(
                RememberRequest(
                    tenant_id=tenant_id,
                    summary=segment.transcript,
                    memory_type=MemoryType.EPISODIC,
                    occurred_at=occurred_at,
                    ended_at=ended_at,
                    evidence_ids=evidence_ids,
                    idempotency_key=_prepared_idempotency_key(
                        adapter_version, video.video_id, segment.segment_id, "transcript"
                    ),
                )
            )


def _prepared_idempotency_key(
    adapter_version: str, video_id: str, segment_id: str, content_kind: str
) -> str:
    source = f"{adapter_version}:{video_id}:{segment_id}:{content_kind}".encode()
    return f"benchmark_{hashlib.sha256(source).hexdigest()}"


def benchmark_tenant_id(tenant_prefix: str, unit_id: str, run_id: str) -> str:
    """Build an isolated tenant so earlier queries cannot see a prior run's future."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    return TypeAdapter(Identifier).validate_python(f"{tenant_prefix}_{unit_id}_{run_id}")


async def wait_for_observation_job(
    memory: MindBridge,
    tenant_id: str,
    job_id: str,
    *,
    poll_interval_seconds: float = 1.0,
    timeout_seconds: float = 1_800.0,
) -> ObservationProcessingJobView:
    """Wait for durable success while allowing failed attempts to be retried."""
    if poll_interval_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("poll interval and timeout must be positive")
    last_state: JobState | None = None

    async def poll() -> ObservationProcessingJobView:
        nonlocal last_state
        while True:
            job = await memory.get_observation_job(tenant_id, job_id)
            last_state = job.state
            if job.state is JobState.SUCCEEDED:
                return job
            await asyncio.sleep(poll_interval_seconds)

    try:
        return await asyncio.wait_for(poll(), timeout_seconds)
    except asyncio.TimeoutError as error:
        state = last_state.value if last_state is not None else "unavailable"
        raise TimeoutError(
            f"observation job {job_id} did not succeed; last state was {state}"
        ) from error


async def ingest_media(
    memory: MindBridge,
    request: ObserveRequest,
    *,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
) -> tuple[str, ...]:
    """Submit one media observation and return its evidence once processing has succeeded.

    Every runner needs the same three steps in the same order, and needs them to stay in that
    order: evidence is only citable after the derived graph is durable.
    """
    receipt = await memory.observe(request)
    await wait_for_observation_job(
        memory,
        request.tenant_id,
        receipt.processing_job_id,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=processing_timeout_seconds,
    )
    return receipt.evidence_ids


def multiple_choice_query(
    question: str,
    choices: tuple[NonEmptyString, ...],
    *,
    rank_all: bool,
) -> str:
    """Format choices without exposing evaluation labels or evidence hints."""
    if not 2 <= len(choices) <= len(OPTION_LABELS):
        raise ValueError("multiple-choice query requires between two and ten choices")
    labels = OPTION_LABELS[: len(choices)]
    options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True))
    instruction = (
        f"Return only all {len(choices)} option labels from best to worst, separated by commas."
        if rank_all
        else "Return only the single best option label."
    )
    return f"{question}\n\nOptions:\n{options}\n\n{instruction}"


def parse_option_ranking(
    answer: str | None,
    choices: tuple[NonEmptyString, ...],
) -> tuple[int, ...]:
    """Parse a constrained model response, refusing ambiguous prose."""
    if not 2 <= len(choices) <= len(OPTION_LABELS):
        raise ValueError("multiple-choice response requires between two and ten choices")
    if answer is None:
        return ()
    normalized = " ".join(answer.split()).casefold()
    exact_matches = tuple(
        index
        for index, choice in enumerate(choices)
        if normalized == " ".join(choice.split()).casefold()
    )
    if len(exact_matches) == 1:
        return exact_matches

    option_labels = OPTION_LABELS[: len(choices)]
    label_pattern = f"[A-{option_labels[-1]}]"
    labels = re.findall(rf"\b{label_pattern}\b", answer.upper())
    if not labels or len(set(labels)) != len(labels):
        return ()
    residual = re.sub(rf"\b{label_pattern}\b", "", answer, flags=re.IGNORECASE)
    residual = _ALLOWED_RESPONSE_WORDS.sub("", residual)
    if re.search(r"[A-Za-z0-9]", residual):
        return ()
    return tuple(option_labels.index(label) for label in labels)


def dot_product(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Cosine similarity for the L2-normalized vectors these runners compare."""
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
