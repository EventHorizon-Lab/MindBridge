"""Produce the prepared-media manifests the media benchmarks read.

These manifests name clips that already live in object storage, which is why they used to be
made by hand: MindBridge has no downloader, clipper, or uploader of its own, and the shapes
encode each benchmark's own clock. This module is that missing step, for the shapes more than one
benchmark shares.

Two things about it are forced rather than chosen.

**A manifest is specific to one run.** `tenant_s3_object_key` accepts a media URI only under
`tenants/<tenant_id>/`, and a benchmark tenant is `<prefix>_<unit>_<run_id>`. So the same clips
staged for a different `--run-id` are unreadable, and preparation belongs inside the run rather
than beside the corpus. `--limit` is what keeps that affordable: it bounds the units prepared as
well as the units answered.

**The selection has to be the runner's own.** Preparing a unit the run will not read wastes an
upload; missing one it will read fails the run. So each producer parses the task's argv with the
runner's own parser and selects with the runner's own helper, rather than keeping a second copy
of rules like Video-MME's duration filter.

Clips are cut by `mindbridge.media.clipping.cut_clips` -- the encoder the product itself stores
evidence with, so what a benchmark ingests is what the product would have produced, down to the
frame sampling and the audio track. It seeks per span, so a 30-minute source costs the length of
the segments taken from it rather than one full decode each.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mindbridge.benchmarks.artifacts import write_text_atomically
from mindbridge.benchmarks.cli_common import report, select_by_id
from mindbridge.benchmarks.runtime import benchmark_tenant_id
from mindbridge.configuration import (
    configuration_source,
    optional_environment_value,
    require_environment_value,
)
from mindbridge.contracts import MediaObjectInput
from mindbridge.core import MediaKind

STAGED_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)
"""The `created_at` every staged object carries.

A real timestamp would change the manifest's digest on every preparation, and that digest is
what a run manifest pins. What the object was created at is not a fact any score depends on;
that two preparations of the same clips agree is.
"""

SEGMENT_SECONDS = 30
"""The official split every shape here uses, and what M3-Bench's schema requires exactly."""


@dataclass(frozen=True, slots=True)
class _Staging:
    """One object store to upload into, resolved from the deployment's own configuration."""

    bucket: str
    client: Any

    def stage(
        self,
        *,
        tenant_id: str,
        key: str,
        content: bytes,
        kind: MediaKind,
        media_object_id: str,
        duration_ms: int | None = None,
    ) -> MediaObjectInput:
        """Upload one object into a tenant's own prefix and describe it as the runners expect."""
        object_key = f"tenants/{tenant_id}/{key}"
        self.client.put_object(Bucket=self.bucket, Key=object_key, Body=content)
        return MediaObjectInput(
            media_object_id=media_object_id,
            kind=kind,
            uri=f"s3://{self.bucket}/{object_key}",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            created_at=STAGED_AT,
            duration_ms=duration_ms,
        )


def staging() -> _Staging:
    """Resolve the bucket the deployment reads, from the same file and variables it reads.

    Not the product's own `S3MediaAccess`: AGENTS.md keeps this package to the public SDK and
    contracts, and that class lives in `infrastructure`. What is shared is the configuration, so
    a benchmark cannot stage into a bucket the deployment will not look in.
    """
    source = configuration_source()
    bucket = require_environment_value(source, "MINDBRIDGE_OBJECT_STORAGE_BUCKET")
    endpoint_url = optional_environment_value(source, "MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL")
    import boto3

    return _Staging(bucket, boto3.client("s3", endpoint_url=endpoint_url))


def video_segments(
    source: Path,
    *,
    seconds: int = SEGMENT_SECONDS,
    limit: int | None = None,
) -> Iterator[tuple[int, int, bytes]]:
    """Cut one source video into contiguous segments, yielding (index, duration_ms, bytes).

    Every segment but the last is exactly `seconds` long, which M3-Bench's schema requires and
    the other shapes are content with. The source is read once and held, then cut span by span:
    `cut_clips` seeks to each span, so this is proportional to what is taken rather than to the
    file per segment.
    """
    from mindbridge.media.clipping import ClipRequest, cut_clips

    duration_ms = _duration_ms(source)
    content = source.read_bytes()
    span_ms = seconds * 1_000
    total = -(-duration_ms // span_ms)
    for index in range(total if limit is None else min(total, limit)):
        start_ms = index * span_ms
        end_ms = min(start_ms + span_ms, duration_ms)
        if end_ms <= start_ms:
            return
        # `cut_clips` keeps the frame at `end_ms` as well: its span is closed at both ends, which
        # is right for evidence -- a span pointing at an instant deserves the frame there -- and
        # wrong for a split. Asking for the millisecond before the next segment starts is what
        # makes these segments disjoint. Left closed, segment 0 of a 30-second split encoded 31
        # seconds and shared its last second with segment 1, so a benchmark that withholds
        # clips after a question's timestamp would have leaked a second of the future into the
        # clip before it.
        clips = cut_clips(
            content,
            ClipRequest(kind=MediaKind.VIDEO, start_ms=start_ms, end_ms=end_ms - 1),
        )
        yield index, end_ms - start_ms, clips[0].content


def _duration_ms(source: Path) -> int:
    """Read a source's duration without decoding it."""
    import av

    with av.open(str(source)) as container:
        if container.duration is None:
            raise ValueError(f"{source} declares no duration, so it cannot be split into segments")
        return int(container.duration / 1_000)


def image_media_objects(
    staging_target: _Staging,
    images: Sequence[tuple[str, Path]],
    *,
    tenant_id: str,
    key_prefix: str,
    quiet: bool,
) -> tuple[tuple[str, MediaObjectInput], ...]:
    """Stage one unit's images, returning each with the release-relative key that references it."""
    staged: list[tuple[str, MediaObjectInput]] = []
    for media_object_id, path in images:
        staged.append(
            (
                media_object_id,
                staging_target.stage(
                    tenant_id=tenant_id,
                    key=f"{key_prefix}/{path.name}",
                    content=path.read_bytes(),
                    kind=MediaKind.IMAGE,
                    media_object_id=media_object_id,
                ),
            )
        )
    report(f"  staged {len(staged)} images into {tenant_id}", quiet=quiet)
    return tuple(staged)


@dataclass(frozen=True, slots=True)
class PrepareRequest:
    """What one producer needs: the argv its runner will get, and where the corpus lives."""

    argv: tuple[str, ...]
    benchmarks_root: Path
    quiet: bool


M3_TIMELINE_ORIGIN = datetime(2000, 1, 1, tzinfo=timezone.utc)
"""M3-Bench annotations are relative to the start of their own video, so the origin is arbitrary.

Fixed rather than the wall clock for the same reason `STAGED_AT` is: two preparations of one
video have to produce the same manifest.
"""


def prepare_mem_gallery(request: PrepareRequest) -> None:
    """Stage the images every selected topic references, keyed as the release references them."""
    from mindbridge.benchmarks.mem_gallery import load_mem_gallery
    from mindbridge.benchmarks.mem_gallery_cli import _parse_arguments
    from mindbridge.benchmarks.mem_gallery_runner import (
        MemGalleryPreparedImage,
        MemGalleryPreparedImages,
    )

    arguments = _parse_arguments(list(request.argv), None)
    topics = select_by_id(
        load_mem_gallery(arguments.dataset_path),
        arguments.topics,
        key=lambda topic: topic.topic,
        label="selected Mem-Gallery topics",
        limit=arguments.limit,
    )
    target = staging()
    images: list[MemGalleryPreparedImage] = []
    for topic in topics:
        tenant_id = benchmark_tenant_id(arguments.tenant_prefix, topic.topic, arguments.run_id)
        references = _mem_gallery_references(topic)
        report(
            f"  {topic.topic}: {len(references)} images -> {tenant_id}",
            quiet=request.quiet,
        )
        for image_key, media_object_id in references.items():
            path = (arguments.dataset_path / image_key).resolve()
            images.append(
                MemGalleryPreparedImage(
                    image_key=image_key,
                    media_object=target.stage(
                        tenant_id=tenant_id,
                        key=f"mem-gallery/{topic.topic}/{path.name}",
                        content=path.read_bytes(),
                        kind=MediaKind.IMAGE,
                        media_object_id=media_object_id,
                    ),
                )
            )
    _write(arguments.prepared_images_path, MemGalleryPreparedImages(images=tuple(images)))


def _mem_gallery_references(topic: object) -> dict[str, str]:
    """Every image key one topic needs, mapped to the ID its media object must carry.

    Read off the topic rather than by walking the image directory: `validate_mem_gallery_images`
    refuses a run whose manifest misses any referenced key, and the directory holds images no
    topic references. A dialogue round's ID is the official one the release assigns; a question
    image has none, so its file name gives one in the same shape.
    """
    references: dict[str, str] = {}
    for session in topic.sessions:  # type: ignore[attr-defined]
        for round_ in session.rounds:
            if round_.image_path is not None and round_.image_id is not None:
                references[round_.image_path] = round_.image_id
    for question in topic.questions:  # type: ignore[attr-defined]
        if question.question_image_path is not None:
            name = Path(question.question_image_path).stem
            references.setdefault(question.question_image_path, name.replace("_", ":", 1))
    return references


def prepare_m3(request: PrepareRequest) -> None:
    """Cut each selected official video into the 30-second clips M3-Bench's schema requires."""
    from mindbridge.benchmarks.m3_bench import load_m3_bench
    from mindbridge.benchmarks.m3_cli import _parse_arguments
    from mindbridge.benchmarks.m3_runner import M3PreparedClip, M3PreparedVideo

    arguments = _parse_arguments(list(request.argv), None)
    videos = select_by_id(
        load_m3_bench(arguments.dataset_path),
        arguments.video_ids,
        key=lambda video: video.video_id,
        label="M3-Bench video IDs",
        limit=arguments.limit,
    )
    target = staging()
    prepared: list[M3PreparedVideo] = []
    for video in videos:
        source = (
            request.benchmarks_root
            / "m3-bench"
            / "videos"
            / arguments.subset
            / f"{video.video_id}.mp4"
        )
        if not source.exists():
            raise FileNotFoundError(
                f"M3-Bench source video {source} is absent; it is part of the "
                "ByteDance-Seed/M3-Bench release and about 2 GB per video"
            )
        tenant_id = benchmark_tenant_id(arguments.tenant_prefix, video.video_id, arguments.run_id)
        clips = tuple(
            M3PreparedClip(
                clip_index=index,
                media_object=target.stage(
                    tenant_id=tenant_id,
                    key=f"m3/{video.video_id}/{index}.mp4",
                    content=content,
                    kind=MediaKind.VIDEO,
                    media_object_id=f"m3_{video.video_id}_{index}",
                    duration_ms=duration_ms,
                ),
            )
            for index, duration_ms, content in video_segments(source)
        )
        report(f"  {video.video_id}: {len(clips)} clips -> {tenant_id}", quiet=request.quiet)
        prepared.append(
            M3PreparedVideo(
                video_id=video.video_id,
                timeline_origin=M3_TIMELINE_ORIGIN,
                clips=clips,
            )
        )
    _write(arguments.prepared_media_path, prepared)


@dataclass(frozen=True, slots=True)
class Producer:
    """One benchmark's prepared-media producer, and the flag whose file it writes."""

    flag: str
    produce: Callable[[PrepareRequest], None]


PREPARERS: dict[str, Producer] = {
    "mem-gallery": Producer("--prepared-images", prepare_mem_gallery),
    "m3": Producer("--prepared-media", prepare_m3),
}
"""The benchmarks whose prepared media this module can produce.

A benchmark absent from this table still needs its manifest made out-of-band, and
`mindbridge-bench eval --list-tasks` says so by naming the file it wants.
"""


def _write(path: Path, prepared: object) -> None:
    """Write one manifest, in whichever of the two shapes its runner reads."""
    if isinstance(prepared, list):
        body = "[\n" + ",\n".join(item.model_dump_json(indent=2) for item in prepared) + "\n]\n"
    else:
        body = prepared.model_dump_json(indent=2) + "\n"  # type: ignore[attr-defined]
    write_text_atomically(path, body)
