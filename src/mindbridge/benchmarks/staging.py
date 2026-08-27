"""Uploading a benchmark's media into the deployment's own object store.

Split out of `prepare.py` when the producers stopped fitting in one module: the table of which
benchmark has a producer is one thing, and the handful of operations every producer performs --
stage an object, cut a source into segments, resolve a release-supplied path -- is another.
Keeping them together made every producer module import the table that imports it.

Two things about staging are forced rather than chosen.

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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mindbridge.benchmarks.artifacts import write_text_atomically
from mindbridge.benchmarks.cli_common import report
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
class Staging:
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


def staging() -> Staging:
    """Resolve the bucket the deployment reads, from the same file and variables it reads.

    Not the product's own `S3MediaAccess`: AGENTS.md keeps this package to the public SDK and
    contracts, and that class lives in `infrastructure`. What is shared is the configuration, so
    a benchmark cannot stage into a bucket the deployment will not look in.
    """
    source = configuration_source()
    bucket = require_environment_value(source, "MINDBRIDGE_OBJECT_STORAGE_BUCKET")
    endpoint_url = optional_environment_value(source, "MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL")
    import boto3

    return Staging(bucket, boto3.client("s3", endpoint_url=endpoint_url))


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

    duration_ms = media_duration_ms(source)
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


def media_duration_ms(source: Path) -> int:
    """Read a source's duration without decoding it."""
    import av

    with av.open(str(source)) as container:
        if container.duration is None:
            raise ValueError(f"{source} declares no duration, so it cannot be split into segments")
        return int(container.duration / 1_000)


def image_media_objects(
    staging_target: Staging,
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
    download: bool = True
    """Whether a producer may fetch the media it is missing, or must refuse.

    `--no-download` is documented as refusing to fetch rather than fetching, and preparation is
    where a media benchmark's bytes are actually obtained -- so without this the flag governed
    the annotations and let the 94 GiB behind them through.
    """


def within(root: Path, *parts: str) -> Path:
    """Resolve a path a release names, refusing one that leaves the corpus root.

    Mem-Gallery's image keys are relative paths out of its own annotations -- `../image/<topic>/`,
    which is why the join is needed at all -- and this command now downloads those annotations
    itself. A key of `../../../../etc/passwd` would otherwise be read and uploaded into the
    deployment's bucket, so the release gets to name a file inside the corpus and nothing else.
    """
    resolved = (root / Path(*parts)).resolve()
    corpus = root.resolve()
    if not resolved.is_relative_to(corpus):
        raise ValueError(
            f"{'/'.join(parts)} resolves to {resolved}, outside the corpus at {corpus}; "
            "a release may only name files inside it"
        )
    return resolved


def key_component(value: str, *, label: str) -> str:
    """Refuse a release-supplied string that would not stay one component of an object key.

    A topic containing `/` or `..` is interpolated straight into the S3 key otherwise, which
    writes outside the prefix the manifest claims and can silently collide across topics.
    """
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} {value!r} is not usable as one object-key component")
    return value


def write_manifest(path: Path, prepared: object) -> None:
    """Write one manifest, in whichever of the two shapes its runner reads."""
    if isinstance(prepared, list):
        body = "[\n" + ",\n".join(item.model_dump_json(indent=2) for item in prepared) + "\n]\n"
    else:
        body = prepared.model_dump_json(indent=2) + "\n"  # type: ignore[attr-defined]
    write_text_atomically(path, body)
