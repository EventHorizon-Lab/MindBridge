"""Which benchmarks can have their prepared media produced, and the producer that does it.

These manifests name clips that already live in object storage, which is why they used to be
made by hand: MindBridge had no downloader, clipper, or uploader of its own, and the shapes
encode each benchmark's own clock. This module is the table that closes that gap; the operations
every producer shares live in `staging.py`, and the producers themselves in the `prepare_*`
modules this imports.

A benchmark registered here must not also name a static manifest path in `task_catalog.py`.
`suite._prepared_media_arguments` substitutes a per-run path only for a task that does not
already carry the flag, so a catalog entry naming one both defeats the substitution and, on the
second run, finds the first run's file and skips preparation as already-done -- leaving the run
to ingest objects under a tenant it cannot address.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from mindbridge.benchmarks.cli_common import flag_value, report, select_by_id
from mindbridge.benchmarks.prepare_archive import prepare_atm
from mindbridge.benchmarks.prepare_lifelong import prepare_mm_lifelong
from mindbridge.benchmarks.prepare_streams import (
    prepare_egolife,
    prepare_egomem,
    prepare_supermemory,
)
from mindbridge.benchmarks.prepare_video import (
    prepare_egotempo,
    prepare_video_mme,
    prepare_video_mme_v2,
)
from mindbridge.benchmarks.releases import ensure_media
from mindbridge.benchmarks.runtime import benchmark_tenant_id
from mindbridge.benchmarks.staging import (
    PrepareRequest,
    key_component,
    staging,
    video_segments,
    within,
    write_manifest,
)
from mindbridge.core import MediaKind

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
    if not topics:
        raise ValueError("Mem-Gallery selection must not be empty")
    target = staging()
    images: list[MemGalleryPreparedImage] = []
    for topic in topics:
        topic_key = key_component(topic.topic, label="Mem-Gallery topic")
        tenant_id = benchmark_tenant_id(arguments.tenant_prefix, topic.topic, arguments.run_id)
        references = _mem_gallery_references(topic)
        report(
            f"  {topic.topic}: {len(references)} images -> {tenant_id}",
            quiet=request.quiet,
        )
        for image_key, media_object_id in references.items():
            path = within(request.benchmarks_root, str(arguments.dataset_path), image_key)
            images.append(
                MemGalleryPreparedImage(
                    image_key=image_key,
                    media_object=target.stage(
                        tenant_id=tenant_id,
                        key=f"mem-gallery/{topic_key}/{path.name}",
                        content=path.read_bytes(),
                        kind=MediaKind.IMAGE,
                        media_object_id=media_object_id,
                    ),
                )
            )
    write_manifest(arguments.prepared_images_path, MemGalleryPreparedImages(images=tuple(images)))


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
    from mindbridge.benchmarks.m3_cli import _parse_arguments, _validate_subset
    from mindbridge.benchmarks.m3_runner import M3PreparedClip, M3PreparedVideo

    arguments = _parse_arguments(list(request.argv), None)
    videos = select_by_id(
        load_m3_bench(arguments.dataset_path),
        arguments.video_ids,
        key=lambda video: video.video_id,
        label="M3-Bench video IDs",
        limit=arguments.limit,
    )
    # The runner's own check, run here too. Selecting the same way it does is not the same as
    # accepting the same input: `--subset web` pointed at `robot.json` selects fine and is then
    # refused by the runner, so without this a producer cuts and uploads every video -- the robot
    # split averages 1.2 GB a file -- before anything says the subset is wrong.
    _validate_subset(videos, arguments.subset)
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
            # On absence rather than up front, and narrowed to the one video that is missing.
            # Both halves matter here and neither is an optimisation. `m3-web` is acquired rather
            # than downloaded -- its 920 videos are the `video_url` of each annotation, not files
            # the release ships -- so an eager call would re-derive a corpus the operator had
            # already filled in, and an unnarrowed one would fetch all 920 to cut the one this
            # run selected. `m3-robot` had the same second problem against the Hub: `--limit 1`
            # pulled all 100 robot videos, 117 GiB, to read one of them.
            ensure_media(
                f"m3-{arguments.subset}",
                root=request.benchmarks_root,
                only=(f"videos/{arguments.subset}/{video.video_id}.mp4",),
                announce=None if request.quiet else partial(report, quiet=False),
                download=request.download,
            )
            if not source.exists():
                # A size per subset, because they are an order apart and one figure for both
                # was wrong about whichever one you were running. `robot` is 100 Hub files
                # totalling 117 GiB, from `repo_info(files_metadata=True)` at the pinned
                # revision: mean 1.2 GiB but a range of 0.28 to 3.13, so no single per-file
                # figure describes it and the total is the number worth quoting. An earlier
                # "about 2 GB each" here was 1.7x high. `web` is 920 downloads of about
                # 100 MB, measured from real ones at the 360p the acquirer selects -- an earlier
                # 20 MB here came from `yt-dlp`'s `filesize_approx`, which is bitrate times
                # duration and was 26x low on a video that arrived as 712 MB. Do not restate a
                # figure from that field.
                whole = (
                    "100 files totalling 117 GiB"
                    if arguments.subset == "robot"
                    else "920 downloads of about 100 MB"
                )
                raise FileNotFoundError(
                    f"M3-Bench source video {source} is absent; the {arguments.subset} subset is "
                    f"{whole} from the ByteDance-Seed/M3-Bench release"
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
    write_manifest(arguments.prepared_media_path, prepared)


@dataclass(frozen=True, slots=True)
class Producer:
    """One benchmark's prepared-media producer, and the flag whose file it writes."""

    flag: str
    produce: Callable[[PrepareRequest], None]
    applies: Callable[[Sequence[str]], bool] | None = None
    """Whether this task's arguments describe an arm that reads the manifest at all.

    `PREPARERS` is keyed by benchmark, but ATM-Bench's `sgm` arms ingest the release's own
    pre-processed captions and never open a prepared-media manifest. Absence of the flag cannot
    stand in for that: absence is precisely what makes the sweep *append* it, after which
    `_prepared_manifest_path` reads it back and the producer stages about 3 GB neither arm will
    read. So an arm that does not want media has to say so, rather than be inferred.
    """


PREPARERS: dict[str, Producer] = {
    "mem-gallery": Producer("--prepared-images", prepare_mem_gallery),
    "m3": Producer("--prepared-media", prepare_m3),
    "video-mme": Producer("--prepared-media", prepare_video_mme),
    "video-mme-v2": Producer("--prepared-media", prepare_video_mme_v2),
    "egotempo": Producer("--prepared-media", prepare_egotempo),
    "egolife": Producer("--prepared-media", prepare_egolife),
    "egomem": Producer("--prepared-media", prepare_egomem),
    "supermemory": Producer("--prepared-media", prepare_supermemory),
    "mm-lifelong": Producer("--prepared-media", prepare_mm_lifelong),
    "atm": Producer(
        "--prepared-media",
        prepare_atm,
        applies=lambda argv: flag_value(argv, "--media-source") != "sgm",
    ),
}
"""The benchmarks whose prepared media this module can produce.

A benchmark absent from this table still needs its manifest made out-of-band, and
`mindbridge-bench eval --list-tasks` says so by naming the file it wants.
"""
