"""Staging ATM-Bench's raw archive, so a `raw` run has bytes to observe rather than captions.

One archive, one tenant, 4,292 images and videos on one clock. Three things are specific to this
release and none of them is optional.

**The archive's own capture time travels in the manifest.** `atm_bench_runner._observe_media`
takes an observation's `occurred_at` straight from `media_object.created_at`, and
`atm_cli._require_one_clock` refuses a manifest whose `created_at` disagrees with the release's
own record. `STAGED_AT` -- right for every other producer, because when a clip was uploaded is
not a fact any score depends on -- would put a three-and-a-half-year archive at the epoch, and
be refused for it. So each object carries `atm_capture_time` of its own stem, which is where
`AtmSgmRecord.occurred_at` comes from as well, and which is as deterministic as a constant.

**The bytes are staged as they are.** `prepare_m3` cuts because one M3-Bench video is 2 GB and
its schema wants 30-second clips; ATM's videos run for seconds and its schema wants exactly one
object per archive item, keyed by the official stem. Re-encoding a 3-second video at the
deployment's one frame per second would replace it with three stills, and an image has no span
to cut at all.

**The file each item lives in is release-supplied.** The two batch-results files name it --
`data/raw_memory/image/20220703_210745.jpg` -- and `atm_bench` keeps only the stem, because the
runner addresses an item by ID and never opens it. Staging is the step that does, so the path is
read here and resolved through `within`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from mindbridge.benchmarks.cli_common import report, select_by_id
from mindbridge.benchmarks.runtime import benchmark_tenant_id
from mindbridge.benchmarks.staging import (
    PrepareRequest,
    key_component,
    staging,
    within,
    write_manifest,
)
from mindbridge.core import MediaKind

ATM_RELEASE = "atm-bench"
"""The directory ATM-Bench occupies under `--benchmarks-root`, and its key in `RELEASES`.

The only ATM path this module fixes. Every other one is read out of the release's own batch
results, so a layout this repository guessed wrong is a wrong guess in one place.
"""

_MEDIA_PATH_FIELDS = {"image_path": MediaKind.IMAGE, "video_path": MediaKind.VIDEO}
"""Which key of a batch-results entry names its file, matching `atm_bench._sgm_record`."""


def prepare_atm(request: PrepareRequest) -> None:
    """Stage every archive item this run's questions rest on, keyed by its official stem."""
    from mindbridge.benchmarks.atm_bench import (
        atm_capture_time,
        atm_evidence_kind,
        load_atm_bench,
        load_atm_sgm,
    )
    from mindbridge.benchmarks.atm_bench_runner import (
        AtmPreparedArchive,
        AtmPreparedMedia,
        validate_prepared_atm,
    )
    from mindbridge.benchmarks.atm_cli import _parse_arguments
    from mindbridge.benchmarks.releases import ensure_media

    arguments = _parse_arguments(list(request.argv), None)
    files = _require_an_arm_that_reads_media(
        arguments.media_source,
        prepared_media_path=arguments.prepared_media_path,
        sgm_image_path=arguments.sgm_image_path,
        sgm_video_path=arguments.sgm_video_path,
    )
    questions = select_by_id(
        load_atm_bench(arguments.dataset_path),
        arguments.question_ids,
        key=lambda question: question.question_id,
        label="selected ATM-Bench question IDs",
        limit=arguments.limit,
    )
    if not questions:
        raise ValueError("ATM-Bench selection must not be empty")
    records = (
        *load_atm_sgm(files.sgm_image_path),
        *load_atm_sgm(files.sgm_video_path),
    )
    paths = {
        **_release_paths(files.sgm_image_path),
        **_release_paths(files.sgm_video_path),
    }
    # Narrowed exactly as `atm_cli._archive_for_run` narrows the emails and the schema-guided
    # records beside them, and for the same reason: one tenant holds the whole archive and every
    # question is asked of it, so `--limit 2` still staged 4,292 items. `niah_evidence_ids` is
    # kept alongside the gold ones because that is the distractor set this release built for the
    # question; a haystack of one item is not the question that was written.
    cited = {
        evidence_id
        for question in questions
        for evidence_id in (*question.evidence_ids, *question.niah_evidence_ids)
    }
    selected = (
        records
        if arguments.limit is None
        else tuple(record for record in records if record.media_id in cited)
    )
    # The runner's own refusal, reached before anything is uploaded rather than after. Its own
    # `validate_prepared_atm` wants an archive that does not exist yet, so the same verdict is
    # taken on the selection; the built archive goes through that function below regardless.
    missing = {
        evidence_id
        for question in questions
        for evidence_id in question.evidence_ids
        if atm_evidence_kind(evidence_id) == "media"
    } - {record.media_id for record in selected}
    if missing:
        raise ValueError(f"missing prepared ATM-Bench media: {', '.join(sorted(missing))}")
    sources = {
        record.media_id: _source(request.benchmarks_root, paths, record.media_id)
        for record in selected
    }
    absent = tuple(media_id for media_id, path in sources.items() if not path.exists())
    if absent:
        # On absence rather than eagerly, because `ensure_media` re-derives an acquired media
        # set before it looks at the disk -- so an operator who placed the files by hand would pay
        # again for the fetch they do not need. Named file by file for the same reason
        # `--limit` exists: a scoped run wants a handful of the 3.1 GB of raw media, and the
        # paths are the release's own, already refused by `_source` if one of them climbs.
        ensure_media(
            ATM_RELEASE,
            root=request.benchmarks_root,
            only=tuple(paths[media_id] for media_id in absent),
            download=request.download,
        )
    _require_every_source_present(sources)
    tenant_id = benchmark_tenant_id(arguments.tenant_prefix, "archive", arguments.run_id)
    report(f"  archive: {len(selected)} media items -> {tenant_id}", quiet=request.quiet)
    target = staging()
    staged: list[AtmPreparedMedia] = []
    for record in selected:
        source = sources[record.media_id]
        stem = key_component(record.media_id, label="ATM-Bench media stem")
        media_object = target.stage(
            tenant_id=tenant_id,
            # The source suffix is kept: `MediaObjectInput` refuses a kind its own URI extension
            # contradicts, and this is a mixed archive.
            key=f"atm/{stem}{source.suffix}",
            content=source.read_bytes(),
            kind=record.media_kind,
            media_object_id=record.media_id,
            duration_ms=(
                None if record.duration_seconds is None else int(record.duration_seconds * 1_000)
            ),
        )
        staged.append(
            AtmPreparedMedia(
                media_id=record.media_id,
                media_object=media_object.model_copy(
                    update={"created_at": atm_capture_time(record.media_id)}
                ),
            )
        )
    prepared = AtmPreparedArchive(media=tuple(staged))
    validate_prepared_atm(questions, prepared, records, media_source="raw")
    write_manifest(files.prepared_media_path, prepared)


class _Files(NamedTuple):
    """The three paths preparing an ATM archive cannot proceed without."""

    prepared_media_path: Path
    sgm_image_path: Path
    sgm_video_path: Path


def _require_an_arm_that_reads_media(
    media_source: str,
    *,
    prepared_media_path: Path | None,
    sgm_image_path: Path | None,
    sgm_video_path: Path | None,
) -> _Files:
    """Refuse to stage an archive for an arm that will not look at it, or without its inventory.

    `PREPARERS` is keyed by benchmark, and `atm-main-sgm`/`atm-hard-sgm` are the same benchmark:
    they pass `--media-source sgm` and ingest the official schema-guided text instead. The
    catalog gives them no `--prepared-media`, but that absence is exactly what makes
    `suite._prepared_media_arguments` append one, so registering a producer for `atm` aims it at
    all four tasks unless the sweep gates on what the task itself declares. Until it does, this
    is a loud stop rather than 3 GB uploaded into a tenant that ingests captions.

    Both batch-results files are required because they are the archive's inventory: they name
    each item's file, and `_require_one_clock` refuses a raw run that stages a video no record
    gives a duration for.
    """
    if media_source != "raw":
        raise ValueError(
            f"ATM-Bench --media-source {media_source!r} ingests the release's own schema-guided "
            "text and reads no prepared media; gate preparation on the task's own "
            "--media-source before registering a producer for this benchmark"
        )
    if prepared_media_path is None:
        raise ValueError("preparing ATM-Bench media needs --prepared-media to write")
    if sgm_image_path is None or sgm_video_path is None:
        raise ValueError(
            "preparing ATM-Bench media needs --sgm-image and --sgm-video: the batch results name "
            "each archive item's file, and a staged video without a record has no duration"
        )
    return _Files(prepared_media_path, sgm_image_path, sgm_video_path)


def _require_every_source_present(sources: dict[str, Path]) -> None:
    """Refuse before uploading anything if the release's raw media half is not on disk."""
    absent = sorted(str(path) for path in sources.values() if not path.exists())
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} ATM-Bench archive files are absent, the first being {absent[0]}; "
            "they are the raw media half of the Jingbiao/ATM-Bench release"
        )


def _release_paths(batch_results_path: Path) -> dict[str, str]:
    """Map each archive item's official stem to the file the release names for it."""
    raw = json.loads(batch_results_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("results", raw)
    if not isinstance(raw, list) or not raw:
        raise ValueError("ATM-Bench batch results must be a non-empty list")
    paths: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("ATM-Bench batch results entries must be objects")
        field = next((name for name in _MEDIA_PATH_FIELDS if name in item), None)
        if field is None:
            raise ValueError("ATM-Bench batch results entry names no media path")
        value = str(item[field])
        paths[Path(value).stem] = value
    return paths


def _source(root: Path, paths: dict[str, str], media_id: str) -> Path:
    """Resolve one item's file, refusing a release-supplied path that leaves the corpus."""
    if media_id not in paths:
        raise KeyError(f"ATM-Bench batch results name no file for {media_id}")
    return within(root, ATM_RELEASE, paths[media_id])
