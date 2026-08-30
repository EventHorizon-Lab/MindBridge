"""Normalize official benchmark releases into one local evaluation protocol."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar, cast

from mindbridge.benchmarks.egomem_reason import EgoMemReasonQuestion
from mindbridge.benchmarks.egotempo import EgoTempoQuestion
from mindbridge.benchmarks.mem_gallery import (
    MemGalleryQuestion,
    MemGalleryRound,
    MemGallerySession,
)
from mindbridge.benchmarks.supermemory_vqa import SuperMemoryQuestion
from mindbridge.benchmarks.task_catalog import TaskSpec

ScoreKind = Literal["choice", "text", "submission"]
Limit: TypeAlias = int | float | None
_T = TypeVar("_T")
_MEDIA_SUFFIXES = frozenset(
    {
        ".aac",
        ".flac",
        ".jpeg",
        ".jpg",
        ".m4a",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".png",
        ".wav",
        ".webm",
    }
)


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """One item added to a unit's physically isolated memory."""

    source_id: str
    content: tuple[str | Path, ...]
    start_seconds: float = 0.0
    end_seconds: float | None = None
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.content:
            raise ValueError("memory items need a source ID and content")
        if (
            not math.isfinite(self.start_seconds)
            or self.start_seconds < 0
            or (
                self.end_seconds is not None
                and (not math.isfinite(self.end_seconds) or self.end_seconds <= self.start_seconds)
            )
        ):
            raise ValueError("memory item interval is invalid")
        if self.occurred_at is not None and (
            self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None
        ):
            raise ValueError("memory item event time must include a timezone")
        if self.occurred_end is not None and (
            self.occurred_end.tzinfo is None
            or self.occurred_end.utcoffset() is None
            or self.occurred_at is None
            or self.occurred_end <= self.occurred_at
        ):
            raise ValueError("memory item event end must follow a timezone-aware event time")


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """One question and the label information kept outside MindBridge."""

    question_id: str
    content: tuple[str | Path, ...]
    references: tuple[str, ...] = ()
    expected_choice: str | None = None
    score_kind: ScoreKind = "text"
    cutoff_seconds: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    reference_at: datetime | None = None
    source_question: str | None = None

    def __post_init__(self) -> None:
        if not self.question_id.strip() or not self.content:
            raise ValueError("evaluation questions need an ID and content")
        if self.score_kind == "choice" and (
            self.expected_choice is None or self.expected_choice not in "ABCDEFGHIJ"
        ):
            raise ValueError("choice questions need an expected A-J option")
        if self.score_kind == "text" and not self.references:
            raise ValueError("text questions need at least one reference")
        if self.cutoff_seconds is not None and (
            not math.isfinite(self.cutoff_seconds) or self.cutoff_seconds < 0
        ):
            raise ValueError("question cutoff must be a non-negative finite number")
        if self.reference_at is not None and (
            self.reference_at.tzinfo is None or self.reference_at.utcoffset() is None
        ):
            raise ValueError("question reference time must include a timezone")
        if self.source_question is not None and not self.source_question.strip():
            raise ValueError("source question must not be blank")


@dataclass(frozen=True, slots=True)
class EvalUnit:
    """One independent corpus that owns one physical ``data_dir``."""

    unit_id: str
    memories: tuple[MemoryItem, ...]
    questions: tuple[EvalQuestion, ...]

    def __post_init__(self) -> None:
        if not self.unit_id.strip() or not self.questions:
            raise ValueError("evaluation units need an ID and questions")
        question_ids = tuple(question.question_id for question in self.questions)
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("evaluation question IDs must be unique within a unit")


@dataclass(frozen=True, slots=True)
class LoadedTask:
    """A pinned task ready for the evaluation engine."""

    spec: TaskSpec
    dataset_path: Path
    dataset_sha256: str
    units: tuple[EvalUnit, ...]
    input_sha256: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unit_ids = tuple(unit.unit_id for unit in self.units)
        if not unit_ids or len(set(unit_ids)) != len(unit_ids):
            raise ValueError("loaded task unit IDs must be non-empty and unique")

    @property
    def evaluation_sha256(self) -> str:
        if not self.input_sha256:
            return self.dataset_sha256
        payload = json.dumps(
            dict(sorted(self.input_sha256.items())),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class _ManifestPart:
    content: tuple[str | Path, ...]
    source_id: str
    start_seconds: float = 0.0
    end_seconds: float | None = None


class MediaResolver:
    """Resolve prepared manifest entries first, then exact local media names."""

    def __init__(
        self,
        task_name: str,
        root: Path | None,
        manifest: Mapping[str, object] | None,
        manifest_directory: Path | None,
    ) -> None:
        self.task_name = task_name
        self.root = root
        self._manifest = _task_manifest(manifest, task_name)
        self._manifest_directory = manifest_directory
        self._index: dict[str, list[Path]] | None = None

    def parts(
        self,
        unit_id: str,
        source_ids: Sequence[str] = (),
        *,
        relative_paths: Sequence[str] = (),
        allow_all: bool = False,
    ) -> tuple[MemoryItem, ...]:
        declared = self._manifest.get(unit_id)
        if declared is not None:
            return tuple(_memory_part(part) for part in self._parse_parts(unit_id, declared))
        if self.root is None:
            raise FileNotFoundError(
                f"{self.task_name} needs media for {unit_id}; pass --media-manifest or --media-root"
            )
        candidates = tuple(dict.fromkeys((*relative_paths, *source_ids)))
        paths = []
        for value in candidates:
            try:
                paths.append(self._resolve(value))
            except FileNotFoundError:
                if not allow_all:
                    raise
        if not paths and allow_all:
            paths.extend(self._scoped_media(candidates))
        if not paths:
            raise FileNotFoundError(f"{self.task_name} has no media for unit {unit_id}")
        return tuple(MemoryItem(Path(path).stem, (path,)) for path in paths)

    def path(self, value: str) -> Path:
        return self._resolve(value)

    def _parse_parts(self, unit_id: str, value: object) -> tuple[_ManifestPart, ...]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"media manifest unit {unit_id} must be a non-empty list")
        return tuple(self._parse_part(unit_id, index, part) for index, part in enumerate(value))

    def _parse_part(self, unit_id: str, index: int, value: object) -> _ManifestPart:
        if isinstance(value, str):
            path = self._manifest_path(value)
            return _ManifestPart((path,), path.stem)
        if not isinstance(value, dict):
            raise ValueError(f"media manifest part {unit_id}[{index}] must be text or an object")
        path_value = value.get("path")
        text_value = value.get("text")
        content: list[str | Path] = []
        if isinstance(text_value, str) and text_value.strip():
            content.append(text_value.strip())
        if isinstance(path_value, str) and path_value.strip():
            content.append(self._manifest_path(path_value))
        if not content:
            raise ValueError(f"media manifest part {unit_id}[{index}] needs path or text")
        start = _finite_number(value.get("start_seconds", 0.0), "start_seconds")
        end_value = value.get("end_seconds")
        end = None if end_value is None else _finite_number(end_value, "end_seconds")
        if start < 0 or (end is not None and end <= start):
            raise ValueError(f"media manifest part {unit_id}[{index}] has an invalid interval")
        source = value.get("source_id")
        source_id = (
            source.strip()
            if isinstance(source, str) and source.strip()
            else f"{unit_id}:{index:05d}"
        )
        return _ManifestPart(tuple(content), source_id, start, end)

    def _manifest_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            if self._manifest_directory is None:
                raise ValueError("relative media manifest paths need a manifest file directory")
            path = self._manifest_directory / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"prepared media does not exist: {path}")
        return path

    def _resolve(self, value: str) -> Path:
        if self.root is None:
            raise FileNotFoundError(value)
        supplied = Path(value)
        root = self.root.resolve()
        direct = tuple(
            candidate
            for candidate in ((root / supplied).resolve(), (root / supplied.name).resolve())
            if candidate.is_relative_to(root)
        )
        for candidate in direct:
            if candidate.is_file():
                return candidate
        key = supplied.with_suffix("").as_posix().casefold()
        matches = self._media_index().get(key, [])
        if not matches and supplied.stem.isdigit():
            matches = self._media_index().get(str(int(supplied.stem)), [])
        if not matches:
            raise FileNotFoundError(f"media not found under {self.root}: {value}")
        unique = tuple(dict.fromkeys(matches))
        if len(unique) != 1:
            raise ValueError(f"media name is ambiguous under {self.root}: {value}")
        return unique[0]

    def _media_index(self) -> dict[str, list[Path]]:
        if self._index is not None:
            return self._index
        index: dict[str, list[Path]] = {}
        for path in self._media_files():
            relative = path.relative_to(cast(Path, self.root)).with_suffix("").as_posix().casefold()
            keys = {relative, path.stem.casefold()}
            if path.stem.isdigit():
                keys.add(str(int(path.stem)))
            for key in keys:
                index.setdefault(key, []).append(path)
        self._index = index
        return index

    def _media_files(self) -> tuple[Path, ...]:
        if self.root is None or not self.root.is_dir():
            return ()
        root = self.root.resolve()
        return tuple(
            resolved
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix.casefold() in _MEDIA_SUFFIXES
            and (resolved := path.resolve()).is_relative_to(root)
        )

    def _scoped_media(self, candidates: Sequence[str]) -> tuple[Path, ...]:
        files = self._media_files()
        if not candidates:
            return files
        keys = {Path(value).stem.casefold() for value in candidates}
        return tuple(
            path
            for path in files
            if keys.intersection(
                part.casefold() for part in path.relative_to(cast(Path, self.root)).parts
            )
        )


def load_media_manifest(path: Path | None) -> tuple[Mapping[str, object] | None, Path | None]:
    """Load one versioned prepared-media manifest."""
    if path is None:
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("media manifest must be a JSON object")
    version = payload.get("version")
    if version != 1:
        raise ValueError(f"unsupported media manifest version: {version}")
    if not isinstance(payload.get("tasks"), dict):
        raise ValueError("media manifest tasks must be an object")
    return payload, path.expanduser().resolve().parent


def load_task(
    spec: TaskSpec,
    *,
    root: Path,
    dataset_path: Path | None = None,
    media_root: Path | None = None,
    media_manifest: Mapping[str, object] | None = None,
    manifest_directory: Path | None = None,
    limit: Limit = None,
    offset: int = 0,
    verify_digest: bool = True,
) -> LoadedTask:
    """Load, verify, and normalize one catalog task."""
    dataset = (dataset_path or spec.dataset_path(root)).expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"benchmark dataset does not exist: {dataset}")
    missing = tuple(root / path for path in spec.auxiliary if not (root / path).exists())
    if missing:
        raise FileNotFoundError(
            f"{spec.name} auxiliary input does not exist: {', '.join(map(str, missing))}"
        )
    digest = dataset_digest(dataset)
    if verify_digest and spec.digest is not None and digest != spec.digest:
        raise ValueError(
            f"{spec.name} dataset digest mismatch: expected {spec.digest}, found {digest}"
        )
    resolved_media = media_root or spec.media_root(root)
    resolver = MediaResolver(spec.name, resolved_media, media_manifest, manifest_directory)
    if limit is not None and limit != -1 and limit <= 0:
        raise ValueError("limit must be -1, a positive count, or a fraction between zero and one")
    if offset < 0:
        raise ValueError("offset must not be negative")
    units = _LOADERS[spec.name](spec, dataset, resolver, root, limit, offset)
    if not units or any(not unit.questions for unit in units):
        raise ValueError(f"{spec.name} produced no evaluation questions")
    inputs = {
        "dataset": digest,
        **{path: dataset_digest(root / path) for path in spec.auxiliary},
    }
    if dataset.is_dir():
        layout = "".join(
            f"{path.relative_to(dataset).as_posix()}\0{dataset_digest(path)}\n"
            for path in sorted(dataset.rglob("*"))
            if path.is_file()
        )
        inputs["dataset_layout"] = hashlib.sha256(layout.encode()).hexdigest()
    task_manifest = _task_manifest(media_manifest, spec.name)
    if task_manifest:
        encoded = json.dumps(
            task_manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        inputs["media_manifest"] = hashlib.sha256(encoded).hexdigest()
    inputs["memory"] = _memory_digest(units)
    return LoadedTask(spec, dataset, digest, units, inputs)


def dataset_digest(path: Path) -> str:
    """Hash one file or the concatenated file digests of a sorted directory tree."""
    if path.is_dir():
        files = tuple(item for item in sorted(path.rglob("*")) if item.is_file())
        if not files:
            raise ValueError(f"benchmark dataset contains no files: {path}")
        joined = "".join(dataset_digest(item) for item in files)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _memory_digest(units: Sequence[EvalUnit]) -> str:
    file_digests: dict[Path, str] = {}

    def identity(value: str | Path) -> tuple[str, ...]:
        if isinstance(value, str):
            return ("text", hashlib.sha256(value.encode("utf-8")).hexdigest())
        path = value.resolve()
        if path not in file_digests:
            file_digests[path] = dataset_digest(path)
        return ("file", path.name, file_digests[path])

    payload = tuple(
        (
            unit.unit_id,
            tuple(
                (
                    memory.source_id,
                    memory.start_seconds,
                    memory.end_seconds,
                    (None if memory.occurred_at is None else memory.occurred_at.isoformat()),
                    (None if memory.occurred_end is None else memory.occurred_end.isoformat()),
                    tuple(identity(content) for content in memory.content),
                )
                for memory in unit.memories
            ),
        )
        for unit in units
    )
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _locomo(
    spec: TaskSpec,
    dataset: Path,
    _media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from mindbridge.benchmarks.locomo_refined import load_locomo_refined

    units = []
    for conversation in _selected(load_locomo_refined(dataset), limit, offset):
        memories = tuple(
            MemoryItem(
                turn.dialog_id,
                (
                    f"[{turn.occurred_at.isoformat()}] {turn.speaker} said: {turn.text}"
                    + (
                        f"\n{turn.speaker} shared an image described as: {turn.image_caption}"
                        if turn.image_caption
                        else ""
                    ),
                ),
                occurred_at=turn.occurred_at,
            )
            for turn in conversation.turns
        )
        questions = tuple(
            EvalQuestion(
                question.question_id,
                (_free_text_prompt(question.question),),
                question.reference_answers,
                metadata={"category": question.category},
                source_question=question.question,
            )
            for question in conversation.questions
        )
        units.append(EvalUnit(conversation.sample_id, memories, questions))
    return tuple(units)


def _m3(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from mindbridge.benchmarks.m3_bench import load_m3_bench

    units = []
    for video in _selected(load_m3_bench(dataset), limit, offset):
        memories = media.parts(
            video.video_id,
            (video.video_id,),
            relative_paths=(video.video_path,),
        )
        questions = tuple(
            EvalQuestion(
                question.question_id,
                (_free_text_prompt(question.question),),
                (question.reference_answer,),
                cutoff_seconds=question.cutoff_seconds,
                metadata={"question_types": question.question_types},
                source_question=question.question,
            )
            for question in video.questions
        )
        _require_timestamped_media(spec.name, video.video_id, memories, questions)
        units.append(EvalUnit(video.video_id, memories, questions))
    return tuple(units)


def _video_mme(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from mindbridge.benchmarks.prompts import VIDEO_MME_QUERY_PROMPT
    from mindbridge.benchmarks.video_mme import load_video_mme

    return tuple(
        EvalUnit(
            video.video_id,
            media.parts(video.video_id, (video.video_id, video.source_video_id)),
            tuple(
                EvalQuestion(
                    question.question_id,
                    (
                        VIDEO_MME_QUERY_PROMPT.text.format(
                            question=question.question,
                            options="\n".join(question.options),
                        ),
                    ),
                    expected_choice=question.answer,
                    score_kind="choice",
                    metadata={
                        "duration": video.duration,
                        "domain": video.domain,
                        "task_type": question.task_type,
                        "choices": question.options,
                    },
                )
                for question in video.questions
            ),
        )
        for video in _selected(load_video_mme(dataset), limit, offset)
    )


def _video_mme_v2(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from mindbridge.benchmarks.prompts import VIDEO_MME_V2_QUERY_PROMPT
    from mindbridge.benchmarks.video_mme_v2 import load_video_mme_v2

    return tuple(
        EvalUnit(
            group.video_id,
            media.parts(group.video_id, (group.video_id,)),
            tuple(
                EvalQuestion(
                    question.question_id,
                    (
                        VIDEO_MME_V2_QUERY_PROMPT.text.format(
                            question=question.question,
                            options="\n".join(question.options),
                        ),
                    ),
                    expected_choice=question.answer,
                    score_kind="choice",
                    metadata={
                        "position": question.position,
                        "group_type": group.group_type,
                        "group_structure": group.group_structure,
                        "level": question.level,
                        "second_head": question.second_head,
                        "third_head": question.third_head,
                        "choices": question.options,
                    },
                )
                for question in group.questions
            ),
        )
        for group in _selected(load_video_mme_v2(dataset), limit, offset)
    )


def _egolife(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from mindbridge.benchmarks.egolife_qa import load_egolife_qa

    questions = load_egolife_qa(dataset)
    identity = dataset.stem.removeprefix("EgoLifeQA_")
    memories = media.parts(identity, (identity,), allow_all=True)
    normalized = tuple(
        EvalQuestion(
            question.question_id,
            (_choice_prompt(question.question, question.choices),),
            expected_choice=question.correct_option,
            score_kind="choice",
            cutoff_seconds=question.query_offset_ms / 1_000,
            metadata={
                "day": question.query_day,
                "question_type": question.question_type,
                "needs_audio": question.needs_audio,
                "choices": question.choices,
            },
        )
        for question in _selected(questions, limit, offset)
    )
    _require_timestamped_media(spec.name, identity, memories, normalized)
    return (EvalUnit(identity, memories, normalized),)


def _egomem(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from mindbridge.benchmarks.egomem_reason import load_egomem_reason
    from mindbridge.benchmarks.prompts import EGOMEM_REASON_QUERY_PROMPT

    grouped: dict[str, list[EgoMemReasonQuestion]] = {}
    for question in _selected(load_egomem_reason(dataset), limit, offset):
        grouped.setdefault(question.identity, []).append(question)
    units = []
    for identity, raw_questions in sorted(grouped.items()):
        memories = media.parts(identity, (identity,), allow_all=True)
        questions = tuple(
            EvalQuestion(
                question.question_id,
                (
                    EGOMEM_REASON_QUERY_PROMPT.text.format(
                        query_time=question.query_time,
                        question_with_options=_choice_prompt(question.question, question.choices),
                    ),
                ),
                score_kind="submission",
                cutoff_seconds=question.query_offset_ms / 1_000,
                metadata={
                    "query_type": question.query_type,
                    "example_id": question.example_id,
                    "choices": question.choices,
                },
            )
            for question in raw_questions
        )
        _require_timestamped_media(spec.name, identity, memories, questions)
        units.append(EvalUnit(identity, memories, questions))
    return tuple(units)


def _egotempo(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from mindbridge.benchmarks.egotempo import load_egotempo
    from mindbridge.benchmarks.prompts import EGOTEMPO_QUERY_PROMPT

    grouped: dict[str, list[EgoTempoQuestion]] = {}
    for question in _selected(load_egotempo(dataset), limit, offset):
        grouped.setdefault(question.clip_id, []).append(question)
    return tuple(
        EvalUnit(
            clip_id,
            media.parts(clip_id, (clip_id,)),
            tuple(
                EvalQuestion(
                    question.question_id,
                    (EGOTEMPO_QUERY_PROMPT.text.format(question=question.question),),
                    (question.reference_answer,),
                    metadata={"question_type": question.question_type},
                    source_question=question.question,
                )
                for question in questions
            ),
        )
        for clip_id, questions in grouped.items()
    )


def _memlens(
    spec: TaskSpec,
    dataset: Path,
    _media: MediaResolver,
    root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from mindbridge.benchmarks.memlens import load_memlens, load_memlens_agent_subset
    from mindbridge.benchmarks.prompts import MEMLENS_QUERY_PROMPT

    questions = load_memlens(dataset)
    subset_path = root / spec.auxiliary[0]
    if subset_path.exists():
        wanted = set(load_memlens_agent_subset(subset_path))
        questions = tuple(question for question in questions if question.question_id in wanted)
    units = []
    for question in _selected(questions, limit, offset):
        memories = tuple(
            MemoryItem(
                turn.turn_id,
                (
                    f"[{session.occurred_at.isoformat()}] {turn.role}: {turn.content}"
                    + "".join(
                        f"\nImage {image.source_file}: {image.caption}"
                        for image in turn.images
                        if image.caption
                    ),
                ),
                occurred_at=session.occurred_at,
            )
            for session in question.sessions
            for turn in session.turns
        )
        prompt = MEMLENS_QUERY_PROMPT.text.format(question=question.question)
        units.append(
            EvalUnit(
                question.question_id,
                memories,
                (
                    EvalQuestion(
                        question.question_id,
                        (prompt,),
                        (question.reference_answer,),
                        metadata={
                            "question_type": question.question_type,
                            "question_subtype": question.question_subtype,
                            "old_answer": question.old_answer,
                        },
                        reference_at=question.question_date,
                        source_question=question.question,
                    ),
                ),
            )
        )
    return tuple(units)


def _mm_lifelong(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from mindbridge.benchmarks.mm_lifelong import MMLifelongSplit, load_mm_lifelong

    split = cast(MMLifelongSplit, spec.variant)
    questions = _selected(load_mm_lifelong(dataset, split), limit, offset)
    memories = media.parts(split, allow_all=True)
    return (
        EvalUnit(
            split,
            memories,
            tuple(
                EvalQuestion(
                    str(question.index),
                    (_free_text_prompt(question.question),),
                    (question.reference_answer,),
                    metadata={
                        "question_type": question.question_type,
                        "temporal_certificate": question.temporal_certificate,
                        "reference_intervals": question.reference_intervals,
                    },
                    source_question=question.question,
                )
                for question in questions
            ),
        ),
    )


def _supermemory(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from mindbridge.benchmarks.supermemory_vqa import load_supermemory_vqa

    grouped: dict[int, list[SuperMemoryQuestion]] = {}
    selected = tuple(
        question
        for question in load_supermemory_vqa(dataset)
        if spec.variant != "subject-1" or question.subject == 1
    )
    for question in _selected(selected, limit, offset):
        grouped.setdefault(question.subject, []).append(question)
    units = []
    for subject, raw_questions in sorted(grouped.items()):
        source_ids = tuple(
            dict.fromkeys(
                source for question in raw_questions for source in question.source_video_ids
            )
        )
        unit_id = f"subject-{subject}"
        memories = media.parts(unit_id, source_ids)
        questions = tuple(
            EvalQuestion(
                str(question.question_id),
                (_choice_prompt(question.question, question.choices),),
                expected_choice="ABCD"[question.correct_option_index],
                score_kind="choice",
                cutoff_seconds=question.question_ended_at.timestamp(),
                metadata={
                    "skill": question.skill,
                    "is_answerable": question.is_answerable,
                    "unanswerable_choice": "ABCD"[question.unanswerable_option_index],
                    "choices": question.choices,
                },
            )
            for question in raw_questions
        )
        _require_timestamped_media(spec.name, unit_id, memories, questions)
        units.append(EvalUnit(unit_id, memories, questions))
    return tuple(units)


def _atm(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from mindbridge.benchmarks.atm_bench import (
        atm_email_block,
        atm_sgm_block,
        load_atm_bench,
        load_atm_emails,
        load_atm_sgm,
    )
    from mindbridge.benchmarks.prompts import ATM_BENCH_QUERY_PROMPT, atm_format_constraint

    email_path = root / spec.auxiliary[0]
    memories = [
        MemoryItem(email.email_id, (atm_email_block(email),), occurred_at=email.occurred_at)
        for email in load_atm_emails(email_path)
    ]
    if cast(str, spec.variant).endswith("_sgm"):
        image_path, video_path = (root / item for item in spec.auxiliary[1:])
        for record in (*load_atm_sgm(image_path), *load_atm_sgm(video_path)):
            memories.append(
                MemoryItem(
                    record.media_id,
                    (atm_sgm_block(record),),
                    occurred_at=record.occurred_at,
                )
            )
    else:
        memories.extend(media.parts(cast(str, spec.variant), allow_all=True))
    questions = tuple(
        EvalQuestion(
            question.question_id,
            _query_parts(
                ATM_BENCH_QUERY_PROMPT.text,
                question.question,
                format_constraint=atm_format_constraint(question.qtype),
            ),
            (question.reference_answer,),
            metadata={"qtype": question.qtype, "evidence_ids": question.evidence_ids},
            source_question=question.question,
        )
        for question in _selected(load_atm_bench(dataset), limit, offset)
    )
    return (EvalUnit(cast(str, spec.variant), tuple(memories), questions),)


def _mem_gallery(
    spec: TaskSpec,
    dataset: Path,
    media: MediaResolver,
    _root: Path,
    limit: Limit,
    offset: int,
) -> tuple[EvalUnit, ...]:
    from mindbridge.benchmarks.mem_gallery import load_mem_gallery
    from mindbridge.benchmarks.prompts import (
        MEM_GALLERY_QUERY_PROMPT,
        mem_gallery_format_constraint,
    )

    units = []
    for topic in _selected(load_mem_gallery(dataset), limit, offset):
        memories = tuple(
            _gallery_memory(topic.profile.name, session, round_, media)
            for session in topic.sessions
            for round_ in session.rounds
        )
        questions = tuple(
            _gallery_question(
                topic.profile.name,
                question,
                media,
                MEM_GALLERY_QUERY_PROMPT.text,
                mem_gallery_format_constraint(question.point),
            )
            for question in topic.questions
        )
        units.append(EvalUnit(topic.topic, memories, questions))
    return tuple(units)


def _gallery_memory(
    profile_name: str,
    session: MemGallerySession,
    round_: MemGalleryRound,
    media: MediaResolver,
) -> MemoryItem:
    text = (
        f"[{session.occurred_at.date().isoformat()}] {profile_name}: {round_.user}"
        f"\nAssistant: {round_.assistant}"
    )
    if round_.image_id is not None:
        text = f"{text}\nImage ID: {round_.image_id}"
    image: Path | None = None
    if round_.image_path:
        image = media.path(round_.image_path)
    return MemoryItem(
        round_.round_id,
        (text,) if image is None else (text, image),
        occurred_at=session.occurred_at,
    )


def _gallery_question(
    profile_name: str,
    question: MemGalleryQuestion,
    media: MediaResolver,
    prompt_template: str,
    format_constraint: str,
) -> EvalQuestion:
    parts = _query_parts(
        prompt_template,
        question.question,
        speaker_a=profile_name,
        speaker_b="the assistant",
        format_constraint=format_constraint,
    )
    image: Path | None = None
    if question.question_image_path:
        image = media.path(question.question_image_path)
    return EvalQuestion(
        question.question_id,
        parts if image is None else (*parts, image),
        (question.reference_answer,),
        metadata={"point": question.point, "clue_ids": question.clue_round_ids},
        source_question=question.question,
    )


def _query_parts(template: str, question: str, **values: str) -> tuple[str, ...]:
    before, marker, after = template.partition("{question}")
    if not marker:
        raise ValueError("benchmark query template needs a question placeholder")
    return tuple(
        part
        for part in (
            question.strip(),
            before.format(**values).strip(),
            after.format(**values).strip(),
        )
        if part
    )


_LOADERS = {
    "locomo-refined": _locomo,
    "m3-bench-robot": _m3,
    "m3-bench-web": _m3,
    "video-mme": _video_mme,
    "video-mme-v2": _video_mme_v2,
    "egolifeqa": _egolife,
    "egomemreason": _egomem,
    "egotempo": _egotempo,
    "memlens-32k": _memlens,
    "memlens-64k": _memlens,
    "memlens-128k": _memlens,
    "memlens-256k": _memlens,
    "mm-lifelong-day-test": _mm_lifelong,
    "mm-lifelong-week-test": _mm_lifelong,
    "mm-lifelong-month-train": _mm_lifelong,
    "mm-lifelong-month-val": _mm_lifelong,
    "supermemory-vqa": _supermemory,
    "atm-bench-main": _atm,
    "atm-bench-hard": _atm,
    "atm-bench-main-sgm": _atm,
    "atm-bench-hard-sgm": _atm,
    "mem-gallery": _mem_gallery,
}


def _task_manifest(manifest: Mapping[str, object] | None, task_name: str) -> Mapping[str, object]:
    if manifest is None:
        return {}
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("media manifest tasks must be an object")
    task = tasks.get(task_name, {})
    if not isinstance(task, dict):
        raise ValueError(f"media manifest task {task_name} must be an object")
    units = task.get("units", task)
    if not isinstance(units, dict):
        raise ValueError(f"media manifest task {task_name} units must be an object")
    return units


def _memory_part(part: _ManifestPart) -> MemoryItem:
    return MemoryItem(part.source_id, part.content, part.start_seconds, part.end_seconds)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"media manifest {name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"media manifest {name} must be a finite number")
    return result


def _selected(values: Sequence[_T], limit: Limit, offset: int) -> Sequence[_T]:
    if limit is None or limit == -1:
        return values[offset:]
    count = max(1, math.ceil(len(values) * limit)) if limit < 1 else int(limit)
    return values[offset : offset + count]


def _require_timestamped_media(
    task_name: str,
    unit_id: str,
    memories: Sequence[MemoryItem],
    questions: Sequence[EvalQuestion],
) -> None:
    if any(question.cutoff_seconds is not None for question in questions) and any(
        memory.end_seconds is None for memory in memories
    ):
        raise ValueError(
            f"{task_name} unit {unit_id} has causal questions; its --media-manifest parts "
            "must declare end_seconds"
        )


def _choice_prompt(question: str, choices: Sequence[str]) -> str:
    options = "\n".join(
        f"{label}. {choice}" for label, choice in zip("ABCDEFGHIJ", choices, strict=False)
    )
    labels = ", ".join("ABCDEFGHIJ"[: len(choices)])
    return (
        f"Select the best answer using only the memories. Reply with one letter ({labels}).\n"
        f"Question: {question}\n{options}\nAnswer:"
    )


def _free_text_prompt(question: str) -> str:
    return f"Answer concisely using only the memories.\nQuestion: {question}\nAnswer:"
