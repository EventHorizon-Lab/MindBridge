"""Thin adapter for the official ATM-Bench release.

One archive, two QA splits, and two representations of the same media. The release carries
two clocks: an image's `timestamp` is local wall clock and agrees with its filename stem,
while a video's is true UTC and sits an hour off the stem for half the year. This adapter
takes the stem as capture time for every modality so one archive has one timeline.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from mindbridge.contracts import ContractModel, Identifier, NonEmptyString
from mindbridge.core import MediaKind

ATM_BENCH_ADAPTER_VERSION = "atm_bench_official_v1"

AtmQuestionType = Literal["number", "list_recall", "open_end"]
AtmEvidenceKind = Literal["email", "media"]

# `fullmatch`-ed against a bare evidence ID, which is always exactly this shape in the
# release. `match`-ed (prefix only) against a media filename stem in `atm_capture_time`,
# because 28 of the 4,292 image and video filenames carry a disambiguating suffix that the
# release's export tooling appended to a duplicate capture — e.g. `20221212_115316_001` or
# `20220627_155122(0)`. The leading 15 characters are still the camera's local wall clock
# either way, and no gold evidence ID ever names one of these 28.
_TIMESTAMP_PATTERN = re.compile(r"(\d{8})_(\d{6})")
_EMAIL_PREFIX = "email"
_MEMORY_CHARACTER_LIMIT = 2_048
_CHUNK_BODY_CHARACTERS = 1_800
_MEDIA_KIND_BY_FIELD = {"image_path": MediaKind.IMAGE, "video_path": MediaKind.VIDEO}


class AtmBenchQuestion(ContractModel):
    """One official question and the evidence a correct answer must rest on."""

    question_id: Identifier
    question: NonEmptyString
    reference_answer: NonEmptyString
    qtype: AtmQuestionType
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1)
    niah_evidence_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def require_consistent_evidence(self) -> AtmBenchQuestion:
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("ATM-Bench evidence IDs must not repeat")
        if len(set(self.niah_evidence_ids)) != len(self.niah_evidence_ids):
            raise ValueError("ATM-Bench NIAH evidence IDs must not repeat")
        return self


class AtmEmail(ContractModel):
    """One email in the archive, cited by 430 of the release's evidence references."""

    email_id: Identifier
    occurred_at: AwareDatetime
    summary: NonEmptyString
    body: str


class AtmSgmRecord(ContractModel):
    """One official schema-guided memory record for an image or a video.

    `raw_timestamp` keeps the record's own timestamp string verbatim so `atm_sgm_block` can
    reproduce the official Oracle baseline's serialization exactly, even though `occurred_at`
    deliberately reads capture time from the filename stem instead — see the module docstring
    on the two clocks.
    """

    media_id: Identifier
    media_kind: MediaKind
    occurred_at: AwareDatetime
    raw_timestamp: str
    location_name: str
    city: str
    short_caption: str
    caption: str
    ocr_text: str
    tags: tuple[NonEmptyString, ...] = ()
    duration_seconds: float | None = None
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def require_duration_for_video_only(self) -> AtmSgmRecord:
        if self.media_kind is MediaKind.VIDEO and self.duration_seconds is None:
            raise ValueError("ATM-Bench video records must carry a duration")
        if self.media_kind is MediaKind.IMAGE and self.duration_seconds is not None:
            raise ValueError("ATM-Bench image records must not carry a duration")
        return self


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    answer: str
    qtype: str
    evidence_ids: list[str] = Field(min_length=1)
    niah_evidence_ids: list[str] = Field(default_factory=list)


class _RawEmail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    timestamp: str
    short_summary: str
    detail: str


class _RawSgmRecord(BaseModel):
    # The media path itself is read straight off the source dict under whichever of
    # `image_path`/`video_path` is present (see `_sgm_record`), not through this model, so
    # `extra="ignore"` is safe here exactly like every other raw model in this file. It is
    # also what silently drops the release's `entities` field: the official Oracle
    # serialization's field set is `type, timestamp, location, short_caption, caption, ocr,
    # tags` — `entities` is not part of the SGM text at all — and all 168,881 image-side
    # entity items are double-encoded garbage (`{'entity': '{"entity"', 'type': '"airplane",
    # "type": "other"}'}`) while only the 970 video-side items are clean. Dropping it is
    # fidelity, not a workaround.
    model_config = ConfigDict(extra="ignore")

    timestamp: str
    location_name: str = ""
    city: str = ""
    short_caption: str = ""
    caption: str = ""
    ocr_text: str = ""
    tags: list[str] = Field(default_factory=list)
    duration: float | None = None
    file_size: int = 0


def atm_evidence_kind(evidence_id: str) -> AtmEvidenceKind:
    """Say which store an evidence ID belongs to, once, so nothing can disagree later."""
    if evidence_id.startswith(_EMAIL_PREFIX):
        return "email"
    if _TIMESTAMP_PATTERN.fullmatch(evidence_id):
        return "media"
    raise ValueError(f"unrecognised ATM-Bench evidence ID: {evidence_id}")


def atm_capture_time(media_name: str) -> datetime:
    """Read capture time out of a `YYYYMMDD_HHMMSS` stem, as UTC.

    The stem is the camera's local wall clock at the place of capture. Reading it as UTC
    keeps images, videos, and email on one timeline; see the design note on the two clocks.
    Only the leading 15 characters are read, so a trailing disambiguator on a duplicate
    capture (e.g. `..._001`, `...(0)`) does not stop the timestamp from being read.
    """
    stem = Path(media_name).stem
    matched = _TIMESTAMP_PATTERN.match(stem)
    if matched is None:
        raise ValueError(f"ATM-Bench media name carries no capture time: {media_name}")
    return datetime.strptime(matched.group(1) + matched.group(2), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


def load_atm_bench(dataset_path: Path) -> tuple[AtmBenchQuestion, ...]:
    """Load `atm-bench.json` or `atm-bench-hard.json`."""
    return _questions(dataset_path, require_pool=False)


def load_atm_niah_pool(pool_path: Path) -> tuple[AtmBenchQuestion, ...]:
    """Load one NIAH pool, refusing a pool that has lost a gold evidence item."""
    return _questions(pool_path, require_pool=True)


def load_atm_emails(emails_path: Path) -> tuple[AtmEmail, ...]:
    """Load the archive's emails, naive timestamps read as UTC."""
    raw = json.loads(emails_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("ATM-Bench emails must be a non-empty list")
    emails = tuple(_email(_RawEmail.model_validate(item)) for item in raw)
    if len({email.email_id for email in emails}) != len(emails):
        raise ValueError("ATM-Bench emails contain duplicate IDs")
    return emails


def load_atm_sgm(batch_results_path: Path) -> tuple[AtmSgmRecord, ...]:
    """Load one official `batch_results.json`, keyed by media stem."""
    raw = json.loads(batch_results_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("results", raw)
    if not isinstance(raw, list) or not raw:
        raise ValueError("ATM-Bench batch results must be a non-empty list")
    records = tuple(_sgm_record(item) for item in raw)
    if len({record.media_id for record in records}) != len(records):
        raise ValueError("ATM-Bench batch results contain duplicate media IDs")
    return records


def atm_sgm_block(record: AtmSgmRecord) -> str:
    """Serialize one record exactly as the official Oracle baseline serializes it."""
    return "\n".join(
        (
            f"ID: {record.media_id}",
            f"Type: {record.media_kind.value}",
            f"Timestamp: {record.raw_timestamp}",
            f"Location: {record.location_name}",
            f"Short Caption: {record.short_caption}",
            f"Caption: {record.caption}",
            f"OCR: {record.ocr_text}",
            f"Tags: {', '.join(record.tags)}",
        )
    )


def atm_email_block(email: AtmEmail) -> str:
    """Serialize one email exactly as the official Oracle baseline serializes it."""
    return "\n".join(
        (
            f"ID: {email.email_id}",
            f"Timestamp: {email.occurred_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Summary: {email.summary}",
            f"Detail: {email.body}",
        )
    )


def atm_memory_chunks(block: str, evidence_id: str) -> tuple[str, ...]:
    """Split one serialized block into writes that fit `RememberRequest.summary`.

    1,063 of the 4,292 SGM blocks exceed the 2,048-character limit, the longest at 9,212.
    Every chunk repeats the ID line, so retrieving any one of them still names its evidence.
    """
    if len(block) <= _MEMORY_CHARACTER_LIMIT:
        return (block,)
    head = f"ID: {evidence_id}\n"
    body = block[len(head) :] if block.startswith(head) else block
    slices = tuple(
        body[offset : offset + _CHUNK_BODY_CHARACTERS]
        for offset in range(0, len(body), _CHUNK_BODY_CHARACTERS)
    )
    return tuple(
        f"{head}Part {index}/{len(slices)}\n{piece}" for index, piece in enumerate(slices, start=1)
    )


def atm_evidence_id_from_block(summary: str) -> str | None:
    """Read the evidence ID a serialized block's leading `ID: <id>` line names, or None.

    Every SGM and email block -- and every chunk `atm_memory_chunks` splits one into -- opens
    with exactly this line, for exactly this purpose: `recall`'s own evidence list only ever
    names media MindBridge itself observed, so it has nothing for an email or an sgm-arm
    write, both of which land as `remember` text instead. Reading this line back out of a
    recalled memory's summary is how those two are not simply invisible to retrieval-recall.
    A summary that does not open with the marker -- e.g. the raw arm's own perception-derived
    text -- names no evidence and returns None.
    """
    first_line, _, _ = summary.partition("\n")
    if not first_line.startswith("ID: "):
        return None
    evidence_id = first_line.removeprefix("ID: ")
    return evidence_id or None


def _questions(path: Path, *, require_pool: bool) -> tuple[AtmBenchQuestion, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("qas", raw)
    if not isinstance(raw, list) or not raw:
        raise ValueError("ATM-Bench annotations must not be empty")
    questions = tuple(_question(_RawQuestion.model_validate(item)) for item in raw)
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("ATM-Bench annotations contain duplicate question IDs")
    for question in questions:
        for evidence_id in question.evidence_ids:
            atm_evidence_kind(evidence_id)
        if require_pool:
            _require_niah_pool(question)
    return questions


def _require_niah_pool(question: AtmBenchQuestion) -> None:
    if not question.niah_evidence_ids:
        raise ValueError("ATM-Bench NIAH pools must carry niah_evidence_ids")
    if not set(question.evidence_ids) <= set(question.niah_evidence_ids):
        raise ValueError(
            f"ATM-Bench NIAH pool must contain every gold evidence for {question.question_id}"
        )


def _question(raw: _RawQuestion) -> AtmBenchQuestion:
    return AtmBenchQuestion(
        question_id=raw.id,
        question=raw.question,
        reference_answer=raw.answer,
        qtype=_question_type(raw.qtype),
        evidence_ids=tuple(raw.evidence_ids),
        niah_evidence_ids=tuple(raw.niah_evidence_ids),
    )


def _question_type(value: str) -> AtmQuestionType:
    if value not in {"number", "list_recall", "open_end"}:
        raise ValueError(f"unknown ATM-Bench question type: {value}")
    return value  # type: ignore[return-value]


def _email(raw: _RawEmail) -> AtmEmail:
    return AtmEmail(
        email_id=raw.id,
        occurred_at=_naive_utc(raw.timestamp),
        summary=raw.short_summary,
        body=raw.detail,
    )


def _sgm_record(item: object) -> AtmSgmRecord:
    if not isinstance(item, dict):
        raise ValueError("ATM-Bench batch results entries must be objects")
    field = next((name for name in _MEDIA_KIND_BY_FIELD if name in item), None)
    if field is None:
        raise ValueError("ATM-Bench batch results entry names no media path")
    kind = _MEDIA_KIND_BY_FIELD[field]
    raw = _RawSgmRecord.model_validate(item)
    return AtmSgmRecord(
        media_id=Path(str(item[field])).stem,
        media_kind=kind,
        occurred_at=atm_capture_time(str(item[field])),
        raw_timestamp=raw.timestamp,
        location_name=raw.location_name,
        city=raw.city,
        short_caption=raw.short_caption,
        caption=raw.caption,
        ocr_text=raw.ocr_text,
        tags=tuple(tag for tag in raw.tags if tag.strip()),
        duration_seconds=raw.duration if kind is MediaKind.VIDEO else None,
        size_bytes=raw.file_size,
    )


def _naive_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
