"""Adapter for the pinned WorldMemArena checkpoint-QA release."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

WORLDMEMARENA_ADAPTER_VERSION = "worldmemarena_checkpoint_qa_v1"


class WorldMemAttachment(ContractModel):
    image_id: Identifier
    file_path: NonEmptyString
    caption: str = ""


class WorldMemTurn(ContractModel):
    role: NonEmptyString
    content: str
    attachments: tuple[WorldMemAttachment, ...] = ()


class WorldMemSession(ContractModel):
    session_id: Identifier
    turns: tuple[WorldMemTurn, ...] = ()


class WorldMemQuestion(ContractModel):
    question_id: Identifier
    question: NonEmptyString
    answer: NonEmptyString
    question_type: NonEmptyString
    question_type_abbrev: str
    difficulty: NonEmptyString
    evidence_ids: tuple[Identifier, ...]
    evidence_contents: tuple[str, ...]


class WorldMemCheckpoint(ContractModel):
    checkpoint_id: Identifier
    covered_sessions: tuple[Identifier, ...] = Field(min_length=1)
    questions: tuple[WorldMemQuestion, ...] = Field(min_length=1)


class WorldMemSample(ContractModel):
    sample_id: Identifier
    source_path: Path
    sessions: tuple[WorldMemSession, ...] = Field(min_length=1)
    checkpoints: tuple[WorldMemCheckpoint, ...] = Field(min_length=1)


class _RawAttachment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_id: str
    file_path: str
    caption: str = ""


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str = ""
    attachments: list[_RawAttachment] = Field(default_factory=list)


class _RawSession(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    session_id: str = Field(alias="_v2_session_id")
    dialogue: list[_RawTurn] = Field(default_factory=list)


class _RawMemoryPoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    memory_id: str
    memory_content: str


class _RawMemoryGroup(BaseModel):
    model_config = ConfigDict(extra="ignore")

    memory_points: list[_RawMemoryPoint] = Field(default_factory=list)


class _RawEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    memory_id: str | None = None
    image_id: str | None = None


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    answer: str
    question_type: str
    question_type_abbrev: str
    difficulty: str
    evidence: list[_RawEvidence] = Field(default_factory=list)


class _RawCheckpoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    checkpoint_id: str
    covered_sessions: list[str] = Field(min_length=1)
    questions: list[_RawQuestion] = Field(min_length=1)


class _RawSample(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sample_id: str
    sessions: list[_RawSession] = Field(min_length=1)
    memory_points: list[_RawMemoryGroup] = Field(default_factory=list)
    qa_checkpoints: list[_RawCheckpoint] = Field(min_length=1)


def load_worldmemarena(dataset: Path) -> tuple[WorldMemSample, ...]:
    """Load every agent and lifelong sample from the official directory tree."""
    paths = tuple(
        path
        for section in ("agent", "lifelong")
        for path in sorted((dataset / section).rglob("*.json"))
        if "images" not in path.parts
    )
    if not paths:
        raise ValueError(f"no WorldMemArena samples under {dataset}")
    samples = tuple(_sample(path) for path in paths)
    ids = tuple(sample.sample_id for sample in samples)
    if len(ids) != len(set(ids)):
        raise ValueError("WorldMemArena contains duplicate sample IDs")
    return samples


def _sample(path: Path) -> WorldMemSample:
    raw = _RawSample.model_validate(json.loads(path.read_text(encoding="utf-8")))
    memory_contents = {
        point.memory_id: point.memory_content
        for group in raw.memory_points
        for point in group.memory_points
    }
    image_contents = {
        attachment.image_id: attachment.caption
        for session in raw.sessions
        for turn in session.dialogue
        for attachment in turn.attachments
    }
    sessions = tuple(
        WorldMemSession(
            session_id=session.session_id,
            turns=tuple(
                WorldMemTurn(
                    role=turn.role,
                    content=turn.content,
                    attachments=tuple(
                        WorldMemAttachment(
                            image_id=item.image_id,
                            file_path=item.file_path,
                            caption=item.caption,
                        )
                        for item in turn.attachments
                    ),
                )
                for turn in session.dialogue
            ),
        )
        for session in raw.sessions
    )
    checkpoints = tuple(
        WorldMemCheckpoint(
            checkpoint_id=checkpoint.checkpoint_id,
            covered_sessions=tuple(checkpoint.covered_sessions),
            questions=tuple(
                _question(
                    question,
                    raw.sample_id,
                    checkpoint.checkpoint_id,
                    index,
                    memory_contents,
                    image_contents,
                )
                for index, question in enumerate(checkpoint.questions, start=1)
            ),
        )
        for checkpoint in raw.qa_checkpoints
    )
    known_sessions = {session.session_id for session in sessions}
    for checkpoint in checkpoints:
        unknown = set(checkpoint.covered_sessions) - known_sessions
        if unknown:
            raise ValueError(
                "WorldMemArena checkpoint names unknown sessions: " + ", ".join(sorted(unknown))
            )
    return WorldMemSample(
        sample_id=raw.sample_id,
        source_path=path,
        sessions=sessions,
        checkpoints=checkpoints,
    )


def _question(
    raw: _RawQuestion,
    sample_id: str,
    checkpoint_id: str,
    index: int,
    memory_contents: dict[str, str],
    image_contents: dict[str, str],
) -> WorldMemQuestion:
    evidence_ids = tuple(item.memory_id or item.image_id or "" for item in raw.evidence)
    if any(not identifier for identifier in evidence_ids):
        raise ValueError("WorldMemArena evidence must contain a memory_id or image_id")
    contents = tuple(
        memory_contents.get(identifier, image_contents.get(identifier, ""))
        for identifier in evidence_ids
    )
    return WorldMemQuestion(
        question_id=f"{sample_id}:{checkpoint_id}:Q{index:03d}",
        question=raw.question,
        answer=raw.answer,
        question_type=raw.question_type,
        question_type_abbrev=raw.question_type_abbrev,
        difficulty=raw.difficulty,
        evidence_ids=evidence_ids,
        evidence_contents=contents,
    )
