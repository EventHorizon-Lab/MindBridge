"""Thin adapter for the official Mem-Gallery release.

Twenty topic files, each one persona's multi-session dialogue plus the questions annotated
over it. Question IDs are not in the release, so they are derived from release order and
pinned, the way the MM-Lifelong adapter pins question indices.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from mindbridge.contracts import ContractModel, Identifier, NonEmptyString
from mindbridge.file_integrity import sha256_file

MEM_GALLERY_ADAPTER_VERSION = "mem_gallery_official_v1"

MemGalleryPoint = Literal["FR", "MR", "TR", "VR", "TTL", "VS", "CD", "KR", "AR"]
_POINTS: frozenset[str] = frozenset(("FR", "MR", "TR", "VR", "TTL", "VS", "CD", "KR", "AR"))
_QA_KEY = "human-annotated QAs"


class MemGalleryProfile(ContractModel):
    """The persona the dialogue belongs to, as the release describes it."""

    name: NonEmptyString
    persona_summary: NonEmptyString
    traits: tuple[NonEmptyString, ...] = ()
    conversation_style: NonEmptyString


class MemGalleryRound(ContractModel):
    """One user turn and its assistant reply, with the round's image when it has one."""

    round_id: Identifier
    user: NonEmptyString
    assistant: NonEmptyString
    image_id: Identifier | None = None
    image_path: NonEmptyString | None = None
    image_caption: NonEmptyString | None = None

    @model_validator(mode="after")
    def require_complete_image(self) -> MemGalleryRound:
        present = (self.image_id is not None, self.image_path is not None)
        if any(present) and not all(present):
            raise ValueError("Mem-Gallery round images need both an ID and a path")
        return self


class MemGallerySession(ContractModel):
    """One dated session, rounds in release order."""

    session_id: Identifier
    occurred_at: AwareDatetime
    rounds: tuple[MemGalleryRound, ...] = Field(min_length=1)


class MemGalleryQuestion(ContractModel):
    """One annotated question, its task type, and the rounds that answer it."""

    question_id: Identifier
    point: MemGalleryPoint
    question: NonEmptyString
    reference_answer: NonEmptyString
    session_ids: tuple[Identifier, ...] = Field(min_length=1)
    clue_round_ids: tuple[Identifier, ...] = ()
    question_image_path: NonEmptyString | None = None
    question_image_caption: NonEmptyString | None = None


class MemGalleryTopic(ContractModel):
    """One topic file: a persona, its sessions, and the questions over them."""

    topic: Identifier
    profile: MemGalleryProfile
    sessions: tuple[MemGallerySession, ...] = Field(min_length=1)
    questions: tuple[MemGalleryQuestion, ...] = Field(min_length=1)


class _RawProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    persona_summary: str
    traits: list[str] = Field(default_factory=list)
    conversation_style: str


class _RawRound(BaseModel):
    model_config = ConfigDict(extra="ignore")

    round: str
    user: str
    assistant: str
    image_id: list[str] = Field(default_factory=list)
    input_image: list[str] = Field(default_factory=list)
    image_caption: list[str] = Field(default_factory=list)


class _RawSession(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str
    date: str
    dialogues: list[_RawRound] = Field(min_length=1)


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    point: str
    question: str
    answer: str
    session_id: list[str] = Field(min_length=1)
    clue: list[str] = Field(default_factory=list)
    question_image: str | None = None
    image_caption: str | None = None


def load_mem_gallery_topic(topic_path: Path) -> MemGalleryTopic:
    """Load one official topic file, keyed by its filename."""
    payload = json.loads(topic_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mem-Gallery topic files must be objects")
    sessions = tuple(
        _session(_RawSession.model_validate(item))
        for item in payload.get("multi_session_dialogues", ())
    )
    if not sessions:
        raise ValueError("Mem-Gallery topics must carry at least one session")
    topic = topic_path.stem
    questions = tuple(
        _question(_RawQuestion.model_validate(item), topic, index)
        for index, item in enumerate(payload.get(_QA_KEY, ()), start=1)
    )
    _require_known_references(questions, sessions)
    return MemGalleryTopic(
        topic=topic,
        profile=_profile(_RawProfile.model_validate(payload["character_profile"])),
        sessions=sessions,
        questions=questions,
    )


def load_mem_gallery(dialog_directory: Path) -> tuple[MemGalleryTopic, ...]:
    """Load every topic file in `data/dialog`, in sorted filename order."""
    paths = sorted(dialog_directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no Mem-Gallery topic files under {dialog_directory}")
    return tuple(load_mem_gallery_topic(path) for path in paths)


def mem_gallery_dialog_digest(dialog_directory: Path) -> str:
    """Digest the concatenated per-file digests of `data/dialog`, in sorted order.

    Shared by the CLI run manifest and the dataset-smoke summary so a run's manifest and a
    smoke row cannot silently disagree about which release digest names -- each used to
    compute this independently.
    """
    joined = "".join(sha256_file(path) for path in sorted(dialog_directory.glob("*.json")))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _require_known_references(
    questions: tuple[MemGalleryQuestion, ...],
    sessions: tuple[MemGallerySession, ...],
) -> None:
    """Refuse a question whose clue or session reference names a round the topic never defined.

    Split out of `load_mem_gallery_topic` to keep that function's cyclomatic complexity under
    the repository's mccabe ceiling; the two set comprehensions below and the two-branch loop
    would otherwise push a single function over it.
    """
    known_rounds = {round_.round_id for session in sessions for round_ in session.rounds}
    known_sessions = {session.session_id for session in sessions}
    for question in questions:
        dangling_rounds = set(question.clue_round_ids) - known_rounds
        if dangling_rounds:
            raise ValueError(
                f"Mem-Gallery clue names an unknown round: {', '.join(sorted(dangling_rounds))}"
            )
        dangling_sessions = set(question.session_ids) - known_sessions
        if dangling_sessions:
            raise ValueError(
                "Mem-Gallery question names an unknown session: "
                f"{', '.join(sorted(dangling_sessions))}"
            )


def _profile(raw: _RawProfile) -> MemGalleryProfile:
    return MemGalleryProfile(
        name=raw.name,
        persona_summary=raw.persona_summary,
        traits=tuple(trait for trait in raw.traits if trait.strip()),
        conversation_style=raw.conversation_style,
    )


def _session(raw: _RawSession) -> MemGallerySession:
    return MemGallerySession(
        session_id=raw.session_id,
        occurred_at=datetime.strptime(raw.date, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        rounds=tuple(_round(item) for item in raw.dialogues),
    )


def _round(raw: _RawRound) -> MemGalleryRound:
    images = tuple(raw.input_image)
    identifiers = tuple(raw.image_id)
    captions = tuple(raw.image_caption)
    if len(images) > 1 or len(identifiers) > 1:
        raise ValueError(f"Mem-Gallery round {raw.round} must carry exactly one image")
    return MemGalleryRound(
        round_id=raw.round,
        user=raw.user,
        assistant=raw.assistant,
        image_id=identifiers[0] if identifiers else None,
        image_path=images[0] if images else None,
        image_caption=captions[0] if captions else None,
    )


def _question(raw: _RawQuestion, topic: str, index: int) -> MemGalleryQuestion:
    if raw.point not in _POINTS:
        raise ValueError(f"unknown Mem-Gallery point: {raw.point}")
    return MemGalleryQuestion(
        question_id=f"{topic}:{index}",
        point=raw.point,  # type: ignore[arg-type]
        question=raw.question,
        reference_answer=raw.answer,
        session_ids=tuple(raw.session_id),
        clue_round_ids=tuple(raw.clue),
        question_image_path=raw.question_image,
        question_image_caption=raw.image_caption,
    )
