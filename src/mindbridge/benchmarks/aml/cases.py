"""Benchmark-neutral case model shared by every AML loader."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

MAX_MESSAGES_PER_CHUNK = 20
MAX_WORDS_PER_CHUNK = 2_000

Message = dict[str, object]


@dataclass(frozen=True, slots=True)
class AmlQuestion:
    """One question plus whatever its official pipeline reads."""

    question_id: str
    question: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AmlCase:
    """One AML retrieval scope: a history and the questions asked against it."""

    user_id: str
    messages: tuple[Message, ...]
    questions: tuple[AmlQuestion, ...]


def chunk_messages(messages: Sequence[Message]) -> tuple[tuple[Message, ...], ...]:
    """Split a history at AML's documented boundary of 20 messages or 2,000 words."""
    chunks: list[tuple[Message, ...]] = []
    current: list[Message] = []
    words = 0
    for message in messages:
        length = len(str(message.get("content") or "").split())
        exceeds = len(current) >= MAX_MESSAGES_PER_CHUNK or (
            current and words + length > MAX_WORDS_PER_CHUNK
        )
        if exceeds:
            chunks.append(tuple(current))
            current, words = [], 0
        current.append(message)
        words += length
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)
