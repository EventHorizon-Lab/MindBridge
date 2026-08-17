"""Turn one AML conversation chunk into retrievable memories."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from mindbridge.application.capabilities import (
    GenerateRequest,
    Generator,
    ModelInput,
    TextPart,
)
from mindbridge.core import MemoryType, ModelOutputError
from mindbridge.prompts import AML_EXTRACT_FACTS_PROMPT


class _AmlMessageLike(Protocol):
    """The subset of `mindbridge.api.aml_contracts.AmlMessage` this module reads."""

    role: str
    content: str
    timestamp: int | None


MAX_EXTRACTION_OUTPUT_TOKENS = 4_096
MAX_SUMMARY_CHARACTERS = 2_048

_MEMORY_TYPES = {
    "semantic": MemoryType.SEMANTIC,
    "episodic": MemoryType.EPISODIC,
    "procedural": MemoryType.PROCEDURAL,
}


@dataclass(frozen=True, slots=True)
class ExtractedMemory:
    """One atomic memory ready for kernel.remember()."""

    summary: str
    memory_type: MemoryType
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    """Memories parsed from a chunk, plus a count of items dropped as malformed."""

    memories: tuple[ExtractedMemory, ...]
    skipped: int


async def extract_memories(
    generator: Generator,
    messages: Sequence[_AmlMessageLike],
    *,
    now: datetime,
) -> ExtractionOutcome:
    """Extract atomic memories, dating them from the chunk's own timestamps."""
    occurred_at = _chunk_time(messages, now=now)
    result = await generator.generate(
        GenerateRequest(
            system_prompt=AML_EXTRACT_FACTS_PROMPT.text,
            input=ModelInput(parts=(TextPart(text=_render_chunk(messages)),)),
            max_output_tokens=MAX_EXTRACTION_OUTPUT_TOKENS,
            json_mode=True,
        )
    )
    parsed, skipped = _parsed_memories(result.text)
    return ExtractionOutcome(
        memories=tuple(
            ExtractedMemory(
                summary=summary,
                memory_type=memory_type,
                occurred_at=occurred_at,
            )
            for summary, memory_type in parsed
        ),
        skipped=skipped,
    )


def _render_chunk(messages: Sequence[_AmlMessageLike]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def _chunk_time(messages: Sequence[_AmlMessageLike], *, now: datetime) -> datetime:
    timestamps = [message.timestamp for message in messages if message.timestamp is not None]
    if not timestamps:
        return now
    return datetime.fromtimestamp(min(timestamps) / 1_000, tz=timezone.utc)


def _parsed_memories(text: str) -> tuple[list[tuple[str, MemoryType]], int]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ModelOutputError("AML extraction output is not JSON") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("memories"), list):
        raise ModelOutputError("AML extraction output has no memories list")
    memories: list[tuple[str, MemoryType]] = []
    skipped = 0
    for item in payload["memories"]:
        if not isinstance(item, dict):
            skipped += 1
            continue
        summary = str(item.get("summary") or "").strip()[:MAX_SUMMARY_CHARACTERS]
        memory_type = _MEMORY_TYPES.get(str(item.get("type") or "").strip().lower())
        if not summary or memory_type is None:
            skipped += 1
            continue
        memories.append((summary, memory_type))
    return memories, skipped
