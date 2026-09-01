"""Thin adapter for the official BEAM conversation tiers and probing questions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

BEAM_ADAPTER_VERSION = "beam_official_v1"

BeamTier = Literal["100K", "500K", "1M", "10M"]
BEAM_TIERS: tuple[BeamTier, ...] = ("100K", "500K", "1M", "10M")

# The ten keys of `probing_questions.json`. Upstream's `run_evaluation.py`
# dispatches on this key and has no default branch, so an unrecognised
# category is a corpus problem rather than something to score generically.
BeamCategory = Literal[
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]
BEAM_CATEGORIES: tuple[BeamCategory, ...] = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)

# Time anchors are published as "March-15-2024". Month names are mapped here
# rather than through `%B`, which `strptime` resolves via the process locale
# and which therefore rejects every anchor under a non-English locale.
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_QUESTIONS_RELATIVE = Path("probing_questions") / "probing_questions.json"
_CHAT_RELATIVE = Path("chat.json")


class BeamTurn(ContractModel):
    """One conversation turn, timestamped only where an anchor was published."""

    turn_id: Identifier
    role: Literal["user", "assistant"]
    # A bare `str` guarded below: assistant turns run past the 2,048-character
    # cap that `NonEmptyString` imposes.
    content: str
    occurred_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def require_content(self) -> BeamTurn:
        if not self.content.strip():
            raise ValueError("BEAM turns require text")
        return self


class BeamQuestion(ContractModel):
    """One probing question with the rubric its official judge scores against."""

    question_id: Identifier
    category: BeamCategory
    question: str
    # `rubric` is the only field every category publishes, and it is the only
    # one the official judge reads: it scores the response against each item
    # in turn. The reference answer below is carried for reporting only -- the
    # release names it differently per category (`answer`, `ideal_answer`,
    # `ideal_summary`, `ideal_response`) and omits it entirely on
    # `instruction_following` and `preference_following`, which publish
    # `expected_compliance` prose instead.
    rubric: tuple[str, ...] = Field(min_length=1)
    reference_answer: str = ""
    difficulty: NonEmptyString | None = None

    @model_validator(mode="after")
    def require_text(self) -> BeamQuestion:
        if not self.question.strip():
            raise ValueError("BEAM questions require question text")
        if any(not item.strip() for item in self.rubric):
            raise ValueError("BEAM rubric items must not be blank")
        return self


class BeamConversation(ContractModel):
    """One conversation directory: a single retrieval scope and its questions."""

    tier: BeamTier
    conversation_id: Identifier
    turns: tuple[BeamTurn, ...] = Field(min_length=1)
    questions: tuple[BeamQuestion, ...] = Field(min_length=1)


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    id: int = Field(ge=0)
    content: str
    time_anchor: str | None = None


class _RawBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    batch_number: int
    turns: list[list[_RawTurn]]


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    rubric: list[str] = Field(min_length=1)
    difficulty: str | None = None
    answer: str | None = None
    ideal_answer: str | None = None
    ideal_response: str | None = None
    ideal_summary: str | None = None


_BATCHES = TypeAdapter(list[_RawBatch])
_QUESTION_FILE = TypeAdapter(dict[str, list[_RawQuestion]])


def load_beam(tier_root: Path, tier: BeamTier) -> tuple[BeamConversation, ...]:
    """Load every conversation directory of one pinned tier, in numeric order."""
    directories = sorted(
        (path for path in tier_root.iterdir() if (path / _CHAT_RELATIVE).is_file()),
        key=_directory_key,
    )
    conversations = tuple(_conversation(path, tier) for path in directories)
    if not conversations:
        raise ValueError(f"BEAM tier {tier} contains no conversation directories")
    return conversations


def _directory_key(path: Path) -> tuple[int, int, str]:
    return (0, int(path.name), "") if path.name.isdigit() else (1, 0, path.name)


def _conversation(directory: Path, tier: BeamTier) -> BeamConversation:
    conversation_id = directory.name
    turns = tuple(
        _turn(conversation_id, raw)
        for batch in _batches(directory / _CHAT_RELATIVE)
        for pair in batch.turns
        for raw in pair
        if raw.content.strip()
    )
    if len({turn.turn_id for turn in turns}) != len(turns):
        raise ValueError(f"BEAM conversation {tier}/{conversation_id} has duplicate turn IDs")
    return BeamConversation(
        tier=tier,
        conversation_id=conversation_id,
        turns=turns,
        questions=_questions(directory / _QUESTIONS_RELATIVE, tier, conversation_id),
    )


def _batches(chat_path: Path) -> tuple[_RawBatch, ...]:
    """Read one `chat.json`, flattening the 10M tier's extra plan nesting.

    The 10M tier -- and only that tier -- wraps each element as a single-key
    `{"plan-N": [...batches...]}` dict. Those plans are consecutive stretches of
    one conversation, not separate conversations: turn IDs ascend straight
    across the boundaries and probing questions cite turns from two plans at
    once. Concatenating them in document order restores the one scope per
    directory that every other tier already has.
    """
    payload = json.loads(chat_path.read_bytes())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"BEAM chat must be a non-empty list: {chat_path}")
    flattened: list[object] = []
    for element in payload:
        if isinstance(element, dict) and "batch_number" not in element:
            if len(element) != 1:
                raise ValueError(f"BEAM plan element must hold exactly one plan: {chat_path}")
            plan = next(iter(element.values()))
            if not isinstance(plan, list):
                raise ValueError(f"BEAM plan must hold a batch list: {chat_path}")
            flattened.extend(plan)
        else:
            flattened.append(element)
    return tuple(_BATCHES.validate_python(flattened))


def _turn(conversation_id: str, raw: _RawTurn) -> BeamTurn:
    return BeamTurn(
        turn_id=f"{conversation_id}_T{raw.id:06d}",
        role=_role(raw.role),
        # The generation harness left a literal "->-> 1,1" index marker on some
        # early turns. It is kept verbatim: it is part of what the official
        # pipeline reads, and stripping it would change the ingested corpus.
        content=raw.content,
        occurred_at=None if raw.time_anchor is None else _anchor(raw.time_anchor),
    )


def _questions(
    questions_path: Path,
    tier: BeamTier,
    conversation_id: str,
) -> tuple[BeamQuestion, ...]:
    grouped = _QUESTION_FILE.validate_json(questions_path.read_bytes())
    unknown = tuple(sorted(set(grouped) - set(BEAM_CATEGORIES)))
    if unknown:
        raise ValueError(f"BEAM {questions_path} has unknown categories: {', '.join(unknown)}")
    questions = tuple(
        BeamQuestion(
            # Individual questions carry no ID; upstream iterates the category
            # key and the list position, so the ID is derived from both.
            question_id=f"{tier}-{conversation_id}-{category}-{index:04d}",
            category=category,
            question=raw.question,
            rubric=tuple(item.strip() for item in raw.rubric if item.strip()),
            reference_answer=_reference_answer(raw),
            difficulty=(raw.difficulty or "").strip() or None,
        )
        for category in BEAM_CATEGORIES
        for index, raw in enumerate(grouped.get(category, ()))
    )
    if not questions:
        raise ValueError(f"BEAM {questions_path} contains no probing questions")
    return questions


def _reference_answer(raw: _RawQuestion) -> str:
    """Return the category's reference answer under whichever key carries it."""
    for value in (raw.answer, raw.ideal_answer, raw.ideal_response, raw.ideal_summary):
        if value and value.strip():
            return value.strip()
    return ""


def _role(value: str) -> Literal["user", "assistant"]:
    normalized = value.strip().casefold()
    if normalized in {"user", "assistant"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"invalid BEAM turn role: {value}")


def _anchor(value: str) -> datetime:
    parts = value.strip().split("-")
    if len(parts) != 3 or parts[0].casefold() not in _MONTHS:
        raise ValueError(f"invalid BEAM time anchor: {value}")
    month = _MONTHS[parts[0].casefold()]
    try:
        return datetime(int(parts[2]), month, int(parts[1]), tzinfo=timezone.utc)
    except ValueError as error:
        raise ValueError(f"invalid BEAM time anchor: {value}") from error
