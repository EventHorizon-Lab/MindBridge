"""LongMemEval -> AML case loader.

Turns the official `longmemeval_s` release into the benchmark-neutral
`AmlCase` model (see `mindbridge.benchmarks.aml.cases`). Unlike LoCoMo, each
record in the raw file already *is* one question plus its own independent
haystack (sessions are not shared across questions), so this loader produces
exactly one `AmlCase` per record rather than grouping several questions under
a shared conversation.

The vendored `longmemeval-s` pipeline is a byte-for-byte copy of the LoCoMo
pipeline (same answer/evaluation contracts), so the payload shape mirrors
`loaders/locomo.py`: rename the dataset's `answer` to `gold_answer` and carry
the dataset's `question_id` as `id`. LongMemEval is single-user, so the
`speaker_2_*` keys the shared pipeline optionally reads are simply omitted
rather than emitted empty.

This loader does not chunk messages and does not evaluate anything — a later
driver replays the full message history through MindBridge and feeds the
retrieved context, plus each question's payload, to the vendored pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion

# LongMemEval's session date, e.g. "2023/05/20 (Sat) 02:21" -- free text, no
# timezone. AML's wire contract (`AmlMessage.timestamp`) requires epoch
# milliseconds, so this is parsed and treated as UTC: the source carries no
# offset, and UTC is the least-wrong assumption available.
_TIMESTAMP_FORMAT = "%Y/%m/%d (%a) %H:%M"


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str


class _RawRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: str
    question: str
    answer: str | int | float | list[str | int | float]
    haystack_dates: list[str]
    haystack_sessions: list[list[_RawTurn]]


def load(path: Path) -> tuple[AmlCase, ...]:
    """Load the official LongMemEval-S corpus into benchmark-neutral AML cases."""
    records = TypeAdapter(list[_RawRecord]).validate_json(path.read_bytes())
    return tuple(_case(record) for record in records)


def _case(record: _RawRecord) -> AmlCase:
    question = AmlQuestion(
        question_id=record.question_id,
        question=record.question,
        payload={"id": record.question_id, "gold_answer": record.answer},
    )
    return AmlCase(
        user_id=f"longmemeval:{record.question_id}",
        messages=_messages(record),
        questions=(question,),
    )


def _messages(record: _RawRecord) -> tuple[dict[str, object], ...]:
    """Flatten haystack sessions in file order, tagging each turn with its session date."""
    messages: list[dict[str, object]] = []
    for session, date in zip(record.haystack_sessions, record.haystack_dates, strict=True):
        timestamp = _parse_timestamp(date)
        for turn in session:
            messages.append({"role": turn.role, "content": turn.content, "timestamp": timestamp})
    return tuple(messages)


def _parse_timestamp(raw: str) -> int:
    """Parse a LongMemEval haystack date string into epoch milliseconds (UTC)."""
    parsed = datetime.strptime(raw, _TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000)
