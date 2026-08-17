"""BEAM -> AML case loader.

Turns one conversation's official `chat.json` + `probing_questions.json`
pair into the benchmark-neutral `AmlCase` model (see
`mindbridge.benchmarks.aml.cases`). `.benchmarks/beam/` is a full upstream
research-repo checkout, not a packaged dataset: the actual per-conversation
transcripts and probing questions live at
`.benchmarks/beam/chats/{100K,500K,1M,10M}/{conv_id}/`, one directory per
conversation, holding `chat.json` and
`probing_questions/probing_questions.json` (plus generation scaffolding this
loader ignores). This loader takes one conversation's two files per call and
always returns a single-element tuple; a driver calls it once per
conversation directory. This loader does not chunk messages and does not
evaluate anything -- a later driver ingests `messages` into MindBridge under
`user_id`, then feeds the retrieved context, plus each question's payload,
to the vendored `beam/pipeline.py`.

### No per-question scope splitting needed (measured, not assumed)

Unlike PersonaMem v1 (questions generated against different prefixes of a
shared history), BEAM's probing questions are generated after the full
conversation is written and reference facts from anywhere across it.
Verified directly against the real 100K corpus (20 conversations, 400
questions total): every question's `source_chat_ids` (or the nested variant
under `original_info`/`updated_info` etc.) points at a message `id` that
exists somewhere in that same conversation's full `chat.json` -- 0
violations across all 400 questions. So one conversation directory is
already a correct, un-leaky retrieval scope: one `AmlCase` per conversation,
no per-question grouping like `personamem_v1` needs.

### The category field's real name and values

The category is not a field on each question object -- it is the **top-level
dict key** of `probing_questions.json` (confirmed against
`answer_generation.py`'s own `for key in data.keys()` iteration). Measured
values across the real corpus, present uniformly on every one of the 20
conversations checked: `abstention`, `contradiction_resolution`,
`event_ordering`, `information_extraction`, `instruction_following`,
`knowledge_update`, `multi_session_reasoning`, `preference_following`,
`summarization`, `temporal_reasoning`. That dict key is copied verbatim into
the payload's `question_type` -- required so the vendored pipeline's
`event_ordering` branch (extra Kendall-tau scoring) fires correctly.

One gotcha found in the real data: the `information_extraction` category's
raw question objects sometimes carry their own inner field also named
`"question_type"` (e.g. `"context_date/time"`) -- a sub-classification, not
the category. That field is ignored (`extra="ignore"`); `question_type` in
the built payload always comes from the outer dict key.

### id and rubric

No raw question carries an `id` -- one is synthesized as
`f"{conversation_id}:{category}:{index:04d}"` (unique within a conversation,
which is all `rows()` requires per answer/eval file). `rubric` is the
dataset's own key, already a non-empty list of plain strings matching the
pipeline's `rubric_nuggets`/`rubrics`/`rubric` fallback -- kept verbatim,
not renamed. An empty or missing rubric list raises rather than being
skipped, per pydantic's `min_length=1` constraint below: the pipeline's
`rubric_items()` would raise on it downstream anyway, so a loader that
silently dropped such a question would just be hiding a real corpus bug.

### Timestamps

Real timestamps (`time_anchor`) are free-text month-day-year anchors (e.g.
`"March-15-2024"`) present on a minority of turns -- typically just the user
turn opening a batch (measured: 5,642 / 5,732 turns across the 100K corpus
have none at all). The `timestamp` key is omitted entirely for turns without
one rather than invented or carried forward from a prior turn. When present,
`time_anchor` is parsed to epoch milliseconds (`%B-%d-%Y`, treated as UTC --
the source carries no timezone) to satisfy AML's wire contract, which types
`timestamp` as `int | None`, not free text.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion

# BEAM's turn timestamp, e.g. "March-15-2024" -- free text, no timezone. AML's
# wire contract (`AmlMessage.timestamp`) requires epoch milliseconds, so this
# is parsed and treated as UTC: the source carries no offset, and UTC is the
# least-wrong assumption available.
_TIMESTAMP_FORMAT = "%B-%d-%Y"


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str
    time_anchor: str | None = None


class _RawBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    turns: list[list[_RawTurn]]


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    rubric: list[str] = Field(min_length=1)


def load(chat: Path, questions: Path) -> tuple[AmlCase, ...]:
    """Load one BEAM conversation's chat + probing questions into an AML case."""
    # Conversation identity is `{size}:{conv_id}`, not `conv_id` alone: BEAM
    # nests conversations under a size bucket
    # (`.benchmarks/beam/chats/{100K,500K,1M,10M}/{conv_id}/`), and conv_id
    # restarts from "1" in every bucket. Dropping the size segment collides
    # unrelated conversations into one `user_id` -- the same MindBridge
    # retrieval scope -- contaminating recall across sizes.
    conversation_id = f"{chat.parent.parent.name}:{chat.parent.name}"
    batches = TypeAdapter(list[_RawBatch]).validate_json(chat.read_bytes())
    by_category = TypeAdapter(dict[str, list[_RawQuestion]]).validate_json(questions.read_bytes())
    case = AmlCase(
        user_id=f"beam:{conversation_id}",
        messages=_messages(batches),
        questions=_questions(conversation_id, by_category),
    )
    return (case,)


def _messages(batches: list[_RawBatch]) -> tuple[dict[str, object], ...]:
    messages: list[dict[str, object]] = []
    for batch in batches:
        for turn_pair in batch.turns:
            for turn in turn_pair:
                message: dict[str, object] = {"role": turn.role, "content": turn.content}
                if turn.time_anchor is not None:
                    message["timestamp"] = _parse_timestamp(turn.time_anchor)
                messages.append(message)
    return tuple(messages)


def _parse_timestamp(raw: str) -> int:
    """Parse a BEAM `time_anchor` string into epoch milliseconds (UTC)."""
    parsed = datetime.strptime(raw, _TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000)


def _questions(
    conversation_id: str, by_category: dict[str, list[_RawQuestion]]
) -> tuple[AmlQuestion, ...]:
    return tuple(
        _question(conversation_id, category, index, raw)
        for category, raws in by_category.items()
        for index, raw in enumerate(raws)
    )


def _question(conversation_id: str, category: str, index: int, raw: _RawQuestion) -> AmlQuestion:
    question_id = f"{conversation_id}:{category}:{index:04d}"
    return AmlQuestion(
        question_id=question_id,
        question=raw.question,
        payload={"id": question_id, "rubric": raw.rubric, "question_type": category},
    )
