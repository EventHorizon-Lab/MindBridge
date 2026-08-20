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

### The 10M tier's plan wrapping is grouping, not scope (measured, not assumed)

The 10M tier alone nests one level deeper: its `chat.json` is a list of
single-key `{"plan-N": [...batches...]}` dicts rather than a list of batch
objects. Measured across all 10 of that tier's conversations: every one is
exactly 10 such elements, keyed `plan-1` ... `plan-10` in document order. (The
sibling `plan-0/` ... `plan-9/` directories on disk are off by one from the
JSON keys; the loader reads only `chat.json`, so that mismatch is irrelevant
here.)

Those plans are consecutive stretches of one conversation, not separate
conversations, so a 10M directory is still exactly one `AmlCase` -- the same
rule as every other tier. Two measurements settle it, the same way
`source_chat_ids` reachability settles the question-scope split above:

- Turn `id`s are conversation-global and strictly increasing across plan
  boundaries in document order -- `0..N-1`, no gaps, no restarts, on all 10
  conversations (18,760 to 23,716 turns each, 208,696 total).
- 47 of the 176 probing questions carrying `source_chat_ids` point at turns
  living in **two different plans** (200 questions total; `abstention` never
  carries source ids, nor do 4 of the 20 `summarization` questions). All 176
  resolve to a turn in the same conversation -- 0 violations, matching the
  100K result above. The questions' own `plan_reference` field says the same
  thing in words, e.g. `"Plan 0-1"`.

One case per plan would therefore push the evidence for 27% of the answerable
questions outside the retrieval scope of the case that asks them --
manufacturing unanswerable questions rather than preventing leakage. So the
plan key is dropped as the generation-pipeline grouping artifact it is, and
the batches under it are concatenated in document order.

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


# The 10M tier's extra nesting level: one `{"plan-N": [...batches...]}` dict per
# plan where the other tiers put a bare batch object. The union of the two in
# `_batches` is unambiguous -- a plan group has no `turns`, and a batch's
# `batch_number` is an int, not a list of batches -- so neither shape can
# validate as the other.
_RawPlanGroup = dict[str, list[_RawBatch]]


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
    batches = _batches(chat)
    by_category = TypeAdapter(dict[str, list[_RawQuestion]]).validate_json(questions.read_bytes())
    case = AmlCase(
        user_id=f"beam:{conversation_id}",
        messages=_messages(batches),
        questions=_questions(conversation_id, by_category),
    )
    return (case,)


def _batches(chat: Path) -> list[_RawBatch]:
    """Read one `chat.json` as batches, flattening the 10M tier's plan grouping.

    Both shapes concatenate in document order, which is the conversation's own
    order -- turn ids ascend straight across plan boundaries (see the module
    docstring). Every real 10M element carries exactly one plan key, so a plan
    key labels a stretch of that timeline and never orders it.
    """
    elements = TypeAdapter(list[_RawBatch | _RawPlanGroup]).validate_json(chat.read_bytes())
    batches: list[_RawBatch] = []
    for element in elements:
        if isinstance(element, _RawBatch):
            batches.append(element)
            continue
        for plan_batches in element.values():
            batches.extend(plan_batches)
    return batches


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
