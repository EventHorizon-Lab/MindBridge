"""PersonaMem v1 -> AML case loader.

Turns the official `questions_32k.csv` + `shared_contexts_32k.jsonl` release
pair into the benchmark-neutral `AmlCase` model (see
`mindbridge.benchmarks.aml.cases`). This loader does not chunk messages and
does not evaluate anything — a later driver replays the message history
through MindBridge and feeds the retrieved context, plus each question's
payload, to the vendored `personamem/pipeline_v1.py`.

### Modelling tension: one persona, many truncation points

One `shared_context_id` is a full simulated chat history for one persona,
and `AmlCase` holds exactly one message list per scope — but every question
asked against that persona was generated against its *own* prefix of that
history (`end_index_in_shared_context`), so different questions in the same
case legitimately see different message counts.

Resolution taken here: `AmlCase.messages` carries the **full, untruncated**
shared context for the persona (a superset of every question's slice), and
each question's `end_index_in_shared_context` rides along in its payload
(alongside the pipeline-required `id`/`all_options`/`correct_answer` keys)
purely as loader metadata — the vendored pipeline itself never reads it, but
a driver needs it to slice `case.messages[:end_index]` into the exact
`context_messages` the question was generated against before calling the
pipeline. The alternative (truncating at the case level) would have to pick
a single cut for every question sharing that persona, silently starving
some questions of context or leaking future context into others; carrying
the per-question boundary forward instead keeps the full history available
and defers the cut to whoever actually builds pipeline input. Consequence:
a driver that naively ingests all of `case.messages` per case without
consulting `end_index_in_shared_context` per question will leak
later-in-conversation content into earlier questions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: str
    user_question_or_message: str
    correct_answer: str
    all_options: str
    shared_context_id: str
    end_index_in_shared_context: int


def load(questions_csv: Path, contexts_jsonl: Path) -> tuple[AmlCase, ...]:
    """Load the official PersonaMem v1 corpus into benchmark-neutral AML cases."""
    shared_contexts = _read_shared_contexts(contexts_jsonl)
    questions_by_scope = _group_questions(_read_questions(questions_csv))
    return tuple(
        _case(shared_context_id, rows, shared_contexts[shared_context_id])
        for shared_context_id, rows in questions_by_scope.items()
    )


def _read_shared_contexts(path: Path) -> dict[str, tuple[dict[str, object], ...]]:
    """Parse `{shared_context_id: [{role, content}, ...]}` per line into messages."""
    shared: dict[str, tuple[dict[str, object], ...]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw: dict[str, object] = json.loads(line)
        for shared_context_id, turns in raw.items():
            parsed = TypeAdapter(list[_RawTurn]).validate_python(turns)
            shared[shared_context_id] = tuple(
                {"role": turn.role, "content": turn.content, "timestamp": None} for turn in parsed
            )
    return shared


def _read_questions(path: Path) -> list[_RawQuestion]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return TypeAdapter(list[_RawQuestion]).validate_python(rows)


def _group_questions(questions: list[_RawQuestion]) -> dict[str, list[_RawQuestion]]:
    """Group rows by `shared_context_id`, preserving first-seen order."""
    grouped: dict[str, list[_RawQuestion]] = {}
    for question in questions:
        grouped.setdefault(question.shared_context_id, []).append(question)
    return grouped


def _case(
    shared_context_id: str,
    rows: list[_RawQuestion],
    messages: tuple[dict[str, object], ...],
) -> AmlCase:
    questions = tuple(
        AmlQuestion(
            question_id=row.question_id,
            question=row.user_question_or_message,
            payload={
                "id": row.question_id,
                "all_options": row.all_options,
                "correct_answer": row.correct_answer,
                "end_index_in_shared_context": row.end_index_in_shared_context,
            },
        )
        for row in rows
    )
    return AmlCase(
        user_id=f"personamem-v1:{shared_context_id}",
        messages=messages,
        questions=questions,
    )
