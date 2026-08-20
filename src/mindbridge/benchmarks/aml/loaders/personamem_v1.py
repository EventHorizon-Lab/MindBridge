"""PersonaMem v1 -> AML case loader.

Turns the official `questions_32k.csv` + `shared_contexts_32k.jsonl` release
pair into the benchmark-neutral `AmlCase` model (see
`mindbridge.benchmarks.aml.cases`). This loader does not chunk messages and
does not evaluate anything — a later driver ingests each case's `messages`
into MindBridge under its `user_id`, then feeds the retrieved context, plus
each question's payload, to the vendored `personamem/pipeline_v1.py`.

### Modelling tension: one persona, many truncation points

One `shared_context_id` is a full simulated chat history for one persona,
and multiple questions reuse prefixes of it (`end_index_in_shared_context`)
— but AML's `Add`/`Search` protocol gives a driver no per-question slicing
opportunity: `Add` writes a history into the memory system once, and
`Search` retrieves against a `user_id` after those memories are already
stored. There is no hook at search time to say "pretend only the first N
messages were ever added." A single case covering the whole persona would
therefore have to ingest everything, leaking later-in-conversation content
into earlier questions' retrieval.

Resolution taken here: split each persona's questions by their exact
truncation point. One `AmlCase` is emitted per `(shared_context_id,
end_index_in_shared_context)` pair, `messages` truncated in the loader to
exactly that prefix, and `user_id` includes the end index
(`f"personamem-v1:{shared_context_id}:{end_index}"`) so cases with
different cuts never collide. This is exactly as correct as one case per
question — every question in a group was generated against precisely that
prefix — while costing far less to ingest, since questions in the same
scope cluster onto relatively few distinct end indices (measured on the
real corpus: 589 questions / 37 scopes collapse to 222 groups, versus 589
if split per question). `end_index_in_shared_context` is intentionally
*not* carried in the question payload: the case structure already encodes
it, and leaving it in the payload would invite a future driver to re-slice
something that has already been sliced.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion
from mindbridge.benchmarks.artifacts import jsonl_lines


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
    questions_by_group = _group_questions(_read_questions(questions_csv))
    return tuple(
        _case(shared_context_id, end_index, rows, shared_contexts[shared_context_id])
        for (shared_context_id, end_index), rows in questions_by_group.items()
    )


def _read_shared_contexts(path: Path) -> dict[str, tuple[dict[str, object], ...]]:
    """Parse `{shared_context_id: [{role, content}, ...]}` per line into messages."""
    shared: dict[str, tuple[dict[str, object], ...]] = {}
    for line in jsonl_lines(path):
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


def _group_questions(questions: list[_RawQuestion]) -> dict[tuple[str, int], list[_RawQuestion]]:
    """Group rows by `(shared_context_id, end_index_in_shared_context)`, first-seen order.

    Splitting on the truncation point (rather than one group per
    `shared_context_id`) is what lets `AmlCase.messages` be truncated in the
    loader to exactly what each group's questions were generated against —
    see the module docstring for why a single per-persona case can't do this.
    """
    grouped: dict[tuple[str, int], list[_RawQuestion]] = {}
    for question in questions:
        key = (question.shared_context_id, question.end_index_in_shared_context)
        grouped.setdefault(key, []).append(question)
    return grouped


def _case(
    shared_context_id: str,
    end_index: int,
    rows: list[_RawQuestion],
    shared_context: tuple[dict[str, object], ...],
) -> AmlCase:
    questions = tuple(
        AmlQuestion(
            question_id=row.question_id,
            question=row.user_question_or_message,
            payload={
                "id": row.question_id,
                "all_options": row.all_options,
                "correct_answer": row.correct_answer,
            },
        )
        for row in rows
    )
    return AmlCase(
        user_id=f"personamem-v1:{shared_context_id}:{end_index}",
        messages=shared_context[:end_index],
        questions=questions,
    )
