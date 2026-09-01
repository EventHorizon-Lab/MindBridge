"""Thin adapter for the official CL-Bench JSONL release.

CL-Bench publishes no `question` field. Every record's final `user` turn mixes
a reference document -- up to ~150,000 characters -- with the query in one
string. This adapter splits that turn at its last blank-line paragraph break:
everything before the break is corpus to remember, the trailing paragraph is
the question.

Measured over the pinned 1,899-record release:

* 1,322 records (69.6%) have a blank-line break. The resulting question has a
  median length of 434 characters and a 90th percentile of 1,533 -- but 75 of
  them still yield a question of 2,000 characters or more, so finding a break
  is not by itself evidence of a clean split.
* 522 (27.5%) have no break and are already short self-contained instructions,
  where taking the whole turn as the question is correct.
* 55 (2.9%) have no break and a long turn. No cheaper machine-checkable signal
  separates the question there, so the whole turn is kept: a "trailing
  sentence" rule, for instance, slices the last entry off a bibliography.

The 130 records in the last two groups -- oversized however they got there --
carry `question_unsliced`, because an oversized question has the same effect
on the answer prompt whichever branch produced it.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

CLBENCH_ADAPTER_VERSION = "clbench_official_v1"

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")

# Separates "a short final turn with no blank-line break, where the whole turn
# is the question" (522 records) from "a long final turn with no break, where
# taking the whole turn blows the answer prompt and voids the retrieval test"
# (55 records). 2,000 characters cleanly separates those two populations in the
# pinned release; naming it keeps the loader and its tests from drifting on
# what "too long to be a question" means.
OVERSIZED_QUESTION_CHARACTERS = 2_000


class CLBenchTurn(ContractModel):
    """One non-system turn kept as corpus rather than as the question.

    `content` is a bare `str` rather than `NonEmptyString`: reference documents
    in this release reach ~150,000 characters, well past that alias's 2,048
    cap, and truncating one would silently delete the material the rubrics
    grade against.
    """

    turn_id: Identifier
    role: NonEmptyString
    content: str

    @model_validator(mode="after")
    def require_content(self) -> CLBenchTurn:
        if not self.content.strip():
            raise ValueError("CL-Bench turns require text")
        return self


class CLBenchTask(ContractModel):
    """One self-contained context-learning task and its official rubrics."""

    task_id: Identifier
    context_id: Identifier
    context_category: NonEmptyString
    sub_category: NonEmptyString
    system_prompt: str
    turns: tuple[CLBenchTurn, ...]
    # Also a bare `str`: the 55 records with no blank-line break keep their
    # whole final turn as the question, which is what `question_unsliced` flags.
    question: str
    rubrics: tuple[str, ...] = Field(min_length=1)
    question_unsliced: bool

    @model_validator(mode="after")
    def require_question_and_rubrics(self) -> CLBenchTask:
        if not self.question.strip():
            raise ValueError("CL-Bench tasks require a question")
        if any(not rubric.strip() for rubric in self.rubrics):
            raise ValueError("CL-Bench rubrics must not be blank")
        return self


class _RawMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str


class _RawMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str
    context_id: str
    context_category: str
    sub_category: str


class _RawRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    messages: list[_RawMessage] = Field(min_length=1)
    rubrics: list[str] = Field(min_length=1)
    metadata: _RawMetadata


_RECORD = TypeAdapter(_RawRecord)


def load_clbench(dataset_path: Path) -> tuple[CLBenchTask, ...]:
    """Load the official `CL-bench.jsonl` release.

    Records are split on the newline alone, never `str.splitlines()`: the
    pinned release carries 343 bare U+2028 characters, which `splitlines()`
    treats as line breaks even though JSON does not, cutting those records in
    half mid-string so they fail to parse.
    """
    records = tuple(
        _RECORD.validate_json(line)
        for line in dataset_path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    )
    tasks = tuple(_task(record) for record in records)
    if not tasks:
        raise ValueError("CL-Bench annotations must not be empty")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("CL-Bench annotations contain duplicate task IDs")
    return tasks


def _task(record: _RawRecord) -> CLBenchTask:
    system_prompt = next(
        (message.content for message in record.messages if message.role == "system"), ""
    )
    conversation = [message for message in record.messages if message.role != "system"]
    if not conversation:
        raise ValueError(
            f"CL-Bench task {record.metadata.task_id} has no turn to derive a question from"
        )
    *prior, last = conversation
    corpus_text, question, unsliced = split_question(last.content)
    turns = [
        CLBenchTurn(turn_id=f"{record.metadata.task_id}_T{index:04d}", role=turn.role, content=text)
        for index, (turn, text) in enumerate(
            [(turn, turn.content) for turn in prior] + [(last, corpus_text)]
        )
        if text.strip()
    ]
    return CLBenchTask(
        task_id=record.metadata.task_id,
        context_id=record.metadata.context_id,
        context_category=record.metadata.context_category,
        sub_category=record.metadata.sub_category,
        system_prompt=system_prompt,
        turns=tuple(turns),
        question=question,
        rubrics=tuple(rubric.strip() for rubric in record.rubrics if rubric.strip()),
        question_unsliced=unsliced,
    )


def split_question(content: str) -> tuple[str, str, bool]:
    """Split a final turn into `(corpus_text, question, question_unsliced)`.

    `question_unsliced` reports the length of the resulting question, not which
    branch produced it: a record that did split at a blank line but whose
    trailing paragraph is itself oversized blows the answer prompt exactly like
    the whole-turn fallback, so it carries the same flag. 75 of the pinned
    release's records are in exactly that state.
    """
    stripped = content.rstrip()
    paragraphs = [part for part in _PARAGRAPH_BREAK.split(stripped) if part.strip()]
    if len(paragraphs) < 2:
        corpus_text, question = "", stripped
    else:
        corpus_text, question = "\n\n".join(paragraphs[:-1]), paragraphs[-1].strip()
    return corpus_text, question, len(question) >= OVERSIZED_QUESTION_CHARACTERS
