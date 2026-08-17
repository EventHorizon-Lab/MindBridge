"""CL-Bench -> AML case loader.

Turns the official `CL-bench.jsonl` release into the benchmark-neutral
`AmlCase` model (see `mindbridge.benchmarks.aml.cases`). This loader does not
chunk messages and does not evaluate anything -- a later driver ingests each
case's `messages` into MindBridge under its `user_id`, then feeds the
retrieved context, plus each question's payload, to the vendored
`clbench/pipeline.py`.

### The central problem: there is no `question` field

Every raw record's `messages` list ends in a `user` turn that mixes a
reference document (sometimes ~150,000 characters) with the actual query, in
one string, with no dedicated field for either half. Feeding that whole turn
through as `question` would both blow the answer prompt and make the
"retrieval" test vacuous (nothing left to memorize ahead of time).

### The slicing rule, measured against the real 1,899-line corpus

Split the final `user` turn on its last blank-line paragraph break
(`\n` + optional whitespace + `\n`): everything before the break folds into
the ingested history (as one message, under the same role, appended after
any real prior turns); the trailing paragraph is the question.

Measured outcomes over the real corpus:

* 69.6% (1,322/1,899) have at least one blank-line break in their final
  turn -- for these the trailing paragraph is a clean, short question
  (median 434 characters, 90th percentile 1,533).
* 30.4% (577/1,899) have no blank-line break at all. Of those, the large
  majority (522/1,899, 27.5% overall) are already short (< 2,000 characters)
  turns with nothing that needs separating -- treating the whole turn as the
  question is harmless (it's a short, self-contained instruction, not a
  document-plus-question blob).
* Only 2.9% of the full corpus (55/1,899) are a genuine failure of this
  rule: a long (>= 2,000 character) final turn with no blank-line break to
  key off of -- e.g. a short leading question followed by an unbroken
  academic paper with no paragraph spacing, or a table/bibliography with no
  paragraph structure near the end. For those, this loader falls back to
  the whole turn as the question and folds in no extra history text: the
  same "blow the prompt" outcome the rule exists to avoid, because no
  cheaper, safe, machine-checkable signal was found in the real data for
  that minority. No further heuristic (e.g. keying off a leading "?", or a
  single-newline fallback) was added on top, because spot-checks of these
  55 records showed no single additional rule that generalizes without
  misfiring on the other patterns already handled correctly above (a
  "trailing sentence" rule, e.g., would wrongly slice off the last citation
  in a reference list rather than a real question).

### The scope rule (measured, not assumed)

`metadata.task_id` is unique across all 1,899 lines (verified: zero
duplicates), and every record's `messages` list is fully self-contained --
one JSONL line supplies its own complete turn sequence, including its own
`assistant` reply where one exists. So one line = one scope = one
`AmlCase` with exactly one question, matching longmemeval.py's pattern
rather than locomo.py's/beam.py's "group several questions under one
shared conversation" pattern.

This does *not* mean lines never share content: `metadata.context_id` is
shared by up to 12 lines (measured: 478 of 500 distinct `context_id`s have
more than one line), and sibling lines under the same `context_id` do reuse
the identical underlying reference document (verified: two lines sharing a
`context_id` had byte-identical 158,789-character first `user` turns).
But siblings are independently-authored tasks built on that shared source
material, not slices of one growing conversation the way PersonaMem v1's
`shared_context_id` is: a 4-message sibling carries its *own* `assistant`
turn, generated for its *own* question, that a 2-message sibling never
sees. Merging siblings into one scope would require picking an ingestion
order and would leak one sibling's assistant answer into another's
retrieval context -- a real correctness bug, not a simplification. Keeping
one case per line (as the schema reference already specifies) avoids that
leak at the cost of re-ingesting the shared document once per sibling.

### No timestamps

The dataset has no timestamps anywhere. The `timestamp` key is omitted
entirely from every message dict (not set to `None`), matching this task's
brief -- the schema reference's own CL-Bench pseudocode sets `"timestamp":
None`, but the task brief is explicit ("No timestamps anywhere; omit the
field"), and this loader follows the brief.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


class _RawMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str


class _RawMetadata(BaseModel):
    """`extra="allow"` (unlike every other loader's raw models) so that
    `model_dump()` reproduces the raw `metadata` dict verbatim, including any
    fields this loader doesn't otherwise care about -- required so the
    vendored pipeline's `row_id()` fallback to `metadata.task_id` keeps
    working even though this loader also sets `id` explicitly.
    """

    model_config = ConfigDict(extra="allow")

    task_id: str


class _RawRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    messages: list[_RawMessage] = Field(min_length=1)
    rubrics: list[str] = Field(min_length=1)
    metadata: _RawMetadata


def load(path: Path) -> tuple[AmlCase, ...]:
    """Load the official CL-Bench corpus into benchmark-neutral AML cases."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    records = [TypeAdapter(_RawRecord).validate_json(line) for line in lines]
    return tuple(_case(record) for record in records)


def _case(record: _RawRecord) -> AmlCase:
    system_prompt = next(
        (message.content for message in record.messages if message.role == "system"), ""
    )
    non_system = [message for message in record.messages if message.role != "system"]
    if not non_system:
        raise ValueError("CL-Bench record has no user/assistant turns to derive a question from")
    *prior, last = non_system

    history_text, question_text = _split_question(last.content)
    messages: list[dict[str, object]] = [
        {"role": message.role, "content": message.content} for message in prior
    ]
    if history_text:
        messages.append({"role": last.role, "content": history_text})

    metadata = record.metadata.model_dump()
    task_id = record.metadata.task_id
    question = AmlQuestion(
        question_id=task_id,
        question=question_text,
        payload={
            "id": task_id,
            "system_prompt": system_prompt,
            "rubrics": record.rubrics,
            "metadata": metadata,
        },
    )
    return AmlCase(
        user_id=f"clbench:{task_id}",
        messages=tuple(messages),
        questions=(question,),
    )


def _split_question(content: str) -> tuple[str, str]:
    """Split a final user turn into `(history_text, question)` at its last
    blank-line paragraph break. See the module docstring for the rule and
    its measured coverage. Returns `("", content)` when no break is found --
    the whole turn becomes the question and nothing is folded into history.
    """
    stripped = content.rstrip()
    paragraphs = [p for p in _PARAGRAPH_BREAK.split(stripped) if p.strip()]
    if len(paragraphs) < 2:
        return "", stripped
    return "\n\n".join(paragraphs[:-1]), paragraphs[-1].strip()
