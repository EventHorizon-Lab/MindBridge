"""LoCoMo-Refined -> AML case loader.

Turns `mem-eval-suite/LoCoMo_refined`'s `data/raw/locomo_refined.json` release into
the benchmark-neutral `AmlCase` model (see `mindbridge.benchmarks.aml.cases`). This
loader does not chunk messages and does not evaluate anything -- a later driver
replays the full message history through MindBridge and feeds the retrieved context,
plus each question's `gold_answer` payload, to the vendored `locomo-refined` scoring
pipeline. That pipeline is the one AML's textual suite actually runs LoCoMo under, so
the refined release is the corpus it expects.

`src/mindbridge/benchmarks/locomo_refined.py` adapts this same corpus for the native
runner, but into pydantic contract models shaped for that runner (speaker names instead
of chat roles, evidence-id parsing, per-category counts). None of that fits the AML case
model, which just needs `{role, content, timestamp}` message dicts and a `gold_answer`
payload, so this loader parses the raw JSON independently rather than reusing it. The one
piece both must agree on -- the release's own `qa_id` -- is imported rather than repeated.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion
from mindbridge.benchmarks.locomo_refined import official_qa_id

_SESSION_KEY = re.compile(r"session_(\d+)$")

# LoCoMo-Refined keeps LoCoMo's session timestamp wording, e.g. "1:56 pm on 8 May,
# 2023" -- free text, no timezone. AML's wire contract (`AmlMessage.timestamp`)
# requires epoch milliseconds, so this is parsed and treated as UTC: the source
# carries no offset, and UTC is the least-wrong assumption available.
_TIMESTAMP_FORMAT = "%I:%M %p on %d %B, %Y"


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    speaker: str
    text: str


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    # Every refined question carries a list of complete gold candidates, any one of
    # which the official evaluator will accept. The vendored pipeline has no way to
    # express alternatives -- its `gold_answer()` renders a list by joining it with
    # newlines, which its own judge would read as "cover all of these" -- so only the
    # first candidate is handed over. That makes an AML LoCoMo-Refined score no higher
    # than, and on the ~200 multi-candidate questions sometimes lower than, the same
    # predictions scored by `mem-eval-suite/LoCoMo_refined`'s own `run_eval.sh`.
    answer: list[str | int | float] = Field(min_length=1)


class _RawSample(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sample_id: str
    conversation: dict[str, object]
    qa: list[_RawQuestion]


def load(path: Path) -> tuple[AmlCase, ...]:
    """Load the official LoCoMo-Refined corpus into benchmark-neutral AML cases."""
    samples = TypeAdapter(list[_RawSample]).validate_json(path.read_bytes())
    return tuple(_case(sample) for sample in samples)


def _case(sample: _RawSample) -> AmlCase:
    questions = tuple(
        AmlQuestion(
            question_id=official_qa_id(sample.sample_id, index),
            question=qa.question,
            # `id` is the release's own `qa_id`, so a row this run writes joins straight
            # onto `data/public/questions.jsonl` -- and `driver.row_id` prefers a
            # payload `id` over its own `{user_id}#{question_id}` format, which is what
            # the CLI's resume check reads back.
            payload={
                "id": official_qa_id(sample.sample_id, index),
                "gold_answer": str(qa.answer[0]),
            },
        )
        for index, qa in enumerate(sample.qa)
    )
    return AmlCase(
        user_id=f"locomo-refined:{sample.sample_id}",
        messages=_messages(sample.conversation),
        questions=questions,
    )


def _messages(conversation: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Flatten `session_N` turns in numeric order, tagging each with its session date."""
    speaker_a = conversation["speaker_a"]
    session_numbers = sorted(
        int(match.group(1)) for key in conversation if (match := _SESSION_KEY.fullmatch(key))
    )
    messages: list[dict[str, object]] = []
    for number in session_numbers:
        date_time = conversation.get(f"session_{number}_date_time")
        timestamp = _parse_timestamp(date_time)
        turns = TypeAdapter(list[_RawTurn]).validate_python(conversation[f"session_{number}"])
        for turn in turns:
            role = "user" if turn.speaker == speaker_a else "assistant"
            messages.append({"role": role, "content": turn.text, "timestamp": timestamp})
    return tuple(messages)


def _parse_timestamp(raw: object) -> int | None:
    """Parse a LoCoMo-Refined session date-time string into epoch milliseconds (UTC)."""
    if not isinstance(raw, str):
        return None
    parsed = datetime.strptime(raw, _TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000)
