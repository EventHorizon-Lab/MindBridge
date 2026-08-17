"""LoCoMo -> AML case loader.

Turns the official `locomo10.json` release into the benchmark-neutral
`AmlCase` model (see `mindbridge.benchmarks.aml.cases`). This loader does not
chunk messages and does not evaluate anything — a later driver replays the
full message history through MindBridge and feeds the retrieved context,
plus each question's `gold_answer` payload, to the vendored LoCoMo scoring
pipeline.

`src/mindbridge/benchmarks/locomo.py` already adapts this same corpus for an
older, unrelated benchmark runner, but into pydantic contract models shaped
for that runner (speaker names instead of chat roles, required non-empty
answers, category-5 answer substitution, evidence-id parsing). None of that
fits the AML case model, which just needs `{role, content, timestamp}`
message dicts and a `gold_answer` payload, so this loader parses the raw
JSON independently rather than reusing it.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion

_SESSION_KEY = re.compile(r"session_(\d+)$")


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    speaker: str
    text: str


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    # Adversarial (category 5) questions in the real corpus drop "answer"
    # entirely and carry "adversarial_answer" instead — both optional here so
    # either shape validates; `_gold_answer` below picks whichever is present.
    answer: str | int | float | list[str | int | float] | None = None
    adversarial_answer: str | int | float | list[str | int | float] | None = None


class _RawSample(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sample_id: str
    conversation: dict[str, object]
    qa: list[_RawQuestion]


def load(path: Path) -> tuple[AmlCase, ...]:
    """Load the official LoCoMo corpus into benchmark-neutral AML cases."""
    samples = TypeAdapter(list[_RawSample]).validate_json(path.read_bytes())
    return tuple(_case(sample) for sample in samples)


def _case(sample: _RawSample) -> AmlCase:
    questions = tuple(
        AmlQuestion(
            question_id=f"{sample.sample_id}:qa{index:04d}",
            question=qa.question,
            payload={"gold_answer": _gold_answer(sample.sample_id, index, qa)},
        )
        for index, qa in enumerate(sample.qa)
    )
    return AmlCase(
        user_id=f"locomo:{sample.sample_id}",
        messages=_messages(sample.conversation),
        questions=questions,
    )


def _gold_answer(
    sample_id: str, index: int, qa: _RawQuestion
) -> str | int | float | list[str | int | float]:
    if qa.answer is not None:
        return qa.answer
    if qa.adversarial_answer is not None:
        return qa.adversarial_answer
    raise ValueError(f"{sample_id}:qa{index:04d} has neither 'answer' nor 'adversarial_answer'")


def _messages(conversation: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Flatten `session_N` turns in numeric order, tagging each with its session date."""
    speaker_a = conversation["speaker_a"]
    session_numbers = sorted(
        int(match.group(1)) for key in conversation if (match := _SESSION_KEY.fullmatch(key))
    )
    messages: list[dict[str, object]] = []
    for number in session_numbers:
        date_time = conversation.get(f"session_{number}_date_time")
        turns = TypeAdapter(list[_RawTurn]).validate_python(conversation[f"session_{number}"])
        for turn in turns:
            role = "user" if turn.speaker == speaker_a else "assistant"
            messages.append({"role": role, "content": turn.text, "timestamp": date_time})
    return tuple(messages)
