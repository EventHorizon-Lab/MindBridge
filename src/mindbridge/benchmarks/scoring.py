"""Who scores each benchmark, declared per benchmark, and the judge that scores the rest.

Ported from lmms-eval, deliberately and including the parts that cost something. There, every
task declares a `metric_list` -- a metric name, an aggregation, and `higher_is_better` -- and the
framework never distinguishes a benchmark it can score from one it cannot: the difference lives
entirely inside the task's own `process_results`. A multiple-choice task compares option letters;
MM-Vet calls `gpt-4o-2024-11-20` from inside `process_results`; MMBench's test split declares
`metric: submission` and its aggregation writes a spreadsheet to upload. `--predict_only`
replaces every metric with `bypass`, whose aggregation returns the literal `999`.

This module is that contract for MindBridge. `SCORING` below is the `metric_list`: one entry per
benchmark saying which mode scores it and, per metric name, whether higher is better. `judge()`
is MM-Vet's `get_chat_response` -- same 0.0-to-1.0 correctness prompt, same `patience = 3`, and
the same floor, which is the consequence worth stating plainly: **a judge that cannot be reached
or cannot be parsed scores the answer 0.0, which is indistinguishable from a wrong answer.** An
upstream outage therefore reads as a lower benchmark score rather than as a failed run. That is
lmms-eval's behaviour and it was adopted on purpose.

Two places where a literal port was impossible, both because MindBridge's `Generator` protocol
has no temperature:

- MM-Vet escalates `temperature += 0.5` when the judge returns something that is not a float, up
  to `2.0`, then floors to 0.0. `GenerateRequest` carries no temperature -- generation here is
  deliberately deterministic -- so retrying the identical request would be three identical calls
  and three identical unparseable answers. The escalation is kept by varying the *prompt* instead:
  each retry appends a stricter instruction. Same purpose, same patience, same floor.
- MM-Vet reads its judge from `MODEL_VERSION`/`API_TYPE`. The equivalents here are
  `MINDBRIDGE_BENCH_JUDGE_MODEL` and its neighbours, with the credential environment-only like
  every other one in this repository.

A judge number lands in the run manifest, so `mindbridge-bench score` is no longer the only way a
benchmark reports. It stays the only way an *official* number does: the judge model this picks is
MindBridge's choice, not the benchmark's, so two runs under different judges are not comparable to
each other and neither is comparable to a leaderboard.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from mindbridge.models import GenerateRequest, Generator, ModelInput, TextPart

BYPASS_VALUE = 999.0
"""What `--predict-only` reports for every metric, copied from lmms-eval's `bypass_agg`.

    @register_aggregation("bypass")
    def bypass_agg(arr):
        return 999

A sentinel in the same column as a real accuracy, which is what it is there too. It is not a
score and no run that reports it has been evaluated.
"""

BYPASS_METRIC = "bypass"
SUBMISSION_METRIC = "submission"
"""lmms-eval's name for a split whose answers are withheld; its aggregation returns no score."""

JUDGE_METRIC = "llm_judge"
JUDGE_PATIENCE = 3
"""MM-Vet's `patience = 3`: three attempts at a parseable score before the 0.0 floor."""

JUDGE_MODEL_VARIABLE = "MINDBRIDGE_BENCH_JUDGE_MODEL"
JUDGE_ENDPOINT_VARIABLE = "MINDBRIDGE_BENCH_JUDGE_ENDPOINT"
JUDGE_API_KEY_VARIABLE = "MINDBRIDGE_BENCH_JUDGE_API_KEY"
DEFAULT_JUDGE_MODEL = "gpt-4o-2024-11-20"
"""MM-Vet's own default, kept so a ported number is comparable to one lmms-eval would produce."""

ScoringMode = Literal["runner", "judge", "external", "bypass"]
"""How one benchmark's numbers come to exist.

`runner` is exact match the runner computes itself. `judge` is this module's judge, called from
inside the run the way MM-Vet calls its own. `external` is an official scorer outside MindBridge,
whose verdict arrives through `mindbridge-bench score`. `bypass` is `--predict-only`.
"""

_JUDGE_PROMPT = """Compare the ground truth and the prediction from an AI model, and give a \
correctness score for the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, \
0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Reply with the score and nothing else.

Question: {question}
Ground truth: {reference}
Prediction: {prediction}
Correctness score:"""
"""MM-Vet's judging prompt, minus its `<AND>`/`<OR>` ground-truth operators, which are its own."""

_RETRY_INSTRUCTIONS = (
    "",
    "\n\nReply with a single number between 0.0 and 1.0. No words, no explanation.",
    "\n\nYour previous reply could not be read as a number. Output exactly one decimal "
    "between 0.0 and 1.0, such as 0.0 or 0.7. Nothing else.",
)
"""What replaces MM-Vet's `temperature += 0.5`, one entry per attempt.

The protocol here has no temperature to raise, so a retry that changed nothing would return the
same unparseable answer. Escalating the instruction is the lever that remains.
"""

_SCORE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class Metric:
    """One declared metric, as lmms-eval's `metric_list` entry declares one."""

    name: str
    higher_is_better: bool = True


@dataclass(frozen=True, slots=True)
class BenchmarkScoring:
    """One benchmark's `metric_list`: who scores it, and which way each metric points."""

    mode: ScoringMode
    metrics: tuple[Metric, ...]

    def higher_is_better(self) -> dict[str, bool]:
        """The `↑`/`↓` column, keyed by metric name."""
        return {metric.name: metric.higher_is_better for metric in self.metrics}


def _judged(*extra: Metric) -> BenchmarkScoring:
    """A free-text benchmark this module's judge scores, as MM-Vet is scored inside lmms-eval."""
    return BenchmarkScoring("judge", (Metric(JUDGE_METRIC), *extra))


_ACCURACY = (Metric("accuracy"), Metric("strict_accuracy"))

SCORING: dict[str, BenchmarkScoring] = {
    # Exact option match, computed by the runner with no network call, the way
    # `videomme_process_results` does: `1.0 if pred_ans.lower() == gt_ans.lower() else 0.0`.
    "egolife": BenchmarkScoring("runner", (Metric("accuracy"),)),
    "video-mme": BenchmarkScoring("runner", _ACCURACY),
    "video-mme-v2": BenchmarkScoring(
        "runner", (Metric("rating.overall"), Metric("accuracy.overall"))
    ),
    "supermemory": BenchmarkScoring(
        "runner",
        (
            Metric("qa_accuracy"),
            Metric("qa_mean_reciprocal_rank"),
            Metric("answerability_precision"),
            Metric("answerability_recall"),
        ),
    ),
    # Free text, so correctness is a judgement. lmms-eval makes that call in-framework and so,
    # now, does this. The official numbers for these still come from their own scorers.
    "locomo-refined": _judged(),
    "m3": _judged(),
    "memlens": _judged(),
    "atm": _judged(),
    "mem-gallery": _judged(),
    "egotempo": _judged(),
    "mm-lifelong": _judged(Metric("unofficial_reference_at_300")),
    # EgoMemReason's answers are held out by its leaderboard, which is MMBench's test split in
    # lmms-eval: `metric: submission`, an aggregation that writes the file to upload, no score.
    "egomem": BenchmarkScoring("external", (Metric(SUBMISSION_METRIC),)),
}
"""The `metric_list` of every dispatchable benchmark, keyed as `RUNNERS` keys them.

A benchmark absent from this table is one the harness cannot say anything about, which is why
`tests/unit/benchmarks/test_scoring.py` requires every entry of `RUNNERS` that is a benchmark to
appear here.
"""


@dataclass(frozen=True, slots=True)
class JudgeOutcome:
    """What one judge pass produced, including how much of it the judge could not read."""

    metrics: dict[str, float]
    failure_count: int
    """Answers floored to 0.0 because no attempt returned a parseable score.

    Counted rather than merely logged, which is the one place this departs from MM-Vet: it prints
    `failed to get a score` per question and the count never reaches the artifact, so a run whose
    judge was down is indistinguishable afterwards from a run that answered badly. The floor is
    kept; the tally is recorded beside it.
    """


def bypass_metrics() -> dict[str, float]:
    """Report `999`, as `override_metric("bypass")` does in lmms-eval.

    Takes no benchmark, because upstream's override replaces whatever the task declared rather
    than mapping each metric to a sentinel of its own.
    """
    return {BYPASS_METRIC: BYPASS_VALUE}


def judge_available() -> bool:
    """Whether a judge is configured, which is what `--predict-only` exists to avoid needing."""
    return bool(os.environ.get(JUDGE_ENDPOINT_VARIABLE, "").strip())


def configured_judge_model() -> str:
    """The judge model id this run would use, recorded in the manifest beside its numbers.

    Read from the environment rather than off the connected generator, because the `Generator`
    protocol deliberately exposes an identity only on a result and a manifest has to name the
    judge even for a pass in which every call failed.
    """
    return os.environ.get(JUDGE_MODEL_VARIABLE, "").strip() or DEFAULT_JUDGE_MODEL


def build_judge(*, request_timeout_seconds: float) -> Generator:
    """Connect the judge this run will score with, from the environment alone.

    The credential never becomes an argument, so a recorded invocation and a process list never
    carry it -- the same rule every other command in this repository follows.
    """
    from mindbridge.models.openai import create_generator

    endpoint = os.environ.get(JUDGE_ENDPOINT_VARIABLE, "").strip()
    if not endpoint:
        raise ValueError(
            f"{JUDGE_ENDPOINT_VARIABLE} must be set to score this benchmark, or pass "
            "--predict-only to write predictions without a score"
        )
    return create_generator(
        {
            "api_key": os.environ.get(JUDGE_API_KEY_VARIABLE, ""),
            "endpoint": endpoint,
            "model_id": configured_judge_model(),
            "request_timeout_seconds": request_timeout_seconds,
            # Every value but "none" makes this endpoint emit reasoning tokens first and then stop
            # on length, which arrives here as an unparseable score and burns all three attempts.
            "reasoning_effort": "none",
        }
    )


@dataclass(frozen=True, slots=True)
class JudgedAnswer:
    """One answer to score: what was asked, what was expected, and what came back."""

    question: str
    reference: str
    prediction: str


def judge_answers(
    answers: Sequence[JudgedAnswer],
    *,
    judge: Generator,
    concurrency: int,
) -> JudgeOutcome:
    """Score free-text answers with the judge, and mean them, as MM-Vet's aggregation does.

    Synchronous because every runner calls this after its own `asyncio.run` has returned, and a
    second entry point would be one more thing for eight runners to get right.
    """
    if not answers:
        raise ValueError("a judge pass needs at least one answer to score")
    scores = asyncio.run(_score_all(answers, judge=judge, concurrency=concurrency))
    return JudgeOutcome(
        metrics={JUDGE_METRIC: sum(score for score, _ in scores) / len(scores)},
        failure_count=sum(not parsed for _, parsed in scores),
    )


async def _score_all(
    answers: Sequence[JudgedAnswer],
    *,
    judge: Generator,
    concurrency: int,
) -> tuple[tuple[float, bool], ...]:
    """Score every answer, bounded, and never let one raised judge call discard the rest.

    `return_exceptions` guards the shape that has cost this repository whole paid-for passes:
    without it, one raising sibling cancels every other still in flight. Here it is belt over
    braces -- `_score_one` catches every `Exception` itself, so nothing should reach this -- and
    removing it changes nothing any test can see. It stays because the thing it guards against is
    an edit inside `_score_one`, not a judge. It does not help against a `BaseException`:
    `gather` re-raises those whatever this flag says, and a `KeyboardInterrupt` mid-pass is meant
    to end the run rather than be scored 0.0.
    """
    semaphore = asyncio.Semaphore(max(concurrency, 1))

    async def one(answer: JudgedAnswer) -> tuple[float, bool]:
        async with semaphore:
            return await _score_one(answer, judge=judge)

    settled = await asyncio.gather(*(one(answer) for answer in answers), return_exceptions=True)
    return tuple(
        (0.0, False) if isinstance(result, BaseException) else result for result in settled
    )


async def _score_one(answer: JudgedAnswer, *, judge: Generator) -> tuple[float, bool]:
    """Ask the judge up to `JUDGE_PATIENCE` times, then floor to 0.0 as MM-Vet does."""
    for instruction in _RETRY_INSTRUCTIONS[:JUDGE_PATIENCE]:
        try:
            result = await judge.generate(
                GenerateRequest(
                    system_prompt="You grade answers against a ground truth.",
                    input=ModelInput((TextPart(_prompt(answer) + instruction),)),
                    max_output_tokens=16,
                )
            )
        except Exception:
            # An unreachable judge is a 0.0 here, as it is upstream: MM-Vet's own retry loop
            # catches everything, and exhausting it scores the answer wrong.
            continue
        score = parse_score(result.text)
        if score is not None:
            return score, True
    return 0.0, False


def _prompt(answer: JudgedAnswer) -> str:
    return _JUDGE_PROMPT.format(
        question=answer.question,
        reference=answer.reference,
        prediction=answer.prediction or "(no answer)",
    )


def parse_score(text: str) -> float | None:
    """Read the judge's reply as a 0.0-to-1.0 correctness score, or say it could not be read.

    MM-Vet does `float(response)` inside a `try` and treats anything else as a retry. The first
    number anywhere in the reply is taken here instead, because a judge that answers "Score: 0.7"
    has said 0.7 and burning an attempt on the prefix helps nobody. A number outside the range is
    still unreadable: a judge replying "7" has not said 0.7 and guessing which it meant would
    quietly invent the score this whole module exists to attribute.
    """
    found = _SCORE.search(text)
    if found is None:
        return None
    value = float(found.group())
    return value if 0.0 <= value <= 1.0 else None
