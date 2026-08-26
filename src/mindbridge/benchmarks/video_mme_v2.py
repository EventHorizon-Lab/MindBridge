"""Official Video-MME-v2 adapter, production-path runner, and grouped non-linear scorer.

Video-MME-v2 is a separate benchmark from Video-MME, not a newer split of it, so it gets its
own adapter rather than widening `video_mme`: the option set is A-H instead of A-D, `options`
arrives as one newline-joined string instead of a list, the short/medium/long bands are gone,
and the headline number is a group score rather than an accuracy.

The unit of scoring is a *group* of four questions over one video, and the released evaluator
slices groups positionally (`all_groups[i // 4]`). That is only safe because the release
happens to order its rows four-per-video; this adapter turns that positional accident into a
checked invariant so a subset run cannot silently produce a meaningless rating.

Scores here are on the released evaluator's 0-100 scale rather than the 0-1 fractions
`video_mme` reports. Reproducing `_rating.json` and `_acc.json` cell for cell is the point of
this module, and mixing units inside one metrics object is how a leaderboard number gets
misquoted by a factor of a hundred.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from mindbridge.benchmarks.prompts import VIDEO_MME_V2_QUERY_PROMPT
from mindbridge.benchmarks.runtime import (
    PreparedVideo,
    answer_failure_trace_id,
    benchmark_tenant_id,
    ingest_prepared_video,
    prepared_video_end,
    settle_answers,
)
from mindbridge.contracts import (
    ContractModel,
    Identifier,
    NonEmptyString,
    RecallFilters,
    RecallQuery,
    RecallRequest,
)
from mindbridge.sdk import MindBridge, MindBridgeError

VIDEO_MME_V2_ADAPTER_VERSION = "video_mme_v2_official_v1"

VideoMMEV2GroupType = Literal["relevance", "logic"]
VideoMMEV2Level = Literal["1", "2", "3"]
VideoMMEV2Option = Literal["A", "B", "C", "D", "E", "F", "G", "H"]
VideoMMEV2LogicStructure = Literal["[1,2,3,4]", "[1,[2,3],4]", "[[1,2],3,4]"]

GROUP_SIZE = 4
"""Questions per group, and the stride the released evaluator slices groups on."""

_OPTION_LABELS = tuple("ABCDEFGH")
# The released `extract_characters_regex_v2` strips these in order with `str.replace`, so the
# order is part of the behaviour: "The answer is" must be removed before "The answer".
_ANSWER_PREFIXES = (
    "Final Answer:",
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
    "Answer:",
    "Option:",
)
# (k/4)^2 * 100: answering more of a group right is worth super-linearly more than answering
# the same questions right spread across groups, which is the mechanism the benchmark exists
# to apply. `relevance` groups score on how many of the four are correct, in any order.
_RELEVANCE_SCORE_MAP = (0.0, 100.0 / 16, 100.0 * 4 / 16, 100.0 * 9 / 16, 100.0)
# `logic` groups score on the longest correct prefix of the dependency chain instead, with a
# denominator per structure: a chain with a parallel pair has fewer distinguishable states, so
# its intermediate steps are worth proportionally more.
_LOGIC_SCORE_MAPS: dict[VideoMMEV2LogicStructure, tuple[float, ...]] = {
    "[1,2,3,4]": (0.0, 100.0 / 16, 100.0 * 4 / 16, 100.0 * 9 / 16, 100.0),
    "[1,[2,3],4]": (0.0, 100.0 / 12, 100.0 * 4 / 12, 100.0 * 7 / 12, 100.0),
    "[[1,2],3,4]": (0.0, 100.0 / 10, 100.0 * 2 / 10, 100.0 * 5 / 10, 100.0),
}
_Score = float


class VideoMMEV2Question(ContractModel):
    """One official multiple-choice question, its offline label, and its taxonomy cells.

    `options` carries between two and eight entries. Most questions offer all eight, but 58 of
    the 3,200 offer fewer — a yes/no/undetermined question offers three — while the official
    instruction still names A through H regardless. Pinning this to eight would reject the
    official release.
    """

    question_id: Identifier
    position: int = Field(ge=1, le=GROUP_SIZE)
    question: NonEmptyString
    options: tuple[NonEmptyString, ...] = Field(min_length=2, max_length=len(_OPTION_LABELS))
    answer: VideoMMEV2Option
    level: VideoMMEV2Level
    second_head: NonEmptyString
    third_head: NonEmptyString

    @model_validator(mode="after")
    def require_official_option_labels(self) -> VideoMMEV2Question:
        if any(
            re.match(rf"^{label}\.\s+\S", option) is None
            for label, option in zip(_OPTION_LABELS, self.options, strict=False)
        ):
            raise ValueError("Video-MME-v2 options must be labelled A. onwards without gaps")
        if _OPTION_LABELS.index(self.answer) >= len(self.options):
            raise ValueError(
                f"Video-MME-v2 answer {self.answer} is past the last option of {self.question_id}"
            )
        return self


class VideoMMEV2Group(ContractModel):
    """One video and the four interdependent questions scored together over it."""

    video_id: Identifier
    source_url: NonEmptyString
    group_type: VideoMMEV2GroupType
    group_structure: NonEmptyString
    questions: tuple[VideoMMEV2Question, ...] = Field(min_length=GROUP_SIZE, max_length=GROUP_SIZE)

    @model_validator(mode="after")
    def require_official_group_shape(self) -> VideoMMEV2Group:
        if tuple(question.position for question in self.questions) != tuple(
            range(1, GROUP_SIZE + 1)
        ):
            raise ValueError(f"Video-MME-v2 group {self.video_id} must hold positions 1 to 4")
        question_ids = tuple(question.question_id for question in self.questions)
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("Video-MME-v2 question IDs must be unique per group")
        # Only `logic` groups have their structure read, and an unknown one would reach the
        # released scorer as an exception mid-run. Rejecting it at load turns a run that dies
        # after hours of ingest into an invocation that refuses to start.
        if self.group_type == "logic" and logic_structure(self.group_structure) is None:
            raise ValueError(
                f"Video-MME-v2 logic group {self.video_id} has unknown structure "
                f"{self.group_structure!r}"
            )
        return self


class VideoMMEV2QuestionResult(ContractModel):
    """One official evaluator row plus the MindBridge diagnostics behind it."""

    question_id: Identifier
    position: int = Field(ge=1, le=GROUP_SIZE)
    question: NonEmptyString
    options: tuple[NonEmptyString, ...] = Field(min_length=2, max_length=len(_OPTION_LABELS))
    answer: VideoMMEV2Option
    level: VideoMMEV2Level
    second_head: NonEmptyString
    third_head: NonEmptyString
    response: str
    mindbridge_model_answer: str
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_evidence_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier
    mindbridge_error_code: NonEmptyString | None = None
    # One ingest covers the whole video before any of its four questions is asked, so every
    # question in a group carries the same count. A non-zero count marks a group answered over
    # incomplete memory, which matters more here than in `video_mme`: one missing segment can
    # break a dependency chain and cost the whole group its score, not one question its point.
    mindbridge_ingest_failure_count: int = Field(default=0, ge=0)


class VideoMMEV2GroupResult(ContractModel):
    """Official result object for one scored group."""

    video_id: Identifier
    group_type: VideoMMEV2GroupType
    group_structure: NonEmptyString
    questions: tuple[VideoMMEV2QuestionResult, ...] = Field(
        min_length=GROUP_SIZE, max_length=GROUP_SIZE
    )


class VideoMMEV2Rating(ContractModel):
    """The leaderboard number: grouped non-linear score, and the cells it breaks into.

    Reproduces the released `_rating.json`. Every cell is a mean over whole groups on a 0-100
    scale, so `group_count` is the denominator rather than a question count.

    The taxonomy cells are keyed on the *fourth* question of each group, because that is what
    the released scorer reads (`group[-1]`). `level`, `second_head`, and `third_head` all vary
    within a group in the official release, so this is a real choice the scorer makes and not
    a detail that happens to be constant.
    """

    group_count: int = Field(gt=0)
    overall: _Score = Field(ge=0.0, le=100.0)
    by_group_type: dict[str, _Score]
    by_level: dict[str, _Score]
    by_second_head: dict[str, _Score]
    by_third_head: dict[str, _Score]


class VideoMMEV2Accuracy(ContractModel):
    """Plain per-question accuracy and its cells, reproducing the released `_acc.json`.

    `overall` counts every question, scoring an unparseable or missing response wrong, which is
    what the released `get_final_acc` writes to disk. `answered_accuracy` is the answered-only
    number the same script prints as "Simple accuracy (valid only)".

    Note the naming runs opposite to `video_mme`, where `accuracy` is the answered-only figure
    and `strict_accuracy` the floor. The official artifacts disagree between the two benchmarks
    and matching each one's own release is what keeps a quoted number checkable.
    """

    question_count: int = Field(gt=0)
    answered_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    overall: _Score = Field(ge=0.0, le=100.0)
    answered_accuracy: _Score = Field(ge=0.0, le=100.0)
    by_group_type: dict[str, _Score]
    by_level: dict[str, _Score]
    by_second_head: dict[str, _Score]
    by_third_head: dict[str, _Score]

    @model_validator(mode="after")
    def require_consistent_counts(self) -> VideoMMEV2Accuracy:
        if self.correct_count > self.answered_count or self.answered_count > self.question_count:
            raise ValueError("Video-MME-v2 accuracy counts are inconsistent")
        if self.error_count > self.question_count - self.answered_count:
            raise ValueError("Video-MME-v2 failed questions must not carry a parsed answer")
        return self


class VideoMMEV2Metrics(ContractModel):
    """Both released scoring views over one run.

    The benchmark exists because these two disagree: a model scattering correct answers across
    groups earns the same accuracy and a much lower rating than one answering whole groups. A
    run reports both or it reports nothing interpretable.
    """

    rating: VideoMMEV2Rating
    accuracy: VideoMMEV2Accuracy


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    url: str
    group_type: VideoMMEV2GroupType
    group_structure: str
    question_id: str
    question: str
    options: str
    answer: VideoMMEV2Option
    level: VideoMMEV2Level
    second_head: str
    third_head: str


def logic_structure(group_structure: str) -> VideoMMEV2LogicStructure | None:
    """Normalize a `group_structure` cell to a known chain, or `None` if it is not one.

    The released scorer runs `ast.literal_eval` and compares against list literals, so spacing
    is not significant. Whitespace is stripped rather than parsed: `relevance` groups carry the
    scalar `"4"` in this column, which `literal_eval` turns into an int that reaches the
    released scorer's `raise` branch, and there is nothing to gain from evaluating a field only
    one of the two group types defines.
    """
    normalized = "".join(group_structure.split())
    return normalized if normalized in _LOGIC_SCORE_MAPS else None


def load_video_mme_v2(annotation_path: Path) -> tuple[VideoMMEV2Group, ...]:
    """Load the official Hugging Face Parquet release without materializing media."""
    try:
        parquet = cast(Any, import_module("pyarrow.parquet"))
    except ModuleNotFoundError as error:
        if error.name is not None and not error.name.startswith("pyarrow"):
            raise
        raise RuntimeError(
            "Video-MME-v2 Parquet support requires `uv sync --extra benchmarks`"
        ) from error
    rows = TypeAdapter(list[_RawQuestion]).validate_python(
        parquet.read_table(annotation_path).to_pylist()
    )
    if not rows:
        raise ValueError("Video-MME-v2 annotations must not be empty")
    if len(rows) % GROUP_SIZE:
        raise ValueError(
            f"Video-MME-v2 annotations must hold whole groups of {GROUP_SIZE}; got {len(rows)} rows"
        )
    groups = tuple(
        _group(rows[offset : offset + GROUP_SIZE]) for offset in range(0, len(rows), GROUP_SIZE)
    )
    video_ids = tuple(group.video_id for group in groups)
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("Video-MME-v2 annotations must not split one video across groups")
    question_ids = tuple(question.question_id for group in groups for question in group.questions)
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Video-MME-v2 annotations contain duplicate question IDs")
    return groups


async def run_video_mme_v2_group(
    memory: MindBridge,
    annotation: VideoMMEV2Group,
    prepared: PreparedVideo,
    *,
    run_id: str,
    tenant_prefix: str = "benchmark_video_mme_v2",
    device_id: str = "video_mme_v2_camera",
    recall_limit: int = 20,
    request_concurrency: int = 4,
    poll_interval_seconds: float = 1.0,
    processing_timeout_seconds: float = 1_800.0,
) -> VideoMMEV2GroupResult:
    """Ingest one source video and answer the four official questions scored over it."""
    if annotation.video_id != prepared.video_id:
        raise ValueError("Video-MME-v2 annotation and prepared video IDs must match")
    if not 1 <= recall_limit <= 100 or request_concurrency <= 0:
        raise ValueError(
            "recall_limit must be between 1 and 100; request_concurrency must be positive"
        )
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")
    tenant_id = benchmark_tenant_id(tenant_prefix, annotation.video_id, run_id)
    ingest_failures = await ingest_prepared_video(
        memory,
        tenant_id,
        device_id,
        prepared,
        adapter_version=VIDEO_MME_V2_ADAPTER_VERSION,
        request_concurrency=request_concurrency,
        poll_interval_seconds=poll_interval_seconds,
        processing_timeout_seconds=processing_timeout_seconds,
    )

    semaphore = asyncio.Semaphore(request_concurrency)
    cutoff = prepared_video_end(prepared)
    answered = await asyncio.gather(
        *(
            _answer_question(
                memory,
                tenant_id,
                question,
                cutoff,
                recall_limit,
                semaphore,
                ingest_failures,
            )
            for question in annotation.questions
        ),
        return_exceptions=True,
    )
    # A raising recall costs its own question, not the group. `VideoMMEV2GroupResult`
    # requires all four rows, so an escaping exception took the whole group's rating with
    # it and then ended the run, because the CLI awaits each group in turn. It matters
    # more here than in any per-question benchmark: the rating is defined over whole
    # groups only, so a group short of a row cannot be scored at all.
    answers = settle_answers(
        annotation.questions, answered, partial(_failed_result, ingest_failures=ingest_failures)
    )
    return VideoMMEV2GroupResult(
        video_id=annotation.video_id,
        group_type=annotation.group_type,
        group_structure=annotation.group_structure,
        questions=tuple(answers),
    )


def evaluate_video_mme_v2(results: tuple[VideoMMEV2GroupResult, ...]) -> VideoMMEV2Metrics:
    """Compute the official grouped rating and per-question accuracy over whole groups."""
    if not results:
        raise ValueError("Video-MME-v2 results must not be empty")
    return VideoMMEV2Metrics(
        rating=_rating(results),
        accuracy=_accuracy(results),
    )


def score_group(result: VideoMMEV2GroupResult) -> _Score:
    """Score one group on the released 0-100 non-linear scale."""
    correct = tuple(_is_correct(question) for question in result.questions)
    if result.group_type == "relevance":
        return _RELEVANCE_SCORE_MAP[sum(correct)]
    structure = logic_structure(result.group_structure)
    if structure is None:
        raise ValueError(
            f"Video-MME-v2 logic group {result.video_id} has unknown structure "
            f"{result.group_structure!r}"
        )
    return _LOGIC_SCORE_MAPS[structure][_chain_progress(correct, structure)]


def parse_video_mme_v2_option(response: str | None) -> VideoMMEV2Option | None:
    """Normalize a response using the released evaluator's first-letter rule.

    The released `extract_characters_regex_v2` also returns early for a long response holding
    no A-H, which is the same empty result its trailing search already produces; the branch is
    not reproduced because reproducing a no-op would only invite someone to "fix" it later.
    """
    if response is None:
        return None
    normalized = response.strip()
    for prefix in _ANSWER_PREFIXES:
        normalized = normalized.replace(prefix, "")
    match = re.search(r"[A-H]", normalized)
    return cast("VideoMMEV2Option | None", match.group() if match is not None else None)


def _chain_progress(correct: tuple[bool, ...], structure: VideoMMEV2LogicStructure) -> int:
    """Count how far the group's dependency chain was answered without a break.

    A parallel pair inside the chain gets one adjustment each, mirroring the released scorer:
    credit for reaching the step a sibling also satisfies. Both adjustments are deliberately
    one-shot — a group that fails the head of the chain earns no credit for anything past the
    parallel pair, even when the tail is correct.
    """
    progress = 0
    for value in correct:
        if not value:
            break
        progress += 1
    if structure == "[1,[2,3],4]" and progress == 1 and correct[2]:
        progress += 1
    if structure == "[[1,2],3,4]" and progress == 0 and correct[1]:
        progress += 1
    return progress


def _rating(results: tuple[VideoMMEV2GroupResult, ...]) -> VideoMMEV2Rating:
    scores = tuple(score_group(result) for result in results)
    # Keyed on the last question of each group, which is what the released scorer reads.
    tails = tuple(result.questions[-1] for result in results)
    return VideoMMEV2Rating(
        group_count=len(results),
        overall=_mean(scores),
        by_group_type=_cells(
            scores, (result.group_type for result in results), key=lambda value: value
        ),
        by_level=_cells(scores, (tail.level for tail in tails), key=_level_key),
        by_second_head=_cells(scores, (tail.second_head for tail in tails)),
        by_third_head=_cells(scores, (tail.third_head for tail in tails)),
    )


def _accuracy(results: tuple[VideoMMEV2GroupResult, ...]) -> VideoMMEV2Accuracy:
    questions = tuple(question for result in results for question in result.questions)
    group_types = tuple(
        result.group_type for result in results for _ in range(len(result.questions))
    )
    scores = tuple(100.0 * _is_correct(question) for question in questions)
    answered = tuple(question for question in questions if question.response in _OPTION_LABELS)
    correct_count = sum(_is_correct(question) for question in answered)
    return VideoMMEV2Accuracy(
        question_count=len(questions),
        answered_count=len(answered),
        correct_count=correct_count,
        error_count=sum(question.mindbridge_error_code is not None for question in questions),
        overall=_mean(scores),
        answered_accuracy=100.0 * correct_count / len(answered) if answered else 0.0,
        by_group_type=_cells(scores, group_types, key=lambda value: value),
        by_level=_cells(scores, (question.level for question in questions), key=_level_key),
        by_second_head=_cells(scores, (question.second_head for question in questions)),
        by_third_head=_cells(scores, (question.third_head for question in questions)),
    )


def _cells(
    scores: tuple[_Score, ...],
    labels: Iterable[str],
    *,
    key: Callable[[str], str] = lambda value: value,
) -> dict[str, _Score]:
    """Average `scores` into one cell per label, sorted so a manifest diff stays readable."""
    grouped: dict[str, list[_Score]] = defaultdict(list)
    for score, label in zip(scores, labels, strict=True):
        grouped[key(label)].append(score)
    return {label: _mean(tuple(values)) for label, values in sorted(grouped.items())}


def _level_key(level: str) -> str:
    """Name a level cell the way the released `_rating.json` does."""
    return f"level_{level}"


def _mean(scores: tuple[_Score, ...]) -> _Score:
    return sum(scores) / len(scores) if scores else 0.0


def _is_correct(question: VideoMMEV2QuestionResult) -> bool:
    """Score one row, counting an unparseable or missing response wrong.

    The released scorer marks those rows `-1` and then folds `-1` into zero in both the rating
    and the accuracy it writes out, so an abstention is a wrong answer to every number here.
    """
    return question.response == question.answer


def _group(rows: list[_RawQuestion]) -> VideoMMEV2Group:
    first = rows[0]
    metadata = (first.video_id, first.url, first.group_type, first.group_structure)
    if any(
        (row.video_id, row.url, row.group_type, row.group_structure) != metadata for row in rows
    ):
        raise ValueError(f"Video-MME-v2 group at video {first.video_id} has inconsistent metadata")
    return VideoMMEV2Group(
        video_id=first.video_id,
        source_url=first.url,
        group_type=first.group_type,
        group_structure=first.group_structure,
        questions=tuple(
            VideoMMEV2Question(
                question_id=row.question_id,
                position=position,
                question=row.question,
                options=tuple(row.options.split("\n")),
                answer=row.answer,
                level=row.level,
                second_head=row.second_head,
                third_head=row.third_head,
            )
            for position, row in enumerate(rows, start=1)
        ),
    )


async def _answer_question(
    memory: MindBridge,
    tenant_id: str,
    question: VideoMMEV2Question,
    cutoff: AwareDatetime,
    recall_limit: int,
    semaphore: asyncio.Semaphore,
    ingest_failures: int,
) -> VideoMMEV2QuestionResult:
    query = VIDEO_MME_V2_QUERY_PROMPT.text.format(
        question=question.question,
        options="\n".join(question.options),
    )
    try:
        async with semaphore:
            result = await memory.recall(
                RecallRequest(
                    tenant_id=tenant_id,
                    query=RecallQuery(text=query),
                    filters=RecallFilters(occurred_before=cutoff),
                    limit=recall_limit,
                )
            )
    except MindBridgeError as error:
        if error.code not in {"model_output_invalid", "model_request_failed"}:
            raise
        return _question_result(
            question,
            model_answer="",
            confidence=0.0,
            memory_ids=(),
            evidence_ids=(),
            trace_id=error.trace_id or f"trace_model_error_{question.question_id}",
            error_code=error.code,
            ingest_failures=ingest_failures,
        )
    return _question_result(
        question,
        model_answer=result.answer or "",
        confidence=result.confidence,
        memory_ids=tuple(item.memory_id for item in result.memories),
        evidence_ids=tuple(item.evidence_id for item in result.evidence),
        trace_id=result.trace_id,
        ingest_failures=ingest_failures,
    )


def _failed_result(
    question: VideoMMEV2Question,
    error_code: str,
    *,
    ingest_failures: int,
) -> VideoMMEV2QuestionResult:
    """One row for a question whose recall raised, so its group still scores four of them.

    `response` is empty, which the released scorer counts wrong, and `mindbridge_error_code`
    is what keeps a transport failure from reading as a model that answered badly.
    """
    return _question_result(
        question,
        model_answer="",
        confidence=0.0,
        memory_ids=(),
        evidence_ids=(),
        trace_id=answer_failure_trace_id(question.question_id),
        error_code=error_code,
        ingest_failures=ingest_failures,
    )


def _question_result(
    question: VideoMMEV2Question,
    *,
    model_answer: str,
    confidence: float,
    memory_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    trace_id: str,
    ingest_failures: int,
    error_code: str | None = None,
) -> VideoMMEV2QuestionResult:
    option = parse_video_mme_v2_option(model_answer)
    return VideoMMEV2QuestionResult(
        question_id=question.question_id,
        position=question.position,
        question=question.question,
        options=question.options,
        answer=question.answer,
        level=question.level,
        second_head=question.second_head,
        third_head=question.third_head,
        response=option or "",
        mindbridge_model_answer=model_answer,
        mindbridge_confidence=confidence,
        mindbridge_memory_ids=memory_ids,
        mindbridge_evidence_ids=evidence_ids,
        mindbridge_trace_id=trace_id,
        mindbridge_error_code=error_code,
        mindbridge_ingest_failure_count=ingest_failures,
    )
