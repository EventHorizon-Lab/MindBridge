"""Official Video-MME-v2 adapter, production-path runner, and grouped non-linear scorer.

The option set is A-H, `options` arrives as one newline-joined string, and the headline number is
a four-question group score rather than question accuracy.

The unit of scoring is a *group* of four questions over one video, and the released evaluator
slices groups positionally (`all_groups[i // 4]`). That is only safe because the release
happens to order its rows four-per-video; this adapter turns that positional accident into a
checked invariant so a subset run cannot silently produce a meaningless rating.

Scores here are on the released evaluator's 0-100 scale. Reproducing `_rating.json` and
`_acc.json` cell for cell is the point of this module, and mixing units inside one metrics object
is how a leaderboard number gets misquoted by a factor of a hundred.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

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


def score_group_answers(group_type: str, group_structure: str, correct: Sequence[bool]) -> _Score:
    """Score four parsed answers without rebuilding the full result contract."""
    answers = tuple(correct)
    if len(answers) != GROUP_SIZE:
        raise ValueError(f"Video-MME-v2 groups need {GROUP_SIZE} answers")
    if group_type == "relevance":
        return _RELEVANCE_SCORE_MAP[sum(answers)]
    if group_type != "logic":
        raise ValueError(f"unknown Video-MME-v2 group type: {group_type}")
    structure = logic_structure(group_structure)
    if structure is None:
        raise ValueError(f"Video-MME-v2 logic group has unknown structure {group_structure!r}")
    return _LOGIC_SCORE_MAPS[structure][_chain_progress(answers, structure)]


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
