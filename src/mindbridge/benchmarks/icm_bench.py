"""Read ICM-Bench annotations without exposing evaluator-only identity labels."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

ICM_ADAPTER_VERSION = "icm_video_speakerless_asr_v1"
ICM_DATA_REVISION = "8670eef08e3a172e3951984cec33a8f4a9ae96a7"
# The pinned release publishes no speakerless transcript for these ten clips.
ICM_NO_ASR_CLIPS = frozenset({0, 51, 168, 287, 331, 468, 517, 669, 722, 786})


def clip_position(value: str) -> int:
    if re.fullmatch(r"clip_\d{3}", value) is None:
        raise ValueError(f"invalid ICM clip ID: {value}")
    return int(value.removeprefix("clip_"))


class ICMQuestion(ContractModel):
    question_id: Identifier
    question: NonEmptyString
    reference_answer: NonEmptyString
    category: Literal[
        "Identity Recall",
        "Cross-Episode Identity Retrieval",
        "Long-Term Identity Profile Inference",
    ]
    target_character_ids: tuple[Identifier, ...]
    evidence_video_ids: tuple[Identifier, ...] = Field(min_length=1)
    before_clip: Identifier | None

    @model_validator(mode="after")
    def check_cutoff(self) -> ICMQuestion:
        positions = [clip_position(item) for item in self.evidence_video_ids]
        profile = self.category == "Long-Term Identity Profile Inference"
        if profile != (self.before_clip is None):
            raise ValueError("only Profile questions may see the full timeline")
        if self.before_clip is not None and max(positions) > clip_position(self.before_clip):
            raise ValueError("ICM evidence lies after the permitted prefix")
        return self


def load_icm_bench(path: Path) -> tuple[ICMQuestion, ...]:
    questions = tuple(
        ICMQuestion.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    identifiers = [item.question_id for item in questions]
    if not questions or len(set(identifiers)) != len(identifiers):
        raise ValueError("ICM questions must be nonempty with unique IDs")
    return questions
