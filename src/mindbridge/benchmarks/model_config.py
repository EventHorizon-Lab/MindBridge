"""Provider settings used only by the benchmark harness."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from mindbridge.types import Modality

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GENERATION_MODEL = "gpt-5-mini"
# An hour bounds nothing a benchmark run cares about: a request the server never answers
# holds its task for the whole hour while the remaining workers idle, and the run reports
# the stall as elapsed time rather than as a failure. The slowest mean model call measured
# across this suite is video grounding at ~36s, so five minutes leaves ample headroom for a
# request that is genuinely slow while still cutting a hung one loose.
DEFAULT_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Reproducible OpenAI SDK inputs for benchmark generation."""

    generation_api_key: str | None = field(default=None, repr=False)
    generation_base_url: str = DEFAULT_OPENAI_BASE_URL
    generation_model: str = DEFAULT_GENERATION_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    generation_capabilities: frozenset[Modality] = frozenset({Modality.TEXT})
    generation_min_video_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.generation_base_url.strip():
            raise ValueError("generation_base_url must not be blank")
        if not self.generation_model.strip():
            raise ValueError("generation_model must not be blank")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.generation_min_video_seconds is not None and (
            not math.isfinite(self.generation_min_video_seconds)
            or self.generation_min_video_seconds <= 0
        ):
            raise ValueError("generation_min_video_seconds must be positive")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> ModelConfig:
        source = os.environ if environ is None else environ
        return cls(
            generation_api_key=(
                source.get("MINDBRIDGE_GENERATION_API_KEY") or source.get("OPENAI_API_KEY")
            ),
            generation_base_url=(
                source.get("MINDBRIDGE_GENERATION_BASE_URL")
                or source.get("OPENAI_BASE_URL")
                or DEFAULT_OPENAI_BASE_URL
            ),
            generation_model=source.get("MINDBRIDGE_GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
            timeout_seconds=_float(
                source.get("MINDBRIDGE_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS
            ),
            generation_capabilities=_modalities(source.get("MINDBRIDGE_GENERATION_MODALITIES")),
        )


def _float(value: str | None, default: float) -> float:
    try:
        return default if value is None else float(value)
    except ValueError:
        raise ValueError("MINDBRIDGE_TIMEOUT_SECONDS must be a number") from None


def _modalities(value: str | None) -> frozenset[Modality]:
    if value is None:
        return frozenset({Modality.TEXT})
    try:
        parsed = {Modality(item.strip().lower()) for item in value.split(",") if item.strip()}
    except ValueError:
        raise ValueError("MINDBRIDGE_GENERATION_MODALITIES is invalid") from None
    if Modality.OMNI in parsed:
        parsed.remove(Modality.OMNI)
        parsed.update({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})
    return frozenset(parsed)
