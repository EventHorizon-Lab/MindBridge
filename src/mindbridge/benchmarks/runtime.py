"""Small runtime helpers shared by production-path benchmark runners."""

from __future__ import annotations

import asyncio
import re

from pydantic import TypeAdapter

from mindbridge.contracts import Identifier, NonEmptyString, ObservationProcessingJobView
from mindbridge.core import JobState
from mindbridge.sdk import MindBridge

OPTION_LABELS = ("A", "B", "C", "D")
_ALLOWED_RESPONSE_WORDS = re.compile(
    r"\b(?:answer|best|choice|choices|from|is|option|options|order|rank|ranking|to|worst)\b",
    re.IGNORECASE,
)


def benchmark_tenant_id(tenant_prefix: str, unit_id: str, run_id: str) -> str:
    """Build an isolated tenant so earlier queries cannot see a prior run's future."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    return TypeAdapter(Identifier).validate_python(f"{tenant_prefix}_{unit_id}_{run_id}")


async def wait_for_observation_job(
    memory: MindBridge,
    tenant_id: str,
    job_id: str,
    *,
    poll_interval_seconds: float = 1.0,
    timeout_seconds: float = 1_800.0,
) -> ObservationProcessingJobView:
    """Wait for durable success while allowing failed attempts to be retried."""
    if poll_interval_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("poll interval and timeout must be positive")
    last_state: JobState | None = None

    async def poll() -> ObservationProcessingJobView:
        nonlocal last_state
        while True:
            job = await memory.get_observation_job(tenant_id, job_id)
            last_state = job.state
            if job.state is JobState.SUCCEEDED:
                return job
            await asyncio.sleep(poll_interval_seconds)

    try:
        return await asyncio.wait_for(poll(), timeout_seconds)
    except asyncio.TimeoutError as error:
        state = last_state.value if last_state is not None else "unavailable"
        raise TimeoutError(
            f"observation job {job_id} did not succeed; last state was {state}"
        ) from error


def multiple_choice_query(
    question: str,
    choices: tuple[NonEmptyString, ...],
    *,
    rank_all: bool,
) -> str:
    """Format choices without exposing evaluation labels or evidence hints."""
    if len(choices) != len(OPTION_LABELS):
        raise ValueError("multiple-choice query requires exactly four choices")
    options = "\n".join(
        f"{label}. {choice}" for label, choice in zip(OPTION_LABELS, choices, strict=True)
    )
    instruction = (
        "Return only all four option labels from best to worst, separated by commas."
        if rank_all
        else "Return only the single best option label."
    )
    return f"{question}\n\nOptions:\n{options}\n\n{instruction}"


def parse_option_ranking(
    answer: str | None,
    choices: tuple[NonEmptyString, ...],
) -> tuple[int, ...]:
    """Parse a constrained model response, refusing ambiguous prose."""
    if answer is None:
        return ()
    normalized = " ".join(answer.split()).casefold()
    exact_matches = tuple(
        index
        for index, choice in enumerate(choices)
        if normalized == " ".join(choice.split()).casefold()
    )
    if len(exact_matches) == 1:
        return exact_matches

    labels = re.findall(r"\b[A-D]\b", answer.upper())
    if not labels or len(set(labels)) != len(labels):
        return ()
    residual = re.sub(r"\b[A-D]\b", "", answer, flags=re.IGNORECASE)
    residual = _ALLOWED_RESPONSE_WORDS.sub("", residual)
    if re.search(r"[A-Za-z0-9]", residual):
        return ()
    return tuple(OPTION_LABELS.index(label) for label in labels)
