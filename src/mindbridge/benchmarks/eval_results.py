"""Per-sample result records and the failure classification that decides retries and error codes."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import (
    Awaitable,
    Callable,
    Mapping,
)
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    pass
from mindbridge import (
    AnswerPolicy,
    MindBridgeError,
)
from mindbridge.benchmarks.eval_cache import EvidenceInterval
from mindbridge.benchmarks.eval_config import DEFAULT_ARM
from mindbridge.models.base import _is_generation_abort_rejection

EVAL_SCHEMA_VERSION = 17


@dataclass(frozen=True, slots=True)
class FailureDetail:
    """Safe, stable benchmark failure diagnostics.

    `message` is the provider's own words when it gave any, else the failure's text: the stable
    fields say a request was rejected, only the words say whether the prompt was malformed or the
    gateway aborted its own generation, and a one-in-two-hundred failure cannot be reproduced on
    demand afterwards. Whitespace-normalized and bounded like `scorer_error`.
    """

    source_id: str | None
    code: str
    reason: str | None
    stage: str | None
    cause_type: str | None
    message: str | None = None

    def json(self) -> dict[str, str | None]:
        return {
            "source_id": self.source_id,
            "code": self.code,
            "reason": self.reason,
            "stage": self.stage,
            "cause_type": self.cause_type,
            "message": self.message,
        }


class _SystemicEmbeddingFailure(RuntimeError):
    """Stop a benchmark unit when the shared embedding service cannot serve requests."""


_SYSTEMIC_EMBEDDING_HTTP_STATUSES = frozenset({401, 403, 404, 405, 408, 429, *range(500, 600)})


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    pending: BaseException | None = error
    seen: set[int] = set()
    while pending is not None and id(pending) not in seen:
        seen.add(id(pending))
        chain.append(pending)
        pending = pending.__cause__ or pending.__context__
    return tuple(chain)


def _systemic_embedding_failure(error: BaseException) -> bool:
    """Recognize failures for which recursive item isolation only amplifies an outage."""
    for current in _exception_chain(error):
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status is None:
            status = getattr(current, "status_code", None)
        if status in _SYSTEMIC_EMBEDDING_HTTP_STATUSES:
            return True
        transport_type = type(current)
        if isinstance(current, (TimeoutError, ConnectionError)) or (
            transport_type.__module__.split(".", maxsplit=1)[0] in {"httpx", "httpcore"}
            and transport_type.__name__
            in {
                "ConnectError",
                "ConnectTimeout",
                "PoolTimeout",
                "ReadTimeout",
                "TimeoutException",
                "WriteTimeout",
            }
        ):
            return True
    return False


def _raise_if_systemic_embedding(error: Exception, message: str) -> None:
    if _systemic_embedding_stage_failure(error):
        raise _SystemicEmbeddingFailure(message) from error


def _systemic_embedding_stage_failure(error: Exception) -> bool:
    return _systemic_embedding_failure(error) and any(
        isinstance(current, MindBridgeError) and current.stage == "embed"
        for current in _exception_chain(error)
    )


_TRANSIENT_RETRY_SECONDS = 600.0


_TRANSIENT_RETRY_CAP_SECONDS = 30.0


_TRANSIENT_REASONS = frozenset({"connection_failed", "timeout", "rate_limited"})


_Retried = TypeVar("_Retried")


def _transient_provider_failure(error: BaseException) -> bool:
    """Recognize a provider failure that time, not a different request, resolves."""
    try:
        import openai
    except ImportError:  # pragma: no cover - every provider path imports the SDK first
        transient: tuple[type[BaseException], ...] = ()
        rate_limited: type[BaseException] | None = None
    else:
        transient = (openai.APIConnectionError, openai.InternalServerError)
        rate_limited = openai.RateLimitError
    for current in _exception_chain(error):
        if isinstance(current, MindBridgeError) and current.reason in _TRANSIENT_REASONS:
            return True
        if rate_limited is not None and isinstance(current, rate_limited):
            # Exhausted billing is a 429 too, and waiting never refills it.
            return getattr(current, "code", None) != "insufficient_quota"
        if isinstance(current, transient):
            return True
        # A 400 is `request_rejected` and permanent, except the one a gateway sends when it
        # aborted its own JSON generation: the answer path streams `response_format` replies,
        # and that rejection is a fact about the provider at that second, not about the prompt.
        if _is_generation_abort_rejection(current):
            return True
    return False


async def _retry_transient(operation: Callable[[], Awaitable[_Retried]]) -> _Retried:
    """Wait out a provider outage with exponential backoff instead of recording an error.

    A connection reset is a fact about the network at that second, not about the answer, the
    memory write, or the judge verdict it interrupted; one fourteen-hour run lost its longmemeval
    score to eight of them. An outage longer than the budget still surfaces as the structured
    error the caller already records, so nothing here hides a dead endpoint.
    """
    deadline = time.monotonic() + _TRANSIENT_RETRY_SECONDS
    attempt = 0
    while True:
        try:
            return await operation()
        except Exception as error:
            delay = min(2.0**attempt, _TRANSIENT_RETRY_CAP_SECONDS)
            if not _transient_provider_failure(error) or time.monotonic() + delay > deadline:
                raise
            attempt += 1
            logging.getLogger(__name__).warning(
                "provider failure (%s); retrying in %.0fs", type(error).__name__, delay
            )
            await asyncio.sleep(delay)


@dataclass(frozen=True, slots=True)
class SampleResult:
    """One answered question plus diagnostics needed for replay and comparison."""

    task: str
    benchmark: str
    dataset_sha256: str
    evaluation_sha256: str
    unit_id: str
    question_id: str
    prediction: str
    parsed_choice: str | None
    score: float | None
    exact_match: float | None
    latency_ms: float
    confidence: float
    memory_ids: tuple[str, ...]
    ingest_failure_count: int
    error_code: str | None
    metadata: Mapping[str, object]
    arm: str = DEFAULT_ARM
    # Two different denominators, both needed. `candidate_count` is the pool a ranker could have
    # drawn from, which is what makes the random-ranker expectation meaningful;
    # `retrieval_candidates` is how deep the retriever's own ranked list actually went.
    candidate_count: int = 0
    retrieval_candidates: int = 0
    # The retriever's own ranked source IDs, deepest first-party list the run produced. Task-level
    # retrieval recall scores this list; `evidence` is what the generator cited, a different
    # quantity that an earlier version of this harness scored under the same name.
    ranked_source_ids: tuple[str, ...] = ()
    ranked_source_ids_complete: bool = False
    dropped_hits: int | None = None
    answer_policy: AnswerPolicy | None = None
    # The shape of the recall plan this answer was grounded on; None unless the run planned.
    recall_shape: str | None = None
    abstained: bool = False
    abstention_reason: str | None = None
    ingest_failures: tuple[FailureDetail, ...] = ()
    error_reason: str | None = None
    error_stage: str | None = None
    error_cause_type: str | None = None
    error_message: str | None = None
    retrieval_diagnostic_error: FailureDetail | None = None
    cached: bool = False
    prompt: tuple[str, ...] | None = None
    references: tuple[str, ...] | None = None
    evidence: tuple[EvidenceInterval, ...] = ()
    # Partial sources stay separate so a scorer cannot silently credit an omitted span as if the
    # full parent had been delivered.
    excerpt_source_ids: tuple[str, ...] = ()
    excerpt_evidence: tuple[EvidenceInterval, ...] = ()
    ref_at_300: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    scorer_protocol: str | None = None
    scorer_details: Mapping[str, str] = field(default_factory=dict)
    scorer_error: str | None = None
    judge_model: str | None = None
    judge_response: str | None = None
    judge_cached: bool = False

    @property
    def sample_id(self) -> str:
        identity = f"{self.task}/{self.unit_id}/{self.question_id}"
        return identity if self.arm == DEFAULT_ARM else f"{self.arm}:{identity}"

    def json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": EVAL_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "arm": self.arm,
            "retrieval_candidates": self.retrieval_candidates,
            "ranked_source_ids": self.ranked_source_ids,
            "ranked_source_ids_complete": self.ranked_source_ids_complete,
            "dropped_hits": self.dropped_hits,
            "recall_shape": self.recall_shape,
            "task": self.task,
            # The policy this sample's request carried; only the product arm reaches `ask`.
            "answer_policy": self.answer_policy,
            "benchmark": self.benchmark,
            "dataset_sha256": self.dataset_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "unit_id": self.unit_id,
            "question_id": self.question_id,
            "prediction": self.prediction,
            "parsed_choice": self.parsed_choice,
            "score": self.score,
            "exact_match": self.exact_match,
            "latency_ms": self.latency_ms,
            "confidence": self.confidence,
            "memory_ids": self.memory_ids,
            "candidate_count": self.candidate_count,
            "evidence": tuple(item.json() for item in self.evidence),
            "excerpt_source_ids": self.excerpt_source_ids,
            "excerpt_evidence": tuple(item.json() for item in self.excerpt_evidence),
            "ref_at_300": self.ref_at_300,
            "metrics": dict(self.metrics),
            "scorer_protocol": self.scorer_protocol,
            "scorer_details": dict(self.scorer_details),
            "scorer_error": self.scorer_error,
            "judge_model": self.judge_model,
            "judge_cached": self.judge_cached,
            "ingest_failure_count": self.ingest_failure_count,
            "ingest_failures": tuple(item.json() for item in self.ingest_failures),
            "error_code": self.error_code,
            "error_reason": self.error_reason,
            "error_stage": self.error_stage,
            "error_cause_type": self.error_cause_type,
            "error_message": self.error_message,
            "retrieval_diagnostic_error": (
                None
                if self.retrieval_diagnostic_error is None
                else self.retrieval_diagnostic_error.json()
            ),
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "cached": self.cached,
            "metadata": dict(self.metadata),
        }
        if self.prompt is not None:
            payload["prompt"] = self.prompt
        if self.references is not None:
            payload["references"] = self.references
        if self.judge_response is not None:
            payload["judge_response"] = self.judge_response
        return payload


@dataclass(frozen=True, slots=True)
class _AnswerOutcome:
    prediction: str
    latency_ms: float
    confidence: float
    memory_ids: tuple[str, ...]
    evidence: tuple[EvidenceInterval, ...]
    abstained: bool = False
    abstention_reason: str | None = None
    cached: bool = False
    error: BaseException | None = None
    # The answer's ranked list before grounding, in score order. Retrieval metrics score this;
    # `evidence` stays the answer's grounded hits.
    ranked_source_ids: tuple[str, ...] = ()
    ranked_source_ids_complete: bool = False
    retrieval_diagnostic_error: BaseException | None = None
    # Set only by the `compile` arm: the bundle's own size, so "useful evidence per token" is
    # computable per answered question without a second call to reconstruct it.
    compiled_chars: int | None = None
    compiled_items: int | None = None
    excerpt_source_ids: tuple[str, ...] = ()
    excerpt_evidence: tuple[EvidenceInterval, ...] = ()


def _restored_failure(payload: Mapping[str, object]) -> FailureDetail:
    return FailureDetail(
        source_id=_optional_text(payload["source_id"]),
        code=str(payload["code"]),
        reason=_optional_text(payload["reason"]),
        stage=_optional_text(payload["stage"]),
        cause_type=_optional_text(payload["cause_type"]),
        message=_optional_text(payload.get("message")),
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _error_code(error: BaseException) -> str:
    return error.code if isinstance(error, MindBridgeError) else type(error).__name__


def _failure_detail(error: BaseException, *, source_id: str | None = None) -> FailureDetail:
    cause = error.__cause__ or error.__context__
    seen = {id(error)}
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        next_cause = cause.__cause__ or cause.__context__
        if next_cause is None:
            break
        cause = next_cause
    return FailureDetail(
        source_id=source_id,
        code=_error_code(error),
        reason=error.reason if isinstance(error, MindBridgeError) else None,
        stage=error.stage if isinstance(error, MindBridgeError) else None,
        cause_type=None if cause is None else type(cause).__name__,
        message=_failure_message(error),
    )


_FAILURE_MESSAGE_CHARS = 500


def _failure_message(error: BaseException) -> str | None:
    """The provider's parsed error body when the chain carries one, else the failure's own text.

    Only the SDK exception's parsed body is read, never a transport exception's text: that text
    names the request URL, and a URL may carry credentials.
    """
    for current in _exception_chain(error):
        body = getattr(current, "body", None)
        message = body.get("message") if isinstance(body, Mapping) else None
        if isinstance(message, str) and message.strip():
            return " ".join(message.split())[:_FAILURE_MESSAGE_CHARS]
    text = " ".join(str(error).split())
    return text[:_FAILURE_MESSAGE_CHARS] or None
