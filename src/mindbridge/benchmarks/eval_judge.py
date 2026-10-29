"""LLM-judge scoring with its response cache and transient-failure retries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import (
    Container,
    Mapping,
    Sequence,
)
from dataclasses import replace
from typing import TYPE_CHECKING, TypedDict, cast

from opentelemetry import trace
from opentelemetry.trace import Tracer

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
    from openai.types.responses import Response
from mindbridge import Modality
from mindbridge._telemetry import (
    MODEL_MODULE,
    SPAN_KIND,
    mark_model_requests,
    model_span,
)
from mindbridge.benchmarks.eval_adapters import EvalQuestion, LoadedTask
from mindbridge.benchmarks.eval_arms import _Arm
from mindbridge.benchmarks.eval_artifacts import _json_bytes
from mindbridge.benchmarks.eval_cache import (
    CachedAnswer,
    ResponseCache,
)
from mindbridge.benchmarks.eval_config import (
    _Arguments,
    _JudgeConfig,
)
from mindbridge.benchmarks.eval_console import (
    _announce,
    _progress,
)
from mindbridge.benchmarks.eval_results import (
    SampleResult,
    _retry_transient,
)
from mindbridge.benchmarks.eval_telemetry import (
    BENCHMARK_ARM,
    BENCHMARK_JUDGE_SPAN,
    BENCHMARK_PURPOSE,
    BENCHMARK_SAMPLE,
    BENCHMARK_TASK,
    JUDGE_PURPOSE,
)
from mindbridge.benchmarks.official_scorers import (
    SCORER_VERSION,
    JudgeMessage,
    JudgePlan,
    combine_judge_scores,
    finalize_scores,
    judge_failure_scores,
    judge_model_is_official,
    judge_plan,
    local_scores,
    parse_judge_response,
    sample_primary_metric,
)
from mindbridge.models._openai_protocol import output_limit_parameter, sampling_supported
from mindbridge.models.openai_sdk import (
    _model_usage,
    _record_openai_provenance,
    _record_usage_batch,
)


class _JudgeRequestRequired(TypedDict):
    model: str
    messages: list[ChatCompletionMessageParam]


class _JudgeRequest(_JudgeRequestRequired, total=False):
    temperature: float
    max_tokens: int
    max_completion_tokens: int
    extra_body: dict[str, object]


class _JudgeRequestError(Exception):
    """An exhausted SDK request failure, distinct from local scoring failures."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


async def _apply_judges(
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
    *,
    arguments: _Arguments,
    config: _JudgeConfig,
    tracer: Tracer | None = None,
    already_judged: Container[str] = (),
) -> tuple[SampleResult, ...]:
    selected_tracer = trace.get_tracer("mindbridge.benchmarks.eval") if tracer is None else tracer
    questions = {
        (task.spec.name, unit.unit_id, question.question_id): question
        for task in tasks
        for unit in task.units
        for question in unit.questions
    }
    planned: dict[str, JudgePlan] = {}
    prepared_samples: dict[str, SampleResult] = {}
    for sample in samples:
        if (
            sample.error_code not in (None, "JudgeError")
            or sample.ingest_failure_count
            or not _Arm(sample.arm).generates
            # `--stream-results` already judged this one when its task finished. Re-judging would
            # spend the calls twice and let a non-deterministic judge overwrite a published score.
            or sample.sample_id in already_judged
        ):
            continue
        question = questions[(sample.task, sample.unit_id, sample.question_id)]
        sample = _restore_local_scores(sample, question)
        prepared_samples[sample.sample_id] = sample
        plan = judge_plan(
            sample.task,
            question=question.source_question,
            references=question.references,
            prediction=sample.prediction,
            metadata=question.metadata,
        )
        if plan is not None:
            planned[sample.sample_id] = plan
    if not planned:
        return tuple(samples)
    if not arguments.quiet:
        _announce(f"judging {len(planned)} answers with {config.model}")
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("official LLM scorers require mindbridge[openai]") from None
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )
    cache = (
        None
        if arguments.use_cache is None
        else ResponseCache(
            arguments.use_cache,
            arguments.run_id,
            _judge_cache_namespace(config),
        )
    )
    semaphore = asyncio.Semaphore(config.concurrency)
    completed = 0

    async def judge(sample: SampleResult) -> SampleResult:
        nonlocal completed
        plan = planned.get(sample.sample_id)
        result = await _judge_sample(
            prepared_samples.get(sample.sample_id, sample),
            plan,
            client=client,
            cache=cache,
            semaphore=semaphore,
            config=config,
            log_samples=arguments.log_samples,
            tracer=selected_tracer,
        )
        if plan is not None:
            completed += 1
            progress(completed, len(planned))
        return result

    try:
        with _progress(
            f"judging with {config.model}",
            "answer",
            total=len(planned),
            enabled=not arguments.quiet,
        ) as progress:
            judged = await asyncio.gather(*(judge(sample) for sample in samples))
    finally:
        await client.close()
        if cache is not None:
            cache.close()
    return tuple(judged)


def _restore_local_scores(sample: SampleResult, question: EvalQuestion) -> SampleResult:
    if sample.error_code != "JudgeError":
        return sample
    # Older artifacts could have overwritten a deterministic headline on judge failure.
    # Recompute it from the saved prediction; retain the original retrieval diagnostics.
    local = local_scores(
        sample.task,
        score_kind=question.score_kind,
        prediction=sample.prediction,
        parsed_choice=sample.parsed_choice,
        expected_choice=question.expected_choice,
        references=question.references,
        question=question.source_question,
        metadata=question.metadata,
        evidence_source_ids=sample.ranked_source_ids,
    )
    metrics = finalize_scores(
        sample.task,
        {
            **sample.metrics,
            **{name: value for name, value in local.items() if not name.startswith("retrieval_")},
        },
    )
    return replace(
        sample,
        metrics=metrics,
        score=metrics.get(sample_primary_metric(sample.task)),
        exact_match=metrics.get("exact_match"),
        error_code=None,
        scorer_error=None,
    )


async def _judge_sample(
    sample: SampleResult,
    plan: JudgePlan | None,
    *,
    client: AsyncOpenAI,
    cache: ResponseCache | None,
    semaphore: asyncio.Semaphore,
    config: _JudgeConfig,
    log_samples: bool,
    tracer: Tracer,
) -> SampleResult:
    if plan is None:
        return sample
    try:
        outcomes = tuple(
            [
                await _traced_judge_call(
                    client,
                    messages,
                    sample=sample,
                    plan=plan,
                    call_index=index,
                    cache=cache,
                    semaphore=semaphore,
                    config=config,
                    tracer=tracer,
                )
                for index, messages in enumerate(plan.calls)
            ]
        )
        scores = combine_judge_scores(plan, tuple(item[0] for item in outcomes))
        metrics = finalize_scores(sample.task, {**sample.metrics, **scores})
        responses = tuple(item[1] for item in outcomes)
        return replace(
            sample,
            score=metrics.get(sample_primary_metric(sample.task)),
            exact_match=metrics.get("exact_match"),
            metrics=metrics,
            scorer_details={**sample.scorer_details, **plan.details},
            judge_model=config.model,
            judge_response=(json.dumps(responses, ensure_ascii=False) if log_samples else None),
            judge_cached=all(item[2] for item in outcomes),
            error_code=None if sample.error_code == "JudgeError" else sample.error_code,
            scorer_error=None,
        )
    except _JudgeRequestError as failure:
        error = failure.error
        message = " ".join(str(error).split())[:500]
        # An unavailable verdict scores zero only for metrics that depend on the judge.
        metrics = judge_failure_scores(sample.task, sample.metrics)
        return replace(
            sample,
            score=metrics.get(sample_primary_metric(sample.task)),
            metrics=metrics,
            error_code=sample.error_code or "JudgeError",
            scorer_error=f"{type(error).__name__}: {message}",
            scorer_details={**sample.scorer_details, **plan.details},
            judge_model=config.model,
        )


async def _traced_judge_call(
    client: AsyncOpenAI,
    messages: Sequence[JudgeMessage],
    *,
    sample: SampleResult,
    plan: JudgePlan,
    call_index: int,
    cache: ResponseCache | None,
    semaphore: asyncio.Semaphore,
    config: _JudgeConfig,
    tracer: Tracer,
) -> tuple[Mapping[str, float], str, bool]:
    cached = _cached_judge_call(
        cache, messages, sample=sample, plan=plan, call_index=call_index, config=config
    )
    if cached is not None:
        return cached
    with model_span(
        tracer,
        BENCHMARK_JUDGE_SPAN,
        attributes={
            BENCHMARK_TASK: sample.task,
            BENCHMARK_SAMPLE: sample.sample_id,
            BENCHMARK_ARM: sample.arm,
            BENCHMARK_PURPOSE: JUDGE_PURPOSE,
            SPAN_KIND: "model",
            MODEL_MODULE: "judge",
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": config.model,
        },
    ):
        return await _judge_call(
            client,
            messages,
            sample=sample,
            plan=plan,
            call_index=call_index,
            cache=cache,
            semaphore=semaphore,
            config=config,
            read_cache=False,
        )


async def _judge_call(  # noqa: C901 - retry and provider fallback belong in one request path
    client: AsyncOpenAI,
    messages: Sequence[JudgeMessage],
    *,
    sample: SampleResult,
    plan: JudgePlan,
    call_index: int,
    cache: ResponseCache | None,
    semaphore: asyncio.Semaphore,
    config: _JudgeConfig,
    read_cache: bool = True,
) -> tuple[Mapping[str, float], str, bool]:
    from httpx2 import TransportError
    from openai import APIError

    request_errors = (APIError, TransportError)
    cache_task, key = _judge_cache_coordinates(
        messages,
        plan=plan,
        call_index=call_index,
        config=config,
    )
    if read_cache:
        cached = _cached_judge_call(
            cache,
            messages,
            sample=sample,
            plan=plan,
            call_index=call_index,
            config=config,
        )
        if cached is not None:
            return cached
    request: _JudgeRequest = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": message.content}
            if message.role == "system"
            else {"role": "user", "content": message.content}
            for message in messages
        ],
    }
    if plan.max_tokens is not None:
        if output_limit_parameter(config.model) == "max_completion_tokens":
            request["max_completion_tokens"] = plan.max_tokens
        else:
            request["max_tokens"] = plan.max_tokens
    extra_body = dict(plan.extra_body or {})
    model_key = re.sub(r"[^a-z0-9]+", "", config.model.casefold())
    if "qwen3" in model_key and not (
        "enable_thinking" in extra_body and judge_model_is_official(sample.task, config.model)
    ):
        extra_body.setdefault("chat_template_kwargs", {"enable_thinking": False})
    if extra_body:
        request["extra_body"] = extra_body
    if sampling_supported(config.model, extra_body):
        request["temperature"] = 0.0
    use_responses_api = plan.parser == "atm" and judge_model_is_official(sample.task, config.model)
    usages = []
    attempted = 0

    async def call() -> ChatCompletion | Response:
        nonlocal attempted
        async with semaphore:
            attempted += 1
            mark_model_requests(attempted)
            try:
                if use_responses_api:
                    return await client.responses.create(
                        model=config.model,
                        input="\n".join(message.content for message in messages),
                        max_output_tokens=plan.max_tokens,
                        reasoning={"effort": "minimal"},
                    )
                completion: ChatCompletion = await client.chat.completions.create(**request)
                return completion
            except request_errors as error:
                raise _JudgeRequestError(error) from error

    try:
        while True:
            try:
                response = await _retry_transient(call)
                break
            except _JudgeRequestError as failure:
                if "temperature" in request and _unsupported_temperature(failure.error):
                    request.pop("temperature")
                    continue
                if "extra_body" in request and _unsupported_extra_body(failure.error):
                    request.pop("extra_body")
                    continue
                raise
        text = (
            str(cast("Response", response).output_text or "").strip()
            if use_responses_api
            else str(cast("ChatCompletion", response).choices[0].message.content or "").strip()
        )
        _record_openai_provenance(response)
        usages.append(
            _model_usage(
                response,
                input_modalities=frozenset({Modality.TEXT}),
                output_modalities=frozenset({Modality.TEXT}),
            )
        )
        scores = parse_judge_response(plan, text)
        if cache is not None:
            cache.put(cache_task, sample.unit_id, key, CachedAnswer(text, 0.0, ()))
        return scores, text, False
    finally:
        _record_usage_batch(usages, request_count=attempted)


def _judge_cache_coordinates(
    messages: Sequence[JudgeMessage],
    *,
    plan: JudgePlan,
    call_index: int,
    config: _JudgeConfig,
) -> tuple[str, str]:
    key = hashlib.sha256(
        _json_bytes(
            {
                "version": SCORER_VERSION,
                "model": config.model,
                "protocol": plan.protocol,
                "call": call_index,
                "messages": tuple((message.role, message.content) for message in messages),
            }
        )
    ).hexdigest()
    return f"judge:{plan.protocol}:{config.model}", key


def _cached_judge_call(
    cache: ResponseCache | None,
    messages: Sequence[JudgeMessage],
    *,
    sample: SampleResult,
    plan: JudgePlan,
    call_index: int,
    config: _JudgeConfig,
) -> tuple[Mapping[str, float], str, bool] | None:
    if cache is None:
        return None
    cache_task, key = _judge_cache_coordinates(
        messages,
        plan=plan,
        call_index=call_index,
        config=config,
    )
    cached = cache.get(cache_task, sample.unit_id, key)
    if cached is None:
        return None
    return parse_judge_response(plan, cached.prediction), cached.prediction, True


def _unsupported_extra_body(error: BaseException) -> bool:
    message = str(error).casefold()
    field = "enable_thinking" in message or "chat_template_kwargs" in message
    marker = any(
        value in message
        for value in (
            "unknown",
            "unsupported",
            "unexpected",
            "unrecognized",
            "not allowed",
            "invalid parameter",
        )
    )
    return field and marker


def _unsupported_temperature(error: BaseException) -> bool:
    if getattr(error, "status_code", None) not in (400, 422):
        return False
    message = str(error).casefold()
    return bool(re.search(r"\btemperature\b", message)) and any(
        marker in message
        for marker in ("unsupported", "does not support", "not supported", "not allowed")
    )


def _judge_cache_namespace(config: _JudgeConfig) -> str:
    return hashlib.sha256(
        _json_bytes(
            {
                "version": SCORER_VERSION,
                "model": config.model,
                "base_url": config.base_url,
                "temperature": 0.0 if sampling_supported(config.model, None) else None,
            }
        )
    ).hexdigest()
