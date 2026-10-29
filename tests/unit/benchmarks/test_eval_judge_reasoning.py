"""Judge request capabilities and replay of saved answers after provider failures."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx2 as httpx
import openai
import pytest
from openai import AsyncOpenAI
from opentelemetry import trace

from mindbridge.benchmarks import eval_judge
from mindbridge.benchmarks.eval_adapters import EvalQuestion, EvalUnit, LoadedTask
from mindbridge.benchmarks.eval_config import _Arguments, _JudgeConfig
from mindbridge.benchmarks.eval_results import SampleResult
from mindbridge.benchmarks.official_scorers import JudgePlan, judge_failure_scores, judge_plan
from mindbridge.benchmarks.task_catalog import TaskSpec


def _sample() -> SampleResult:
    return SampleResult(
        task="mem-gallery",
        benchmark="Mem-Gallery",
        dataset_sha256="dataset",
        evaluation_sha256="evaluation",
        unit_id="unit",
        question_id="question",
        prediction="red door",
        parsed_choice=None,
        score=1.0,
        exact_match=1.0,
        latency_ms=0.0,
        confidence=1.0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=None,
        metadata={},
        metrics={"f1": 1.0, "bleu_1": 1.0, "retrieval_recall@10": 0.5},
    )


def _plan() -> JudgePlan:
    plan = judge_plan(
        "mem-gallery",
        question="What did I see?",
        references=("red door",),
        prediction="red door",
        metadata={},
    )
    assert plan is not None
    return plan


def _reply() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "judge",
            "object": "chat.completion",
            "created": 0,
            "model": "served-judge",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"score":1.0}'}}],
        },
    )


async def _judge(client: AsyncOpenAI, model: str, *, plan: JudgePlan | None = None) -> SampleResult:
    return await eval_judge._judge_sample(
        _sample(),
        plan or _plan(),
        client=client,
        cache=None,
        semaphore=asyncio.Semaphore(1),
        config=_JudgeConfig(model, "http://judge.invalid/v1"),
        log_samples=False,
        tracer=trace.get_tracer(__name__),
    )


@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-5-mini-2025-08-07", "o3-mini"])
async def test_known_reasoning_judge_omits_sampling_and_uses_completion_limit(model: str) -> None:
    requests: list[dict[str, object]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _reply()

    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply))
    ) as client:
        result = await _judge(client, model, plan=replace(_plan(), max_tokens=512))
    assert result.error_code is None
    assert result.metrics["llm_judge"] == 1.0
    assert requests[0]["model"] == model
    assert "temperature" not in requests[0]
    assert "max_tokens" not in requests[0]
    assert requests[0]["max_completion_tokens"] == 512


async def test_unknown_alias_retries_only_explicit_unsupported_temperature_once() -> None:
    requests: list[dict[str, object]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "Unsupported value: temperature does not support 0.0"}},
            )
        return _reply()

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply)),
    ) as client:
        result = await _judge(client, "custom-reasoning-alias")
    assert result.error_code is None
    assert len(requests) == 2
    assert requests[0].pop("temperature") == 0.0
    assert requests[0] == requests[1]
    assert requests[1]["model"] == "custom-reasoning-alias"


@pytest.mark.parametrize(
    "message",
    ["Invalid value temperature: outside range", "unsupported parameter unrelated_field"],
)
async def test_other_provider_rejections_do_not_change_request(message: str) -> None:
    calls = 0

    def reply(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": message}})

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply)),
    ) as client:
        result = await _judge(client, "custom-judge")
    assert calls == 1
    assert result.error_code == "JudgeError"
    assert result.score == result.metrics["f1"] == 1.0
    assert result.metrics["bleu_1"] == 1.0
    assert result.metrics["retrieval_recall@10"] == 0.5
    assert result.metrics["llm_judge"] == 0.0


def test_judge_failure_zeros_judge_derived_metrics_without_erasing_local_metrics() -> None:
    scores = judge_failure_scores(
        "atm-bench", {"accuracy": 1.0, "retrieval_recall@5": 0.5, "joint_partial@5": 0.5}
    )
    assert scores["accuracy"] == scores["joint_partial@5"] == 0.0
    assert scores["retrieval_recall@5"] == 0.5
    scores = judge_failure_scores("es-memeval", {"f1": 0.75, "llm_judge": 1.0})
    assert scores["f1"] == 0.75
    assert scores["llm_judge"] == 0.0


async def test_saved_judge_error_replays_prediction_and_restores_deterministic_f1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, object]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _reply()

    client = AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply))
    )
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_kwargs: client)
    question = EvalQuestion(
        "question", ("What did I see?",), ("red door",), source_question="What did I see?"
    )
    spec = TaskSpec("mem-gallery", "Mem-Gallery", "fixture.json", "v1", "owner/repo", "0" * 40)
    task = LoadedTask(
        spec, tmp_path / "fixture.json", "1" * 64, (EvalUnit("unit", (), (question,)),)
    )
    sample = replace(
        _sample(),
        score=0.0,
        error_code="JudgeError",
        scorer_error="BadRequestError: old unsupported temperature",
        metrics={"f1": 0.0, "bleu_1": 1.0, "retrieval_recall@10": 0.5},
    )
    arguments = cast(
        _Arguments, SimpleNamespace(quiet=True, use_cache=None, run_id="replay", log_samples=False)
    )
    (result,) = await eval_judge._apply_judges(
        (task,),
        (sample,),
        arguments=arguments,
        config=_JudgeConfig("gpt-5-mini", "http://judge.invalid/v1", api_key="test"),
    )
    assert len(requests) == 1
    assert result.prediction == sample.prediction
    assert result.latency_ms == sample.latency_ms
    assert result.error_code is None
    assert result.scorer_error is None
    assert result.score == result.metrics["f1"] == result.metrics["llm_judge"] == 1.0
    assert result.metrics["retrieval_recall@10"] == 0.5
    assert "temperature" not in requests[0]
