"""Provider outages may score zero; failures in the local scorer must surface."""

from __future__ import annotations

import asyncio
import json
from contextlib import closing
from pathlib import Path

import httpx as download_httpx
import httpx2 as httpx
import pytest
from openai import APIConnectionError, AsyncOpenAI
from opentelemetry import trace

from mindbridge.benchmarks import eval_judge, eval_results
from mindbridge.benchmarks.eval_cache import ResponseCache
from mindbridge.benchmarks.eval_config import _JudgeConfig
from mindbridge.benchmarks.eval_results import SampleResult
from mindbridge.benchmarks.official_scorers import finalize_scores, judge_plan


def _sample() -> SampleResult:
    return SampleResult(
        task="m3-bench-robot",
        benchmark="m3",
        dataset_sha256="dataset",
        evaluation_sha256="evaluation",
        unit_id="unit",
        question_id="question",
        prediction="yes",
        parsed_choice=None,
        score=None,
        exact_match=None,
        latency_ms=0.0,
        confidence=1.0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=None,
        metadata={},
        metrics={"retrieval_recall@5": 1.0},
    )


async def _judge(client: AsyncOpenAI, cache: ResponseCache | None = None) -> SampleResult:
    plan = judge_plan(
        "m3-bench-robot",
        question="Is the door open?",
        references=("yes",),
        prediction="yes",
        metadata={},
    )
    assert plan is not None
    return await eval_judge._judge_sample(
        _sample(),
        plan,
        client=client,
        cache=cache,
        semaphore=asyncio.Semaphore(1),
        config=_JudgeConfig("qwen3", "http://judge.invalid/v1"),
        log_samples=False,
        tracer=trace.get_tracer(__name__),
    )


def _reply(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "answer",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen3",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "yes"}}],
        },
    )


@pytest.mark.parametrize(
    "stage",
    ["parse_judge_response", "combine_judge_scores", "finalize_scores", "_record_usage_batch"],
)
async def test_local_judge_failure_propagates(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("local scorer failed")

    if stage == "finalize_scores":
        original = finalize_scores
        calls = 0

        def finalize(task: str, scores: dict[str, float]) -> dict[str, float]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("local scorer failed")
            return original(task, scores)

        monkeypatch.setattr(eval_judge, stage, finalize)
    else:
        monkeypatch.setattr(eval_judge, stage, fail)
    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(_reply))
    ) as client:
        with pytest.raises(RuntimeError, match="local scorer failed"):
            await _judge(client)


@pytest.mark.parametrize("operation", ["get", "put"])
async def test_local_cache_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("cache failed")

    with closing(ResponseCache(tmp_path, "run", "namespace")) as cache:
        monkeypatch.setattr(cache, operation, fail)
        async with AsyncOpenAI(
            api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(_reply))
        ) as client:
            with pytest.raises(OSError, match="cache failed"):
                await _judge(client, cache)


async def test_exhausted_provider_failure_still_scores_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("judge unavailable", request=request)

    monkeypatch.setattr(eval_results, "_TRANSIENT_RETRY_SECONDS", 0.0)
    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fail)),
    ) as client:
        result = await _judge(client)
    assert result.error_code == "JudgeError"
    assert result.score == 0.0
    assert result.metrics["retrieval_recall@5"] == 1.0
    assert result.scorer_error is not None
    assert result.scorer_error.startswith("APIConnectionError:")


async def test_local_failure_with_provider_parameter_words_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def reply(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _reply(request)

    def fail(*_args: object) -> None:
        raise ValueError("unsupported enable_thinking in local parser")

    monkeypatch.setattr(eval_judge, "parse_judge_response", fail)
    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply))
    ) as client:
        with pytest.raises(ValueError, match="local parser"):
            await _judge(client)
    assert calls == 1


@pytest.mark.parametrize(
    "error",
    [
        APIConnectionError(request=httpx.Request("POST", "http://local.invalid")),
        httpx.ConnectError("local parser failed"),
        download_httpx.ReadTimeout("local parser failed"),
    ],
)
async def test_provider_exception_raised_by_local_parser_propagates(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*_args: object) -> None:
        raise error

    monkeypatch.setattr(eval_judge, "parse_judge_response", fail)
    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(_reply))
    ) as client:
        with pytest.raises(type(error)):
            await _judge(client)


async def test_local_request_accounting_failure_is_not_a_provider_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_count: int) -> None:
        raise APIConnectionError(request=httpx.Request("POST", "http://local.invalid"))

    monkeypatch.setattr(eval_judge, "mark_model_requests", fail)
    monkeypatch.setattr(eval_results, "_TRANSIENT_RETRY_SECONDS", 0.0)
    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(_reply))
    ) as client:
        with pytest.raises(APIConnectionError):
            await _judge(client)


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout])
async def test_raw_request_transport_failure_scores_zero(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    async def fail(**_kwargs: object) -> None:
        raise failure("request transport failed")

    async with AsyncOpenAI(api_key="test") as client:
        monkeypatch.setattr(client.chat.completions, "create", fail)
        result = await _judge(client)
    assert result.error_code == "JudgeError"
    assert result.score == 0.0
    assert result.scorer_error == f"{failure.__name__}: request transport failed"


async def test_unrelated_http_client_exception_in_request_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(**_kwargs: object) -> None:
        raise download_httpx.ReadTimeout("unrelated transport")

    async with AsyncOpenAI(api_key="test") as client:
        monkeypatch.setattr(client.chat.completions, "create", fail)
        with pytest.raises(download_httpx.ReadTimeout, match="unrelated transport"):
            await _judge(client)


@pytest.mark.parametrize("status", [400, 503])
async def test_provider_fallback_and_transient_retry_preserve_answer(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    requests: list[dict[str, object]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                status,
                json={"error": {"message": "unsupported enable_thinking", "type": "error"}},
            )
        return _reply(request)

    async def sleep(_delay: float) -> None:
        pass

    monkeypatch.setattr(asyncio, "sleep", sleep)
    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply)),
    ) as client:
        result = await _judge(client)
    assert result.score == 1.0
    assert result.error_code is None
    assert len(requests) == 2
    if status == 400:
        assert "chat_template_kwargs" in requests[0]
        assert "chat_template_kwargs" not in requests[1]
    else:
        assert requests[0] == requests[1]
