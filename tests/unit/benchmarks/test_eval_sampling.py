"""The product and baseline arms send the same resolved sampling controls."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast

import openai
import pytest

import mindbridge.configuration as configuration_module
from mindbridge import EmbedTask, MindBridgeConfig, Modality, ModelInput, SearchHit
from mindbridge.benchmarks import eval as eval_module
from mindbridge.benchmarks.eval_arms import _BaselineGenerator
from mindbridge.benchmarks.eval_config import (
    _Arguments,
    _evaluation_memory_config,
    _generation_kwargs,
    _generation_temperature,
)
from mindbridge.benchmarks.model_config import ModelConfig


@pytest.mark.parametrize(
    "configured_style", [None, "default", "concise"], ids=["raw", "configured", "concise"]
)
@pytest.mark.parametrize("temperature", [0.0, 0.7], ids=["default", "sampled"])
def test_every_generation_arm_emits_the_resolved_temperature(  # noqa: C901 - sync and async clients
    monkeypatch: pytest.MonkeyPatch, configured_style: str | None, temperature: float
) -> None:
    requests: list[dict[str, object]] = []

    def complete(request: dict[str, object]) -> object:
        requests.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(content="blue", refusal=None),
                )
            ],
            usage=None,
        )

    class Completions:
        def create(self, **request: object) -> object:
            return complete(request)

    class AsyncCompletions:
        async def create(self, **request: object) -> object:
            return complete(request)

    class Client:
        def __init__(self, **_options: object) -> None:
            self.chat = SimpleNamespace(completions=Completions())

        def close(self) -> None: ...

    class AsyncClient:
        def __init__(self, **_options: object) -> None:
            self.chat = SimpleNamespace(completions=AsyncCompletions())

        async def close(self) -> None: ...

    class Embedder:
        embedding_model = "sampling-fixture-v1"
        embedding_space = "sampling-fixture-v1:2"
        embedding_dimension = 2
        embedding_capabilities = frozenset({Modality.TEXT})

        def __init__(self, **_options: object) -> None: ...

        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask
        ) -> tuple[tuple[float, ...], ...]:
            del task
            return tuple((1.0, 0.0) for _value in inputs)

        def close(self) -> None: ...

    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setattr(openai, "AsyncOpenAI", AsyncClient)
    monkeypatch.setattr(eval_module, "SentenceTransformersEmbedder", Embedder)
    monkeypatch.setattr(configuration_module, "SentenceTransformersEmbedder", Embedder)
    options = _generation_kwargs(f"temperature={temperature}", 7)
    arguments = cast(_Arguments, SimpleNamespace(gen_kwargs=options, seed=7, device=None))
    model = ModelConfig(generation_model="qwen3.8-27b")
    declared = (
        MindBridgeConfig.model_validate(
            {
                "embedding": {"provider": "sentence-transformers", "model": "fixture"},
                "generation": {
                    "provider": "openai",
                    "model": model.generation_model,
                    "answer_style": configured_style,
                },
            }
        )
        if configured_style is not None
        else None
    )
    effective = _evaluation_memory_config(declared, model, arguments)
    pool = eval_module._BackendPool(
        model,
        device=None,
        batch_size=1,
        needs_speech=False,
        seed=7,
        gen_kwargs=options,
        memory_config=effective,
    )
    baseline = _BaselineGenerator(model, seed=7, gen_kwargs=options, generation=effective)
    try:
        assert pool._answerer is not None
        hit = SearchHit(
            id="toolbox",
            content="The toolbox is blue",
            score=1.0,
            created_at=datetime(2026, 10, 28, tzinfo=timezone.utc),
        )
        assert pool._answerer.answer(ModelInput(text="What colour?"), (hit,)).answer == "blue"
        assert asyncio.run(baseline.answer("What colour?", hit.content)) == "blue"
        recorded = eval_module._model_result(arguments, model, effective)
    finally:
        pool.close()
        asyncio.run(baseline.close())

    assert len(requests) == 2
    assert all(request["temperature"] == temperature for request in requests)
    assert all(request["seed"] == 7 for request in requests)
    assert recorded["generation_temperature"] == temperature
    assert recorded["generation_kwargs"] == options
    assert recorded["generation_answer_style"] == (configured_style or "default")
    for request in requests:
        messages = cast(list[dict[str, str]], request["messages"])
        assert ("Give only the answer to the question" in messages[0]["content"]) == (
            configured_style == "concise"
        )


def test_explicit_temperature_is_not_silently_dropped_for_an_unsupported_model() -> None:
    with pytest.raises(ValueError, match="does not support"):
        _generation_temperature("gpt-5-mini", {"temperature": "0.7"})
    assert (
        _generation_temperature("gpt-5.4", {"temperature": "0.7"}, {"reasoning_effort": "none"})
        == 0.7
    )
