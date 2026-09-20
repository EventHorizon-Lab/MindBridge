from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import httpx2 as httpx
from openai import OpenAI

from mindbridge import AnswerPolicy, AnswerResult, Modality, SearchHit
from mindbridge.models.base import EmbedTask, ModelInput


class _FakeModels:
    embedding_model = "deterministic"
    embedding_space = "deterministic:2"
    embedding_dimension = 2
    embedding_capabilities = generation_capabilities = frozenset({Modality.TEXT})

    def __init__(self) -> None:
        self.answer_calls: list[tuple[ModelInput, tuple[SearchHit, ...]]] = []

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in inputs)

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        self.answer_calls.append((question, tuple(hits)))
        return AnswerResult("A caliper from Niko.", tuple(hits))

    def close(self) -> None:
        pass


def _driver() -> ModuleType:
    path = Path(__file__).parents[3] / "benchmarks/complementary_trial.py"
    spec = importlib.util.spec_from_file_location("complementary_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_trial_uses_one_shared_plan_and_keeps_labels_out(tmp_path: Path) -> None:
    driver = _driver()
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        requests.append(data)
        if "response_format" in data:
            payload = json.loads(data["messages"][1]["content"])
            assert "HIDDEN_REFERENCE" not in str(payload)
            text = json.dumps(
                {
                    "facets": ["person", "tool"],
                    "records": [
                        {"index": r["index"], "relevance": 3, "covers": [r["index"] % 2]}
                        for r in payload["records"]
                    ],
                }
            )
        else:
            text = "Yes"
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 0,
                "model": "mock",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    audit = driver.Audit(tmp_path / "http.jsonl")
    with OpenAI(
        api_key="test",
        base_url="http://mock/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(respond),
            event_hooks={"response": [audit.response]},
        ),
    ) as client:
        models = _FakeModels()
        reader = driver.SelectorBackend(
            models,
            client,
            "mock",
            audit,
            {"reader_rows": 2, "reader_chars": 1000, "candidate_rows": 60},
            tmp_path,
        )
        with driver.open_memory(tmp_path / "store", models, reader) as memory:
            memory.add("Mara borrowed the caliper.", metadata={"source_id": "one"})
            memory.add("Niko lent the tool.", metadata={"source_id": "two"})
            rows = driver.evaluate(
                memory,
                reader,
                audit,
                {
                    "id": "independent",
                    "dataset": "constructed-text-diagnostics",
                    "question": "What tool and from whom?",
                    "reference": "HIDDEN_REFERENCE",
                    "gold": ["one", "two"],
                },
                tmp_path,
            )
    assert len(rows) == 3 and all(row["plan_valid"] for row in rows)
    assert sum("response_format" in request for request in requests) == 1
    assert len(models.answer_calls) == 3
    assert len({tuple(row["candidate_ids"]) for row in rows}) == 1
    assert "HIDDEN_REFERENCE" not in repr(models.answer_calls)


def test_media_audit_hashes_payload_without_logging_media_bytes() -> None:
    driver = _driver()
    result = driver.redact_media({"url": "data:image/jpeg;base64,SECRET_FRAME"})
    assert "SECRET_FRAME" not in json.dumps(result)
    assert result["url"]["encoded_chars"] > 0
