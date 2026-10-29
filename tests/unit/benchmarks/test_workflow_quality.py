"""Exercise the workflow gate through the public SDK without a provider connection."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx2 as httpx
import pytest
from openai import OpenAI

from mindbridge import Modality, OpenAIModels
from mindbridge.benchmarks import workflow_quality


def _responder(broken: str | None) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["store"] is False
        name = payload.get("response_format", {}).get("json_schema", {}).get("name")
        if name == "formation":
            content = json.dumps(
                {
                    "items": [
                        {
                            "observation_id": "observation_0",
                            "proposals": [
                                {
                                    "kind": "event",
                                    "content": "I moved to Paris.",
                                    "confidence": 0.9,
                                    "valid_from": "2099-01-01T00:00:00Z"
                                    if broken == "formation"
                                    else "2026-03-01T00:00:00Z",
                                    "evidence_observation_ids": ["observation_0"],
                                }
                            ],
                        }
                    ]
                }
            )
        elif name == "recall_plan":
            content = (
                "invalid"
                if broken == "planning"
                else json.dumps(
                    {
                        "shape": "set",
                        "steps": [{"op": "match", "terms": ["toolbox delivery"], "any_of": True}],
                    }
                )
            )
        elif name == "vision":
            content = json.dumps({"descriptions": ["Shown: a large red rectangle"]})
        elif name == "consolidation":
            content = json.dumps({"operations": [{"intent": "forget", "targets": [0, 1, 2]}]})
        else:
            user = payload["messages"][1]["content"]
            text = user if isinstance(user, str) else json.dumps(user)
            content = "blue" if broken == "vision" else "red"
            if "deliveries" in text:
                content = "3"
            if "工具箱" in text:
                content = "紫色" if broken == "answering" else "蓝色"
        if payload.get("stream"):
            chunks = [
                {
                    "model": "fixture-returned-snapshot",
                    "choices": [
                        {"index": 0, "delta": {"content": content}, "finish_reason": "stop"}
                    ],
                },
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
                },
            ]
            body = (
                "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "id": "fixture-response",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture-returned-snapshot",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )

    return respond


def test_workflow_gate_checks_all_stages_in_distinct_physical_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths: list[Path] = []

    def tracked_directory(*, prefix: str) -> TemporaryDirectory[str]:
        directory = TemporaryDirectory(prefix=prefix)
        paths.append(Path(directory.name))
        return directory

    monkeypatch.setattr(workflow_quality, "TemporaryDirectory", tracked_directory)
    with (
        httpx.Client(transport=httpx.MockTransport(_responder(None))) as transport,
        OpenAI(api_key="test", http_client=transport, max_retries=0) as client,
    ):
        models = OpenAIModels(
            client,
            generation_model="fixture-snapshot",
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
        )
        vision = OpenAIModels(
            client,
            generation_model="fixture-snapshot",
            generation_capabilities=frozenset({Modality.IMAGE}),
        )
        report = workflow_quality.run_workflow_quality_eval(models, vision, repeats=2)

    assert report.passed, [
        (case.case_id, case.error_reason) for case in report.cases if not case.passed
    ]
    assert report.structured_output_capability_evaluated
    assert report.response_models == ("fixture-returned-snapshot",)
    assert len(report.cases) == len(set(paths)) == 12
    assert all(not path.exists() for path in paths)
    assert "SYSTEM OVERRIDE" not in json.dumps(asdict(report))


@pytest.mark.parametrize("broken", ["formation", "planning", "vision", "answering"])
def test_workflow_gate_rejects_wrong_dates_planning_fallback_and_ungrounded_answers(
    broken: str,
) -> None:
    with (
        httpx.Client(transport=httpx.MockTransport(_responder(broken))) as transport,
        OpenAI(api_key="test", http_client=transport, max_retries=0) as client,
    ):
        models = OpenAIModels(
            client,
            generation_model="fixture-snapshot",
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
        )
        vision = OpenAIModels(
            client,
            generation_model="fixture-snapshot",
            generation_capabilities=frozenset({Modality.IMAGE}),
        )
        report = workflow_quality.run_workflow_quality_eval(models, vision, repeats=1)

    assert not report.passed
    expected = {
        "formation": "formation_relative_time",
        "planning": "recall_collection_count",
        "vision": "vision_prompt_injection",
        "answering": "answering_chinese",
    }[broken]
    assert not next(case for case in report.cases if case.case_id == expected).passed


@pytest.mark.parametrize("repeats", [0, 11, True])
def test_workflow_gate_bounds_repeat_counts(repeats: int) -> None:
    with pytest.raises(ValueError, match="repeats"):
        workflow_quality.run_workflow_quality_eval(OpenAIModels(), OpenAIModels(), repeats=repeats)


def test_workflow_command_without_credentials_makes_no_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = tmp_path / "report.json"
    assert workflow_quality.main(["--model", "fixture-snapshot", "--output", str(output)]) == 2
    assert not output.exists()
