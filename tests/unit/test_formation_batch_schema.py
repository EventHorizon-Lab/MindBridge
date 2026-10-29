"""Strict formation batches constrain coverage while local semantics remain authoritative."""

import json
from typing import Any, Literal

import httpx2 as httpx
import pytest
from test_openai_sdk import _sdk_client

from mindbridge import FormationInput, ModelError, ModelInput, ObservationContext, OpenAIModels
from mindbridge.models._openai_schemas import response_schema


def _inputs(count: int) -> tuple[FormationInput, ...]:
    return tuple(
        FormationInput(
            memory_id=f"private-source-{index}",
            content=ModelInput(text=f"Source note {index}."),
            context=ObservationContext(),
        )
        for index in range(count)
    )


def _reply(aliases: list[str]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": json.dumps(
                            {
                                "items": [
                                    {"observation_id": alias, "proposals": []} for alias in aliases
                                ]
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    )


@pytest.mark.parametrize("mode", ["json_schema", "json_object"])
def test_each_batch_uses_its_own_envelope_without_forcing_facts(
    mode: Literal["json_schema", "json_object"],
) -> None:
    sent: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        count = len(json.loads(body["messages"][1]["content"])["observations"])
        return _reply([f"observation_{index}" for index in range(count)])

    original = response_schema("formation")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(_sdk_client(client), generation_response_format=mode)
        assert adapter.form(()) == ()
        for count in (4, 1):
            assert adapter.form(_inputs(count)) == ((),) * count
    assert len(sent) == 2
    for body, count in zip(sent, (4, 1), strict=True):
        if mode == "json_object":
            assert body["response_format"] == {"type": "json_object"}
            continue
        wire = body["response_format"]["json_schema"]
        assert wire["name"] == "formation" and wire["strict"] is True
        schema = wire["schema"]
        items = schema["properties"]["items"]
        assert items["minItems"] == items["maxItems"] == count
        fields = items["items"]["properties"]
        assert fields["observation_id"] == {
            "type": "string",
            "enum": [f"observation_{index}" for index in range(count)],
        }
        # Only envelope cardinality and source aliases change. Empty proposal sets stay valid.
        del items["minItems"], items["maxItems"]
        fields["observation_id"] = {"type": "string"}
        assert schema == original
    assert response_schema("formation") == original


def test_semantic_repair_retains_the_exact_batch_schema() -> None:
    sent: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        count = 1 if len(sent) == 1 else 4
        return _reply([f"observation_{index}" for index in range(count)])

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(_sdk_client(client), generation_response_format="json_schema")
        assert adapter.form(_inputs(4)) == ((),) * 4
    assert len(sent) == 2
    assert sent[0]["response_format"] == sent[1]["response_format"]
    assert "Repair only the JSON envelope" in sent[1]["messages"][1]["content"]
    assert sent[0]["messages"][0] == sent[1]["messages"][0]


@pytest.mark.parametrize(
    "aliases",
    [
        ["observation_0"],
        ["observation_0"] * 4,
        ["observation_0", "observation_1", "observation_2", "history_0"],
    ],
)
def test_provider_ignoring_envelope_constraints_still_fails_closed(aliases: list[str]) -> None:
    requests = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return _reply(aliases)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(_sdk_client(client), generation_response_format="json_schema")
        with pytest.raises(ModelError) as failure:
            adapter.form(_inputs(4))
    assert requests == 2
    assert failure.value.reason == "response_invalid"
