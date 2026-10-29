"""Identical history is shared on the wire without sharing witness authorization."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Literal, cast

import httpx2 as httpx
import pytest
from test_formation_history import _history_input
from test_openai_sdk import _sdk_client

import mindbridge.models.openai_sdk as backend
from mindbridge import AssetRef, FormationInput, Modality, ModelInput, OpenAIModels


def _response(witnesses: list[list[str]]) -> str:
    return json.dumps(
        {
            "items": [
                {
                    "observation_id": f"observation_{index}",
                    "proposals": [
                        {
                            "kind": "event",
                            "content": "The visitor left a notebook.",
                            "confidence": 0.9,
                            "evidence_observation_ids": aliases,
                        }
                    ],
                }
                for index, aliases in enumerate(witnesses)
            ]
        }
    )


def _payload(inputs: tuple[FormationInput, ...]) -> dict[str, Any]:
    content = backend._formation_content(inputs)
    assert isinstance(content, str)
    return cast(dict[str, Any], json.loads(content))


@pytest.mark.parametrize("mode", ["json_schema", "json_object"])
def test_http_request_sends_four_identical_thirty_record_histories_once(
    mode: Literal["json_schema", "json_object"],
) -> None:
    first = _history_input()
    history = tuple(
        replace(
            first.history[0],
            id=f"private-history-{index:02}",
            content=f"Visitor note {index}: a notebook was left on the desk.",
        )
        for index in range(30)
    )
    inputs = tuple(
        replace(first, memory_id=f"private-source-{i}", history=history) for i in range(4)
    )
    sent: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": _response(
                                [[f"observation_{i}", "history_0"] for i in range(4)]
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        adapter = OpenAIModels(
            _sdk_client(transport),
            generation_response_format=mode,
        )
        results = adapter.form(inputs)
    assert len(sent) == 1
    wire = sent[0]["messages"][1]["content"]
    payload = json.loads(wire)
    assert tuple(payload["read_only_history_groups"]) == ("history_group_0",)
    shared = payload["read_only_history_groups"]["history_group_0"]
    assert [item["content"] for item in shared] == [record.content for record in history]
    for index, observation in enumerate(payload["observations"]):
        assert observation["observation_id"] == f"observation_{index}"
        assert observation["read_only_history_group"] == "history_group_0"
        assert "read_only_history" not in observation
        assert observation["history_truncated"] is False
        assert results[index][0].evidence_ids == (inputs[index].memory_id, history[0].id)
    inline = dict(payload)
    del inline["read_only_history_groups"]
    inline["observations"] = [
        {key: value for key, value in observation.items() if key != "read_only_history_group"}
        | {"read_only_history": shared}
        for observation in payload["observations"]
    ]
    assert len(wire.encode()) < len(backend._json_text(inline).encode()) // 2
    assert "private-history-" not in wire and "private-source-" not in wire


@pytest.mark.parametrize(
    "difference",
    [
        "content",
        "confidence",
        "recipe",
        "hidden_witness",
        "visible",
        "retired",
        "order",
        "truncated",
        "iso",
    ],
)
def test_only_complete_equal_ordered_snapshots_are_shared(difference: str) -> None:
    first = _history_input()
    old = first.history[0]
    assert old.context is not None
    first = replace(first, history=(old, replace(old, id="other")))
    second = replace(first, memory_id="second")
    changed = old
    if difference == "content":
        changed = replace(old, content="A corrected visitor note.")
    elif difference == "confidence":
        changed = replace(old, context=replace(old.context, confidence=0.5))
    elif difference == "recipe":
        changed = replace(old, context=replace(old.context, recipe="another-private-recipe"))
    elif difference == "hidden_witness":
        first = replace(
            first,
            history=(
                replace(old, context=replace(old.context, evidence_ids=("other-hidden",))),
                first.history[1],
            ),
        )
        changed = replace(old, context=replace(old.context, evidence_ids=("not-in-history",)))
        assert backend._formation_history_payload(
            first.history[0], "history_0", {}
        ) == backend._formation_history_payload(changed, "history_0", {})
    elif difference == "visible":
        changed = replace(old, context=replace(old.context, visible=False))
    elif difference == "retired":
        changed = replace(
            old, context=replace(old.context, retired_at=old.created_at + timedelta(days=1))
        )
    elif difference == "iso":
        changed = replace(old, created_at=old.created_at.astimezone(timezone(timedelta(hours=8))))
        assert changed == old  # Equality alone would merge distinct model-visible timestamps.
    if difference == "order":
        second = replace(second, history=first.history[::-1])
    elif difference == "truncated":
        second = replace(second, history_truncated=True)
    else:
        second = replace(second, history=(changed, first.history[1]))
    payload = _payload((first, second))
    assert "read_only_history_groups" not in payload
    assert all("read_only_history" in item for item in payload["observations"])
    aliases = backend._formation_aliases((first, second))
    assert set(aliases[0].values()).isdisjoint(aliases[1].values())


def test_shared_aliases_do_not_authorize_a_private_or_missing_history() -> None:
    first = _history_input()
    second = replace(first, memory_id="second")
    retired = replace(_history_input(retired=True), memory_id="third")
    empty = replace(first, memory_id="fourth", history=())
    inputs = (first, second, retired, empty)
    payload = _payload(inputs)
    assert payload["observations"][2]["read_only_history"][0]["observation_id"] == "history_2"
    response = _response(
        [
            ["observation_0", "history_0"],
            ["observation_1", "history_0"],
            ["observation_2", "history_0"],
            ["observation_3", "history_0"],
        ]
    )
    results = backend._formation_results(response, inputs)
    assert tuple(len(items) for items in results) == (1, 1, 0, 0)
    assert results[0][0].evidence_ids == (first.memory_id, first.history[0].id)
    assert results[1][0].evidence_ids == (second.memory_id, first.history[0].id)
    # Its own retired snapshot cannot support a current assertion either.
    response = _response(
        [["observation_0"], ["observation_1"], ["observation_2", "history_2"], ["observation_3"]]
    )
    assert backend._formation_results(response, inputs)[2] == ()


def test_current_batch_history_keeps_observation_alias_and_own_source_once() -> None:
    first = _history_input()
    history = replace(first.history[0], id="third", context=None)
    first = replace(first, history=(history,))
    second = replace(first, memory_id="second")
    third = replace(first, memory_id="third", history=())
    inputs = (first, second, third)
    shared = _payload(inputs)["read_only_history_groups"]["history_group_0"]
    assert shared[0]["observation_id"] == "observation_2"
    results = backend._formation_results(
        _response(
            [
                ["observation_0", "observation_2"],
                ["observation_1", "observation_2"],
                ["observation_2"],
            ]
        ),
        inputs,
    )
    assert results[0][0].evidence_ids == ("current", "third")
    assert results[1][0].evidence_ids == ("second", "third")
    assert results[2][0].evidence_ids == ("third",)
    assert (
        backend._formation_results(
            _response([["observation_0"], ["observation_1"], ["observation_2", "observation_2"]]),
            inputs,
        )[2]
        == ()
    )


def test_native_parts_keep_observation_and_media_order_after_shared_header(tmp_path: Path) -> None:
    first = _history_input()
    assets = tuple(
        AssetRef(
            id=f"image-{index}",
            path=tmp_path / f"image-{index}.png",
            modality=Modality.IMAGE,
            media_type="image/png",
            size_bytes=1,
            sha256=str(index) * 64,
        )
        for index in range(2)
    )
    inputs = tuple(
        replace(
            first, memory_id=f"source-{index}", content=ModelInput(text="A photo.", assets=(asset,))
        )
        for index, asset in enumerate(assets)
    )
    image_parts: dict[str, tuple[dict[str, object], ...]] = {
        asset.id: ({"type": "image_url", "image_url": {"url": f"fixture:{asset.id}"}},)
        for asset in assets
    }
    content = backend._formation_content(inputs, image_parts=image_parts)
    assert isinstance(content, list)
    assert "read_only_history_groups" in json.loads(str(content[0]["text"]))
    for index in range(2):
        observation = json.loads(str(content[1 + 2 * index]["text"]))["observation"]
        assert observation["observation_id"] == f"observation_{index}"
        assert observation["media_order"] == [f"media_{index}"]
        assert content[2 + 2 * index] == image_parts[assets[index].id][0]


def test_nonshared_and_empty_history_keep_the_inline_shape() -> None:
    first = _history_input()
    empty = replace(first, memory_id="empty", history=())
    truncated = replace(empty, memory_id="truncated", history_truncated=True)
    payload = _payload((first, empty, truncated))
    assert set(payload) == {"observations"}
    assert payload["observations"][0]["read_only_history"][0]["observation_id"] == "history_0"
    assert "read_only_history" not in payload["observations"][1]
    assert "history_truncated" not in payload["observations"][1]
    assert payload["observations"][2]["read_only_history"] == []
    assert payload["observations"][2]["history_truncated"] is True


def test_shared_history_prompt_changes_both_formation_recipe_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = (OpenAIModels(), OpenAIModels(generation_response_format="json_schema"))
    monkeypatch.setattr(backend, "_FORMATION_SYSTEM_PROMPT", "previous formation presentation")
    after = (OpenAIModels(), OpenAIModels(generation_response_format="json_schema"))
    for old, new in zip(before, after, strict=True):
        assert old.formation_space != new.formation_space
        assert old.vision_space == new.vision_space
        assert old.consolidation_recipe == new.consolidation_recipe
