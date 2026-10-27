"""Formation retains the observation clock independently of assertion validity."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx2 as httpx
import pytest
from _feature_support import TinyEmbedder
from openai import OpenAI

from mindbridge import (
    Blob,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryKind,
    ObservationContext,
    OpenAIModels,
)
from mindbridge.models.openai_sdk import _formation_content

OBSERVED_AT = datetime(2026, 3, 2, 12, tzinfo=timezone.utc)


class RecordingFormer:
    formation_capabilities = TinyEmbedder.embedding_capabilities
    formation_model = "formation-clock-fixture"
    formation_space = "formation-clock-fixture:v1"

    def __init__(self) -> None:
        self.inputs: list[FormationInput] = []

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        self.inputs.extend(inputs)
        return tuple(() for _ in inputs)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("has_event_time", [False, True])
def test_sdk_passes_source_times_to_former(tmp_path: Path, has_event_time: bool) -> None:
    former = RecordingFormer()
    occurred_at = OBSERVED_AT if has_event_time else None
    occurred_end = OBSERVED_AT + timedelta(hours=1) if has_event_time else None
    with Memory(tmp_path, embedder=TinyEmbedder(), former=former) as memory:
        source = memory.add(
            "I moved to Paris yesterday", occurred_at=occurred_at, occurred_end=occurred_end
        )

    (target,) = former.inputs
    assert getattr(target, "created_at", None) == source.created_at
    assert getattr(target, "occurred_at", None) == occurred_at
    assert getattr(target, "occurred_end", None) == occurred_end
    assert target.context.valid_from is None
    assert target.history == ()


@pytest.mark.parametrize("with_media", [False, True])
def test_formation_payload_keeps_observation_times_separate_from_validity(
    tmp_path: Path, with_media: bool
) -> None:
    former = RecordingFormer()
    valid_from = datetime(2026, 2, 1, tzinfo=timezone.utc)
    occurred_end = OBSERVED_AT + timedelta(hours=1)
    with Memory(tmp_path, embedder=TinyEmbedder(), former=former) as memory:
        source = memory.add(
            ("The shop has been open since February", Blob(b"image", "image/png"))
            if with_media
            else "The shop has been open since February",
            occurred_at=OBSERVED_AT,
            occurred_end=occurred_end,
            context=ObservationContext(valid_from=valid_from),
        )
        content = _formation_content(former.inputs)

    payload = (
        json.loads(content)["observations"][0]
        if isinstance(content, str)
        else json.loads(str(content[0]["text"]))["observation"]
    )
    assert payload.get("occurred_at") == "2026-03-02T12:00:00+00:00"
    assert payload.get("occurred_end") == "2026-03-02T13:00:00+00:00"
    assert payload.get("created_at") == source.created_at.isoformat()
    assert payload["context"]["valid_from"] == "2026-02-01T00:00:00+00:00"


def test_resolved_yesterday_survives_formation_persistence(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        observation = json.loads(payload["messages"][1]["content"])["observations"][0]
        assert observation["context"]["valid_from"] is None
        assert "read_only_history" not in observation
        # This deterministic provider checks the transport and persistence path, not model quality.
        proposal: dict[str, object] = {
            "kind": "event",
            "content": "The user moved to Paris.",
            "confidence": 0.9,
            "evidence_observation_ids": ["observation_0"],
        }
        if observation.get("occurred_at") == "2026-03-02T12:00:00+00:00":
            proposal["valid_from"] = "2026-03-01T00:00:00Z"
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
                                        {"observation_id": "observation_0", "proposals": [proposal]}
                                    ]
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as transport,
        OpenAI(api_key="fixture", http_client=transport, max_retries=0) as client,
    ):
        former = OpenAIModels(client, generation_model="formation-clock-fixture")
        with Memory(tmp_path, embedder=TinyEmbedder(), former=former) as memory:
            source = memory.add("I moved to Paris yesterday", occurred_at=OBSERVED_AT)
            event = next(
                record
                for record in memory.list().items
                if record.context is not None and record.context.kind is MemoryKind.EVENT
            )

    assert event.context is not None
    assert event.context.evidence_ids == (source.id,)
    assert event.context.valid_from == datetime(2026, 3, 1, tzinfo=timezone.utc)
