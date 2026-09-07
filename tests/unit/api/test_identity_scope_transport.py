"""Both network transports carry the identity scope axis to the SDK unchanged.

REST and MCP translate `scope` into a `RetrievalScope` and pass it through, so a new axis is a
contract question rather than a routing one: does the payload the caller sends still reach the
predicate, and does it still narrow the answer? These drive a real `Memory` for that reason.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient
from mcp import Client
from mcp.types import CallToolResult, TextContent

from mindbridge import (
    AssetRef,
    Blob,
    EmbedTask,
    FaceAnalysis,
    FaceEmbedding,
    Memory,
    Modality,
    ModelInput,
)
from mindbridge.api.app import create_app
from mindbridge.api.mcp import build_mcp_server


class _Embedder:
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
    embedding_model = "identity-scope-transport"
    embedding_space = "identity-scope-transport:2"
    embedding_dimension = 2

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        del task
        return tuple((1.0, 0.0) for _value in inputs)

    def close(self) -> None:
        pass


class _Face:
    face_capabilities = frozenset({Modality.IMAGE})
    face_model = "identity-scope-transport-face"
    face_space = "identity-scope-transport-face:2"
    face_analysis_space = "identity-scope-transport-detector:1"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        return tuple(
            FaceAnalysis((FaceEmbedding("face-0", (0.0, 1.0), (0.1, 0.1, 0.4, 0.5)),))
            for _asset in assets
        )

    def close(self) -> None:
        pass


def _library(tmp_path: Path) -> tuple[Memory, str, str]:
    memory = Memory(
        tmp_path,
        embedder=_Embedder(),
        face_analyzer=_Face(),
        minimum_relevance=0,
    )
    seen = memory.add(Blob(b"one", "image/png", "alice.png"))
    memory.add("a kettle nobody was photographed beside")
    return memory, seen.id, memory.faces(seen.id)[0].identity_id


def test_rest_search_accepts_and_applies_an_identity_scope(tmp_path: Path) -> None:
    memory, seen_id, alice = _library(tmp_path)
    with memory, TestClient(create_app(memory=memory)) as client:
        response = client.post(
            "/v1/memories/search",
            json={"query": "kettle", "limit": 10, "scope": {"identity_id": alice}},
        )
        unscoped = client.post("/v1/memories/search", json={"query": "kettle", "limit": 10})

    assert response.status_code == 200
    assert [hit["id"] for hit in response.json()["hits"]] == [seen_id]
    assert len(unscoped.json()["hits"]) == 2


def test_rest_rejects_a_blank_identity_scope(tmp_path: Path) -> None:
    memory, _seen_id, _alice = _library(tmp_path)
    with memory, TestClient(create_app(memory=memory)) as client:
        response = client.post(
            "/v1/memories/search",
            json={"query": "kettle", "scope": {"identity_id": " "}},
        )

    assert response.status_code == 422


def test_rest_answers_a_scope_value_the_index_filter_cannot_spell(tmp_path: Path) -> None:
    """A scope value Zvec's filter grammar cannot express is a narrower answer, not a 503."""
    memory, _seen_id, _alice = _library(tmp_path)
    with memory, TestClient(create_app(memory=memory)) as client:
        response = client.post(
            "/v1/memories/search",
            json={"query": "kettle", "scope": {"identity_id": "attic\\"}},
        )

    assert response.status_code == 200
    assert response.json()["hits"] == []


async def test_mcp_search_accepts_and_applies_an_identity_scope(tmp_path: Path) -> None:
    memory, seen_id, alice = _library(tmp_path)
    with memory:
        async with Client(build_mcp_server(memory)) as client:
            scoped = await client.call_tool(
                "search_memories",
                {"query": "kettle", "limit": 10, "scope": {"identity_id": alice}},
            )
            unscoped = await client.call_tool("search_memories", {"query": "kettle", "limit": 10})

    assert [hit["id"] for hit in _hits(scoped)] == [seen_id]
    assert len(_hits(unscoped)) == 2


def _hits(result: CallToolResult) -> list[dict[str, str]]:
    content = result.content[0]
    assert isinstance(content, TextContent)
    return cast(list[dict[str, str]], json.loads(content.text)["hits"])
