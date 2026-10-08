"""AML transport contracts against physically isolated local SDK stores."""

from __future__ import annotations

import base64
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindbridge import AsyncMemory, Memory, MindBridgeConfig, Modality
from mindbridge.benchmarks import eval as evaluation
from mindbridge.benchmarks import eval_leaderboard as aml
from mindbridge.benchmarks.eval_artifacts import _atomic_replace
from mindbridge.infrastructure.local._lock import DataDirectoryInUseError
from mindbridge.models.base import EmbedTask, ModelInput

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQAAAABJRU5ErkJggg=="
)
IMAGE = {
    "type": "image_url",
    "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()},
}
HEADERS = {"Authorization": "Bearer secret"}


class Embedder:
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
    embedding_model = "aml-contract"
    embedding_space = "aml-contract:2"
    embedding_dimension = 2

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in inputs)

    def close(self) -> None:
        pass


def memory_factory(path: Path) -> AsyncMemory:
    return AsyncMemory(Memory(path, embedder=Embedder()))


def payload(
    content: object = "Alice's favorite color is red.", *, user: str = "../用户"
) -> dict[str, object]:
    return {
        "request_id": "request-1",
        "user_id": user,
        "session_id": " session-1 ",
        "messages": [{"role": "user", "content": content, "timestamp": 1704067200000}],
    }


def search(
    client: TestClient, user: str = "../用户", *, top_k: int = 100
) -> list[dict[str, object]]:
    response = client.post(
        "/search",
        headers=HEADERS,
        json={
            "query": "favorite color",
            "options": ["A. red", "B. blue"],
            "user_id": user,
            "top_k": top_k,
        },
    )
    assert response.status_code == 200, response.text
    return cast(list[dict[str, object]], response.json()["data"])


def test_add_search_retry_restart_and_physical_isolation(tmp_path: Path) -> None:
    app = aml.create_app(data_root=tmp_path, memory_factory=memory_factory, api_key="secret")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert search(client) == []
        response = client.post("/add", headers=HEADERS, json=payload())
        assert response.status_code == 200, response.text
        assert response.json() == {
            "success": True,
            "request_id": "request-1",
            "user_id": "../用户",
            "session_id": " session-1 ",
        }
        assert client.post("/add", headers=HEADERS, json=payload()).json() == response.json()
        rows = search(client, top_k=1)
        assert len(rows) == 1
        assert rows[0]["content"] == "Alice's favorite color is red."
        assert rows[0]["created_at"] == "2024-01-01T00:00:00+00:00"
        assert search(client, "other") == []
        assert (
            client.post(
                "/add", headers=HEADERS, json=payload("Bob likes blue.", user="other")
            ).status_code
            == 200
        )
        assert [row["content"] for row in search(client)] == [rows[0]["content"]]
        assert [row["content"] for row in search(client, "other")] == ["Bob likes blue."]
    with TestClient(app) as client:
        assert search(client)[0]["id"] == rows[0]["id"]
        assert client.post("/add", headers=HEADERS, json=payload()).status_code == 200
        assert len(search(client)) == 1
        assert client.post("/add", headers=HEADERS, json=payload("changed")).status_code == 422
    directories = list((tmp_path / "users").iterdir())
    assert len(directories) == 2
    assert all(len(directory.name) == 64 for directory in directories)


@pytest.mark.parametrize(
    "headers", [{"Authorization": "Token secret"}, HEADERS, {"X-Api-Key": "secret"}]
)
def test_supported_authentication(tmp_path: Path, headers: dict[str, str]) -> None:
    with TestClient(
        aml.create_app(data_root=tmp_path, memory_factory=memory_factory, api_key="secret")
    ) as client:
        assert client.post("/add", headers=headers, json=payload()).status_code == 200
        assert (
            client.post(
                "/search",
                headers=headers,
                json={"query": "color", "user_id": "../用户", "top_k": 100},
            ).status_code
            == 200
        )
        assert client.post("/add", json=payload()).status_code == 401
        assert (
            client.post(
                "/search",
                headers={"X-Api-Key": "wrong"},
                json={"query": "color", "user_id": "../用户", "top_k": 100},
            ).status_code
            == 401
        )


def test_ordered_multimodal_content_and_search_query(tmp_path: Path) -> None:
    content = [{"type": "text", "text": " before "}, IMAGE, {"type": "text", "text": "after"}]
    with TestClient(
        aml.create_app(data_root=tmp_path, memory_factory=memory_factory, api_key=None)
    ) as client:
        response = client.post("/add", json=payload(content))
        assert response.status_code == 200, response.text
        response = client.post(
            "/search",
            json={
                "query": [IMAGE, {"type": "text", "text": "before"}],
                "user_id": "../用户",
                "top_k": 100,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"][0]["content"] == content


@pytest.mark.parametrize(
    "content",
    [
        " ",
        [],
        [{"type": "text", "text": ""}],
        [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,!"}}],
    ],
)
def test_invalid_content_is_rejected_before_storage(tmp_path: Path, content: object) -> None:
    with TestClient(
        aml.create_app(data_root=tmp_path, memory_factory=memory_factory, api_key="secret")
    ) as client:
        assert client.post("/add", headers=HEADERS, json=payload(content)).status_code == 422
        assert not (tmp_path / "users").exists()


def test_media_limits_and_numeric_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with TestClient(
        aml.create_app(data_root=tmp_path, memory_factory=memory_factory, api_key=None)
    ) as client:
        for timestamp in (True, 1.5, 10**30):
            document = payload()
            document["messages"] = [{"role": "user", "content": "hello", "timestamp": timestamp}]
            assert client.post("/add", json=document).status_code == 422
        for top_k in (0, 101, 1.5, True):
            assert (
                client.post(
                    "/search", json={"query": "color", "user_id": "a", "top_k": top_k}
                ).status_code
                == 422
            )
        monkeypatch.setattr(aml, "_TOTAL_IMAGE_BYTES", len(PNG))
        assert client.post("/add", json=payload([IMAGE, IMAGE])).status_code == 422
        assert client.post("/add", json=payload([IMAGE])).status_code == 200
        assert (
            client.post("/add", json={**payload([IMAGE]), "request_id": "request-2"}).status_code
            == 200
        )
        assert (
            client.post(
                "/search", json={"query": "image", "user_id": "../用户", "top_k": 100}
            ).status_code
            == 422
        )
        monkeypatch.setattr(aml, "_IMAGE_BYTES", len(PNG) - 1)
        assert client.post("/add", json=payload([IMAGE])).status_code == 422


def test_failed_acknowledgement_can_be_retried_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail = True

    def interrupted(files: Sequence[tuple[Path, bytes]]) -> None:
        nonlocal fail
        if fail and any(b'"complete":true' in content for _, content in files):
            fail = False
            raise OSError("simulated receipt failure")
        _atomic_replace(files)

    monkeypatch.setattr(aml, "_atomic_replace", interrupted)
    app = aml.create_app(data_root=tmp_path, memory_factory=memory_factory, api_key="secret")
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/add", headers=HEADERS, json=payload()).status_code == 500
        assert client.post("/add", headers=HEADERS, json=payload()).status_code == 200
        assert len(search(client)) == 1


def test_concurrent_users_and_same_user_retries(tmp_path: Path) -> None:
    app = aml.create_app(data_root=tmp_path, memory_factory=memory_factory, api_key="secret")
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=4) as workers:

        def add(user: str) -> int:
            return client.post(
                "/add", headers=HEADERS, json=payload(f"{user} likes red.", user=user)
            ).status_code

        assert list(workers.map(add, ["a", "a", "b", "b"])) == [200] * 4
        assert [row["content"] for row in search(client, "a")] == ["a likes red."]
        assert [row["content"] for row in search(client, "b")] == ["b likes red."]


def test_missing_index_rebuild_uses_stored_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = aml.create_app(data_root=tmp_path, memory_factory=memory_factory, api_key="secret")
    with TestClient(app) as client:
        assert client.post("/add", headers=HEADERS, json=payload()).status_code == 200
        directory = next((tmp_path / "users").iterdir())
        shutil.rmtree(directory / "zvec")
        embed = Embedder.embed

        def query_only(
            self: Embedder, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            assert task is EmbedTask.QUERY, "rebuilding must not embed stored documents"
            return embed(self, inputs, task)

        monkeypatch.setattr(Embedder, "embed", query_only)
        assert search(client)[0]["content"] == "Alice's favorite color is red."
        assert (directory / "zvec").is_dir()


def test_cli_loads_config_runs_server_and_closes_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    configs: list[MindBridgeConfig] = []
    closed: list[bool] = []
    configuration = tmp_path / "config.yaml"
    configuration.write_text(
        "generation:\n  provider: openai\nbenchmark:\n  run:\n    tasks: ignored\n",
        encoding="utf-8",
    )

    class Pool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            configs.append(cast(MindBridgeConfig, kwargs["memory_config"]))

        def memory(self, path: Path) -> AsyncMemory:
            return memory_factory(path)

        def close(self) -> None:
            closed.append(True)

    def run(app: object, **kwargs: object) -> None:
        assert kwargs == {"host": "127.0.0.1", "port": 8123, "access_log": False}
        with TestClient(cast(FastAPI, app)) as client:
            assert client.post("/add", headers=HEADERS, json=payload()).status_code == 200
            assert len(search(client)) == 1

    monkeypatch.setattr(evaluation, "_BackendPool", Pool)
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setenv("MINDBRIDGE_AML_API_KEY", "secret")
    assert (
        evaluation.main(
            [
                "serve",
                "--config",
                str(configuration),
                "--data-root",
                str(tmp_path / "data"),
                "--port",
                "8123",
            ]
        )
        == 0
    )
    assert configs[0].generation is None
    assert configs[0].embedding.provider == "sentence-transformers"
    assert closed == [True]


def test_second_owner_fails_and_different_roots_work(tmp_path: Path) -> None:
    first = aml.create_app(data_root=tmp_path / "a", memory_factory=memory_factory, api_key=None)
    second = aml.create_app(data_root=tmp_path / "a", memory_factory=memory_factory, api_key=None)
    other = aml.create_app(data_root=tmp_path / "b", memory_factory=memory_factory, api_key=None)
    with TestClient(first), TestClient(other) as client:
        assert client.get("/health").status_code == 200
        with pytest.raises(DataDirectoryInUseError), TestClient(second):
            pass


def test_eval_dispatch_and_server_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[Sequence[str] | None, str | None]] = []

    def serve(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
        calls.append((argv, prog))
        return 0

    monkeypatch.delenv("MINDBRIDGE_AML_API_KEY", raising=False)
    with pytest.raises(SystemExit) as error:
        aml.main(["--config", str(tmp_path / "config.yaml"), "--data-root", str(tmp_path)])
    assert error.value.code == 2
    assert "MINDBRIDGE_AML_API_KEY" in capsys.readouterr().err
    monkeypatch.setattr(aml, "main", serve)
    assert evaluation.main(["serve", "--help"], prog="mindbridge-bench eval") == 0
    assert calls == [(["--help"], "mindbridge-bench eval serve")]
