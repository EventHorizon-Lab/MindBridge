"""Checks for lazy, capability-safe model plugin loading."""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from typing import cast

import pytest

import mindbridge.models.openai as openai_models
import mindbridge.models.plugins as plugins
from mindbridge.models import GenerateRequest, GenerateResult


class _Generator:
    async def generate(self, request: GenerateRequest) -> GenerateResult:
        raise AssertionError(f"not invoked by discovery: {request}")


@dataclass(frozen=True)
class _Point:
    name: str
    factory: object

    def load(self) -> object:
        return self.factory


def test_loader_loads_only_the_selected_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []
    selected = _Generator()

    def factory(config: Mapping[str, object]) -> object:
        loaded.append(str(config["model"]))
        return selected

    _install_points(
        monkeypatch,
        _Point("unused", lambda _config: (_ for _ in ()).throw(AssertionError("loaded"))),
        _Point("selected", factory),
    )

    assert plugins.load_generator("selected", {"model": "exact"}) is selected
    assert loaded == ["exact"]


@pytest.mark.parametrize(
    "points",
    [(), (_Point("same", lambda _config: _Generator()),) * 2],
)
def test_loader_rejects_missing_or_duplicate_plugins(
    monkeypatch: pytest.MonkeyPatch,
    points: tuple[_Point, ...],
) -> None:
    _install_points(monkeypatch, *points)

    with pytest.raises(LookupError):
        plugins.load_generator("same", {})


def test_loader_rejects_the_wrong_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_points(monkeypatch, _Point("wrong", lambda _config: object()))

    with pytest.raises(TypeError, match="Generator"):
        plugins.load_generator("wrong", {})


def test_loader_rejects_non_json_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_points(monkeypatch, _Point("selected", lambda _config: _Generator()))

    with pytest.raises(ValueError, match="JSON values"):
        plugins.load_generator("selected", {"model": float("nan")})


def test_loader_rejects_non_callable_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_points(monkeypatch, _Point("broken", object()))

    with pytest.raises(TypeError, match="must be callable"):
        plugins.load_generator("broken", {})


def _install_points(monkeypatch: pytest.MonkeyPatch, *points: _Point) -> None:
    monkeypatch.setattr(
        plugins,
        "_entry_points",
        lambda _group: cast(tuple[EntryPoint, ...], points),
    )


def _embedder_config(**changes: object) -> dict[str, object]:
    config: dict[str, object] = {
        "api_key": "key",
        "endpoint": "https://embeddings.example.test/v1",
        "model_id": "omni",
        "space_id": "space",
    }
    config.update(changes)
    return config


def test_bundled_embedder_factory_accepts_a_complete_configuration() -> None:
    embedder = openai_models.create_embedder(_embedder_config(request_timeout_seconds=30))

    assert embedder.space_reference.space_id == "space"


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"unexpected": 1}, id="unknown-key"),
        pytest.param({"api_key": "   "}, id="blank-text"),
        pytest.param({"api_key": 1}, id="non-text"),
        pytest.param({"dimension": True}, id="bool-for-integer"),
        pytest.param({"dimension": 1024.0}, id="float-for-integer"),
        pytest.param({"max_retries": "2"}, id="quoted-integer"),
        pytest.param({"request_timeout_seconds": "30"}, id="quoted-number"),
    ],
)
def test_bundled_embedder_factory_rejects_a_malformed_configuration(
    changes: dict[str, object],
) -> None:
    """A plugin factory fails on the operator's mistake instead of ignoring the key."""
    with pytest.raises(ValueError):
        openai_models.create_embedder(_embedder_config(**changes))
