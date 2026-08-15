"""Checks for lazy, capability-safe model plugin loading."""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from typing import cast

import pytest

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
