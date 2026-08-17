"""Checks for the shared client lifecycle and prepared-media index."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from mindbridge.benchmarks.cli_common import CoreArguments, connected_memory, index_prepared
from mindbridge.sdk import MindBridge


@dataclass(frozen=True, slots=True)
class _Prepared:
    unit_id: str


@dataclass(frozen=True, slots=True)
class _Indexed:
    index: int


async def test_connected_memory_closes_the_client_when_the_run_raises() -> None:
    """A failed benchmark must not leak the connection pool it opened."""
    opened: list[MindBridge] = []
    async with connected_memory(_arguments()) as memory:
        opened.append(memory)
        assert not memory._client.is_closed
        with pytest.raises(RuntimeError):
            async with connected_memory(_arguments()) as inner:
                inner_client = inner._client
                raise RuntimeError("benchmark failed mid-run")
        assert inner_client.is_closed
    assert opened[0]._client.is_closed


async def test_connected_memory_reads_the_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded invocation never carries the credential the run used."""
    monkeypatch.setenv("MINDBRIDGE_API_KEY", "runtime-secret-000000000000000000")
    async with connected_memory(_arguments()) as memory:
        assert memory._client.headers["authorization"] == "Bearer runtime-secret-000000000000000000"


async def test_connected_memory_allows_an_unauthenticated_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MINDBRIDGE_API_KEY", raising=False)
    async with connected_memory(_arguments()) as memory:
        assert "authorization" not in memory._client.headers


def test_index_prepared_keys_every_prepared_unit() -> None:
    prepared = (_Prepared("second"), _Prepared("first"))

    indexed = index_prepared(
        ("first", "second"),
        prepared,
        key=lambda item: item.unit_id,
        label="test units",
    )

    assert indexed == {"second": prepared[0], "first": prepared[1]}


def test_index_prepared_rejects_a_run_missing_a_required_unit() -> None:
    with pytest.raises(ValueError, match="missing prepared test units: absent"):
        index_prepared(
            ("present", "absent"),
            (_Prepared("present"),),
            key=lambda item: item.unit_id,
            label="test units",
        )


def test_index_prepared_reports_integer_units_in_numeric_order() -> None:
    """String ordering would report these as "10, 100, 9" and read as a data error."""
    with pytest.raises(ValueError, match=r"missing prepared examples: 9, 10, 100$"):
        index_prepared(
            (9, 10, 100),
            (_Indexed(1),),
            key=lambda item: item.index,
            label="examples",
        )


def test_index_prepared_accepts_a_run_that_requires_nothing() -> None:
    assert index_prepared((), (_Prepared("spare"),), key=lambda i: i.unit_id, label="u") == {
        "spare": _Prepared("spare")
    }


def _arguments() -> CoreArguments:
    return CoreArguments(
        dataset_path=Path("dataset.json"),
        output_path=Path("predictions.json"),
        api_base_url="https://memory.example.test",
        code_revision="mindbridge-commit",
        deployment_config_path=Path("deployment.json"),
        run_id="run_01",
        tenant_prefix="benchmark_test",
        recall_limit=20,
        request_concurrency=4,
        request_timeout_seconds=1_800.0,
        overwrite=False,
    )
