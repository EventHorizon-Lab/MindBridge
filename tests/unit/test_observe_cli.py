"""Checks for `mindbridge observe`, the command that hands MindBridge a file already on disk.

The upload dance itself is `sdk.observe_file`'s and is tested against a fake deployment in
`test_sdk_observe_file.py`. What is checked here is the part this command owns: that every flag
reaches that call rather than being parsed and dropped, and that the receipt reaches stdout in a
shape the next command can read.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge import cli, observe_cli
from mindbridge.contracts import ObservationReceipt
from mindbridge.core import MediaKind, SensorKind

RECEIPT = ObservationReceipt.model_validate(
    {
        "observation_id": "observation_01",
        "processing_job_id": "job_01",
        "evidence_ids": ["evidence_01"],
        "idempotency_key": "derived-key",
        "status": "accepted",
        "trace_id": "trace_01",
    }
)


class _FakeClient:
    """Stands in for the SDK client, recording the one call this command makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.closed = False

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_error: object) -> None:
        self.closed = True

    async def observe_file(self, path: Path, **keywords: object) -> ObservationReceipt:
        self.calls.append((path, keywords))
        return RECEIPT


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    """Replace the connection, keeping how it was built observable."""
    import mindbridge.sdk

    fake = _FakeClient()
    connections: list[dict[str, object]] = []

    def connect(**keywords: object) -> _FakeClient:
        connections.append(keywords)
        return fake

    monkeypatch.setattr(mindbridge.sdk.MindBridge, "connect", connect)
    fake.connections = connections  # type: ignore[attr-defined]
    return fake


def test_the_dispatcher_can_reach_this_command() -> None:
    """`mindbridge observe` has to resolve, or none of the rest of this file is reachable."""
    command = cli.COMMANDS[("observe",)]

    assert command.handler() is observe_cli.main


def test_a_local_file_is_observed_with_nothing_but_a_path_and_a_tenant(
    tmp_path: Path, client: _FakeClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """The user's complaint verbatim: a file on disk, and one command that takes it.

    Everything else defaults, and the defaults are the SDK's own -- passing `None` is what lets
    `observe_file` read the kind off the extension and the timestamps off the file, rather than
    this command guessing and handing down a worse answer.
    """
    path = tmp_path / "capture.mp4"
    path.write_bytes(b"frames")

    observe_cli.main([str(path), "--tenant-id", "tenant_01"])

    handed, keywords = client.calls[0]
    assert handed == path
    assert keywords == {
        "tenant_id": "tenant_01",
        "kind": None,
        "occurred_at": None,
        "ended_at": None,
        "device_id": "cli",
        "boot_id": "cli",
        "sequence": None,
        "sensor": None,
    }
    assert json.loads(capsys.readouterr().out) == {
        "observation_id": "observation_01",
        "processing_job_id": "job_01",
        "evidence_ids": ["evidence_01"],
        "status": "accepted",
    }
    assert client.closed, "the pools have to be released whether it succeeded or not"


def test_every_flag_reaches_the_call_rather_than_being_parsed_and_dropped(
    tmp_path: Path, client: _FakeClient
) -> None:
    """A flag argparse accepts and nobody forwards is worse than one that does not exist.

    The kind and the sensor arrive as their own enums rather than as the strings argparse
    collected, because `observe_file` compares the kind against the one the extension implies --
    a string would never equal it, and the suffix would be silently dropped from the object key.
    """
    path = tmp_path / "sound.bin"
    path.write_bytes(b"pcm")

    observe_cli.main(
        [
            str(path),
            "--tenant-id",
            "tenant_01",
            "--kind",
            "audio",
            "--occurred-at",
            "2026-08-26T10:00:00+00:00",
            "--ended-at",
            "2026-08-26T10:30:00+00:00",
            "--device-id",
            "recorder_01",
            "--boot-id",
            "boot_07",
            "--sequence",
            "42",
            "--sensor",
            "microphone",
            "--timeout-seconds",
            "30",
        ]
    )

    _, keywords = client.calls[0]
    assert keywords == {
        "tenant_id": "tenant_01",
        "kind": MediaKind.AUDIO,
        "occurred_at": datetime(2026, 8, 26, 10, 0, tzinfo=timezone.utc),
        "ended_at": datetime(2026, 8, 26, 10, 30, tzinfo=timezone.utc),
        "device_id": "recorder_01",
        "boot_id": "boot_07",
        "sequence": 42,
        "sensor": SensorKind.MICROPHONE,
    }
    assert client.connections[0]["timeout_seconds"] == 30.0  # type: ignore[attr-defined]


def test_the_api_key_is_read_from_the_environment_and_not_from_a_flag(
    tmp_path: Path, client: _FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A credential in argv is a credential in the process list and in shell history.

    Read through `configuration_source` rather than `os.environ` directly, so this command sees
    the key the same way every other MindBridge process does.
    """
    path = tmp_path / "capture.mp4"
    path.write_bytes(b"frames")
    monkeypatch.setenv("MINDBRIDGE_API_KEY", "secret-token")

    observe_cli.main([str(path), "--tenant-id", "tenant_01"])

    assert client.connections[0]["api_key"] == "secret-token"  # type: ignore[attr-defined]
    with pytest.raises(SystemExit):
        observe_cli.main([str(path), "--tenant-id", "tenant_01", "--api-key", "secret-token"])


def test_a_tenant_is_required_because_nothing_here_can_default_one(
    tmp_path: Path, client: _FakeClient
) -> None:
    """An observation is written under exactly one tenant, and guessing wrong is a leak."""
    path = tmp_path / "capture.mp4"
    path.write_bytes(b"frames")

    with pytest.raises(SystemExit):
        observe_cli.main([str(path)])

    assert client.calls == []
