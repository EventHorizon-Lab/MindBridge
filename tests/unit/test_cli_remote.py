"""Round-trip checks for `--url` against one real `/v1` application.

`urlopen` is replaced by the FastAPI test client over a real `Memory`, so every assertion here
runs the actual route: a wrong path answers 404 and a body the route does not accept answers 422,
neither of which a recorded-request test can catch. What is asserted is that each command reaches
its route and that the document it prints is the one the same command prints locally.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from email.message import Message
from io import BytesIO
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from _feature_support import TinyEmbedder
from fastapi.testclient import TestClient

from mindbridge import Memory, cli
from mindbridge.api.app import create_app
from mindbridge.cli import EXIT_CODES, main
from mindbridge.exceptions import IdentityNotFoundError
from mindbridge.types import (
    FormationProposal,
    MemoryIntent,
    MemoryKind,
    MemoryOperation,
)

URL = "http://owner:8000"


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, object, list[object]]:
    status = main(argv)
    captured = capsys.readouterr()
    stdout = json.loads(captured.out) if captured.out.strip() else None
    stderr = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
    return status, stdout, stderr


def _serve(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Route the CLI's own transport into one test client, leaving its encoding untouched."""

    def urlopen(request: Request, *, timeout: float) -> BytesIO:
        del timeout
        response = client.request(
            request.get_method(),
            request.full_url,
            content=cast(bytes | None, request.data),
            headers=dict(request.headers),
        )
        if response.status_code >= 400:
            raise HTTPError(
                request.full_url,
                response.status_code,
                "",
                Message(),
                BytesIO(response.content),
            )
        return BytesIO(response.content)

    monkeypatch.setattr(cli, "urlopen", urlopen)


@pytest.fixture
def owner(tmp_path: Path) -> Iterator[Memory]:
    with Memory(tmp_path / "store", embedder=TinyEmbedder()) as memory:
        yield memory


@pytest.fixture
def open_owner(owner: Memory, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """An owner that opted in to both switches, so every gated route is registered."""
    app = create_app(memory=owner, identity_operations=True, embodied_operations=True)
    with TestClient(app) as client:
        _serve(monkeypatch, client)
        yield client


@pytest.fixture
def closed_owner(owner: Memory, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with TestClient(create_app(memory=owner)) as client:
        _serve(monkeypatch, client)
        yield client


def test_capture_settle_and_pending_captures_round_trip(
    open_owner: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    status, captured, _ = _run(capsys, "--url", URL, "-q", "capture", "the kettle is empty")
    assert status == 0
    memory_id = cast(dict[str, object], captured)["id"]

    status, pending, _ = _run(capsys, "--url", URL, "-q", "pending-captures", cast(str, memory_id))
    assert status == 0
    # `pending`, not the route's own `items`: one command prints one shape on both transports.
    rows = cast(list[dict[str, object]], cast(dict[str, object], pending)["pending"])
    assert [row["memory_id"] for row in rows] == [memory_id]

    status, settled, _ = _run(capsys, "--url", URL, "-q", "settle")
    assert status == 0
    assert settled == {"settled": 1}
    assert _run(capsys, "--url", URL, "-q", "pending-captures")[1] == {"pending": []}


def test_reinforce_and_traced_search_round_trip(
    open_owner: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    _status, added, _ = _run(capsys, "--url", URL, "-q", "add", "the spare key is in the toolbox")
    memory_id = cast(str, cast(dict[str, object], added)["id"])

    status, reinforced, _ = _run(capsys, "--url", URL, "-q", "reinforce", memory_id)
    assert status == 0
    assert reinforced == {"reinforced": 1}

    status, traced, _ = _run(capsys, "--url", URL, "-q", "search-with-trace", "spare key")
    assert status == 0
    # The traced operation runs only when the request carries `explain`, so a null trace here
    # means the CLI reached the plain search route with the same body.
    assert cast(dict[str, object], traced)["trace"] is not None


def test_an_operation_row_prints_the_same_keys_on_both_transports(
    owner: Memory, open_owner: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """The REST row carried no `proposal`, so one command printed two shapes.

    `export` is the routed command whose document holds log rows, and the row it prints under
    `--url` comes from the REST operation object while a local run prints the CLI's own.
    """
    observation = owner.add("Ana asked at the door")
    owner.apply(
        MemoryOperation(
            intent=MemoryIntent.CONSOLIDATE,
            evidence_ids=(observation.id,),
            proposal=FormationProposal(
                kind=MemoryKind.ENTITY,
                content="Ana is a person",
                subject="Ana",
                confidence=0.9,
            ),
            rationale="she is named in the evidence",
        )
    )

    status, bundle, _ = _run(capsys, "--url", URL, "-q", "export", "--memory-id", observation.id)

    assert status == 0
    rows = cast(list[dict[str, object]], cast(dict[str, object], bundle)["operations"])
    local = cli._operation_document(owner.operations()[0])
    assert [row["operation_id"] for row in rows] == [local["operation_id"]]
    assert set(rows[0]) == set(local)
    assert rows[0]["proposal"] == local["proposal"]


def test_a_scope_carries_the_place_and_identity_axes_to_the_route(
    open_owner: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """A body the route refuses answers 422, so reaching a real answer is the whole assertion."""
    _run(capsys, "--url", URL, "-q", "add", "the kettle is on the shelf")

    status, found, _ = _run(
        capsys,
        "--url",
        URL,
        "-q",
        "search",
        "kettle",
        "--scope",
        json.dumps({"place_id": "kitchen", "identity_id": "nobody"}),
    )

    assert status == 0
    # Nothing stored is in that room or about that person, which the local command also answers.
    assert found == {"hits": [], "trace": None}


def test_export_and_retention_round_trip(
    open_owner: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    _status, added, _ = _run(capsys, "--url", URL, "-q", "add", "the kettle is on the shelf")
    memory_id = cast(str, cast(dict[str, object], added)["id"])

    status, bundle, _ = _run(capsys, "--url", URL, "-q", "export", "--memory-id", memory_id)
    assert status == 0
    exported = cast(dict[str, object], bundle)
    assert [record["id"] for record in cast(list[dict[str, object]], exported["records"])] == [
        memory_id
    ]

    status, report, _ = _run(capsys, "--url", URL, "-q", "apply-retention", "--dry-run")
    assert status == 0
    assert cast(dict[str, object], report)["dry_run"] is True
    assert cast(dict[str, object], report)["deleted"] == 0


@pytest.mark.parametrize(
    ("command", "operands", "expected"),
    (
        # No identity exists yet, so the owner answers with the kernel's own judgement on the
        # subject the command named. That is the assertion: a mistyped path answers "route does
        # not exist" and a body the route refuses answers `validation_error`, neither of which
        # is a statement about `identity-1`.
        ("register-identity", ("identity-1", "Ana"), None),
        ("record-consent", ("identity-1", "withdrawn"), None),
        ("forget-identity", ("identity-1",), None),
        ("identity", ("identity-1",), {"identity": None}),
        ("consent", ("identity-1",), {"consent": None}),
        ("unlink-identity", ("identity-1",), {"restored_identity_id": None}),
    ),
)
def test_the_identity_commands_reach_their_route(
    command: str,
    operands: tuple[str, ...],
    expected: dict[str, object] | None,
    open_owner: TestClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status, stdout, stderr = _run(capsys, "--url", URL, "-q", command, *operands)
    if expected is None:
        assert stdout is None
        assert status == EXIT_CODES[IdentityNotFoundError.code]
        assert cast(dict[str, object], stderr[0])["code"] == IdentityNotFoundError.code
        return
    assert status == 0
    assert stdout == expected


@pytest.mark.parametrize(
    ("command", "response", "expected"),
    (
        (
            "identity",
            {
                "identity": {
                    "identity_id": "i1",
                    "name": "Ana",
                    "relationship": "neighbour",
                    "confirmed": True,
                    "evidence_ids": ["memory_1"],
                }
            },
            {"identity": {"identity_id": "i1", "name": "Ana", "relationship": "neighbour"}},
        ),
        ("register-identity", {"registered": True}, {}),
        (
            "forget-identity",
            {"erasure": {"identity_id": "i1", "alias_ids": [], "face_exemplars": 2}},
            {"identity_id": "i1", "alias_ids": [], "face_exemplars": 2},
        ),
    ),
)
def test_a_reshaped_response_prints_the_document_the_command_prints_locally(
    command: str, response: dict[str, object], expected: dict[str, object]
) -> None:
    """The three routes whose response shape is not the command's own document.

    Reaching these through a real owner needs an enrolled person, which needs a face analyzer;
    the reshaping itself is what a `--url` caller would otherwise see differ from local mode.
    """
    assert cli._REMOTE_RESULT[command](response) == expected


@pytest.mark.parametrize(
    ("command", "operands", "switch"),
    (
        ("speech", ("memory-1",), "embodied_operations"),
        ("faces", ("memory-1",), "embodied_operations"),
        ("register-identity", ("identity-1", "Ana"), "identity_operations"),
        ("export", ("--identity-id", "identity-1"), "identity_operations"),
        ("apply-retention", (), "identity_operations"),
    ),
)
def test_a_gated_route_the_owner_never_registered_names_the_switch(
    command: str,
    operands: tuple[str, ...],
    switch: str,
    closed_owner: TestClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status, stdout, stderr = _run(capsys, "--url", URL, "-q", command, *operands)
    assert status == 10
    assert stdout is None
    envelope = cast(dict[str, object], stderr[0])
    assert envelope["reason"] == "unsupported_in_remote_mode"
    assert envelope["subject"] == command
    assert switch in cast(str, envelope["message"])


def test_a_missing_memory_still_forwards_the_owners_own_envelope(
    open_owner: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real 404 from a registered route must not be reported as a disabled switch."""
    status, _stdout, stderr = _run(capsys, "--url", URL, "-q", "get", "memory-missing")
    assert status == EXIT_CODES["memory_not_found"]
    assert cast(dict[str, object], stderr[0])["code"] == "memory_not_found"
