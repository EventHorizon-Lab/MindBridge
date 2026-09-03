"""Contract tests for the product command line.

The interesting assertions here are the mechanical ones: the command set is derived from `Memory`,
the exit table is derived from the exception hierarchy, and every JSON document and every decoded
content part is compared against the REST models rather than against a hand-written expectation.
Those are what stop the CLI drifting away from the surfaces it shares a vocabulary with.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import inspect
import json
import re
import sys
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from email.message import Message
from pathlib import Path
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest
from pydantic import TypeAdapter

from mindbridge import Memory, recipes
from mindbridge.api import app as rest
from mindbridge.api import content as rest_content
from mindbridge.api.errors import ErrorEnvelope
from mindbridge.cli import COMMANDS, EXIT_CODES, OPERATIONS, main
from mindbridge.exceptions import MindBridgeError, ValidationError
from mindbridge.memory import declared_capabilities
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.models.funasr import DEFAULT_FUNASR_MODEL_ID
from mindbridge.models.jina import DEFAULT_JINA_MODEL_ID, DEFAULT_JINA_REVISION
from mindbridge.types import (
    AbstentionReason,
    AnswerResult,
    AssetRef,
    MemoryRecord,
    MemoryType,
    Modality,
    Page,
    SearchHit,
)

# The REST content union itself, so the two decoders are compared over one schema.
_REST_CONTENT: TypeAdapter[rest_content.Content] = TypeAdapter(rest_content.Content)
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
WAV = base64.b64encode(b"RIFF____WAVE").decode("ascii")
APP_SOURCE = '''
"""A minimal application composition, the way `--app` expects to find one."""

from mindbridge import Memory
from mindbridge.models.base import EmbedTask
from mindbridge.types import Modality

DATA_DIR = {data_dir!r}


class Embedder:
    embedding_capabilities = frozenset({{Modality.TEXT}})
    embedding_model = "cli-test-embedding"
    embedding_space = "cli-test-embedding:2:v1"
    embedding_dimension = 2

    def embed(self, inputs, task=EmbedTask.DOCUMENT):
        return tuple(
            (1.0, 0.0) if "red" in value.text.casefold() else (0.0, 1.0) for value in inputs
        )

    def close(self):
        return None


def build_memory():
    return Memory(DATA_DIR, embedder=Embedder())


not_a_memory = 7
'''


class _Embedder:
    """The same backend `APP_SOURCE` composes, for the tests that call `Memory` directly."""

    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "cli-test-embedding"
    embedding_space = "cli-test-embedding:2:v1"
    embedding_dimension = 2

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (1.0, 0.0) if "red" in value.text.casefold() else (0.0, 1.0) for value in inputs
        )

    def close(self) -> None:
        return None


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Write an importable application module and return its `MODULE:ATTR` spec."""
    module = tmp_path / "cli_test_application.py"
    module.write_text(APP_SOURCE.format(data_dir=str(tmp_path / "store")), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    monkeypatch.delitem(sys.modules, "cli_test_application", raising=False)
    return "cli_test_application:build_memory"


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, object, list[object]]:
    status = main(argv)
    captured = capsys.readouterr()
    stdout = json.loads(captured.out) if captured.out.strip() else None
    stderr = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
    return status, stdout, stderr


# ---------------------------------------------------------------------------------------------
# Surface derivation


def test_commands_are_the_sdk_operations_kebab_cased() -> None:
    published = {
        name
        for name, value in inspect.getmembers(Memory, inspect.isfunction)
        if not name.startswith("_") and name != "close"
    }
    assert set(OPERATIONS) == published
    assert set(COMMANDS) == {name.replace("_", "-") for name in published} | {"doctor"}


def test_every_public_error_code_has_a_stable_exit_status() -> None:
    def codes(root: type[MindBridgeError]) -> set[str]:
        found = {root.code}
        for subclass in root.__subclasses__():
            found |= codes(subclass)
        return found

    assert codes(MindBridgeError) - {MindBridgeError.code} <= set(EXIT_CODES)
    assert len(set(EXIT_CODES.values())) == len(EXIT_CODES)


def test_error_envelope_matches_the_rest_envelope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status, stdout, stderr = _run(capsys, "add", "anything")
    assert status == EXIT_CODES["configuration_error"] == 10
    assert stdout is None
    envelope = cast(dict[str, object], stderr[0])
    assert set(envelope) == set(ErrorEnvelope.model_fields)
    assert re.fullmatch(r"trace_[0-9a-f]{32}", cast(str, envelope["trace_id"]))
    message = cast(str, envelope["message"])
    assert all(flag in message for flag in ("--app", "--embedder", "--url"))


# ---------------------------------------------------------------------------------------------
# Output parity with the REST response models


def _record(**overrides: object) -> MemoryRecord:
    values: dict[str, object] = {
        "id": "memory-1",
        "content": "the spare key is in the blue toolbox",
        "created_at": datetime(2026, 1, 2, 3, 4, 5, 600_000, tzinfo=timezone.utc),
        "occurred_at": datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        "occurred_end": datetime(2026, 1, 1, 1, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        "metadata": {"source": "inspection", "count": 3},
        "modality": Modality.IMAGE,
        "memory_type": MemoryType.EPISODIC,
        "assets": (
            AssetRef(
                id="asset-1",
                modality=Modality.IMAGE,
                media_type="image/png",
                size_bytes=8,
                sha256="a" * 64,
                name="panel.png",
                path=Path("/srv/assets/panel.png"),
            ),
        ),
    }
    values.update(overrides)
    return MemoryRecord(**values)  # type: ignore[arg-type]


def test_memory_documents_equal_the_rest_response_models() -> None:
    from mindbridge.cli import _ask, _memory_document

    record = _record()
    assert _memory_document(record) == rest.MemoryResponse.model_validate(record).model_dump(
        mode="json"
    )
    hit = SearchHit(
        id=record.id,
        content=record.content,
        score=0.75,
        created_at=record.created_at,
        occurred_at=record.occurred_at,
        occurred_end=record.occurred_end,
        metadata=record.metadata,
        assets=record.assets,
        modality=record.modality,
        memory_type=record.memory_type,
    )
    assert _memory_document(hit) == rest.SearchHitResponse.model_validate(hit).model_dump(
        mode="json"
    )
    page = Page(items=(record,), next_cursor="djEiXQ")
    assert {
        "items": [_memory_document(item) for item in page.items],
        "next_cursor": page.next_cursor,
    } == rest.PageResponse.model_validate(page).model_dump(mode="json")
    assert rest.DeleteResponse(deleted=True).model_dump(mode="json") == {"deleted": True}

    answer = AnswerResult(
        "unknown",
        hits=(hit,),
        abstained=True,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
    )

    class AnsweringMemory:
        def ask(self, *_args: object, **_kwargs: object) -> AnswerResult:
            return answer

    document = _ask(
        cast(Memory, AnsweringMemory()),
        argparse.Namespace(
            content=["question"],
            content_json=None,
            limit=5,
            memory_type=None,
            reference_at=None,
        ),
    )
    assert document == rest.AnswerResponse.model_validate(answer).model_dump(mode="json")


# ---------------------------------------------------------------------------------------------
# Input parity with the REST discriminated part union


PART_CASES: tuple[list[dict[str, object]], ...] = (
    [{"type": "input_text", "text": "inspection evidence"}],
    [{"type": "input_image", "image_url": f"data:image/png;base64,{PNG}"}],
    [{"type": "input_image", "file_id": "asset-1"}],
    [{"type": "input_file", "file_url": f"data:audio/wav;base64,{WAV}", "filename": "note.wav"}],
    [{"type": "input_file", "file_data": PNG, "media_type": "image/png", "filename": "p.png"}],
    [{"type": "input_file", "file_id": "asset-1"}],
    [{"type": "input_file", "file_id": "asset-1", "media_type": "image/*"}],
    [{"type": "input_file", "file_id": "asset-1", "media_type": "image/png"}],
    [
        {"type": "input_text", "text": "inspection evidence"},
        {"type": "input_image", "image_url": f"data:image/png;base64,{PNG}"},
        {"type": "input_file", "file_url": f"data:audio/wav;base64,{WAV}"},
    ],
)


@pytest.mark.parametrize("parts", PART_CASES, ids=range(len(PART_CASES)))
def test_parts_decode_exactly_as_rest_decodes_them(parts: list[dict[str, object]]) -> None:
    from mindbridge.cli import _parts_input

    assert _parts_input(parts) == rest_content.content_input(_REST_CONTENT.validate_python(parts))


@pytest.mark.parametrize(
    "parts",
    (
        [{"type": "input_text", "text": "a", "extra": 1}],
        [{"type": "input_image"}],
        [{"type": "input_image", "image_url": "https://example.invalid/a.png"}],
        [{"type": "input_image", "image_url": f"data:audio/wav;base64,{WAV}"}],
        [
            {
                "type": "input_file",
                "file_url": f"data:audio/wav;base64,{WAV}",
                "media_type": "image/png",
            }
        ],
        [{"type": "input_file", "file_data": PNG}],
        [{"type": "input_file", "file_data": PNG, "media_type": "image/*"}],
        [{"type": "input_audio", "text": "a"}],
        [],
    ),
)
def test_invalid_parts_are_rejected_by_both_surfaces(parts: list[dict[str, object]]) -> None:
    from mindbridge.cli import _parts_input

    with pytest.raises(ValidationError):
        _parts_input(parts)
    with pytest.raises(ValueError):
        rest_content.content_input(_REST_CONTENT.validate_python(parts))


def test_the_cli_only_path_part_reaches_memory_as_a_path(tmp_path: Path) -> None:
    from mindbridge.cli import _parts_input

    media = tmp_path / "panel.png"
    assert _parts_input([{"type": "input_file", "path": str(media)}]) == (media,)
    with pytest.raises(ValueError):
        _REST_CONTENT.validate_python([{"type": "input_file", "path": str(media)}])


# ---------------------------------------------------------------------------------------------
# End-to-end through an application composition


def test_add_search_get_list_and_delete_round_trip(
    app: str, capsys: pytest.CaptureFixture[str]
) -> None:
    status, added, stderr = _run(capsys, "--app", app, "add", "a red wrench on the bench")
    assert status == 0
    record = cast(dict[str, object], added)
    assert record["content"] == "a red wrench on the bench"
    assert cast(dict[str, object], stderr[0])["source"] == f"--app {app}"

    status, replayed, _ = _run(capsys, "--app", app, "-q", "add", "a red wrench on the bench")
    assert status == 0
    assert cast(dict[str, object], replayed)["id"] == record["id"]

    status, found, _ = _run(capsys, "--app", app, "-q", "search", "red wrench")
    assert status == 0
    hits = cast(dict[str, list[dict[str, object]]], found)["hits"]
    assert [hit["id"] for hit in hits] == [record["id"]]

    status, traced, _ = _run(
        capsys,
        "--app",
        app,
        "-q",
        "search-with-trace",
        "red wrench",
    )
    assert status == 0
    traced_document = cast(dict[str, object], traced)
    assert cast(list[dict[str, object]], traced_document["hits"])[0]["id"] == record["id"]
    trace = cast(dict[str, object], traced_document["trace"])
    candidate = cast(list[dict[str, object]], trace["candidates"])[0]
    assert candidate["memory_id"] == record["id"]
    assert {"lexical_relevance", "lexical_rerank_bonus", "gate_relevance"} <= candidate.keys()

    status, page, _ = _run(capsys, "--app", app, "-q", "list", "--limit", "1")
    assert status == 0
    assert cast(dict[str, object], page)["next_cursor"] is None

    status, one, _ = _run(capsys, "--app", app, "-q", "get", cast(str, record["id"]))
    assert status == 0 and one == record

    status, removed, _ = _run(capsys, "--app", app, "-q", "delete", cast(str, record["id"]))
    assert status == 0 and removed == {"deleted": True}
    status, missing, envelope = _run(capsys, "--app", app, "-q", "get", cast(str, record["id"]))
    assert status == EXIT_CODES["memory_not_found"] == 4
    assert missing is None
    assert cast(dict[str, object], envelope[0])["code"] == "memory_not_found"


def test_typed_context_and_scope_round_trip(app: str, capsys: pytest.CaptureFixture[str]) -> None:
    spatial = {
        "frame_id": "room",
        "anchor": "subject",
        "x": 1.0,
        "y": 2.0,
        "position_uncertainty_m": 0.25,
    }
    context = {
        "basis": "user_statement",
        "source_id": "turn-7",
        "confidence": 0.9,
        "valid_from": "2026-08-01T00:00:00Z",
        "spatial": spatial,
    }
    status, added, _ = _run(
        capsys,
        "--app",
        app,
        "-q",
        "add",
        "a red wrench on the bench",
        "--context",
        json.dumps(context),
    )
    assert status == 0
    stored_context = cast(dict[str, object], cast(dict[str, object], added)["context"])
    assert stored_context["source_id"] == "turn-7"
    assert stored_context["spatial"] == {**spatial, "z": 0.0, "orientation_xyzw": None}

    scope = {
        "valid_at": "2026-08-02T00:00:00Z",
        "near": {**spatial, "position_uncertainty_m": 0.0},
        "radius_m": 0.5,
    }
    status, found, _ = _run(
        capsys,
        "--app",
        app,
        "-q",
        "search",
        "red wrench",
        "--scope",
        json.dumps(scope),
    )
    assert status == 0
    assert len(cast(dict[str, list[object]], found)["hits"]) == 1


def test_quiet_suppresses_only_the_banner(app: str, capsys: pytest.CaptureFixture[str]) -> None:
    _status, quiet_stdout, quiet_stderr = _run(capsys, "--app", app, "-q", "add", "a red wrench")
    assert quiet_stderr == []
    _status, loud_stdout, loud_stderr = _run(capsys, "--app", app, "add", "a red wrench")
    assert loud_stdout == quiet_stdout
    assert [cast(dict[str, object], line)["source"] for line in loud_stderr] == [f"--app {app}"]


def test_ordered_atoms_and_stdin_reach_memory_in_order(
    app: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mindbridge import cli

    captured: list[object] = []

    def record(_memory: Memory, arguments: object) -> dict[str, object]:
        captured.append(cli._content_input(cast("argparse.Namespace", arguments)))
        return {}

    monkeypatch.setitem(cli._LOCAL, "add", record)
    monkeypatch.setattr("sys.stdin", _Stdin("piped text"))
    media = tmp_path / "panel.png"
    _run(capsys, "--app", app, "-q", "add", "before", f"@{media}", "@@literal", "-")
    assert captured == [("before", media, "@literal", "piped text")]


def test_add_many_reads_jsonl_with_per_item_time_and_metadata(
    app: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    source = tmp_path / "batch.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps({"content": "a red wrench", "metadata": {"n": 1}}),
                json.dumps(
                    {
                        "content": "a blue toolbox",
                        "occurred_at": "2026-01-01T00:00:00Z",
                        "context": {"source_id": "camera-1", "confidence": 0.8},
                    }
                ),
            )
        ),
        encoding="utf-8",
    )
    status, document, _ = _run(capsys, "--app", app, "-q", "add-many", f"@{source}")
    assert status == 0
    memories = cast(dict[str, list[dict[str, object]]], document)["memories"]
    assert [item["metadata"] for item in memories] == [{"n": 1}, {}]
    assert memories[1]["occurred_at"] == "2026-01-01T00:00:00Z"
    assert cast(dict[str, object], memories[1]["context"])["source_id"] == "camera-1"


def test_add_stream_commits_before_reading_a_later_invalid_line(
    app: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    source = tmp_path / "stream.jsonl"
    source.write_text(
        f"{json.dumps({'content': 'a red wrench', 'metadata': {'sequence': 0}})}\nnot-json\n",
        encoding="utf-8",
    )

    status, document, stderr = _run(
        capsys,
        "--app",
        app,
        "-q",
        "add-stream",
        f"@{source}",
        "--memory-type",
        "episodic",
    )

    assert status == EXIT_CODES[ValidationError.code]
    assert document is None
    assert "line 2" in cast(dict[str, str], stderr[0])["message"]
    assert cast(dict[str, object], stderr[0])["subject"] == "contents[1]"
    with Memory(tmp_path / "store", embedder=_Embedder()) as memory:
        records = memory.list().items
    assert [(record.content, record.metadata, record.memory_type) for record in records] == [
        ("a red wrench", {"sequence": 0}, MemoryType.EPISODIC)
    ]


def test_a_second_owner_of_the_directory_exits_nine(
    app: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with Memory(tmp_path / "store", embedder=_Embedder()):
        status, stdout, stderr = _run(capsys, "--app", app, "-q", "add", "a red wrench")
    assert status == 9
    assert stdout is None
    envelope = cast(dict[str, object], stderr[0])
    assert envelope["code"] == "storage_error"
    assert envelope["reason"] == "data_dir_in_use"
    assert envelope["retryable"] is True
    assert envelope["subject"] == str(tmp_path / "store")


def test_an_application_target_that_is_not_a_memory_exits_ten(
    app: str, capsys: pytest.CaptureFixture[str]
) -> None:
    status, _stdout, stderr = _run(
        capsys, "--app", "cli_test_application:not_a_memory", "-q", "list"
    )
    assert status == 10
    assert cast(dict[str, object], stderr[0])["reason"] == "app_invalid"


@pytest.mark.parametrize(
    ("option", "value"),
    (("--answerer", "openai"), ("--former", "openai"), ("--timeout", "1")),
)
def test_composition_specific_options_are_refused_by_other_compositions(
    option: str,
    value: str,
    app: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status, _stdout, stderr = _run(capsys, "--app", app, "-q", option, value, "list")
    assert status == 10
    envelope = cast(dict[str, object], stderr[0])
    assert envelope["reason"] == "option_not_applicable"
    assert envelope["subject"] == option


def test_explain_reports_the_composition_and_executes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "store"
    status, document, _ = _run(
        capsys, "--embedder", "jina-omni", "--data-dir", str(store), "--explain", "-q", "reindex"
    )
    assert status == 0
    composition = cast(dict[str, object], document)
    assert composition["source"] == "--embedder jina-omni"
    assert cast(dict[str, object], composition["embedder"])["models"] == {
        "embedder": DEFAULT_JINA_MODEL_ID
    }
    assert not store.exists()


# ---------------------------------------------------------------------------------------------
# Recipes


def test_recipes_pin_identity_to_the_constants_in_the_source() -> None:
    assert recipes.names() == ("funasr", "jina-omni", "openai")
    jina = recipes.describe("jina-omni")
    assert jina["models"] == {"embedder": DEFAULT_JINA_MODEL_ID}
    assert jina["revision"] == DEFAULT_JINA_REVISION
    assert jina["license"] == "CC BY-NC 4.0"
    assert recipes.describe("funasr")["models"] == {"transcriber": DEFAULT_FUNASR_MODEL_ID}
    assert recipes.describe("openai:gpt-5-mini")["models"] == {
        "embedder": "gpt-5-mini",
        "answerer": "gpt-5-mini",
        "former": "gpt-5-mini",
        "consolidator": "gpt-5-mini",
        "transcriber": "gpt-5-mini",
    }
    assert "OPENAI_API_KEY" in cast(str, recipes.describe("openai")["credential"])


def test_the_former_flag_reaches_the_memory_it_composes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--former` is only wired if the constructed backend arrives as `Memory(former=...)`.

    Asserting the `--explain` document instead would stay green while `_open_memory` dropped the
    slot on the floor, which is the failure this flag was added to prevent.
    """
    from mindbridge import cli

    sentinel = object()
    captured: dict[str, object] = {}
    # Parse before patching: `_parser` derives its defaults from `Memory`'s own signature.
    arguments = cli._parser().parse_args(
        ["--data-dir", str(tmp_path), "--embedder", "openai", "--former", "openai", "list"]
    )
    monkeypatch.setattr(recipes, "embedder", lambda name, **kw: object())
    monkeypatch.setattr(recipes, "former", lambda name, **kw: sentinel)
    monkeypatch.setattr(cli, "Memory", lambda *args, **kwargs: captured.update(kwargs))

    cli._open_memory(arguments)

    assert captured["former"] is sentinel


def test_the_consolidator_flag_reaches_the_memory_it_composes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this the shipped `mindbridge consolidate` command needed `--app` to work at all."""
    from mindbridge import cli

    sentinel = object()
    captured: dict[str, object] = {}
    arguments = cli._parser().parse_args(
        ["--data-dir", str(tmp_path), "--embedder", "openai", "--consolidator", "openai", "list"]
    )
    monkeypatch.setattr(recipes, "embedder", lambda name, **kw: object())
    monkeypatch.setattr(recipes, "consolidator", lambda name, **kw: sentinel)
    monkeypatch.setattr(cli, "Memory", lambda *args, **kwargs: captured.update(kwargs))

    cli._open_memory(arguments)

    assert captured["consolidator"] is sentinel


def test_every_recipe_slot_has_a_command_line_flag() -> None:
    """The four literals this used to need are now one list; keep the flag set derived from it."""
    from mindbridge import cli

    flags = {action.dest for action in cli._parser()._actions}
    assert set(cli._SLOTS) <= flags


def test_recipes_return_an_object_the_caller_owns() -> None:
    embedder = recipes.embedder("jina-omni")
    try:
        assert embedder.embedding_model == DEFAULT_JINA_MODEL_ID
        assert embedder.embedding_dimension == 1024
    finally:
        embedder.close()


class _SdkClient:
    """Stands in for the SDK client `recipes` constructs, which needs a credential to build."""

    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1


@pytest.mark.parametrize("slot", ("embedder", "answerer", "former", "transcriber"))
def test_a_recipe_closes_the_sdk_client_it_constructed(
    slot: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clients: list[_SdkClient] = []

    def build() -> _SdkClient:
        clients.append(_SdkClient())
        return clients[-1]

    monkeypatch.setattr(recipes, "_openai_client", build)
    backend = getattr(recipes, slot)("openai")
    assert [client.closed for client in clients] == [False]

    backend.close()
    backend.close()

    # `OpenAIModels.close()` leaves a caller-supplied client open; a recipe-built one is the
    # recipe's, so the connection pool goes when the returned backend does.
    assert [client.closed for client in clients] == [True]
    assert [client.close_calls for client in clients] == [1]


@pytest.mark.parametrize(
    "name",
    ("unknown", "jina-omni:something", "openai:", "funasr:x"),
)
def test_unknown_or_misused_recipe_names_are_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        recipes.describe(name)


def test_a_recipe_cannot_fill_a_slot_it_does_not_serve() -> None:
    with pytest.raises(ValidationError):
        recipes.answerer("jina-omni")
    with pytest.raises(ValidationError):
        recipes.embedder("funasr")


def test_doctor_reports_a_failed_loader_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(name: str, *, load: bool = False) -> object:
        raise ImportError("no module named torchaudio", name="torchaudio")

    monkeypatch.setattr(recipes, "transcriber", refuse)
    store = tmp_path / "store"
    status, document, _ = _run(
        capsys,
        "--embedder",
        "openai",
        "--transcriber",
        "funasr",
        "--data-dir",
        str(store),
        "-q",
        "doctor",
    )
    report = cast(dict[str, object], document)
    composition = cast(dict[str, object], report["composition"])
    transcriber = cast(dict[str, object], composition["transcriber"])
    assert transcriber["loader"] == "failed"
    assert transcriber["reason"] == "missing_dependency"
    assert transcriber["detail"] == "torchaudio"
    assert transcriber["probe"] == "import"
    assert report["data_dir_state"] == "absent"
    assert not store.exists()
    assert status == 0


def test_doctor_publishes_the_capability_document_without_opening_a_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`/healthz`, the MCP greeting, and `doctor` must render one document, not three views."""
    monkeypatch.setattr(recipes, "embedder", lambda name, *, load=False: _Embedder())
    store = tmp_path / "store"
    _status, document, _ = _run(
        capsys, "--embedder", "jina-omni", "--data-dir", str(store), "-q", "doctor"
    )
    report = cast(dict[str, object], document)

    assert report["capabilities"] == declared_capabilities(embedder=_Embedder()).document()
    assert cast(dict[str, object], report["capabilities"])["operations"] == []
    # Declared by the backend, so the summary neither opens nor creates the data directory.
    assert report["data_dir_state"] == "absent"
    assert not store.exists()


def test_doctor_declares_no_capabilities_for_an_application_it_must_not_call(
    app: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _status, document, _ = _run(capsys, "--app", app, "-q", "doctor")

    assert cast(dict[str, object], document)["capabilities"] is None


def test_doctor_sees_a_directory_another_process_owns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "store"
    with Memory(store, embedder=_Embedder()):
        _status, document, _ = _run(
            capsys, "--embedder", "jina-omni", "--data-dir", str(store), "-q", "--explain", "doctor"
        )
        assert cast(dict[str, object], document)["source"] == "--embedder jina-omni"
        _status, report, _ = _run(
            capsys, "--url", "http://127.0.0.1:0", "-q", "--explain", "doctor"
        )
    from mindbridge.cli import _data_dir_state

    assert _data_dir_state(store) == "free"
    assert cast(dict[str, object], report)["url"] == "http://127.0.0.1:0"
    assert cast(dict[str, object], report)["timeout_seconds"] == 30.0


# ---------------------------------------------------------------------------------------------
# Remote mode


class _Stdin:
    def __init__(self, text: str) -> None:
        self._text = text

    def isatty(self) -> bool:
        return False

    def read(self) -> str:
        return self._text


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_error: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[str, str, object]]]:
    from mindbridge import cli

    seen: list[tuple[str, str, object]] = []

    def fake(request: Request, *, timeout: float) -> _Response:
        assert timeout == 30.0
        data = request.data
        body = None if data is None else json.loads(cast(bytes, data))
        seen.append((request.get_method(), request.full_url, body))
        return _Response(b'{"hits": []}')

    monkeypatch.setattr(cli, "urlopen", fake)
    yield seen


def test_remote_mode_posts_to_v1_and_echoes_the_body(
    calls: list[tuple[str, str, object]], capsys: pytest.CaptureFixture[str]
) -> None:
    status, document, _ = _run(
        capsys,
        "--url",
        "http://owner:8000/",
        "-q",
        "search",
        "red wrench",
        "--limit",
        "3",
        "--occurred-from",
        "2026-08-27T00:00:00Z",
        "--occurred-until",
        "2026-08-28T00:00:00Z",
        "--scope",
        json.dumps({"valid_at": "2026-08-27T12:00:00Z"}),
    )
    assert status == 0
    assert document == {"hits": []}
    assert calls == [
        (
            "POST",
            "http://owner:8000/v1/memories/search",
            {
                "query": "red wrench",
                "limit": 3,
                "occurred_from": "2026-08-27T00:00:00Z",
                "occurred_until": "2026-08-28T00:00:00Z",
                "scope": {
                    "valid_at": "2026-08-27T12:00:00Z",
                    "known_at": None,
                    "near": None,
                    "radius_m": None,
                },
            },
        )
    ]


def test_remote_mode_serializes_typed_observation_context(
    calls: list[tuple[str, str, object]], capsys: pytest.CaptureFixture[str]
) -> None:
    context = {
        "basis": "observation",
        "source_id": "microphone-2",
        "confidence": 0.7,
        "valid_from": "2026-08-27T00:00:00Z",
    }
    status, _document, _ = _run(
        capsys,
        "--url",
        "http://owner:8000",
        "-q",
        "add",
        "door closed",
        "--context",
        json.dumps(context),
    )
    assert status == 0
    body = cast(dict[str, object], calls[0][2])
    assert body["context"] == {
        **context,
        "valid_until": None,
        "spatial": None,
    }


def test_remote_mode_compiles_over_v1(
    calls: list[tuple[str, str, object]], capsys: pytest.CaptureFixture[str]
) -> None:
    _run(
        capsys,
        "--url",
        "http://owner:8000",
        "-q",
        "compile",
        "what should I bring",
        "--max-items",
        "8",
        "--memory-type",
        "episodic",
        "--freshness-seconds",
        "3600",
        "--max-latency-ms",
        "250",
    )

    assert calls == [
        (
            "POST",
            "http://owner:8000/v1/context",
            {
                "goal": "what should I bring",
                "budget": {
                    "max_chars": 16000,
                    "max_items": 8,
                    "memory_types": ["episodic"],
                    "min_confidence": 0.0,
                    "freshness_seconds": 3600.0,
                    "max_latency_ms": 250,
                },
            },
        ),
    ]


def test_remote_mode_passes_the_cursor_through_unparsed(
    calls: list[tuple[str, str, object]], capsys: pytest.CaptureFixture[str]
) -> None:
    _run(capsys, "--url", "http://owner:8000", "-q", "list", "--cursor", "WyJ2MSIsIngiXQ")
    assert calls[0][1] == "http://owner:8000/v1/memories?limit=100&cursor=WyJ2MSIsIngiXQ"


@pytest.mark.parametrize(
    ("command", "operands"),
    (
        ("add-stream", ("[]",)),
        ("speech", ("memory-1",)),
        ("faces", ("memory-1",)),
        ("register-speaker", ("speaker-1", "Ana")),
        ("register-identity", ("identity-1", "Ana")),
        ("reinforce", ("memory-1",)),
        ("consolidate", ("why",)),
        ("forget", ("memory-1",)),
        ("rollback", ("1",)),
        ("operations", ()),
        ("reindex", ()),
        ("optimize", ()),
    ),
)
def test_operations_without_a_rest_route_exit_ten_remotely(
    command: str, operands: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    status, stdout, stderr = _run(capsys, "--url", "http://owner:8000", "-q", command, *operands)
    assert status == 10
    assert stdout is None
    envelope = cast(dict[str, object], stderr[0])
    assert envelope["reason"] == "unsupported_in_remote_mode"
    assert envelope["subject"] == command


def test_a_local_path_is_never_sent_to_a_remote_owner(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    status, _stdout, stderr = _run(
        capsys, "--url", "http://owner:8000", "-q", "add", f"@{tmp_path / 'panel.png'}"
    )
    assert status == 10
    assert cast(dict[str, object], stderr[0])["reason"] == "unsupported_in_remote_mode"
    status, _stdout, stderr = _run(
        capsys,
        "--url",
        "http://owner:8000",
        "-q",
        "add",
        "--content-json",
        json.dumps([{"type": "input_file", "path": "/srv/panel.png"}]),
    )
    assert status == 10
    assert cast(dict[str, object], stderr[0])["subject"] == "input_file.path"


def test_a_batched_local_path_is_never_sent_to_a_remote_owner(
    calls: list[tuple[str, str, object]],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    source = tmp_path / "batch.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps({"content": "a red wrench"}),
                json.dumps({"content": [{"type": "input_file", "path": "/srv/media/panel.png"}]}),
            )
        ),
        encoding="utf-8",
    )
    status, stdout, stderr = _run(
        capsys, "--url", "http://owner:8000", "-q", "add-many", f"@{source}"
    )
    assert status == 10
    assert stdout is None
    envelope = cast(dict[str, object], stderr[0])
    assert envelope["reason"] == "unsupported_in_remote_mode"
    assert envelope["subject"] == "input_file.path"
    # The refusal happens before the request, so no local path reached the owner.
    assert calls == []


@pytest.mark.parametrize("command", ("add", "search", "ask"))
def test_two_sources_for_one_operand_are_refused_before_anything_runs(
    command: str,
    app: str,
    calls: list[tuple[str, str, object]],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    parts = json.dumps([{"type": "input_text", "text": "a blue toolbox"}])
    for composition in (("--app", app), ("--url", "http://owner:8000")):
        status, stdout, stderr = _run(
            capsys, *composition, "-q", command, "a red wrench", "--content-json", parts
        )
        assert status == EXIT_CODES[ValidationError.code]
        assert stdout is None
        envelope = cast(dict[str, object], stderr[0])
        assert envelope["code"] == ValidationError.code
        assert envelope["subject"] == "--content-json"
    # Refused during argument validation: nothing was composed locally and nothing was sent.
    assert not (tmp_path / "store").exists()
    assert calls == []


def test_a_remote_failure_envelope_is_forwarded_and_mapped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mindbridge import cli

    envelope = {
        "code": "model_error",
        "reason": "rate_limited",
        "retryable": True,
        "stage": "generate",
        "subject": None,
        "message": "the provider rejected the request",
        "trace_id": "trace_" + "0" * 32,
        "issues": [],
    }

    class _Failure(HTTPError):
        """A 503 carrying a body, without the file object `HTTPError` would otherwise own."""

        def __init__(self) -> None:
            super().__init__(
                "http://owner:8000/v1/answers", 503, "Service Unavailable", Message(), None
            )

        def read(self, _size: int = -1) -> bytes:
            return json.dumps(envelope).encode("utf-8")

    def fake(request: object, *, timeout: float) -> _Response:
        raise _Failure

    monkeypatch.setattr(cli, "urlopen", fake)
    status, stdout, stderr = _run(capsys, "--url", "http://owner:8000", "-q", "ask", "why")
    assert status == EXIT_CODES["model_error"] == 6
    assert stdout is None
    assert stderr[0] == envelope


def test_an_unreachable_owner_is_a_retryable_storage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mindbridge import cli

    def fake(request: object, *, timeout: float) -> _Response:
        raise URLError("connection refused")

    monkeypatch.setattr(cli, "urlopen", fake)
    status, _stdout, stderr = _run(capsys, "--url", "http://owner:8000", "-q", "list")
    assert status == EXIT_CODES["storage_error"] == 7
    envelope = cast(dict[str, object], stderr[0])
    assert envelope["reason"] == "connection_failed"
    assert envelope["retryable"] is True


@pytest.mark.parametrize(
    "failure", (TimeoutError("socket detail"), URLError(TimeoutError("socket detail")))
)
def test_a_remote_timeout_is_a_retryable_storage_error(
    failure: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mindbridge import cli

    def fake(request: object, *, timeout: float) -> _Response:
        assert timeout == 0.25
        raise failure

    monkeypatch.setattr(cli, "urlopen", fake)
    status, stdout, stderr = _run(
        capsys, "--url", "http://owner:8000", "--timeout", "0.25", "-q", "list"
    )
    assert status == EXIT_CODES["storage_error"] == 7
    assert stdout is None
    envelope = cast(dict[str, object], stderr[0])
    assert envelope["reason"] == "timeout"
    assert envelope["retryable"] is True
    assert envelope["stage"] == "request"
    assert envelope["subject"] is None
    assert "socket detail" not in cast(str, envelope["message"])


@pytest.mark.parametrize("value", ("0", "-1", "nan", "inf", "-inf"))
def test_remote_timeout_must_be_positive_and_finite(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(("--url", "http://owner:8000", "--timeout", value, "list"))
    assert raised.value.code == 2
    assert "--timeout" in capsys.readouterr().err
