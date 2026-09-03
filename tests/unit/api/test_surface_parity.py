"""Mechanical parity checks between the Python SDK, REST, and MCP surfaces.

`AGENTS.md` requires every surface that shares an operation to preserve the same
IDs, field meanings, pagination, idempotency, defaults, and error semantics. Everything here is
derived from the code so a drifting default, field, or error code fails instead of rotting.
"""

import inspect
import json
import re
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, cast, get_type_hints

import pytest
from fastapi import status
from fastapi.routing import APIRoute
from pydantic import BaseModel

import mindbridge
from mindbridge import Memory, cli
from mindbridge.api import app as rest
from mindbridge.api import content, errors
from mindbridge.api import mcp as mcp_adapter
from mindbridge.api.errors import REASON_STATUS, ErrorEnvelope, _public_error
from mindbridge.exceptions import (
    RETRYABLE_REASONS,
    IndexUnavailableError,
    MemoryNotFoundError,
    MindBridgeError,
    ModelError,
    ModelOutputTruncatedError,
    StorageError,
    ValidationError,
)
from mindbridge.memory import declared_capabilities
from mindbridge.types import (
    AssetRef,
    Blob,
    ContextBudget,
    ContextBundle,
    ContextConflict,
    ContextUnknown,
    FaceObservation,
    IdentityErasure,
    IdentityProfile,
    MemoryCapabilities,
    MemoryRecord,
    Modality,
    Page,
    SearchHit,
    SpeakerSegment,
)

# The only hand-written mapping: which SDK operation each transport route and tool serves. Every
# `/v1` route and every published tool must appear, so a new surface cannot skip this file.
SHARED_OPERATIONS: dict[str, tuple[str | None, str | None]] = {
    "add": ("createMemory", "add_memory"),
    "add_many": ("createMemories", None),
    "search": ("searchMemories", "search_memories"),
    "ask": ("answer", "ask_memory"),
    "compile": ("compileContext", "compile_context"),
    "get": ("getMemory", "get_memory"),
    "list": ("listMemories", "list_memories"),
    "delete": ("deleteMemory", "delete_memory"),
    "reinforce": ("reinforceMemories", "reinforce_memories"),
    "speech": (None, "analyze_speech"),
    "faces": (None, "analyze_faces"),
    "register_speaker": (None, "register_speaker"),
    "register_identity": (None, "register_identity"),
    "identity": (None, "get_identity"),
    "unlink_identity": (None, "unlink_identity"),
    "forget_identity": (None, "forget_identity"),
}
# Operations no transport exposes, each with the reason. `docs/design-principles.md` requires a
# transport gap to be documented rather than silently left out, and the union of this and
# `SHARED_OPERATIONS` must be every public operation, so a new SDK operation cannot arrive without
# someone deciding which of the two it belongs in.
UNEXPOSED_OPERATIONS: dict[str, str] = {
    "add_stream": "a lazy generator does not fit one finite request or response",
    "search_with_trace": "candidate-level retrieval diagnostics with no agent or client action",
    "reindex": "unbounded index maintenance an operator schedules, not a caller",
    "optimize": "index maintenance an operator schedules, not a caller",
    # Fast capture is a two-call contract whose second half the host schedules; a transport
    # caller cannot be handed the first half without also owning the settle loop.
    "capture": "acknowledges before enrichment, so the host must own the matching settle loop",
    "settle": "deferred-enrichment work an owner schedules, not a caller",
    "pending_captures": "queue depth for the process that runs settle",
    # The control plane rewrites derived memory under policy. `docs/context-os.md` keeps that
    # authority with the host: an agent must not gain it by holding ordinary recall access.
    "consolidation_candidates": "the control plane's own due-work queue, read by the host loop",
    "consolidate": "derives and retires memory under host authority, not agent authority",
    "forget": "cognitive forgetting is a policy decision the host owns",
    "rollback": "reverses a committed operation, so it stays with the host that authorized it",
    "operations": "the control-plane audit log, read by an operator rather than a caller",
}
# Not operations: construction, lifecycle, and the capability declaration REST reports in
# `/healthz`.
NON_OPERATIONS = frozenset({"capabilities", "close", "from_config", "from_plugins"})
# Transport fields that select between two SDK operations instead of naming an argument to one.
# `explain` routes the same search to `search_with_trace`, which takes no extra argument itself.
ROUTING_FIELDS: dict[str, frozenset[str]] = {"search": frozenset({"explain"})}
# Prose in `docs/api/mcp.md` spells the count out, so the check has to spell it out too.
_COUNT_WORDS = {
    10: "Ten",
    11: "Eleven",
    12: "Twelve",
    13: "Thirteen",
    14: "Fourteen",
}
# `search_with_trace` has no route or tool of its own; the search surfaces reach it through
# `explain`, so the adapter protocol must still declare it exactly as the SDK does. The MCP-only
# operations are absent from the REST protocol by design, so they are not part of it.
PROTOCOL_OPERATIONS = (
    *(operation for operation, (route, _tool) in SHARED_OPERATIONS.items() if route is not None),
    "search_with_trace",
)
# One declared composition, so the capability view an MCP server publishes at construction is
# built from a real value rather than a stub.
CAPABILITIES = MemoryCapabilities(
    embedding=frozenset({Modality.TEXT, Modality.IMAGE}),
    embedding_model="jina-v5-omni",
    embedding_space="space_1",
    embedding_dimension=1024,
    generation=frozenset({Modality.TEXT}),
    generation_model="qwen3-omni",
    speaker_recognition=True,
)
# Body models are matched to their route by `operation_id`; a route without one takes its arguments
# from query or path parameters instead.
REST_REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "createMemory": rest.MemoryCreate,
    "createMemories": rest.MemoryBatchCreate,
    "searchMemories": rest.QueryRequest,
    "answer": rest.AnswerRequest,
    "reinforceMemories": rest.ReinforceRequest,
    "compileContext": rest.ContextRequest,
}


def _sdk_defaults(operation: str) -> dict[str, object]:
    signature = inspect.signature(getattr(Memory, operation))
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _sdk_parameters(operation: str) -> set[str]:
    signature = inspect.signature(getattr(Memory, operation))
    return set(signature.parameters) - {"self"}


def _rest_route(operation_id: str) -> APIRoute:
    app = rest.create_app(memory=cast(rest._Memory, _UnusedMemory()))
    routes = [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.operation_id == operation_id
    ]
    assert len(routes) == 1, operation_id
    return routes[0]


def _rest_defaults(operation_id: str) -> dict[str, object]:
    model = REST_REQUEST_MODELS.get(operation_id)
    if model is not None:
        return {
            name: field.get_default(call_default_factory=True)
            for name, field in model.model_fields.items()
            if not field.is_required()
        }
    return {
        name: parameter.default
        for name, parameter in inspect.signature(
            _rest_route(operation_id).endpoint
        ).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _rest_fields(operation_id: str) -> set[str]:
    model = REST_REQUEST_MODELS.get(operation_id)
    if model is not None:
        return set(model.model_fields)
    return set(inspect.signature(_rest_route(operation_id).endpoint).parameters)


def _mcp_defaults(tool: str) -> dict[str, object]:
    server = mcp_adapter.build_mcp_server(_UnusedMemory())  # type: ignore[arg-type]
    tools = {published.name: published for published in server._tool_manager.list_tools()}
    assert tool in tools, tool
    signature = inspect.signature(tools[tool].fn)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


class _UnusedMemory:
    """Adapter construction reads shapes and the capability declaration, nothing else.

    MCP fixes `instructions` at construction, so building a server reads `capabilities`; every
    other member is only inspected.
    """

    @property
    def capabilities(self) -> MemoryCapabilities:
        return CAPABILITIES


@pytest.mark.parametrize("operation", sorted(SHARED_OPERATIONS))
def test_transport_defaults_match_the_sdk(operation: str) -> None:
    operation_id, tool = SHARED_OPERATIONS[operation]
    expected = _sdk_defaults(operation)

    for surface, defaults in (
        ("rest", {} if operation_id is None else _rest_defaults(operation_id)),
        ("mcp", {} if tool is None else _mcp_defaults(tool)),
    ):
        for name in sorted(set(defaults) & set(expected)):
            assert defaults[name] == expected[name], f"{surface}.{operation}.{name}"


@pytest.mark.parametrize("operation", sorted(SHARED_OPERATIONS))
def test_transport_fields_name_sdk_arguments(operation: str) -> None:
    operation_id, tool = SHARED_OPERATIONS[operation]
    allowed = _sdk_parameters(operation) | ROUTING_FIELDS.get(operation, frozenset())

    if operation_id is not None:
        assert _rest_fields(operation_id) <= allowed, operation
    if tool is not None:
        assert mcp_adapter._TOOL_ARGUMENTS[tool] <= allowed, operation


def test_every_transport_operation_is_covered() -> None:
    app = rest.create_app(memory=cast(rest._Memory, _UnusedMemory()))
    routes = {
        route.operation_id
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith("/v1")
    }
    server = mcp_adapter.build_mcp_server(_UnusedMemory())  # type: ignore[arg-type]
    tools = {published.name for published in server._tool_manager.list_tools()}

    assert routes == {
        operation_id for operation_id, _tool in SHARED_OPERATIONS.values() if operation_id
    }
    assert tools == {tool for _id, tool in SHARED_OPERATIONS.values() if tool is not None}
    assert tools == set(mcp_adapter._TOOL_ARGUMENTS)


def test_every_sdk_operation_is_exposed_or_documented_as_a_gap() -> None:
    """A new SDK operation must land in one of the two tables, not in neither."""
    published = {name for name in dir(Memory) if not name.startswith("_")} - NON_OPERATIONS

    assert published == set(SHARED_OPERATIONS) | set(UNEXPOSED_OPERATIONS)
    assert not set(SHARED_OPERATIONS) & set(UNEXPOSED_OPERATIONS)
    # Every embodied and identity operation is agent-callable over MCP: the adapter runs in the
    # process that owns `Memory`, so "owner-process concern" was never a transport boundary.
    for operation in (
        "speech",
        "faces",
        "register_speaker",
        "register_identity",
        "identity",
        "forget_identity",
    ):
        assert SHARED_OPERATIONS[operation][1] is not None, operation


def test_mcp_tool_descriptions_state_the_contract_an_agent_needs() -> None:
    """The published description is the contract, and 3.13 strips docstring indentation."""
    server = mcp_adapter.build_mcp_server(_UnusedMemory())  # type: ignore[arg-type]

    for tool in server._tool_manager.list_tools():
        description = tool.description
        assert description is not None, tool.name
        # `inspect.cleandoc` output can never contain an indented continuation line, so this
        # fails on any interpreter that publishes the raw docstring.
        assert not any(line.startswith(" ") for line in description.splitlines()), tool.name
        assert description == description.strip(), tool.name


def test_tool_descriptions_are_normalized_whatever_the_interpreter_did() -> None:
    """CPython 3.13 strips docstring indentation and earlier versions do not.

    `test_mcp_tool_descriptions_state_the_contract_an_agent_needs` cannot see the difference on
    3.13 or later, where the compiler already stripped it, so the normalization is checked here
    against a docstring the compiler never touched.
    """

    def operation() -> None: ...

    operation.__doc__ = "Summary line.\n\n    An indented continuation.\n    "

    assert mcp_adapter._stable_errors(operation).__doc__ == (
        "Summary line.\n\nAn indented continuation."
    )


def test_the_rest_adapter_protocol_matches_the_sdk_it_dispatches_to() -> None:
    """D3: mypy does not compare defaults across a structural protocol, so this does."""
    for operation in PROTOCOL_OPERATIONS:
        declared = inspect.signature(getattr(rest._Memory, operation))
        real = inspect.signature(getattr(Memory, operation))
        assert set(declared.parameters) == set(real.parameters), operation
        for name, parameter in declared.parameters.items():
            assert parameter.default == real.parameters[name].default, f"{operation}.{name}"


def _public_exceptions(root: type[MindBridgeError]) -> list[type[MindBridgeError]]:
    found = [root]
    for subclass in root.__subclasses__():
        found.extend(_public_exceptions(subclass))
    return found


def test_every_public_exception_is_mapped_on_both_transports() -> None:
    for error_type in _public_exceptions(MindBridgeError):
        error = error_type("failure detail")
        status_code, message = _public_error(error)

        assert error.code in mcp_adapter._STABLE_ERROR_CODES, error_type
        # Both transports read one table, so the check is that the table covers the class, not
        # that two calls to the same function agree.
        assert (error_type.code in errors.MESSAGE_BY_CODE) is (error_type is not MindBridgeError), (
            error_type
        )
        # The raise site's own words always survive the mapping.
        assert message == "failure detail" or error_type is MindBridgeError, error_type
        # Only the abstract base falls through to "unexpected"; every raised class is classified.
        assert (status_code == 500) is (error_type is MindBridgeError), error_type


def test_a_reason_gets_one_status_whatever_raised_it() -> None:
    """The status is a function of `reason`, so no raise site can answer a mapped condition twice.

    This is the whole point of the table: `payload_too_large` used to mean 413 from the request
    middleware and 502 from a provider rejecting an oversized asset.
    """
    carriers = (
        MindBridgeError,
        ValidationError,
        MemoryNotFoundError,
        ModelError,
        ModelOutputTruncatedError,
        StorageError,
        IndexUnavailableError,
    )
    for reason, expected in errors.REASON_STATUS.items():
        for error_type in carriers:
            error = error_type("failure detail", reason=reason)
            assert _public_error(error)[0] == expected, (reason, error_type)


def test_every_retryable_reason_says_503_and_nothing_else_does() -> None:
    """503 and `RETRYABLE_REASONS` must be the same set in both directions.

    503 tells a client the condition is transient. A non-retryable reason mapped to 503 sends an
    agent into a retry loop it can never win, and a retryable reason mapped elsewhere loses the
    retry. `io_failed` was 503 while `retryable` was false.
    """
    transient = {
        reason
        for reason, status_code in errors.REASON_STATUS.items()
        if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    }

    assert transient == RETRYABLE_REASONS


def test_every_reason_a_raise_site_can_produce_has_a_row() -> None:
    """A new reason must land in the table, or one condition gets two statuses again.

    `cli.py` and `benchmarks/` are excluded on purpose: their reasons are theirs, reported in the
    CLI's own JSON document, and never reach an HTTP status or a tool error envelope.
    """
    source = Path(mindbridge.__file__).parent
    literal = re.compile(r'\breason="([a-z_]+)"|\bdefault_reason\s*[:=][^"]*"([a-z_]+)"')
    raised: dict[str, str] = {}
    for path in sorted(source.rglob("*.py")):
        if path.name == "cli.py" or "benchmarks" in path.parts:
            continue
        for match in literal.finditer(path.read_text(encoding="utf-8")):
            raised[match.group(1) or match.group(2)] = path.name

    missing = {reason: origin for reason, origin in raised.items() if reason not in REASON_STATUS}
    assert not missing, (
        f"add a row to mindbridge.api.errors.REASON_STATUS for {missing}, "
        "or the condition falls back to a coarse per-code status"
    )
    assert "backend_not_configured" in raised, "the scan matched nothing; check the pattern"


def test_the_error_envelope_has_one_shape() -> None:
    assert set(ErrorEnvelope.model_fields) == mcp_adapter._ERROR_FIELDS
    assert set(ErrorEnvelope.model_fields) == {
        "code",
        "reason",
        "retryable",
        "stage",
        "subject",
        "message",
        "trace_id",
        "issues",
    }


@pytest.mark.parametrize(
    ("record", "rest_model", "mcp_model"),
    [
        (MemoryRecord, rest.MemoryResponse, mcp_adapter.MemoryResult),
        (SearchHit, rest.SearchHitResponse, mcp_adapter.SearchHitResult),
    ],
)
def test_result_records_expose_the_same_fields_on_both_transports(
    record: type,
    rest_model: type[BaseModel],
    mcp_model: type[BaseModel],
) -> None:
    expected = {field.name for field in dataclass_fields(record)}

    assert set(rest_model.model_fields) == expected
    assert set(mcp_model.model_fields) == expected


def test_serialized_assets_drop_the_local_path_on_both_transports() -> None:
    expected = {field.name for field in dataclass_fields(AssetRef)} - {"path"}

    assert set(rest.AssetResponse.model_fields) == expected
    assert set(mcp_adapter.AssetResult.model_fields) == expected


def test_page_results_expose_the_same_fields_on_both_transports() -> None:
    expected = {field.name for field in dataclass_fields(Page)}

    assert set(rest.PageResponse.model_fields) == expected
    assert set(mcp_adapter.PageResult.model_fields) == expected


@pytest.mark.parametrize(
    ("record", "model", "field"),
    [
        (SpeakerSegment, mcp_adapter.SpeechResult, "segments"),
        (FaceObservation, mcp_adapter.FacesResult, "observations"),
        (IdentityProfile, mcp_adapter.IdentityResult, "identity"),
        (IdentityErasure, mcp_adapter.ForgetResult, "erasure"),
    ],
)
def test_embodied_tool_results_publish_every_public_field(
    record: type,
    model: type[BaseModel],
    field: str,
) -> None:
    schema = model.model_json_schema()

    assert field in schema["properties"]
    assert set(schema["$defs"][record.__name__]["properties"]) == {
        declared.name for declared in dataclass_fields(record)
    }


def test_health_reports_every_declared_capability() -> None:
    """`/healthz` says what the composition can do, so a new capability cannot go unreported."""
    assert set(rest.CapabilitiesResponse.model_fields) == set(CAPABILITIES.document())
    assert set(CAPABILITIES.document()) == {
        field.name for field in dataclass_fields(MemoryCapabilities)
    } | {"operations"}


class _TinyEmbedder:
    """The smallest thing `declared_capabilities` accepts, so the CLI path needs no store."""

    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
    embedding_model = "jina-v5-omni"
    embedding_space = "space_1"
    embedding_dimension = 1024


def test_every_surface_publishes_one_capability_document() -> None:
    """REST, MCP, and the CLI must render `MemoryCapabilities.document()`, not three views."""
    served = rest._capabilities_response(CAPABILITIES).model_dump(mode="json")
    instructions = mcp_adapter.build_mcp_server(_UnusedMemory()).instructions  # type: ignore[arg-type]
    assert instructions is not None
    greeted = json.loads(instructions[instructions.index("{") :])
    probed = cli._doctor_capabilities({"embedder": _TinyEmbedder()})

    assert served == CAPABILITIES.document()
    assert greeted == CAPABILITIES.document()
    assert probed == declared_capabilities(embedder=cast(Any, _TinyEmbedder())).document()
    # The one composition the three surfaces share here differs only in its declared backends.
    assert set(probed or {}) == set(CAPABILITIES.document())


def test_the_capability_document_names_the_backend_each_operation_needs() -> None:
    """`operations` is derived, so an agent never has to know that `ask` means generation."""
    assert CAPABILITIES.operations == frozenset({"ask"})
    assert MemoryCapabilities(
        embedding=frozenset({Modality.TEXT}),
        embedding_model="m",
        embedding_space="s",
        embedding_dimension=8,
        transcription=frozenset({Modality.AUDIO}),
        face=frozenset({Modality.IMAGE}),
        formation=frozenset({Modality.TEXT}),
        consolidation_model="c",
        speaker_recognition=True,
    ).operations == frozenset({"speech", "transcribe", "faces", "formation", "consolidate"})


def test_the_compiled_bundle_mirrors_the_sdk_value_on_both_transports() -> None:
    """A field added to `ContextBundle` must reach both transports instead of being dropped."""
    for record, rest_model, mcp_model, renamed in (
        (ContextBundle, rest.ContextBundleResponse, mcp_adapter.ContextBundleResult, {}),
        (ContextConflict, rest.ContextConflictResponse, mcp_adapter.ContextConflictResult, {}),
        (ContextUnknown, rest.ContextUnknownResponse, mcp_adapter.ContextUnknownResult, {}),
        (
            ContextBudget,
            rest.ContextBudgetResponse,
            mcp_adapter.ContextBudgetResult,
            {"freshness": "freshness_seconds"},
        ),
    ):
        expected = {renamed.get(field.name, field.name) for field in dataclass_fields(record)}
        # `rendered` is the transports' serialization of `render()`, not a stored field.
        if record is ContextBundle:
            expected |= {"rendered"}
        assert set(rest_model.model_fields) == expected, record.__name__
        assert set(mcp_model.model_fields) == expected, record.__name__


def test_the_context_budget_transport_defaults_come_from_the_sdk_value() -> None:
    """`ContextBudget()` is the one source of the published default, on both surfaces."""
    budget = ContextBudget()
    expected = {
        "max_chars": budget.max_chars,
        "max_items": budget.max_items,
        "memory_types": budget.memory_types,
        "min_confidence": budget.min_confidence,
        "freshness_seconds": budget.freshness,
        "max_latency_ms": budget.max_latency_ms,
    }

    for model in (rest.ContextBudgetRequest, mcp_adapter.ContextBudgetInput):
        assert {
            name: field.get_default(call_default_factory=True)
            for name, field in model.model_fields.items()
        } == expected, model.__name__


def test_the_mcp_instructions_publish_the_declared_capability_view() -> None:
    """MCP has no capability tool: the declaration REST serves at `/healthz` is the greeting."""
    server = mcp_adapter.build_mcp_server(_UnusedMemory())  # type: ignore[arg-type]
    instructions = server.instructions

    assert instructions is not None
    assert instructions.startswith("MindBridge is local multimodal memory")
    assert "Prefer compile_context for task-ready context" in instructions
    # The whole document, so a new capability cannot go unmentioned in the greeting.
    assert json.loads(instructions[instructions.index("{") :]) == CAPABILITIES.document()


def test_the_documented_count_of_operations_without_a_tool_is_the_real_one() -> None:
    """`docs/api/mcp.md` states a count in prose; the prose must not drift from the tables."""
    without_a_tool = len(UNEXPOSED_OPERATIONS) + sum(
        1 for _route, tool in SHARED_OPERATIONS.values() if tool is None
    )
    page = (Path(mindbridge.__file__).parents[2] / "docs" / "api" / "mcp.md").read_text(
        encoding="utf-8"
    )

    assert f"{_COUNT_WORDS[without_a_tool]} Python operations have no MCP tool." in page


def test_both_transports_decode_content_parts_with_one_implementation() -> None:
    """The 83-line decoder existed twice; identical inputs must still normalize identically."""
    parts = [
        {"type": "input_text", "text": "  At the station.  "},
        {"type": "input_image", "image_url": "data:image/png;base64,cG5n"},
        {"type": "input_file", "file_data": "d2F2", "media_type": "audio/wav"},
    ]
    request = rest.MemoryCreate.model_validate({"content": parts})

    # Neither transport may define its own decoder again.
    for module in (rest, mcp_adapter):
        assert module.__dict__["content_input"] is content.content_input
    assert content.content_input(request.content) == (
        "At the station.",
        Blob(data=b"png", media_type="image/png"),
        Blob(data=b"wav", media_type="audio/wav"),
    )


def test_delete_reports_the_same_state_on_both_transports() -> None:
    assert set(rest.DeleteResponse.model_fields) == set(mcp_adapter.DeleteResult.model_fields)
    assert get_type_hints(rest.DeleteResponse) == get_type_hints(mcp_adapter.DeleteResult)
