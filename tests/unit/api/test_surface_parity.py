"""Mechanical parity checks between the Python SDK, REST, and MCP surfaces.

`AGENTS.md` requires every surface that shares an operation to preserve the same
IDs, field meanings, pagination, idempotency, defaults, and error semantics. Everything here is
derived from the code so a drifting default, field, or error code fails instead of rotting.
"""

import inspect
from dataclasses import fields as dataclass_fields
from typing import cast, get_type_hints

import pytest
from fastapi.routing import APIRoute
from pydantic import BaseModel

from mindbridge import Memory
from mindbridge.api import app as rest
from mindbridge.api import mcp as mcp_adapter
from mindbridge.api.errors import ErrorEnvelope, _public_error
from mindbridge.exceptions import MindBridgeError
from mindbridge.types import (
    AssetRef,
    ContextBudget,
    ContextBundle,
    ContextConflict,
    MemoryCapabilities,
    MemoryRecord,
    Modality,
    Page,
    SearchHit,
)

CAPABILITIES = MemoryCapabilities(
    modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
    answer=True,
    transcribe=False,
    faces=False,
    describe_vision=False,
    form=True,
    consolidate=False,
    decay=True,
)

# The only hand-written mapping: which SDK operation each transport route and tool serves. Every
# `/v1` route and every published tool must appear, so a new surface cannot skip this file.
SHARED_OPERATIONS: dict[str, tuple[str, str | None]] = {
    "add": ("createMemory", "add_memory"),
    "add_many": ("createMemories", None),
    "search": ("searchMemories", "search_memories"),
    "ask": ("answer", "ask_memory"),
    "compile": ("compileContext", "compile_context"),
    "capabilities": ("capabilities", None),
    "get": ("getMemory", "get_memory"),
    "list": ("listMemories", "list_memories"),
    "delete": ("deleteMemory", "delete_memory"),
}
# Body models are matched to their route by `operation_id`; a route without one takes its arguments
# from query or path parameters instead.
REST_REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "createMemory": rest.MemoryCreate,
    "createMemories": rest.MemoryBatchCreate,
    "searchMemories": rest.QueryRequest,
    "answer": rest.AnswerRequest,
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
    """Adapter construction reads shapes and the capability advertisement, nothing else."""

    def capabilities(self) -> MemoryCapabilities:
        return CAPABILITIES


@pytest.mark.parametrize("operation", sorted(SHARED_OPERATIONS))
def test_transport_defaults_match_the_sdk(operation: str) -> None:
    operation_id, tool = SHARED_OPERATIONS[operation]
    expected = _sdk_defaults(operation)

    for surface, defaults in (
        ("rest", _rest_defaults(operation_id)),
        ("mcp", {} if tool is None else _mcp_defaults(tool)),
    ):
        for name in sorted(set(defaults) & set(expected)):
            assert defaults[name] == expected[name], f"{surface}.{operation}.{name}"


@pytest.mark.parametrize("operation", sorted(SHARED_OPERATIONS))
def test_transport_fields_name_sdk_arguments(operation: str) -> None:
    operation_id, tool = SHARED_OPERATIONS[operation]
    allowed = _sdk_parameters(operation)

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

    assert routes == {operation_id for operation_id, _tool in SHARED_OPERATIONS.values()}
    assert tools == {tool for _id, tool in SHARED_OPERATIONS.values() if tool is not None}
    assert tools == set(mcp_adapter._TOOL_ARGUMENTS)


def test_the_rest_adapter_protocol_matches_the_sdk_it_dispatches_to() -> None:
    """D3: mypy does not compare defaults across a structural protocol, so this does."""
    for operation in SHARED_OPERATIONS:
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
        status_code, rest_message = _public_error(error)
        mcp_message = mcp_adapter._error_message(error)

        assert error.code in mcp_adapter._STABLE_ERROR_CODES, error_type
        assert rest_message == mcp_message, error_type
        # Only the abstract base falls through to "unexpected"; every raised class is classified.
        assert (status_code == 500) is (error_type is MindBridgeError), error_type


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


def test_delete_reports_the_same_state_on_both_transports() -> None:
    assert set(rest.DeleteResponse.model_fields) == set(mcp_adapter.DeleteResult.model_fields)
    assert get_type_hints(rest.DeleteResponse) == get_type_hints(mcp_adapter.DeleteResult)


@pytest.mark.parametrize(
    ("record", "rest_model", "mcp_model", "renamed"),
    [
        (ContextBundle, rest.ContextBundleResponse, mcp_adapter.ContextBundleResult, {}),
        (ContextConflict, rest.ContextConflictResponse, mcp_adapter.ContextConflictResult, {}),
        (
            ContextBudget,
            rest.ContextBudgetResponse,
            mcp_adapter.ContextBudgetResult,
            {"freshness": "freshness_seconds"},
        ),
    ],
)
def test_the_compiled_bundle_mirrors_the_sdk_value_on_both_transports(
    record: type,
    rest_model: type[BaseModel],
    mcp_model: type[BaseModel],
    renamed: dict[str, str],
) -> None:
    """A new `ContextBundle` field must reach both transports instead of being dropped."""
    expected = {renamed.get(field.name, field.name) for field in dataclass_fields(record)}
    # `rendered` is the transports' serialization of `ContextBundle.render()`, not a stored field.
    if record is ContextBundle:
        expected |= {"rendered"}

    assert set(rest_model.model_fields) == expected
    assert set(mcp_model.model_fields) == expected


def test_the_context_budget_input_defaults_come_from_the_sdk_value() -> None:
    budget = ContextBudget()
    expected = {
        "max_chars": budget.max_chars,
        "max_items": budget.max_items,
        "memory_types": budget.memory_types,
        "min_confidence": budget.min_confidence,
        "freshness_seconds": budget.freshness,
    }

    for model in (rest.ContextBudgetRequest, mcp_adapter.ContextBudgetInput):
        assert {
            name: field.get_default(call_default_factory=True)
            for name, field in model.model_fields.items()
        } == expected


def test_capabilities_reports_the_same_flags_on_rest_and_in_the_mcp_instructions() -> None:
    expected = {field.name for field in dataclass_fields(MemoryCapabilities)}

    assert set(rest.CapabilitiesResponse.model_fields) == expected
    assert set(mcp_adapter._CAPABILITY_FLAGS) == expected - {"modalities"}
