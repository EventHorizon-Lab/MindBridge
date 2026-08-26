"""Detect accidental REST or MCP contract changes before release."""

import json
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from mcp import Client

from mindbridge.api.aml import AmlSettings
from mindbridge.api.app import build_app
from mindbridge.api.auth import TenantApiKeyAuthenticator
from mindbridge.api.errors import ERRORS, RUNTIME_ERROR_CODES
from mindbridge.api.mcp import build_mcp_server
from mindbridge.application.kernel import MemoryKernel
from mindbridge.models import Generator

SNAPSHOT_DIRECTORY = Path(__file__).with_name("snapshots")


def test_openapi_schema_matches_snapshot() -> None:
    # Built with the AML routes mounted, because that is the app a deployment running the
    # leaderboard adapter actually serves. Snapshotting the variant without them left the two
    # /aml operations outside the only place `app.openapi()` is ever called, so their error
    # surface could regress without any gate noticing -- which is how they kept FastAPI's
    # default 422 through a change whose whole subject was the documented error surface.
    assert _built_app().openapi() == _read_snapshot("openapi.json")


def test_every_documented_status_is_one_the_error_table_can_produce() -> None:
    """A route may not inherit FastAPI's default error model for a status it overrides."""
    # `responses()` is what puts ErrorResponse in the document; a route registered without it
    # silently publishes `HTTPValidationError`, whose only field is `detail`, while the
    # app-wide handler answers with the envelope. Checking every operation rather than a
    # named list is what makes a newly registered route fail here instead of shipping.
    schema = _built_app().openapi()
    envelope = "#/components/schemas/ErrorResponse"
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            for status_code, response in operation["responses"].items():
                if not status_code.startswith(("4", "5")):
                    continue
                reference = response["content"]["application/json"]["schema"]["$ref"]
                assert reference == envelope, f"{method.upper()} {path} {status_code}"


def test_every_operation_documents_unexpected_errors() -> None:
    for path, operations in _built_app().openapi()["paths"].items():
        for method, operation in operations.items():
            response = operation["responses"]["500"]
            assert "`internal_error`" in response["description"], f"{method.upper()} {path}"


def test_complete_deletion_only_claims_central_storage() -> None:
    schema = _built_app().openapi()["components"]["schemas"]["DeletionTombstoneView"]
    description = schema["properties"]["propagation_state"]["description"]

    assert "central PostgreSQL and object storage" in description
    assert "offline edge devices apply" in description
    assert "every copy" not in description


def _built_app() -> FastAPI:
    return build_app(
        cast(MemoryKernel, object()),
        authenticator=TenantApiKeyAuthenticator(
            {"tenant_01": ("tenant-api-key-000000000000000000",)}
        ),
        aml=(
            AmlSettings(api_key="aml-api-key-0000000000000000000000", tenant_prefix="bench_aml"),
            cast(Generator, object()),
        ),
    )


async def test_mcp_tool_schemas_match_snapshot() -> None:
    server = build_mcp_server(cast(MemoryKernel, object()))

    async with Client(server) as client:
        result = await client.list_tools()

    actual = [
        tool.model_dump(mode="json", by_alias=True, exclude_none=True)
        for tool in sorted(result.tools, key=lambda tool: tool.name)
    ]
    assert actual == _read_snapshot("mcp-tools.json")


async def test_every_published_field_is_described() -> None:
    """A snapshot cannot hold this line: a new field changes it either way.

    Regenerating the snapshot is the documented fix for a snapshot diff, so a field shipped
    without a description would simply be recorded. This is the assertion that fails instead.

    Asserted over the two published documents rather than over the contract classes, because
    what has to be self-describing is what a caller actually reads. Internal models that never
    reach a caller are out of scope by construction.
    """
    server = build_mcp_server(cast(MemoryKernel, object()))
    async with Client(server) as client:
        tools = (await client.list_tools()).tools

    schemas = _built_app().openapi().get("components", {}).get("schemas", {})
    undescribed = _undescribed("openapi", schemas)
    for tool in tools:
        undescribed += _undescribed(tool.name, tool.input_schema)
        undescribed += _undescribed(tool.name, tool.output_schema or {})

    assert sorted(set(undescribed)) == []


def _undescribed(source: str, schema: object) -> list[str]:
    """Every property anywhere in one JSON Schema document that carries no description."""
    missing: list[str] = []
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            missing += [
                f"{source}:{schema.get('title', '?')}.{name}"
                for name, value in properties.items()
                if isinstance(value, dict) and not value.get("description") and "$ref" not in value
            ]
        for key, value in schema.items():
            if key in {"$defs", "properties", "items", "anyOf", "allOf"}:
                for nested in value.values() if isinstance(value, dict) else value:
                    missing += _undescribed(source, nested)
    return missing


def test_every_documented_error_code_is_a_real_code() -> None:
    """Every code a route publishes, and every code a handler renders, is in the one table."""
    unknown_runtime = sorted(set(RUNTIME_ERROR_CODES.values()) - set(ERRORS))

    assert unknown_runtime == []


def _read_snapshot(name: str) -> object:
    return json.loads((SNAPSHOT_DIRECTORY / name).read_text(encoding="utf-8"))
