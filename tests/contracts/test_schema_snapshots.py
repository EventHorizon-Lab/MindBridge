"""Detect accidental REST or MCP contract changes before release."""

import json
from pathlib import Path
from typing import cast

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
    app = build_app(
        cast(MemoryKernel, object()),
        authenticator=TenantApiKeyAuthenticator(
            {"tenant_01": ("tenant-api-key-000000000000000000",)}
        ),
    )

    assert app.openapi() == _read_snapshot("openapi.json")


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
    app = build_app(
        cast(MemoryKernel, object()),
        authenticator=TenantApiKeyAuthenticator(
            {"tenant_01": ("tenant-api-key-000000000000000000",)}
        ),
    )
    server = build_mcp_server(cast(MemoryKernel, object()))
    async with Client(server) as client:
        tools = (await client.list_tools()).tools

    undescribed = _undescribed("openapi", app.openapi().get("components", {}).get("schemas", {}))
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


def test_aml_routes_publish_the_shared_error_envelope() -> None:
    """The default snapshot builds without AML, so these two routes need their own check.

    They were the last pair still republishing FastAPI's `HTTPValidationError` for a body the
    app never sends, and no snapshot could see it.
    """
    app = build_app(
        cast(MemoryKernel, object()),
        authenticator=TenantApiKeyAuthenticator(
            {"tenant_01": ("tenant-api-key-000000000000000000",)}
        ),
        aml=(
            AmlSettings(api_key="aml-api-key-0000000000000000000000", tenant_prefix="aml"),
            cast(Generator, object()),
        ),
    )
    schema = app.openapi()

    assert "HTTPValidationError" not in schema["components"]["schemas"]
    for path in ("/aml/add", "/aml/search"):
        responses = schema["paths"][path]["post"]["responses"]
        assert "401" in responses, path
        bodies = {
            code: body["content"]["application/json"]["schema"]["$ref"].rsplit("/", 1)[-1]
            for code, body in responses.items()
            if code != "200"
        }
        assert set(bodies.values()) == {"ErrorResponse"}, (path, bodies)


def _read_snapshot(name: str) -> object:
    return json.loads((SNAPSHOT_DIRECTORY / name).read_text(encoding="utf-8"))
