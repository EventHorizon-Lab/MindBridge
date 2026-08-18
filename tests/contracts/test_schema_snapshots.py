"""Detect accidental REST or MCP contract changes before release."""

import json
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from mcp import Client

from mindbridge.api.aml import AmlSettings
from mindbridge.api.app import build_app
from mindbridge.api.auth import TenantApiKeyAuthenticator
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


def _read_snapshot(name: str) -> object:
    return json.loads((SNAPSHOT_DIRECTORY / name).read_text(encoding="utf-8"))
