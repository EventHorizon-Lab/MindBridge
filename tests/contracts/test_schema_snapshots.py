"""Detect accidental REST or MCP contract changes before release."""

import json
from pathlib import Path
from typing import cast

from mcp import Client

from mindbridge.api import create_app, create_mcp_server
from mindbridge.application import MemoryKernel

SNAPSHOT_DIRECTORY = Path(__file__).with_name("snapshots")


def test_openapi_schema_matches_snapshot() -> None:
    app = create_app(cast(MemoryKernel, object()))

    assert app.openapi() == _read_snapshot("openapi.json")


async def test_mcp_tool_schemas_match_snapshot() -> None:
    server = create_mcp_server(cast(MemoryKernel, object()))

    async with Client(server) as client:
        result = await client.list_tools()

    actual = [
        tool.model_dump(mode="json", by_alias=True, exclude_none=True)
        for tool in sorted(result.tools, key=lambda tool: tool.name)
    ]
    assert actual == _read_snapshot("mcp-tools.json")


def _read_snapshot(name: str) -> object:
    return json.loads((SNAPSHOT_DIRECTORY / name).read_text(encoding="utf-8"))
