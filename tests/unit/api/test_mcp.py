"""Contract tests for the official MCP protocol adapter."""

from datetime import datetime, timezone
from typing import cast

from mcp import Client

from mindbridge.api import create_mcp_server
from mindbridge.application import MemoryKernel
from mindbridge.contracts import (
    FeedbackReceipt,
    FeedbackRequest,
    ForgetReceipt,
    ForgetRequest,
    GetMemoryRequest,
    MemoryView,
    ObservationReceipt,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import MemoryState, MemoryType, VerificationStatus

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


class StubKernel:
    """Small protocol stub proving that MCP contains no memory business logic."""

    async def remember(self, request: RememberRequest) -> MemoryView:
        return _memory_view(request.summary, request.memory_type)

    async def recall(self, request: RecallRequest) -> RecallResult:
        return RecallResult(
            answer=request.query.text,
            confidence=1.0,
            memories=(),
            evidence=(),
            trace_id="trace_recall",
        )

    async def get_memory(self, tenant_id: str, memory_id: str) -> MemoryView:
        return _memory_view(f"{tenant_id}:{memory_id}", MemoryType.EPISODIC)

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        raise AssertionError("not called")

    async def record_feedback(self, request: FeedbackRequest) -> FeedbackReceipt:
        raise AssertionError("not called")

    async def forget(self, request: ForgetRequest) -> ForgetReceipt:
        raise AssertionError("not called")


async def test_mcp_lists_stable_tools_from_shared_contracts() -> None:
    server = create_mcp_server(cast(MemoryKernel, StubKernel()))

    async with Client(server) as client:
        result = await client.list_tools()

    tools = {tool.name: tool for tool in result.tools}
    assert set(tools) == {
        "memory_feedback",
        "memory_forget",
        "memory_get",
        "memory_observe",
        "memory_recall",
        "memory_remember",
    }
    assert (
        tools["memory_remember"].input_schema["$defs"]["RememberRequest"]["properties"]
        == RememberRequest.model_json_schema()["properties"]
    )
    assert (
        tools["memory_get"].input_schema["$defs"]["GetMemoryRequest"]["properties"]
        == GetMemoryRequest.model_json_schema()["properties"]
    )
    assert tools["memory_forget"].annotations is not None
    assert tools["memory_forget"].annotations.destructive_hint is True


async def test_mcp_calls_shared_kernel_and_returns_structured_output() -> None:
    server = create_mcp_server(cast(MemoryKernel, StubKernel()))
    request = RememberRequest(
        tenant_id="tenant_01",
        summary="A red cup was left on the table",
        memory_type=MemoryType.SEMANTIC,
        occurred_at=NOW,
    )

    async with Client(server) as client:
        result = await client.call_tool(
            "memory_remember",
            {"request": request.model_dump(mode="json")},
        )

    assert result.is_error is False
    assert result.structured_content is not None
    assert MemoryView.model_validate(result.structured_content).summary == request.summary


def _memory_view(summary: str, memory_type: MemoryType) -> MemoryView:
    return MemoryView(
        memory_id="memory_01",
        memory_type=memory_type,
        summary=summary,
        evidence_ids=(),
        occurred_at=NOW,
        ended_at=NOW,
        created_at=NOW,
        verification_status=VerificationStatus.ATTESTED,
        state=MemoryState.ACTIVE,
    )
