"""Official MCP adapter for the shared MindBridge memory use cases."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from mcp.server import MCPServer
from mcp_types import ToolAnnotations

from mindbridge.application.kernel import MemoryKernel
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

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_IDEMPOTENT_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_DESTRUCTIVE_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)
_McpLifespan = Callable[[MCPServer[None]], AbstractAsyncContextManager[None]]


def create_mcp_server(
    kernel: MemoryKernel,
    *,
    lifespan: _McpLifespan | None = None,
) -> MCPServer[None]:
    """Expose one memory kernel through typed, agent-friendly MCP tools."""
    server: MCPServer[None] = MCPServer(
        "mindbridge",
        title="MindBridge Memory",
        description="Evidence-grounded embodied Memory as a Service.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @server.tool(annotations=_IDEMPOTENT_WRITE)
    async def memory_observe(request: ObserveRequest) -> ObservationReceipt:
        """Store one timestamped multimodal observation and return its durable job ID."""
        return await kernel.observe(request)

    @server.tool(annotations=_IDEMPOTENT_WRITE)
    async def memory_remember(request: RememberRequest) -> MemoryView:
        """Retain one explicit memory, preserving evidence and temporal provenance."""
        return await kernel.remember(request)

    @server.tool(annotations=_READ_ONLY)
    async def memory_recall(request: RecallRequest) -> RecallResult:
        """Recall relevant memories and return inspectable evidence with the answer."""
        return await kernel.recall(request)

    @server.tool(annotations=_READ_ONLY)
    async def memory_get(request: GetMemoryRequest) -> MemoryView:
        """Read one tenant-owned memory by its stable identifier."""
        return await kernel.get_memory(request.tenant_id, request.memory_id)

    @server.tool(annotations=_IDEMPOTENT_WRITE)
    async def memory_feedback(request: FeedbackRequest) -> FeedbackReceipt:
        """Record a useful, wrong, missing, or correction signal for future recall."""
        return await kernel.record_feedback(request)

    @server.tool(annotations=_DESTRUCTIVE_WRITE)
    async def memory_forget(request: ForgetRequest) -> ForgetReceipt:
        """Recoverably erase one exact memory or source observation and its derivatives."""
        return await kernel.forget(request)

    return server
