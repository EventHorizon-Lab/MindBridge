"""Contract tests for the official MCP protocol adapter."""

from datetime import datetime, timezone
from typing import cast

from mcp import Client
from mcp_types import TextContent

from mindbridge.api.mcp import build_mcp_server
from mindbridge.application.kernel import MemoryKernel
from mindbridge.contracts import (
    FeedbackReceipt,
    FeedbackRequest,
    ForgetReceipt,
    ForgetRequest,
    GetMemoryRequest,
    MemoryResult,
    MemoryWriteStatus,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
    RememberResult,
)
from mindbridge.core import JobState, MemoryState, MemoryType, VerificationStatus

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


class StubKernel:
    """Small protocol stub proving that MCP contains no memory business logic."""

    async def remember(self, request: RememberRequest) -> RememberResult:
        return RememberResult.model_validate(
            _memory_view(request.summary, request.memory_type).model_dump()
            | {"status": MemoryWriteStatus.CREATED}
        )

    async def recall(self, request: RecallRequest) -> RecallResult:
        return RecallResult(
            answer=request.query.text,
            confidence=1.0,
            memories=(),
            evidence=(),
            trace_id="trace_recall",
        )

    async def get_memory(self, tenant_id: str, memory_id: str) -> MemoryResult:
        return _memory_view(f"{tenant_id}:{memory_id}", MemoryType.EPISODIC)

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        raise AssertionError("not called")

    async def record_feedback(self, request: FeedbackRequest) -> FeedbackReceipt:
        raise AssertionError("not called")

    async def forget(self, request: ForgetRequest) -> ForgetReceipt:
        raise AssertionError("not called")

    async def get_observation_job(
        self,
        tenant_id: str,
        job_id: str,
    ) -> ObservationProcessingJobView:
        return ObservationProcessingJobView(
            job_id=job_id,
            observation_id=f"observation_for_{tenant_id}",
            state=JobState.SUCCEEDED,
            attempt=1,
            error_code=None,
            memory_ids=("memory_01",),
            created_at=NOW,
            updated_at=NOW,
            trace_id="trace_job",
        )


async def test_mcp_lists_stable_tools_from_shared_contracts() -> None:
    server = build_mcp_server(cast(MemoryKernel, StubKernel()))

    async with Client(server) as client:
        result = await client.list_tools()

    tools = {tool.name: tool for tool in result.tools}
    assert set(tools) == {
        "memory_feedback",
        "memory_forget",
        "memory_get",
        "memory_job",
        "memory_observe",
        "memory_recall",
        "memory_remember",
    }
    # The contract's own fields are the tool's arguments, not one nested `request` object:
    # a caller reaches for the flat shape first, and there is no second contract to keep in
    # step with this one.
    assert (
        tools["memory_remember"].input_schema["properties"]
        == RememberRequest.model_json_schema()["properties"]
    )
    assert (
        tools["memory_get"].input_schema["properties"]
        == GetMemoryRequest.model_json_schema()["properties"]
    )
    assert "request" not in tools["memory_recall"].input_schema["properties"]
    assert tools["memory_forget"].annotations is not None
    assert tools["memory_forget"].annotations.destructive_hint is True


async def test_mcp_calls_shared_kernel_and_returns_structured_output() -> None:
    server = build_mcp_server(cast(MemoryKernel, StubKernel()))
    request = RememberRequest(
        tenant_id="tenant_01",
        summary="A red cup was left on the table",
        memory_type=MemoryType.SEMANTIC,
        occurred_at=NOW,
    )

    async with Client(server) as client:
        result = await client.call_tool("memory_remember", request.model_dump(mode="json"))

    assert result.is_error is False
    assert result.structured_content is not None
    response = RememberResult.model_validate(result.structured_content)
    assert response.summary == request.summary
    assert response.trace_id == "trace_memory"
    # A write says whether it is the write that stored the content, the way observe always has.
    assert response.status is MemoryWriteStatus.CREATED


async def test_mcp_resolves_the_job_id_observe_hands_back() -> None:
    """The receipt's `processing_job_id` has to be usable from the face that returned it.

    `memory_observe` answers before any memory exists, so without this tool an Agent holds an
    ID it cannot redeem and has to poll recall until something appears.
    """
    server = build_mcp_server(cast(MemoryKernel, StubKernel()))

    async with Client(server) as client:
        result = await client.call_tool(
            "memory_job",
            {"tenant_id": "tenant_01", "job_id": "job_01"},
        )

    assert result.is_error is False
    job = ObservationProcessingJobView.model_validate(result.structured_content)
    assert job.state is JobState.SUCCEEDED
    assert job.memory_ids == ("memory_01",)


async def test_mcp_rejects_the_nested_request_shape() -> None:
    """A wrapper object is not a second accepted spelling; it is a missing-field error.

    The flat shape is what a caller produces from the published schema, so the wrapped one
    reaching the kernel would mean the schema and the accepted input had drifted apart.
    """
    server = build_mcp_server(cast(MemoryKernel, StubKernel()))
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

    assert result.is_error is True


async def test_mcp_enforces_cross_field_contract_rules() -> None:
    """The invariants the docstrings promise are enforced on the flattened arguments too.

    Synthesizing the signature from the model's fields could have published them while
    validating something looser; `missing` feedback is the cheapest proof that it does not.
    """
    server = build_mcp_server(cast(MemoryKernel, StubKernel()))

    async with Client(server) as client:
        result = await client.call_tool(
            "memory_feedback",
            {
                "tenant_id": "tenant_01",
                "feedback_type": "missing",
                "recall_trace_id": "trace_recall",
                "memory_id": "memory_01",
            },
        )

    assert result.is_error is True
    reported = result.content[0]
    assert isinstance(reported, TextContent)
    assert "missing feedback must not provide memory_id" in reported.text


def _memory_view(summary: str, memory_type: MemoryType) -> MemoryResult:
    return MemoryResult(
        memory_id="memory_01",
        memory_type=memory_type,
        summary=summary,
        evidence_ids=(),
        occurred_at=NOW,
        ended_at=NOW,
        created_at=NOW,
        verification_status=VerificationStatus.ATTESTED,
        state=MemoryState.ACTIVE,
        trace_id="trace_memory",
    )
