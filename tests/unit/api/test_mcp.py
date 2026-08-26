"""Contract tests for the official MCP protocol adapter."""

import json
from datetime import datetime, timezone
from typing import cast

from mcp import Client
from mcp_types import TextContent

from mindbridge.api.mcp import build_mcp_server
from mindbridge.application.kernel import MemoryKernel
from mindbridge.contracts import (
    ContractModel,
    FeedbackReceipt,
    FeedbackRequest,
    ForgetReceipt,
    ForgetRequest,
    GetMemoryRequest,
    GetObservationJobRequest,
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
from mindbridge.core import (
    JobState,
    MemoryDeletedError,
    MemoryState,
    MemoryType,
    VerificationStatus,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)

TOOL_CONTRACTS: dict[str, type[ContractModel]] = {
    "memory_observe": ObserveRequest,
    "memory_remember": RememberRequest,
    "memory_recall": RecallRequest,
    "memory_get": GetMemoryRequest,
    "memory_job": GetObservationJobRequest,
    "memory_feedback": FeedbackRequest,
    "memory_forget": ForgetRequest,
}
"""Every published tool and the contract whose fields are its arguments.

Adding a tool without adding a row here fails the tool-set assertion, which is what makes the
per-tool shape check below cover the new one too.
"""


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
    assert set(tools) == set(TOOL_CONTRACTS)
    # The contract's own fields are the tool's arguments, not one nested `request` object:
    # a caller reaches for the flat shape first, and there is no second contract to keep in
    # step with this one. Asserted for every tool, because the way the library documents --
    # one `request: SomeContract` parameter -- silently produces the nested shape, and a new
    # tool written that way would otherwise only move the snapshot.
    for name, model in TOOL_CONTRACTS.items():
        assert tools[name].input_schema["properties"] == model.model_json_schema()["properties"], (
            name
        )
        assert "request" not in tools[name].input_schema["properties"], name
    assert tools["memory_forget"].annotations is not None
    assert tools["memory_forget"].annotations.destructive_hint is True


async def test_mcp_rejects_an_unknown_argument() -> None:
    """An unknown key must fault rather than be dropped.

    MCP's generated argument model ignores extras, so `ContractModel`'s `extra="forbid"` no
    longer covers the flattened fields. Dropping one is not a lost value but a different
    question: `Mode` ignored leaves `mode` at its default, so a caller asking to enumerate
    gets a truncating ranked answer and no indication it happened.
    """
    server = build_mcp_server(cast(MemoryKernel, StubKernel()))

    async with Client(server) as client:
        result = await client.call_tool(
            "memory_recall",
            {"tenant_id": "tenant_01", "query": {"text": "how many mugs"}, "Mode": "enumerate"},
        )

    assert result.is_error is True
    reported = result.content[0]
    assert isinstance(reported, TextContent)
    envelope = json.loads(reported.text[reported.text.index("{") :])
    assert envelope["code"] == "request_validation_failed"
    assert envelope["issues"][0]["location"] == ["Mode"]
    assert envelope["issues"][0]["code"] == "extra_forbidden"


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
    envelope = json.loads(reported.text[reported.text.index("{") :])
    assert envelope["code"] == "request_validation_failed"
    assert "missing feedback must not provide memory_id" in envelope["issues"][0]["message"]


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


async def test_mcp_errors_carry_the_code_and_trace_id_rest_callers_get() -> None:
    """An agent has to branch on a code, not grep a sentence.

    `ErrorResponse.code` tells callers to branch on it rather than on the message, and the
    tool descriptions promise specific codes. Over MCP the only channel is the error text, so
    the envelope goes there: `memory_deleted` and `memory_not_found` are a substring apart in
    prose and a different decision in practice, and without `trace_id` an MCP failure cannot
    be correlated with the telemetry the other two faces correlate with.
    """

    class DeletedKernel(StubKernel):
        async def get_memory(self, tenant_id: str, memory_id: str) -> MemoryResult:
            raise MemoryDeletedError("memory has been explicitly forgotten")

    server = build_mcp_server(cast(MemoryKernel, DeletedKernel()))

    async with Client(server) as client:
        result = await client.call_tool(
            "memory_get",
            {"tenant_id": "tenant_01", "memory_id": "memory_01"},
        )

    assert result.is_error is True
    reported = result.content[0]
    assert isinstance(reported, TextContent)
    # MCP prefixes the raised text with "Error executing tool <name>: "; the envelope is the
    # remainder, so a caller recovers it without depending on that prefix's wording.
    envelope = json.loads(reported.text[reported.text.index("{") :])
    assert envelope["code"] == "memory_deleted"
    assert envelope["message"] == "memory content was explicitly deleted"
    assert envelope["trace_id"]


async def test_mcp_sanitizes_an_unmapped_failure_as_internal_error() -> None:

    class BrokenKernel(StubKernel):
        async def get_memory(self, tenant_id: str, memory_id: str) -> MemoryResult:
            raise ZeroDivisionError("a genuine bug")

    server = build_mcp_server(cast(MemoryKernel, BrokenKernel()))

    async with Client(server) as client:
        result = await client.call_tool(
            "memory_get",
            {"tenant_id": "tenant_01", "memory_id": "memory_01"},
        )

    assert result.is_error is True
    reported = result.content[0]
    assert isinstance(reported, TextContent)
    envelope = json.loads(reported.text[reported.text.index("{") :])
    assert envelope["code"] == "internal_error"
    assert envelope["message"] == "the request failed for a reason the server did not anticipate"
    assert envelope["trace_id"]
    assert "a genuine bug" not in reported.text
