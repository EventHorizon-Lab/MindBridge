"""Official MCP adapter for the shared MindBridge memory use cases."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Annotated, Any, TypeVar, get_type_hints

from mcp.server import MCPServer
from mcp.server.context import CallNext, HandlerResult, ServerMiddleware, ServerRequestContext
from mcp_types import CallToolResult, TextContent, ToolAnnotations

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
    ObservationProcessingJobView,
    ObservationReceipt,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
    RememberResult,
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

_RequestT = TypeVar("_RequestT", bound=ContractModel)
_ResultT = TypeVar("_ResultT", bound=ContractModel)


def _flattened(
    model: type[_RequestT],
    handler: Callable[[_RequestT], Awaitable[_ResultT]],
) -> Callable[..., Awaitable[_ResultT]]:
    """Publish `model`'s own fields as the tool's arguments instead of nesting them.

    MCP derives a tool's input schema from its function signature, so a handler that takes one
    model parameter publishes `{"request": {...}}` and rejects the flat object a caller
    reaches for first. Synthesizing the signature from `model.model_fields` keeps the contract
    single-sourced -- constraints, defaults, nested models, and cross-field validators all
    still come from the same Pydantic model REST and Python use -- while the published
    arguments match every other face of the API.

    Two consequences of the fields being top-level rather than nested. Unknown keys reach an
    argument model MCP builds on its own base, which ignores them instead of rejecting them;
    `_reject_unknown_arguments` restores that check before validation runs, because dropping
    a misspelled `mode` or `memory_ids` silently answers a different question than the one
    asked. And MCP JSON-decodes any top-level argument whose annotation is not exactly `str`,
    which every `X | None` string field trips: `idempotency_key="null"` arrives as None, and
    `correction_summary="123"` fails as a non-string. That one is not fixable from here -- the
    check reads the field's own annotation -- so callers must send those values as JSON
    strings only when they mean them as JSON.
    """
    parameters = [
        # `Annotated[..., field]` already carries requiredness and the default, including
        # `default_factory`; a `default=` here would only restate it, and wrongly for the
        # factory case, where `field.default` is `PydanticUndefined`.
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=Annotated[field.annotation, field],
        )
        for name, field in model.model_fields.items()
    ]
    if len(inspect.signature(handler).parameters) != 1:
        # MCP finds `Context` and resolved parameters through `__annotations__`, which cannot
        # describe the synthesized signature. Such a handler would register clean and fail on
        # its first call, so refuse it here instead.
        raise TypeError(f"{handler.__name__} must take exactly one contract parameter")

    async def tool(**fields: object) -> _ResultT:
        return await handler(model.model_validate(fields))

    # Resolved rather than reflected: `from __future__ import annotations` leaves every
    # handler's own annotations as strings, and MCP reads this signature verbatim when it
    # decides whether the tool has a structured output schema.
    tool.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters,
        return_annotation=get_type_hints(handler)["return"],
    )
    tool.__name__ = handler.__name__
    tool.__doc__ = handler.__doc__
    return tool


def _reject_unknown_arguments(
    fields_by_tool: Mapping[str, frozenset[str]],
) -> ServerMiddleware[Any]:
    """Fault an unknown tool argument instead of letting MCP drop it.

    `ContractModel` forbids extras, but flattening moves the fields out of it into an
    argument model MCP generates on a base whose config ignores them, and there is no
    supported way to tighten that model. Middleware sees the raw params before validation,
    which is early enough to reject the key rather than answer a different question with it.
    """

    async def reject_unknown_arguments(
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> HandlerResult:
        params = ctx.params if isinstance(ctx.params, dict) else {}
        if ctx.method == "tools/call":
            known = fields_by_tool.get(str(params.get("name", "")))
            arguments = params.get("arguments")
            if known is not None and isinstance(arguments, dict):
                unknown = sorted(set(arguments) - known)
                if unknown:
                    # A tool result rather than a raise: raising here becomes a protocol-level
                    # "Internal server error", which tells the caller nothing it can correct.
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=(
                                    f"unknown arguments: {', '.join(unknown)}. "
                                    f"This tool accepts: {', '.join(sorted(known))}."
                                ),
                            )
                        ],
                        is_error=True,
                    )
        return await call_next(ctx)

    return reject_unknown_arguments


def build_mcp_server(
    kernel: MemoryKernel,
    *,
    lifespan: _McpLifespan | None = None,
) -> MCPServer[None]:
    """Expose one memory kernel through typed, agent-friendly MCP tools."""
    fields_by_tool: dict[str, frozenset[str]] = {}
    server: MCPServer[None] = MCPServer(
        "mindbridge",
        title="MindBridge Memory",
        description="Evidence-grounded embodied Memory as a Service.",
        version="0.1.0",
        lifespan=lifespan,
        middleware=[_reject_unknown_arguments(fields_by_tool)],
    )

    def register(
        model: type[_RequestT],
        handler: Callable[[_RequestT], Awaitable[_ResultT]],
        annotations: ToolAnnotations,
    ) -> None:
        """Publish one tool and record the arguments it accepts, from the one model."""
        fields_by_tool[handler.__name__] = frozenset(model.model_fields)
        server.add_tool(_flattened(model, handler), annotations=annotations)

    async def memory_observe(request: ObserveRequest) -> ObservationReceipt:
        """Store one timestamped multimodal observation and return its durable job ID.

        Preconditions this tool cannot do for you: every `media_objects[*].uri` must already
        be readable at `s3://<bucket>/tenants/<tenant_id>/<key>`, with `sha256` and
        `size_bytes` matching those exact bytes. This is the first-party device path. An Agent
        holding content rather than stored media objects wants `memory_remember` instead.

        Rejected combinations: `ended_at` before `occurred_at`; a repeated `media_object_id`;
        a media `duration_ms` longer than the observation's own span; an identity span with
        `end_ms` past that span or `start_ms` after its own `end_ms`; `transcript` on anything
        but a `voice` identity; `visual_bbox_xyxy` on anything but a `face` identity, or one
        whose 0..1 normalized corners have no positive width and height.

        `status` returns `duplicate` when a retry matched an earlier `idempotency_key`.
        Deriving memory from raw media outlives this request, so no memory exists yet when
        this receipt returns: read the returned `processing_job_id` with `memory_job` until it
        reaches `succeeded`, then use the `memory_ids` it carries.
        """
        return await kernel.observe(request)

    register(ObserveRequest, memory_observe, _IDEMPOTENT_WRITE)

    async def memory_remember(request: RememberRequest) -> RememberResult:
        """Retain one explicit memory, preserving evidence and temporal provenance.

        Choose `memory_type` by the role the content will serve: `episodic` for something
        that happened at a time, `semantic` for a durable fact, `procedural` for how to do
        something, `prospective` for a future intention, `working` for short-lived task
        state, `perceptual` for a raw sensory detail.

        `ended_at` defaults to `occurred_at` and must not precede it. `evidence_ids` must
        already exist and must not repeat. Omitting `idempotency_key` derives one from the
        content, so an identical retry returns the same memory rather than a second copy --
        and says so: `status` is `created` only when this call is what stored it.
        """
        return await kernel.remember(request)

    register(RememberRequest, memory_remember, _IDEMPOTENT_WRITE)

    async def memory_recall(request: RecallRequest) -> RecallResult:
        """Recall relevant memories and return inspectable evidence with the answer.

        `query` needs `text`, `media_object_ids`, or both -- neither modality is privileged.

        `mode` selects what work is done. `answer` reasons over the retrieved memories and
        fills `answer`. `search` ranks and returns memories with `answer` left null. Use
        `enumerate` for count and timeline questions: it scans the complete structured-filter
        scope and verifies each candidate against original media, and fails with
        `enumeration_limit_exceeded` rather than silently truncating an oversized scope.

        For a grounded follow-up, pass IDs from a previous result in `memory_ids`; they become
        the strict candidate scope instead of a ranking hint. `filters` applies before
        ranking, and `occurred_before` must not precede `occurred_after`.
        """
        return await kernel.recall(request)

    register(RecallRequest, memory_recall, _READ_ONLY)

    async def memory_get(request: GetMemoryRequest) -> MemoryResult:
        """Read one tenant-owned memory by its stable identifier.

        Evidence arrives with the memory as short-lived signed URLs, so verifying an answer
        needs no second storage call. A memory erased through `memory_forget` fails as
        deleted rather than as missing, which distinguishes "was removed" from "never was".
        """
        return await kernel.get_memory(request.tenant_id, request.memory_id)

    register(GetMemoryRequest, memory_get, _READ_ONLY)

    async def memory_job(request: GetObservationJobRequest) -> ObservationProcessingJobView:
        """Read how far one observation's processing has got.

        `memory_observe` returns a `processing_job_id` because deriving memory from raw media
        outlives the request that submitted it. This resolves that ID. Once `state` is
        `succeeded`, `memory_ids` names exactly what the observation produced, so the derived
        memories can be read directly rather than searched for.

        `failed` settles this attempt, not the job: a stale-job sweep can retry it, and
        `attempt` counts how often it has been claimed. Following every intermediate state as
        a stream is REST-only; this tool answers one read at a time.
        """
        return await kernel.get_observation_job(request.tenant_id, request.job_id)

    register(GetObservationJobRequest, memory_job, _READ_ONLY)

    async def memory_feedback(request: FeedbackRequest) -> FeedbackReceipt:
        """Record a useful, wrong, missing, or correction signal for future recall.

        Which fields are required depends on `feedback_type`, and the wrong combination is
        rejected rather than ignored. `missing` reports that a recall returned nothing
        usable: it needs `recall_trace_id` -- the `trace_id` from that recall -- and must omit
        `memory_id`. `useful`, `wrong`, and `correction` each judge one memory and need
        `memory_id`. Only `correction` may carry `correction_summary`, and it must.

        A correction supersedes the original with a new version rather than editing it, so the
        receipt names both the `corrected_memory_id` and the original's resulting state.
        """
        return await kernel.record_feedback(request)

    register(FeedbackRequest, memory_feedback, _IDEMPOTENT_WRITE)

    async def memory_forget(request: ForgetRequest) -> ForgetReceipt:
        """Recoverably erase one exact memory or source observation and its derivatives.

        `target_type` decides what `target_id` names: `memory_record` erases that one memory,
        `observation` erases a source observation and everything derived from it. Idempotent
        -- a repeat returns the same tombstone rather than failing.

        Erasure propagates across storage and offline edge devices after this reply, so read
        `propagation_state` on the receipt instead of assuming `complete`. Confirming that it
        finished uses the REST deletion routes, which MCP does not mirror.
        """
        return await kernel.forget(request)

    register(ForgetRequest, memory_forget, _DESTRUCTIVE_WRITE)

    return server
