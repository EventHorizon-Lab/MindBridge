"""The typed MCP tool surface over one local memory boundary."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from typing import Annotated, Any, ParamSpec, TypeVar, cast
from uuid import uuid4

from mcp.server import MCPServer
from mcp.server.context import CallNext, HandlerResult, ServerMiddleware, ServerRequestContext
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import AwareDatetime, BaseModel, Field, JsonValue, StringConstraints

from mindbridge import Memory
from mindbridge.api.content import (
    Content,
    ContextBudgetInput,
    Limit,
    content_input,
    context_budget,
)
from mindbridge.api.messages import error_message
from mindbridge.exceptions import MindBridgeError, ValidationError
from mindbridge.types import (
    AbstentionReason,
    AssetRef,
    ContextBudget,
    ContextBundle,
    ContextConflict,
    ContextUnknown,
    ContextUnknownKind,
    FaceObservation,
    IdentityErasure,
    IdentityProfile,
    MemoryCapabilities,
    MemoryContext,
    MemoryRecord,
    MemoryType,
    Modality,
    ObservationContext,
    ProvisionalActor,
    RetrievalScope,
    RetrievalTrace,
    SearchHit,
    SpeakerSegment,
)

_LOGGER = logging.getLogger(__name__)
_Identifier = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S(?:.*\S)?$")]
_Cursor = Annotated[str, StringConstraints(min_length=1)]
# The advertised tool defaults are read from the SDK value, so the prose and the schema cannot
# drift from `ContextBudget`.
_BUDGET = ContextBudget()
_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_NON_IDEMPOTENT_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
_IDEMPOTENT_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
# Reinforcement accumulates: each call raises `access_count` and moves the ranking factor, so a
# repeated call is not the same as one call and the honest hint is `idempotent_hint=False`.
_COUNTED_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
_DELETE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)
# Erasure is destructive and, unlike a delete, cannot be repeated: the second call reports the
# person as unknown, so a retry after a dropped response is not the same call.
_ERASE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=False,
)
_TOOL_ARGUMENTS = {
    "add_memory": frozenset(
        {"content", "occurred_at", "occurred_end", "metadata", "memory_type", "context"}
    ),
    "search_memories": frozenset(
        {
            "query",
            "limit",
            "memory_type",
            "reference_at",
            "occurred_from",
            "occurred_until",
            "scope",
            "explain",
        }
    ),
    "ask_memory": frozenset({"question", "limit", "memory_type", "reference_at", "scope"}),
    "compile_context": frozenset({"goal", "budget", "reference_at", "scope"}),
    "get_memory": frozenset({"memory_id"}),
    "list_memories": frozenset({"limit", "cursor"}),
    "delete_memory": frozenset({"memory_id"}),
    "analyze_speech": frozenset({"memory_id"}),
    "analyze_faces": frozenset({"memory_id"}),
    "register_speaker": frozenset({"speaker_id", "name", "relationship"}),
    "register_identity": frozenset({"identity_id", "name", "relationship"}),
    "get_identity": frozenset({"identity_id"}),
    "unlink_identity": frozenset({"alias_id"}),
    "forget_identity": frozenset({"identity_id"}),
    "reinforce_memories": frozenset({"memory_ids"}),
}
# Prose the tool schemas share. MCP publishes these verbatim, so an agent reads them before it
# guesses at frames, units, or media sources.
_CONTENT_DESCRIPTION = (
    "The memory content: a non-blank string, or 1 through 16 ordered parts (`input_text`,"
    " `input_image`, `input_file`). Media must arrive as a base64 `data:` URL or as a `file_id`"
    " already stored in this data directory; remote URLs, local paths, and `file:` URLs are"
    " rejected, so fetch media before calling."
)
_QUERY_DESCRIPTION = (
    "What to look for: a non-blank string, or 1 through 16 ordered parts for a mixed-modal query"
    " (an image plus words). Media follows the same base64 `data:` URL or stored `file_id` rule as"
    " `add_memory`."
)
_MEMORY_TYPE_FILTER_DESCRIPTION = (
    "Restrict results to `semantic`, `episodic`, or `procedural` memories. Null considers every"
    " type; a wrong guess here is a common cause of an empty result."
)
_REFERENCE_AT_DESCRIPTION = (
    "The timezone-aware instant that relative language in the query ('yesterday', 'last week')"
    " is resolved against, and the instant recency is scored from. Defaults to now."
)
_SCOPE_DESCRIPTION = (
    "Optional retrieval filter. `valid_at` selects world validity and `known_at` selects the"
    " transaction version known at that time. `near` and `radius_m` must be supplied together:"
    " `near` is a pose in a named coordinate frame (`frame_id`, `anchor` of observer or subject,"
    " metres for `x`/`y`/`z`, `orientation_xyzw` as a unit quaternion) and `radius_m` is metres."
    " Spatial filtering only matches memories stored with a spatial context in the same frame and"
    " anchor, so it excludes every plain text memory."
)
_GOAL_DESCRIPTION = (
    "The task the context is for, phrased the way the memories would be worded: a non-blank"
    " string, or 1 through 16 ordered parts. It is the retrieval query as well as the bundle's"
    " heading, so state the goal rather than naming a topic."
)
_BUDGET_DESCRIPTION = (
    "What this compilation may spend. `max_chars` (1 through 65,536) caps the evidence text and"
    " `max_items` (1 through 100) the number of memories; `memory_types` keeps only the named"
    " types; `min_confidence` drops typed memories below it, counting an untyped record as 1.0;"
    " `freshness_seconds` keeps only memories anchored within that many seconds of the reference"
    " clock; `max_latency_ms` is a deadline after which optional work is skipped rather than a"
    " timeout that aborts, and the bundle reports `elapsed_ms` and `deadline_exceeded`. Null uses"
    f" the defaults, which are {_BUDGET.max_chars:,} characters and {_BUDGET.max_items} items"
    " with no deadline."
)
_MEMORY_ID_DESCRIPTION = (
    "The `id` a previous `add_memory`, `search_memories`, or `list_memories` result returned."
)
# Prose the identity tool schemas share. An identity ID and a memory ID are different vocabularies,
# and passing one where the other belongs is the mistake these strings exist to prevent.
_SPEAKER_ID_DESCRIPTION = (
    "A `speaker_id` from an `analyze_speech` segment. It names one recognized voice, never a"
    " stored memory."
)
_IDENTITY_ID_DESCRIPTION = (
    "An `identity_id` from `analyze_faces` or `analyze_speech`, or a `speaker_id`: a merged alias"
    " resolves to the person it was bound into. It never names a stored memory."
)
_PERSON_NAME_DESCRIPTION = (
    "What to call this person, as a non-blank name. Registering again replaces the stored name."
)
_RELATIONSHIP_DESCRIPTION = (
    "How this person relates to the caller ('sister', 'colleague'). Null leaves any recorded"
    " relationship as it is rather than clearing it."
)
# The error envelope MCP shares with REST.
_ERROR_FIELDS = frozenset(
    {"code", "reason", "retryable", "stage", "subject", "message", "trace_id", "issues"}
)


def _error_codes(root: type[MindBridgeError]) -> frozenset[str]:
    codes = {root.code}
    for subclass in root.__subclasses__():
        codes |= _error_codes(subclass)
    return frozenset(codes)


# Derived rather than listed. A hand-maintained set silently loses every new exception class to the
# middleware's `validation_error` overwrite, which is how `model_output_truncated` was destroyed.
_STABLE_ERROR_CODES = _error_codes(MindBridgeError) | {"internal_error"}

_P = ParamSpec("_P")
_T = TypeVar("_T")


class AssetResult(BaseModel):
    id: str
    modality: Modality
    media_type: str
    size_bytes: int
    sha256: str
    name: str | None = None


class MemoryResult(BaseModel):
    id: str
    content: str
    modality: Modality
    memory_type: MemoryType
    assets: tuple[AssetResult, ...]
    created_at: AwareDatetime
    occurred_at: AwareDatetime | None
    occurred_end: AwareDatetime | None
    metadata: dict[str, JsonValue]
    context: MemoryContext | None = None
    place_id: str | None = None
    forgotten_at: AwareDatetime | None = None


class SearchHitResult(MemoryResult):
    score: Annotated[float, Field(ge=0.0, le=1.0)]


class PageResult(BaseModel):
    items: tuple[MemoryResult, ...]
    next_cursor: str | None = None


class SearchResult(BaseModel):
    hits: tuple[SearchHitResult, ...]
    # Null unless the call asked for it, so the default result keeps the shape agents already read.
    trace: RetrievalTrace | None = None


class AnswerResponse(BaseModel):
    answer: str
    hits: tuple[SearchHitResult, ...]
    abstained: bool
    abstention_reason: AbstentionReason | None


class DeleteResult(BaseModel):
    deleted: bool


# The embodied and identity results carry the public dataclasses directly. Pydantic derives their
# schema from the same definition the SDK returns, so a field cannot drift between the two.
# `forget_identity` wraps rather than returning `IdentityErasure` directly because the MCP SDK
# cannot build a schema for a slotted dataclass as a top-level return type: it reads the slot
# descriptors as defaults, rejects them, and silently publishes no output schema at all.
class SpeechResult(BaseModel):
    segments: tuple[SpeakerSegment, ...]


class FacesResult(BaseModel):
    observations: tuple[FaceObservation, ...]


class IdentityResult(BaseModel):
    identity: IdentityProfile | None


class RegisterResult(BaseModel):
    registered: bool


class UnlinkResult(BaseModel):
    restored_identity_id: str | None


class ForgetResult(BaseModel):
    erasure: IdentityErasure


class ReinforceResult(BaseModel):
    reinforced: int


class ContextBudgetResult(BaseModel):
    max_chars: int
    max_items: int
    memory_types: tuple[MemoryType, ...] | None
    min_confidence: float
    freshness_seconds: float | None
    max_latency_ms: int | None


class ContextConflictResult(BaseModel):
    lineage_id: str
    subject: str | None
    predicate: str | None
    values: tuple[str, ...]
    memory_ids: tuple[str, ...]


class ProvisionalActorResult(BaseModel):
    """A recognized person in the evidence whom no visible naming assertion names."""

    identity_id: str
    memory_ids: tuple[str, ...]


class ContextUnknownResult(BaseModel):
    kind: ContextUnknownKind
    detail: str


class ContextBundleResult(BaseModel):
    goal: str
    reference_at: AwareDatetime
    budget: ContextBudgetResult
    # An unnamed person present in the evidence is reported here beside the ranked entity
    # hits, labelled, rather than omitted: what an agent may say depends on knowing they are
    # there and that nobody has named them.
    actors: tuple[SearchHitResult | ProvisionalActorResult, ...]
    relationships: tuple[SearchHitResult, ...]
    scene: tuple[SearchHitResult, ...]
    episodes: tuple[SearchHitResult, ...]
    facts: tuple[SearchHitResult, ...]
    procedures: tuple[SearchHitResult, ...]
    affect: tuple[SearchHitResult, ...]
    traits: tuple[SearchHitResult, ...]
    conflicts: tuple[ContextConflictResult, ...]
    unknowns: tuple[ContextUnknownResult, ...]
    occurred_from: AwareDatetime | None
    occurred_until: AwareDatetime | None
    frames: tuple[str, ...]
    places: tuple[str, ...]
    omitted: int
    chars: int
    elapsed_ms: int
    deadline_exceeded: bool
    rendered: str


def build_mcp_server(
    memory: Memory,
    *,
    identity_operations: bool = True,
) -> MCPServer[None]:
    """Expose the typed agent tool surface without taking ownership of ``memory``.

    Every tool is one call on the injected ``Memory``: the embodied and identity tools are
    reachable here for the same reason the common-path tools are, because this server runs in the
    process that holds it.

    ``identity_operations`` is the host's choice about naming and erasing people. It defaults to
    True, which is the published fifteen-tool surface. Pass False and the five identity tools are
    not registered at all -- naming, reading, unlinking and erasing a person stay with the
    process that owns the memory -- and the server instructions say so, because a tool an agent
    cannot see is better than one it must be trusted not to call.
    """
    server: MCPServer[None] = MCPServer(
        "mindbridge",
        title="MindBridge Memory",
        description="Fast local memory for agents.",
        # Read once here because MCP fixes `instructions` at construction: an agent learns the
        # configured composition when it connects instead of spending a tool call or discovering
        # a missing backend by failing. `/healthz` reports the same view per request, so a
        # composition swapped behind a long-lived server stays visible there.
        instructions=_instructions(memory.capabilities, identity_operations=identity_operations),
        version="0.2.0",
        middleware=[cast(ServerMiddleware[Any], _strict_tool_arguments)],
    )

    @server.tool(annotations=_IDEMPOTENT_WRITE)
    @_stable_errors
    def add_memory(
        content: Annotated[Content, Field(description=_CONTENT_DESCRIPTION)],
        occurred_at: Annotated[
            AwareDatetime | None,
            Field(
                description=(
                    "When the remembered event happened, as a timezone-aware timestamp. Omit for a"
                    " timeless fact; without it the memory never matches a search occurrence"
                    " window."
                )
            ),
        ] = None,
        occurred_end: Annotated[
            AwareDatetime | None,
            Field(
                description=(
                    "End of the remembered interval, timezone-aware. Requires `occurred_at` and"
                    " must be later than it."
                )
            ),
        ] = None,
        metadata: Annotated[
            dict[str, JsonValue] | None,
            Field(
                description=(
                    "Application-owned JSON stored verbatim with the record, at most 262,144 UTF-8"
                    " bytes serialized. It is not searched, and it is not an access-control or"
                    " isolation boundary."
                )
            ),
        ] = None,
        memory_type: Annotated[
            MemoryType,
            Field(
                description=(
                    "`semantic` for a standing fact, `episodic` for something that happened,"
                    " `procedural` for how to do something. Search can filter on it."
                )
            ),
        ] = MemoryType.SEMANTIC,
        context: Annotated[
            ObservationContext | None,
            Field(
                description=(
                    "Typed provenance for this observation: basis, source ID, confidence, validity"
                    " window, and optional spatial pose. Omit it unless the caller actually knows"
                    " these; `scope` filters at search time only match memories that carry them."
                )
            ),
        ] = None,
    ) -> MemoryResult:
        """Store one memory and return its stable record.

        Call this to persist something the agent should be able to recall in a later session.
        Writes durable local state and may call the configured embedding backend. Storage is
        content-addressed, so re-adding identical content returns the existing record instead of a
        duplicate and retrying a call whose response was lost is safe.
        """
        record = memory.add(
            content_input(content),
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )
        return _memory_result(record)

    @server.tool(annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def search_memories(
        query: Annotated[Content, Field(description=_QUERY_DESCRIPTION)],
        limit: Annotated[
            Limit,
            Field(
                description=(
                    "Maximum hits to return, 1 through 100. It caps the result; it cannot raise a"
                    " candidate above the relevance gate, so it never turns an empty result into a"
                    " non-empty one."
                )
            ),
        ] = 10,
        memory_type: Annotated[
            MemoryType | None, Field(description=_MEMORY_TYPE_FILTER_DESCRIPTION)
        ] = None,
        reference_at: Annotated[
            AwareDatetime | None, Field(description=_REFERENCE_AT_DESCRIPTION)
        ] = None,
        occurred_from: Annotated[
            AwareDatetime | None,
            Field(
                description=(
                    "Keep only memories whose event interval overlaps at or after this"
                    " timezone-aware instant. Memories stored without `occurred_at` never match."
                )
            ),
        ] = None,
        occurred_until: Annotated[
            AwareDatetime | None,
            Field(
                description=(
                    "Exclusive upper bound of the same half-open overlap filter; with"
                    " `occurred_from` it must be strictly later."
                )
            ),
        ] = None,
        scope: Annotated[RetrievalScope | None, Field(description=_SCOPE_DESCRIPTION)] = None,
        explain: Annotated[
            bool,
            Field(
                description=(
                    "Also return `trace`: every candidate considered, with its score components"
                    " and, when it did not become a hit, the `rejected_by` reason"
                    " (`minimum_relevance`, `ambiguity`, `memory_type`, `occurrence_range`,"
                    " `missing_memory`, `stale_index`, or `limit`). Ask for it when `hits` is"
                    " empty or shorter than expected; `hits` itself is identical either way."
                )
            ),
        ] = False,
    ) -> SearchResult:
        """Rank stored memories against a query and return the matching records.

        Call this to find out what is stored before answering from your own context, or to gather
        evidence you will read yourself; use `ask_memory` when you want MindBridge to write the
        answer. Returns at most `limit` hits, best first, and never invents one: empty `hits` means
        nothing cleared the relevance gate, and `explain=true` reports which stage discarded each
        candidate. Stores no memory and is safe to retry, though it may cache a transcript for
        media it had to transcribe.
        """
        content = content_input(query)
        if not explain:
            hits = memory.search(
                content,
                limit=limit,
                memory_type=memory_type,
                reference_at=reference_at,
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                scope=scope,
            )
            return SearchResult(hits=tuple(_search_hit_result(hit) for hit in hits))
        traced = memory.search_with_trace(
            content,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
        )
        return SearchResult(
            hits=tuple(_search_hit_result(hit) for hit in traced.hits),
            trace=traced.trace,
        )

    @server.tool(annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def ask_memory(
        question: Annotated[
            Content,
            Field(
                description=(
                    "The question to answer, as a non-blank string or 1 through 16 ordered parts."
                    " It is also the retrieval query, so phrase it the way the memory would be"
                    " worded."
                )
            ),
        ],
        limit: Annotated[
            Limit,
            Field(
                description=(
                    "Maximum memories read as evidence, 1 through 100. More evidence costs more"
                    " model input and can be truncated by the configured evidence budget."
                )
            ),
        ] = 5,
        memory_type: Annotated[
            MemoryType | None, Field(description=_MEMORY_TYPE_FILTER_DESCRIPTION)
        ] = None,
        reference_at: Annotated[
            AwareDatetime | None, Field(description=_REFERENCE_AT_DESCRIPTION)
        ] = None,
        scope: Annotated[RetrievalScope | None, Field(description=_SCOPE_DESCRIPTION)] = None,
    ) -> AnswerResponse:
        """Answer a question using only the memories retrieved for it.

        Requires a generation backend in the host process: without one every call fails with
        `model_error/backend_not_configured`, so use `search_memories` if you are not sure one is
        configured. Returns the answer with the hits it used; `abstained` is true when the answerer
        reported no usable evidence, and the answer is then a fixed sentence rather than a guess.
        Stores no memory, but each call spends another generation, so retry only on a retryable
        error.
        """
        result = memory.ask(
            content_input(question),
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
        )
        return AnswerResponse(
            answer=result.answer,
            hits=tuple(_search_hit_result(hit) for hit in result.hits),
            abstained=result.abstained,
            abstention_reason=result.abstention_reason,
        )

    # The same annotation `search_memories` carries, because it is the same side effect: the
    # shared retrieval path may cache a transcript for spoken query media. Compiling stores no
    # memory, but it is not read-only, so the honest hint is the one search already publishes.
    @server.tool(annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def compile_context(
        goal: Annotated[Content, Field(description=_GOAL_DESCRIPTION)],
        budget: Annotated[ContextBudgetInput | None, Field(description=_BUDGET_DESCRIPTION)] = None,
        reference_at: Annotated[
            AwareDatetime | None, Field(description=_REFERENCE_AT_DESCRIPTION)
        ] = None,
        scope: Annotated[RetrievalScope | None, Field(description=_SCOPE_DESCRIPTION)] = None,
    ) -> ContextBundleResult:
        """Compile the stored memories that bear on a goal into one bounded context bundle.

        Prefer this tool whenever you need context to act on: it returns the actors, facts,
        episodes, procedures, affect cues and traits that matter, each with its provenance, inside
        a budget you declare, plus `rendered` text you can read straight into your own reasoning.
        Every non-empty section gets a slot before any section gets a second one, so a small
        budget still describes the whole scene, and `omitted` counts what did not fit. Lineage
        disagreements are reported in `conflicts` and never resolved for you. `ask_memory` remains
        a convenience for when you want one grounded sentence instead of the evidence.

        Sections it cannot fill and evidence a bound excluded are named in `unknowns`, so a thin
        bundle says why it is thin instead of looking like an empty store.

        Calls no generation model and stores no memory, so it needs no generation backend and is
        safe to retry. Like `search_memories`, and through the same code path, it may cache a
        transcript for spoken query media -- a cache of your own input, never a new memory --
        which is why it is not annotated read-only.
        """
        return _bundle_result(
            memory.compile(
                content_input(goal),
                budget=context_budget(budget),
                reference_at=reference_at,
                scope=scope,
            )
        )

    @server.tool(annotations=_READ_ONLY)
    @_stable_errors
    def get_memory(
        memory_id: Annotated[_Identifier, Field(description=_MEMORY_ID_DESCRIPTION)],
    ) -> MemoryResult:
        """Read one memory by its stable identifier.

        Call this to re-read a record whose ID you already hold, not to discover memories. Fails
        with `memory_not_found` when the ID never existed or was deleted, and that verdict is
        permanent, so retrying an unknown ID never starts working. Changes nothing.
        """
        return _memory_result(memory.get(memory_id))

    @server.tool(annotations=_READ_ONLY)
    @_stable_errors
    def list_memories(
        limit: Annotated[
            Limit,
            Field(description="Maximum records in this page, 1 through 100."),
        ] = 100,
        cursor: Annotated[
            _Cursor | None,
            Field(
                description=(
                    "The `next_cursor` from the previous page, passed back unchanged. Its contents"
                    " are opaque; a null `next_cursor` means the listing is complete."
                )
            ),
        ] = None,
    ) -> PageResult:
        """Page through every stored memory, newest first.

        Call this to inventory what is stored when you have no query; use `search_memories` when
        you know what you are looking for. Changes nothing and is safe to retry, but a page is a
        snapshot: memories added or deleted while you page can appear twice or not at all.
        """
        page = memory.list(limit=limit, cursor=cursor)
        return PageResult(
            items=tuple(_memory_result(record) for record in page.items),
            next_cursor=page.next_cursor,
        )

    @server.tool(annotations=_DELETE)
    @_stable_errors
    def delete_memory(
        memory_id: Annotated[_Identifier, Field(description=_MEMORY_ID_DESCRIPTION)],
    ) -> DeleteResult:
        """Permanently delete one memory and any media only it referenced.

        Call this when the caller asked for the memory to be forgotten. There is no undo and no
        tombstone to read back. It is idempotent, so retrying is safe: `deleted` reports whether a
        record existed, and a second call on the same ID returns `false`.
        """
        return DeleteResult(deleted=memory.delete(memory_id))

    @server.tool(annotations=_COUNTED_WRITE)
    @_stable_errors
    def reinforce_memories(
        memory_ids: Annotated[
            tuple[_Identifier, ...],
            Field(
                min_length=1,
                max_length=100,
                description=(
                    "IDs of memories that were actually useful, from a previous result. Duplicates"
                    " count once; IDs that no longer exist are skipped."
                ),
            ),
        ],
    ) -> ReinforceResult:
        """Record that these memories were useful, so retrieval favours them later.

        Call this after a memory helped you answer, and only then: it is the sole way to move the
        ranker's reinforcement factor away from its neutral value. It writes durable state and its
        effect accumulates, so two calls count twice; do not repeat one whose outcome you are
        unsure of. Content, timestamps, and metadata are untouched. Returns how many of the
        supplied IDs existed.
        """
        return ReinforceResult(reinforced=memory.reinforce(memory_ids))

    _register_embodied_tools(server, memory)
    _register_identity_tools(server, memory, enabled=identity_operations)
    return server


def _register_embodied_tools(server: MCPServer[None], memory: Memory) -> None:
    """Register the embodied tools on an existing server.

    Split from `build_mcp_server` only to keep each registration function readable; every set
    dispatches to the same injected memory.
    """

    @server.tool(annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def analyze_speech(
        memory_id: Annotated[_Identifier, Field(description=_MEMORY_ID_DESCRIPTION)],
    ) -> SpeechResult:
        """Transcribe one stored memory's audio and video, and resolve who spoke.

        Call this to learn what was said in a memory whose content is speech, or which person
        said it. Segments are returned per asset in time order and cover only this one memory's
        audio and video assets; there is no page cursor because the bound is the stored media. A
        memory with no audio or video returns no segments rather than failing.

        Side effects: caches the transcript and records the voice as identity evidence, so an
        identical retry is safe but may resolve a speaker an earlier call left unnamed.
        `speaker_id` is stable and accepted by `register_speaker` and `get_identity`. Fails with
        `model_error` when the configured transcription backend cannot recognize speakers.
        """
        return SpeechResult(segments=memory.speech(memory_id))

    @server.tool(annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def analyze_faces(
        memory_id: Annotated[_Identifier, Field(description=_MEMORY_ID_DESCRIPTION)],
    ) -> FacesResult:
        """Detect the faces in one stored memory's images and video, and resolve who was seen.

        Call this to learn which people appear in a visual memory. Observations cover only this
        one memory's image and video assets, each with a frame-normalized bounding box and, for
        video, the offset it was observed at; there is no page cursor because the bound is the
        stored media. A memory with no image or video returns no observations rather than failing.

        Side effects: records the face as identity evidence and may bind it to a voice already
        seen with it, so an identical retry is safe but may resolve an identity an earlier call
        left unnamed; `unlink_identity` reverses a binding. `identity_id` is stable and accepted
        by `register_identity` and `get_identity`. Fails with `model_error` when no face backend
        is configured or when it does not support the stored modality.
        """
        return FacesResult(observations=memory.faces(memory_id))


def _register_identity_tools(
    server: MCPServer[None],
    memory: Memory,
    *,
    enabled: bool,
) -> None:
    """Register the identity tools, unless the host withheld them.

    Naming a person, reading who they are, splitting a wrong merge and erasing them are host
    authority: every one of them is recorded in the operation log and, apart from erasure,
    reversible. A host that does not want an agent holding that authority builds the server
    with `identity_operations=False`, and then these five tools are never registered at all --
    an agent cannot call a tool it cannot see, which is stronger than trusting it not to.
    """
    if not enabled:
        return

    @server.tool(annotations=_IDEMPOTENT_WRITE)
    @_stable_errors
    def register_speaker(
        speaker_id: Annotated[_Identifier, Field(description=_SPEAKER_ID_DESCRIPTION)],
        name: Annotated[_Identifier, Field(description=_PERSON_NAME_DESCRIPTION)],
        relationship: Annotated[
            _Identifier | None, Field(description=_RELATIONSHIP_DESCRIPTION)
        ] = None,
    ) -> RegisterResult:
        """Name the person behind one `speaker_id` from `analyze_speech`.

        Naming a person is host authority, exercised through you: this asserts, on the host's
        behalf, that this voice belongs to this name. Call it once a conversation establishes
        who a recognized voice belongs to, and not to record a guess. The assertion is a
        versioned memory record, it is written to the operation log, and it is reversible:
        naming again supersedes it and a rollback restores the previous name.

        Repeating the same name changes nothing, so retrying is safe; omitting `relationship`
        leaves any recorded relationship intact rather than clearing it. Naming a speaker also
        rewrites the stored memories that quote them, so later searches can find the person by
        name. Fails with `speaker_not_found` when the ID is not a recognized speaker.
        """
        memory.register_speaker(speaker_id, name, relationship=relationship)
        return RegisterResult(registered=True)

    @server.tool(annotations=_IDEMPOTENT_WRITE)
    @_stable_errors
    def register_identity(
        identity_id: Annotated[_Identifier, Field(description=_IDENTITY_ID_DESCRIPTION)],
        name: Annotated[_Identifier, Field(description=_PERSON_NAME_DESCRIPTION)],
        relationship: Annotated[
            _Identifier | None, Field(description=_RELATIONSHIP_DESCRIPTION)
        ] = None,
    ) -> RegisterResult:
        """Name the person behind one `identity_id` from `analyze_faces` or `analyze_speech`.

        Naming a person is host authority, exercised through you: this asserts, on the host's
        behalf, that this recognized person is called this. Call it for an identity that may
        have been seen as well as heard, and only once something establishes the name; use
        `register_speaker` only for a voice-only speaker ID. The assertion is a versioned
        memory record, it is written to the operation log, and it is reversible: naming again
        supersedes it and a rollback restores the previous name.

        Repeating the same name changes nothing, so retrying is safe; omitting `relationship`
        leaves any recorded relationship intact rather than clearing it. Fails with
        `identity_not_found` when the ID is not a recognized identity.
        """
        memory.register_identity(identity_id, name, relationship=relationship)
        return RegisterResult(registered=True)

    @server.tool(annotations=_READ_ONLY)
    @_stable_errors
    def get_identity(
        identity_id: Annotated[_Identifier, Field(description=_IDENTITY_ID_DESCRIPTION)],
    ) -> IdentityResult:
        """Read the name and relationship recorded for one identity or speaker ID.

        Call this to turn an `identity_id` or `speaker_id` from a face or speech result into who
        the person is. Merge aliases are followed, so an ID that was bound into another identity
        resolves to that person. Returns `identity: null` when the ID exists but nothing has been
        registered for it; reads nothing else and changes nothing.
        """
        return IdentityResult(identity=memory.identity(identity_id))

    @server.tool(annotations=_DELETE)
    @_stable_errors
    def unlink_identity(
        alias_id: Annotated[
            _Identifier,
            Field(
                description=(
                    "The identity that was merged into another and should become its own again:"
                    " the `identity_id` the person says is not them."
                )
            ),
        ],
    ) -> UnlinkResult:
        """Reverse one face-and-voice merge, restoring `alias_id` as its own identity.

        Call this when a person says the system has confused them with someone else. Returns the
        restored identity ID, or `restored_identity_id: null` when the merge cannot be reversed
        because no record names which modality was contributed, and repeating the call is safe
        because a reversed merge stays reversed.

        Side effects: the restored identity keeps no name or relationship and the pair's
        accumulated evidence is reset. This does not suppress the pair: if the same voice and face
        keep appearing together they will be corroborated and merged again.
        """
        return UnlinkResult(restored_identity_id=memory.unlink_identity(alias_id))

    @server.tool(annotations=_ERASE)
    @_stable_errors
    def forget_identity(
        identity_id: Annotated[_Identifier, Field(description=_IDENTITY_ID_DESCRIPTION)],
    ) -> ForgetResult:
        """Erase one person: their biometric template, their aliases, and their indexed name.

        Call this when a person asks to be forgotten, and `delete_memory` to forget an event
        instead. Memories, their content and their media survive: a transcript keeps its words
        with the speaker attribution dropped, because forgetting a person is not forgetting the
        evening. The result counts what was destroyed, as the audit record for the request.

        Side effects: this destroys the face and voice templates and cannot be undone, so a later
        encounter mints a fresh unnamed identity for the same person rather than recognizing them
        as forgotten. Retrying is not safe in the sense that matters here: the second call fails
        with `identity_not_found` because the person is already gone, which is not evidence that
        the first call failed.
        """
        return ForgetResult(erasure=memory.forget_identity(identity_id))


def _stable_errors(
    operation: Callable[_P, _T],
) -> Callable[_P, _T]:
    @wraps(operation)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return operation(*args, **kwargs)
        except Exception as error:
            raise ToolError(_error_json(error)) from None

    # The MCP SDK publishes `__doc__` verbatim, and CPython 3.13 strips a docstring's common
    # indentation at compile time while earlier versions keep it. Without this, the published
    # tool description -- which is public contract -- would differ by interpreter.
    guarded.__doc__ = None if operation.__doc__ is None else inspect.cleandoc(operation.__doc__)
    return guarded


async def _strict_tool_arguments(
    context: ServerRequestContext[Any, Any],
    call_next: CallNext,
) -> HandlerResult:
    params = context.params or {}
    if context.method == "tools/call":
        allowed = _TOOL_ARGUMENTS.get(str(params.get("name", "")))
        if allowed is None:
            return _error_result("validation_error", "tool does not exist", reason="unknown_field")
        arguments = params.get("arguments")
        if isinstance(arguments, Mapping):
            unknown = set(arguments).difference(allowed)
            if unknown:
                return _error_result(
                    "validation_error",
                    "tool arguments contain unknown fields",
                    reason="unknown_field",
                    issues=[
                        {
                            "location": ["arguments", name],
                            "message": "Extra inputs are not permitted",
                            "type": "extra_forbidden",
                        }
                        for name in sorted(unknown)
                    ],
                )
    result = await call_next(context)
    if (
        context.method == "tools/call"
        and isinstance(result, dict)
        and result.get("isError") is True
    ):
        # A tool body signals failure by raising, and the MCP runtime prefixes that message with
        # "Error executing tool <name>: ". Re-emit the envelope on its own so one bare `json.loads`
        # of the text works for every failure, including the recoverable ones an agent parses to
        # decide whether to retry.
        envelope = _stable_envelope(result)
        return _envelope_result(
            envelope
            if envelope is not None
            else _envelope("validation_error", "tool arguments are invalid", reason="input_invalid")
        )
    return result


def _memory_result(record: MemoryRecord) -> MemoryResult:
    return MemoryResult(
        id=record.id,
        content=record.content,
        modality=record.modality,
        memory_type=record.memory_type,
        assets=tuple(_asset_result(asset) for asset in record.assets),
        created_at=record.created_at,
        occurred_at=record.occurred_at,
        occurred_end=record.occurred_end,
        metadata=cast(dict[str, JsonValue], dict(record.metadata)),
        context=record.context,
        place_id=record.place_id,
        forgotten_at=record.forgotten_at,
    )


def _search_hit_result(hit: SearchHit) -> SearchHitResult:
    return SearchHitResult(
        id=hit.id,
        content=hit.content,
        modality=hit.modality,
        memory_type=hit.memory_type,
        assets=tuple(_asset_result(asset) for asset in hit.assets),
        score=hit.score,
        created_at=hit.created_at,
        occurred_at=hit.occurred_at,
        occurred_end=hit.occurred_end,
        metadata=cast(dict[str, JsonValue], dict(hit.metadata)),
        context=hit.context,
        place_id=hit.place_id,
        forgotten_at=hit.forgotten_at,
    )


def _bundle_result(bundle: ContextBundle) -> ContextBundleResult:
    """Publish `ContextBundle.document()`, the same projection REST and the CLI publish.

    Only the sections need translating -- every other value is already what the tool schema
    declares -- so a section added to the bundle reaches this tool without being named here.
    """
    document = bundle.document()
    for name, value in document.items():
        if isinstance(value, tuple):
            document[name] = tuple(_bundle_entry(entry) for entry in value)
    return ContextBundleResult.model_validate(document)


def _bundle_entry(entry: object) -> object:
    """Translate one entry of one bundle section; `frames` and `places` carry plain strings."""
    if isinstance(entry, SearchHit):
        return _search_hit_result(entry)
    # A recognized person in the evidence whom no visible naming assertion names travels in
    # `actors` beside the ranked hits.
    if isinstance(entry, ProvisionalActor):
        return ProvisionalActorResult(identity_id=entry.identity_id, memory_ids=entry.memory_ids)
    if isinstance(entry, ContextConflict):
        return ContextConflictResult(
            lineage_id=entry.lineage_id,
            subject=entry.subject,
            predicate=entry.predicate,
            values=entry.values,
            memory_ids=entry.memory_ids,
        )
    if isinstance(entry, ContextUnknown):
        return ContextUnknownResult(kind=entry.kind, detail=entry.detail)
    return entry


def _instructions(
    capabilities: MemoryCapabilities,
    *,
    identity_operations: bool = True,
) -> str:
    """Render the declared capability view an agent reads when it connects.

    The document is `MemoryCapabilities.document()` verbatim -- the same object `/healthz` serves
    and `mindbridge doctor` prints -- rather than a prose subset that used to omit every model
    identity and every non-embedding modality set. MCP fixes `instructions` at construction, so
    this is a snapshot of the composition the server was built with; `/healthz` is the live one.
    """
    return "\n".join(
        (
            "MindBridge is local multimodal memory for one physical data directory.",
            "Prefer compile_context for task-ready context; ask_memory remains a convenience for"
            " one grounded sentence.",
            "Cognitive forgetting, consolidation and operation rollback have no tool here; they"
            " stay with the process that owns this memory.",
            (
                "Naming a person is host authority exercised through you: register_speaker and"
                " register_identity assert a name on the host's behalf, are recorded in the"
                " operation log, and are reversible."
                if identity_operations
                else "Identity operations are not exposed here: naming, reading, unlinking and"
                " erasing a person stay with the process that owns this memory."
            ),
            "Declared capabilities of this composition, the same JSON document GET /healthz"
            " serves. `operations` names the optional operations these backends can serve:",
            json.dumps(capabilities.document(), indent=2, sort_keys=True),
        )
    )


def _asset_result(asset: AssetRef) -> AssetResult:
    if (
        asset.modality is None
        or asset.media_type is None
        or asset.size_bytes is None
        or asset.sha256 is None
    ):
        raise ValidationError("stored asset metadata is incomplete")
    return AssetResult(
        id=asset.id,
        modality=asset.modality,
        media_type=asset.media_type,
        size_bytes=asset.size_bytes,
        sha256=asset.sha256,
        name=asset.name,
    )


def _error_json(error: Exception) -> str:
    return json.dumps(_tool_error(error), separators=(",", ":"))


def _tool_error(error: Exception) -> dict[str, object]:
    if isinstance(error, MindBridgeError):
        return _envelope(
            error.code,
            error_message(error),
            reason=error.reason,
            retryable=error.retryable,
            stage=error.stage,
            # MCP runs in the owner process as the same user, so a subject naming local state is
            # already visible to the caller. Unauthenticated REST withholds the same value.
            subject=error.subject,
        )
    _LOGGER.exception("unhandled MCP tool error")
    return _envelope(
        "internal_error",
        "the memory operation failed unexpectedly",
        reason="unexpected",
    )


def _envelope(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    retryable: bool = False,
    stage: str | None = None,
    subject: str | None = None,
    issues: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    return {
        "code": code,
        "reason": reason,
        "retryable": retryable,
        "stage": stage,
        "subject": subject,
        "message": message,
        "trace_id": f"trace_{uuid4().hex}",
        "issues": list(issues),
    }


def _error_result(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    issues: Sequence[Mapping[str, object]] = (),
) -> CallToolResult:
    return _envelope_result(_envelope(code, message, reason=reason, issues=issues))


def _envelope_result(envelope: Mapping[str, object]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(dict(envelope)))],
        is_error=True,
    )


def _stable_envelope(result: Mapping[str, object]) -> dict[str, object] | None:
    """Recover the envelope a tool raised, ignoring any prefix the MCP runtime added to it."""
    content = result.get("content")
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
            continue
        text = item["text"]
        try:
            envelope: object = json.loads(text[text.index("{") :])
        except (ValueError, json.JSONDecodeError):
            continue
        if (
            isinstance(envelope, dict)
            and set(envelope) == _ERROR_FIELDS
            and envelope.get("code") in _STABLE_ERROR_CODES
            and isinstance(envelope.get("message"), str)
        ):
            return envelope
    return None
