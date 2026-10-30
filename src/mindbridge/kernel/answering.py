"""The answer plane: grounding a question and generating from what was grounded.

Retrieval ranks; this module decides how much of the ranking to ground, whether a recall program
adds exhaustive reads beside it, how the question and its evidence are routed into the
generation backend, and how a streamed answer is validated.
"""

from __future__ import annotations

import builtins
from collections.abc import Generator, Sequence
from contextlib import closing, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from time import perf_counter
from typing import cast

from opentelemetry import trace
from opentelemetry.trace import Tracer

from mindbridge._telemetry import (
    ASYNC_QUEUE_TIME,
    EXPANSION_DROPPED_EDGES,
    EXPANSION_EDGES,
    EXPANSION_LINKED_ROWS,
    EXPANSION_ROWS,
    MODEL_TTFT,
    OPERATION_TTFT,
    RECALL_COMPLETE,
    RECALL_EXHAUSTIVE_ROWS,
    RECALL_FALLBACK,
    RECALL_NON_SELECTIVE_STEPS,
    RECALL_OPS,
    RECALL_QUERY_BUDGET_DROPPED,
    RECALL_QUERY_CANDIDATES,
    RECALL_QUERY_COUNT,
    RECALL_QUERY_FAILURES,
    RECALL_QUERY_GROUNDED,
    RECALL_QUERY_LIMITED,
    RECALL_QUERY_NEW_CANDIDATES,
    RECALL_QUERY_OMITTED,
    RECALL_QUERY_SELECTED,
    RECALL_REPLAN,
    RECALL_SHAPE,
    _record_generation_inputs,
    _record_retrieval_results,
    current_model_request_count,
    mark_model_requests,
)
from mindbridge.exceptions import MindBridgeError, ModelError, StorageError
from mindbridge.infrastructure.local.store import LocalStore, RecallRead, datetime_text
from mindbridge.kernel import grounding
from mindbridge.kernel.content import PreparedContent, memory_modality
from mindbridge.kernel.contracts import Backends, fallback_unsupported, require_audio_transcription
from mindbridge.kernel.derived import derived_text, has_stream_description, has_stream_transcript
from mindbridge.kernel.execution import Execution
from mindbridge.kernel.hydration import Hydrator
from mindbridge.kernel.lifecycle import Lifecycle, OperationAssets
from mindbridge.kernel.materialization import Materializer
from mindbridge.kernel.retrieval import RERANK_CANDIDATES, Retrieval
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.settings import Settings
from mindbridge.kernel.speech import Speech
from mindbridge.kernel.temporal import (
    query_temporal_range,
    resolved_reference_at,
    rows_time_span,
    temporal_context,
)
from mindbridge.kernel.tracing import Traced
from mindbridge.kernel.validation import (
    optional_memory_type,
    pushed_memory_types,
    validate_answer_policy,
    validate_limit,
    validated_retrieval_scope,
)
from mindbridge.kernel.vision import Vision
from mindbridge.models.base import (
    GenerationBackend,
    ModelInput,
    RecallPlanningBackend,
    StreamingGenerationBackend,
)
from mindbridge.recall import (
    SEMANTIC_QUERY_CAPABILITY,
    RecallPlan,
    RecallResult,
    RecallRows,
    execute,
    fallback_plan,
    parse_recall_plan,
    recall_note,
)
from mindbridge.types import (
    AnswerChunk,
    AnswerPolicy,
    AnswerResult,
    ContentInput,
    MemoryType,
    Modality,
    RetrievalScope,
    SearchHit,
)

# How many media rows an exhaustive plan may ground on, as a multiple of the window the caller
# asked for. Media is the one kind of evidence whose grounding costs writes -- face and speech
# recognition run on each row before the answer call, and again on a replan -- so a set read that
# matched every clip in the corpus is bounded here rather than by the character budget alone.
_RECALL_MEDIA_FACTOR = 2

# A per-ask ceiling, shared by all steps and replans, including a deeper original-query read.
_RECALL_QUERY_LIMIT = 2


ASYNC_QUEUE_TIME_MS: ContextVar[float | None] = ContextVar(
    "mindbridge_async_queue_time_ms",
    default=None,
)


@dataclass(frozen=True, slots=True)
class _RecallContext:
    """What one answering round needs, so a second round repeats nothing but the reading."""

    answerer: GenerationBackend
    prepared: PreparedContent
    routed: ModelInput
    assets: OperationAssets
    # The original limit-sized window stays the grounding floor. Semantic expansion can replace
    # its supplemental ranked tail without comparing scores from different queries.
    ranked: tuple[SearchHit, ...]
    ranked_depth: int
    limit: int
    reference: datetime
    temporal_range: tuple[datetime, datetime] | None
    scope: RetrievalScope | None
    memory_type: MemoryType | None
    link_identities: bool
    answer_policy: AnswerPolicy


class _RecallReads:
    """Run a recall plan's reads against one store, inside the caller's open operation.

    The exhaustive ops read SQLite through the store's own primitives and are hydrated the way
    every other hit is. They carry no relevance score, because they earned none: they are in the
    evidence set because they match the question's predicate, not because they ranked.
    """

    def __init__(
        self,
        store: LocalStore,
        hydrator: Hydrator,
        context: _RecallContext,
        retrieval: Retrieval | None = None,
    ) -> None:
        self._store = store
        self._hydrator = hydrator
        self._context = context
        self._retrieval = retrieval
        original = context.prepared.text.strip()
        self._cache = {original: context.ranked}
        self._depths = {original: context.ranked_depth}
        self._semantic_depths: builtins.dict[str, int] = {}
        self._unavailable: set[str] = set()
        self.queries = 0
        self.failures = 0
        self.limited: set[str] = set()

    def similar(self, query: str, *, k: int) -> RecallRows:
        if self._retrieval is None:
            rows = self._context.ranked[:k]
            return RecallRows(rows, len(rows))
        query = query.strip()
        if k > self._depths.get(query, 0):
            if self.queries >= _RECALL_QUERY_LIMIT:
                self.limited.add(query)
                self._unavailable.add(query)
                return RecallRows(self._cache.get(query, ())[:k], k)
            self.queries += 1
            # Query strings are untrusted model output, interpreted only as text. A rewrite
            # cannot replace the original media, reference clock, temporal intent or scope.
            prepared = (
                self._context.prepared
                if query == self._context.prepared.text.strip()
                else PreparedContent(query, (), Modality.TEXT, (("text", query),))
            )
            try:
                rows = self._retrieval.search_prepared(
                    prepared,
                    limit=RERANK_CANDIDATES,
                    operation=self._context.assets,
                    memory_types=pushed_memory_types(self._context.memory_type),
                    reference_at=self._context.reference,
                    temporal_range=self._context.temporal_range,
                    occurred_from=None,
                    occurred_until=None,
                    scope=self._context.scope,
                    require_unambiguous=False,
                    capture_trace=False,
                ).hits
            except ModelError:
                # An optional expansion failure must not cost the evidence already earned.
                # Cache the failure too: repeated steps and replans do not retry a paid call.
                self.failures += 1
                self._unavailable.add(query)
                rows = self._cache.get(query, ())
            else:
                self._semantic_depths.setdefault(query, 0)
            self._cache[query] = rows
            self._depths[query] = RERANK_CANDIDATES
        if query in self._semantic_depths:
            self._semantic_depths[query] = max(self._semantic_depths[query], k)
        rows = self._cache[query][:k]
        return RecallRows(rows, len(rows))

    def semantic_routes(self) -> tuple[tuple[SearchHit, ...], ...]:
        """Retain successful reads across rounds, in discovery order at requested depth.

        A later query may depend on an earlier read's bridge fact. Keep both routes eligible
        for the same bounded packing pass even when the later plan names only the new query.
        """
        return tuple(self._cache[query][:depth] for query, depth in self._semantic_depths.items())

    def mark_unread(self, program: RecallResult) -> RecallResult:
        """A failed anchor cannot make its dependent exhaustive reads complete."""
        if not self._unavailable:
            return program
        unread: set[int] = set()
        executed = list(program.executed)
        for index, step in enumerate(program.plan.steps):
            if (
                (step.op == "similar" and (step.query or "").strip() in self._unavailable)
                or step.of in unread
                or step.time_of in unread
            ):
                unread.add(index)
                executed[index] = replace(executed[index], bounded=True)
        return replace(program, executed=tuple(executed))

    def match(
        self,
        terms: Sequence[str],
        *,
        any_of: bool,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        modality: Modality | None,
        memory_type: MemoryType | None,
        max_rows: int,
        entity_id: str | None = None,
    ) -> RecallRows:
        scope = self._context.scope or RetrievalScope()
        return self._hits(
            self._store.recall.match_memories(
                tuple(terms),
                any_of=any_of,
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                modality=None if modality is None else modality.value,
                memory_type=self._memory_type(memory_type),
                max_rows=max_rows,
                valid_at=scope.valid_at,
                known_at=scope.known_at,
                near=scope.near,
                radius_m=scope.radius_m,
                place_id=scope.place_id,
                identity_id=scope.identity_id,
                entity_id=entity_id,
            )
        )

    def window(
        self,
        *,
        occurred_from: datetime,
        occurred_until: datetime,
        modality: Modality | None,
        memory_type: MemoryType | None,
        max_rows: int,
    ) -> RecallRows:
        scope = self._context.scope or RetrievalScope()
        return self._hits(
            self._store.recall.memories_in_window(
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                modality=None if modality is None else modality.value,
                memory_type=self._memory_type(memory_type),
                max_rows=max_rows,
                valid_at=scope.valid_at,
                known_at=scope.known_at,
                near=scope.near,
                radius_m=scope.radius_m,
                place_id=scope.place_id,
                identity_id=scope.identity_id,
            )
        )

    def neighbors(
        self,
        memory_ids: Sequence[str],
        *,
        before: int,
        after: int,
        max_rows: int,
    ) -> RecallRows:
        scope = self._context.scope or RetrievalScope()
        return self._hits(
            self._store.recall.neighbor_memories(
                tuple(memory_ids),
                before=before,
                after=after,
                max_rows=max_rows,
                valid_at=scope.valid_at,
                known_at=scope.known_at,
                near=scope.near,
                radius_m=scope.radius_m,
                place_id=scope.place_id,
                identity_id=scope.identity_id,
            )
        )

    def entity(self, name: str, *, max_rows: int) -> RecallRows:
        """Read what is known about one person, by identity when the name resolves to one.

        A name no identity carries is not an error: the store may hold the person only as words
        in other people's memories, so the read degrades to matching the name as text.
        """
        identity_id = self._store.recall.identity_id_for_name(
            name,
            identity_id=None if self._context.scope is None else self._context.scope.identity_id,
        )
        return self.match(
            () if identity_id is not None else (name,),
            any_of=True,
            occurred_from=None,
            occurred_until=None,
            modality=None,
            memory_type=None,
            max_rows=max_rows,
            entity_id=identity_id,
        )

    def related(
        self,
        memory_ids: Sequence[str],
        *,
        ceiling: int,
        max_rows: int,
    ) -> grounding.EvidenceExpansion:
        """Read what the grounded evidence is structurally linked to, under the caller's scope."""
        scope = self._context.scope or RetrievalScope()
        read = self._store.recall.related_memories(
            tuple(memory_ids),
            ceiling=ceiling,
            max_rows=max_rows,
            valid_at=scope.valid_at,
            known_at=scope.known_at,
            near=scope.near,
            radius_m=scope.radius_m,
            place_id=scope.place_id,
            identity_id=scope.identity_id,
        )
        return grounding.EvidenceExpansion(
            hits=self._hits(read).rows,
            edges=read.edges,
            dropped=read.dropped,
        )

    def time_span(self, rows: Sequence[SearchHit]) -> tuple[datetime, datetime] | None:
        """Bound a later recall read by these rows' event times and stated dates."""
        return rows_time_span(rows, self._context.reference)

    def chronological(self, rows: Sequence[SearchHit]) -> tuple[SearchHit, ...]:
        """Keep SQLite's corpus order when timestamps tie across different reads."""
        by_id = {hit.id: hit for hit in rows}
        return tuple(
            by_id[memory_id] for memory_id in self._store.recall.order_memory_ids(tuple(by_id))
        )

    def _memory_type(self, requested: MemoryType | None) -> str | None:
        """The caller's own `memory_type` wins: a plan may not widen the question's scope."""
        chosen = self._context.memory_type or requested
        return None if chosen is None else chosen.value

    def _hits(self, read: RecallRead) -> RecallRows:
        """Hydrate one primitive's rows, keeping the count its predicate selected.

        Metric scope may reduce the rows selected before SQL's bound. Only the selection count
        says whether that bound truncated anything.
        """
        return RecallRows(
            tuple(self._hydrator.search_hit(memory, 0.0) for memory in read),
            read.selected,
        )


class Answering(Traced):
    """Ground a question, optionally run its recall program, and generate the answer."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        backends: Backends,
        settings: Settings,
        lifecycle: Lifecycle,
        hydrator: Hydrator,
        materializer: Materializer,
        speech: Speech,
        vision: Vision,
        retrieval: Retrieval,
        execution: Execution,
    ) -> None:
        super().__init__(tracer)
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._backends = backends
        self._settings = settings
        self._lifecycle = lifecycle
        self._hydrator = hydrator
        self._materializer = materializer
        self._speech = speech
        self._vision = vision
        self._retrieval = retrieval
        self._execution = execution

    def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        link_identities: bool = True,
        answer_policy: AnswerPolicy = "strict",
    ) -> AnswerResult:
        # Draining `ask_stream()` keeps one implementation of the whole path, so a buffered and
        # a streamed answer cannot drift apart in grounding, abstention, or reinforcement.
        stream = self.ask_stream(
            question,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            link_identities=link_identities,
            answer_policy=answer_policy,
        )
        while True:
            try:
                next(stream)
            except StopIteration as complete:
                return cast(AnswerResult, complete.value)

    def ask_stream(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        link_identities: bool = True,
        answer_policy: AnswerPolicy = "strict",
    ) -> Generator[AnswerChunk, None, AnswerResult]:
        validate_limit(limit, maximum=100)
        validate_answer_policy(answer_policy)
        answerer = self._backends.answerer
        if answerer is None:
            raise ModelError(
                "answer backend is not configured",
                reason="backend_not_configured",
                stage="generate",
            )
        return self._ask_chunks(
            question,
            answerer=answerer,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            link_identities=link_identities,
            answer_policy=answer_policy,
        )

    def _ask_chunks(
        self,
        question: ContentInput,
        *,
        answerer: GenerationBackend,
        limit: int,
        memory_type: MemoryType | None,
        reference_at: datetime | None,
        scope: RetrievalScope | None,
        link_identities: bool,
        answer_policy: AnswerPolicy,
    ) -> Generator[AnswerChunk, None, AnswerResult]:
        answer = yield from self._ask_operation(
            question,
            answerer=answerer,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            link_identities=link_identities,
            answer_policy=answer_policy,
        )
        # The operation and its span close before the result is handed over. A generator stays
        # suspended at whichever yield the caller stops on, so a terminal chunk yielded inside
        # the operation would pin it open for every caller that reads the result and stops --
        # the ordinary way to consume this -- and `close()` waits on open operations.
        yield AnswerChunk(result=answer)
        return answer

    def _ask_operation(
        self,
        question: ContentInput,
        *,
        answerer: GenerationBackend,
        limit: int,
        memory_type: MemoryType | None,
        reference_at: datetime | None,
        scope: RetrievalScope | None,
        link_identities: bool,
        answer_policy: AnswerPolicy,
    ) -> Generator[AnswerChunk, None, AnswerResult]:
        started = perf_counter()
        operation_ttft_recorded = False
        with (
            self._trace("mindbridge.ask", kind="operation") as span,
            self._lifecycle.operation(model_work=True) as assets,
        ):
            queue_time_ms = ASYNC_QUEUE_TIME_MS.get()
            if queue_time_ms is not None:
                span.set_attribute(ASYNC_QUEUE_TIME, queue_time_ms)
            # `search()` opens a `mindbridge.search` operation span; `ask` reaches the same
            # retrieval plane directly. Keep its stage around every prerequisite through ranked
            # hits so the reported leg matches what answering actually waits for.
            with self._trace("mindbridge.retrieve", kind="stage"):
                with self._trace("mindbridge.content.prepare", kind="stage"):
                    prepared = self._materializer.prepare(question, assets)
                explicit_reference = resolved_reference_at(reference_at)
                reference, temporal_text = temporal_context(
                    prepared.text,
                    explicit_reference or datetime.now(timezone.utc),
                    infer_reference=explicit_reference is None,
                )
                temporal_range = query_temporal_range(temporal_text, reference)
                scope = validated_retrieval_scope(scope)
                memory_type = optional_memory_type(memory_type)
                ranked_depth = (
                    RERANK_CANDIDATES
                    if self._settings.evidence_budget is not None
                    else min(100, limit * 3)
                )
                prepared_search = partial(
                    self._retrieval.search_prepared,
                    prepared,
                    # Without a budget the answer grounds on `limit` hits, so ranking three times
                    # that is enough headroom for the modality round robin. With one, the budget is
                    # what decides depth, so the whole rerank pool has to be ranked for it to spend.
                    limit=ranked_depth,
                    operation=assets,
                    memory_types=pushed_memory_types(memory_type),
                    reference_at=reference,
                    temporal_range=temporal_range,
                    occurred_from=None,
                    occurred_until=None,
                    scope=scope,
                    require_unambiguous=limit == 1,
                    capture_trace=False,
                    capture_visual_matches=True,
                )
                speech_assets = self._speech.answer_speech_assets(prepared.assets)
                if speech_assets and all(
                    Modality(asset.modality) in self._backends.embedding_capabilities
                    for asset in speech_assets
                ):
                    with self._execution.overlap_speech(
                        self._speech.recognize, speech_assets, assets
                    ) as identities:
                        outcome = prepared_search()
                        hits = outcome.hits
                        _record_retrieval_results(hits)
                        identities.result()
                else:
                    if speech_assets:
                        self._speech.recognize(speech_assets, assets)
                    outcome = prepared_search()
                    hits = outcome.hits
                    _record_retrieval_results(hits)
            routed = self._route_generation(prepared, assets)
            image_ids = {asset.id for asset in routed.assets if asset.modality is Modality.IMAGE}
            round_context = _RecallContext(
                answerer=answerer,
                prepared=prepared,
                # Every question the answerer reads as text gets the reference time, not just the
                # ones whose phrasing a parser recognized. "How long ago did grandpa visit?"
                # narrows no retrieval window and names no date, yet it is exactly the question
                # the reader cannot answer without knowing when it is being asked. Applied after
                # routing, because that is what decides whether the reader gets text at all -- a
                # spoken question becomes text there -- and because routing appends speech
                # identities and transcripts, which would otherwise land after the line
                # documented as final.
                routed=replace(
                    routed,
                    image_retrieval_matches=tuple(
                        match
                        for match in outcome.image_retrieval_matches
                        if match.query_asset_id in image_ids
                    ),
                ),
                assets=assets,
                ranked=hits,
                ranked_depth=ranked_depth,
                limit=limit,
                reference=reference,
                temporal_range=temporal_range,
                scope=scope,
                memory_type=memory_type,
                link_identities=link_identities,
                answer_policy=answer_policy,
            )
            # `closing` rather than plain iteration: an abandoned stream throws `GeneratorExit`
            # at the yield below, and without this the inner generator would only be closed when
            # the cleared frame drops its last reference. That defers closing the provider's
            # response and ends `mindbridge.model.generation` after its own parent, which loses
            # the call's model usage and emits a malformed trace.
            with closing(self._answer_rounds(round_context)) as deltas:
                while True:
                    try:
                        delta = next(deltas)
                    except StopIteration as complete:
                        result, hits = complete.value
                        break
                    if delta.strip() and not operation_ttft_recorded:
                        operation_ttft_ms = (perf_counter() - started) * 1_000.0
                        if queue_time_ms is not None:
                            operation_ttft_ms += queue_time_ms
                        span.set_attribute(OPERATION_TTFT, operation_ttft_ms)
                        operation_ttft_recorded = True
                    yield AnswerChunk(text=delta)
            used_ids = {hit.id for hit in result.hits}
            grounding = tuple(hit for hit in hits if hit.id in used_ids)
            # Reinforcement is part of answering, so it stays inside the operation and runs
            # before the terminal chunk: a caller that sees the result sees a settled store.
            if not result.abstained:
                self._reinforce_answered(grounding)
            # An abstention is the answering plane reporting that the evidence did not support
            # an answer, which is the same durable signal as an empty search: the memory the
            # question needed is missing or unreachable.
            self._retrieval.note_query_failure(prepared.text, failed=result.abstained)
            return AnswerResult(
                answer=result.answer,
                hits=grounding,
                abstained=result.abstained,
                abstention_reason=result.abstention_reason,
            )

    def _route_generation(
        self,
        prepared: PreparedContent,
        operation: OperationAssets,
    ) -> ModelInput:
        value = self._hydrator.model_input(prepared)
        missing = value.modalities - self._backends.generation_capabilities
        if (
            missing == {Modality.TEXT}
            and (prepared.audio_transcript or prepared.visual_description)
            and value.modalities - {Modality.TEXT} <= self._backends.generation_capabilities
        ):
            return ModelInput(assets=value.assets)
        unsupported = fallback_unsupported(
            prepared,
            self._backends.generation_capabilities,
            "generation",
        )
        if self._speech.answer_speech_assets(prepared.assets):
            prepared = self._speech.with_speech_identities(prepared, operation)
        if Modality.AUDIO in unsupported and not prepared.audio_transcript:
            require_audio_transcription(self._backends.transcription_capabilities)
            prepared = self._speech.with_audio_transcripts(prepared, operation)
        value = self._hydrator.model_input(prepared)
        unsupported = value.modalities - self._backends.generation_capabilities
        if not unsupported:
            return value
        text = derived_text(value.text, prepared.assets)
        fallback = {Modality.AUDIO}
        if prepared.visual_description:
            fallback.update((Modality.IMAGE, Modality.VIDEO))
        assets = tuple(asset for asset in value.assets if asset.modality not in unsupported)
        if unsupported <= fallback and (text or assets):
            routed = ModelInput(
                text=text,
                assets=assets,
            )
            if not routed.modalities - self._backends.generation_capabilities:
                return routed
        names = ", ".join(sorted(modality.value for modality in unsupported))
        raise ModelError(
            f"configured generation model does not support: {names}",
            reason="unsupported_modality",
        )

    def _route_generation_hits(
        self,
        hits: Sequence[SearchHit],
        operation: OperationAssets,
        *,
        link_identities: bool = True,
    ) -> tuple[SearchHit, ...]:
        asset_ids = tuple(asset.id for hit in hits for asset in hit.assets)
        with (
            self._trace("mindbridge.storage.hydrate", kind="stage"),
            translate_storage_errors("hydrate media for answer generation"),
        ):
            stored = self._store.media.read_assets(asset_ids)
        by_id = {asset.asset_id: asset for asset in stored}
        prepared_hits = []
        for hit in hits:
            try:
                assets = tuple(by_id[asset.id] for asset in hit.assets)
            except KeyError:
                raise StorageError(
                    "memory references missing media metadata", reason="asset_unavailable"
                ) from None
            prepared_hits.append(
                PreparedContent(
                    text=hit.content,
                    assets=assets,
                    modality=memory_modality(assets),
                    canonical_parts=(),
                    audio_transcript=has_stream_transcript(hit.content, assets),
                    visual_description=has_stream_description(hit.content, assets),
                )
            )
        if Modality.AUDIO not in self._backends.generation_capabilities:
            for prepared in prepared_hits:
                fallback_unsupported(
                    prepared,
                    self._backends.generation_capabilities,
                    "generation",
                )
        speech_assets = self._speech.answer_speech_assets(
            tuple(asset for prepared in prepared_hits for asset in prepared.assets)
        )
        if speech_assets:
            self._speech.recognize(speech_assets, operation)
        elif Modality.AUDIO not in self._backends.generation_capabilities:
            self._speech.cache_audio_transcripts(
                tuple(
                    asset
                    for prepared in prepared_hits
                    if not prepared.audio_transcript
                    for asset in prepared.assets
                ),
                operation,
            )
        face_assets = self._vision.answer_face_assets(
            tuple(asset for prepared in prepared_hits for asset in prepared.assets)
        )
        if face_assets:
            self._vision.recognize(face_assets, operation, link_identities=link_identities)
        routed = []
        for hit, prepared in zip(hits, prepared_hits, strict=True):
            if self._vision.answer_face_assets(prepared.assets):
                prepared = self._vision.with_face_identities(
                    prepared, operation, link_identities=link_identities
                )
            model_input = self._route_generation(prepared, operation)
            routed.append(
                replace(
                    hit,
                    content=model_input.text,
                    assets=model_input.assets,
                    modality=model_input.modality,
                )
            )
        return tuple(routed)

    def _answer_rounds(
        self,
        context: _RecallContext,
    ) -> Generator[str, None, tuple[AnswerResult, tuple[SearchHit, ...]]]:
        """Answer this question, replanning once when the first round's own answer asked for it.

        A round another round may replace is held rather than streamed. Two complete answers on
        one wire with no boundary between them is not something a caller can render, and the
        terminal result has to be the text that was streamed -- so the deltas of a replannable
        round are buffered and yielded only if that round is the one that stands. Every other
        round, the common case and always the last one, streams as the provider produces it.
        """
        attempted = ""
        reads = _RecallReads(
            self._store,
            self._hydrator,
            context,
            self._retrieval if self._settings.recall_query_expansion else None,
        )
        for attempt in range(self._settings.recall_rounds if self._settings.recall_planning else 1):
            held: builtins.list[str] = []
            replannable = self._replan_possible(context, attempt=attempt)
            with closing(self._recall_round(context, reads, attempted=attempted)) as deltas:
                while True:
                    try:
                        delta = next(deltas)
                    except StopIteration as complete:
                        result, hits = complete.value
                        break
                    if replannable:
                        held.append(delta)
                    else:
                        yield delta
            if not (result.abstained and replannable):
                yield from held
                break
            attempted = grounding.attempt_note(
                hits, result, include_evidence=self._settings.recall_query_expansion
            )
        return result, hits

    def _recall_round(
        self,
        context: _RecallContext,
        reads: _RecallReads,
        *,
        attempted: str,
    ) -> Generator[str, None, tuple[AnswerResult, tuple[SearchHit, ...]]]:
        """Ground one attempt at this question and yield its answer deltas.

        With `recall_planning` off there is exactly one attempt and this is the sequence `ask`
        has always run: the ranking's own grounding window, the routed question with the
        reference clock last, then the answer.
        """
        program = self._recall_program(context, reads, attempted=attempted)
        semantic = (
            program is not None
            and self._settings.recall_query_expansion
            and any(reads.semantic_routes())
        )
        query_context = (
            self._query_media_context(context, reads) if self._settings.evidence_expansion else ()
        )
        native_grounding = (
            self._ground_query_media_recall(program, reads, context, query_context)
            if query_context
            else None
        )
        if native_grounding is not None:
            hits, note = native_grounding
        else:
            hits, note = self._grounded_recall(program, context, defer_tail=semantic)
            if program is not None and semantic:
                hits, note = self._ground_semantic_recall(program, reads, context, hits, note)
        sequence = program is not None and program.plan.shape == "sequence"
        if sequence:
            # Ordering is a presentation step after every budget/route decision. Predicate
            # rows alone omit ranked anchors and semantic bridges, which may share their event
            # timestamp: SQLite's insertion order must cover the entire selected union.
            hits = reads.chronological(hits)
            order_note = (
                "The supplied memory hits are in corpus time order, including the store's "
                "insertion order when event times tie. This orders only the evidence shown; "
                "it does not establish a complete event history or answer set."
            )
            note = f"{note} {order_note}" if note else order_note
        routed = context.routed
        question = (
            grounding.with_reference_time(
                grounding.with_recall_note(routed, note), context.reference
            )
            if routed.text
            else routed
        )
        routed_hits = (
            self._route_generation_hits(
                hits,
                context.assets,
                link_identities=context.link_identities,
            )
            if hits
            else ()
        )
        if question.image_retrieval_matches:
            delivered = {(hit.id, asset.id) for hit in routed_hits for asset in hit.assets}
            question = replace(
                question,
                image_retrieval_matches=tuple(
                    match
                    for match in question.image_retrieval_matches
                    if (match.memory_id, match.source_asset_id) in delivered
                ),
            )
        self._speech.persist_transcripts(context.assets)
        with closing(
            self._answer_chunks(
                question,
                routed_hits,
                answerer=context.answerer,
                answer_policy=context.answer_policy,
                # This existing backend flag selects the program-order prompt, not a promise
                # of completeness. Sequence hits are ordered even for a pure similarity plan;
                # calling them rank order would contradict the question's chronology note.
                exhaustive=program is not None and (program.plan.exhaustive or sequence),
            )
        ) as deltas:
            result = yield from deltas
        return result, hits

    def _recall_program(
        self, context: _RecallContext, reads: _RecallReads, *, attempted: str
    ) -> RecallResult | None:
        """Plan and run this question's reads, or None when planning is off.

        Inside its own stage: the `mindbridge.retrieve` leg closes on the ranked hits, and the
        planning call and the exhaustive reads happen after it -- a model call and a set of table
        scans that a trace without this span attributes to nothing, leaving the operation's own
        duration as the only evidence they ran at all.
        """
        if not self._settings.recall_planning:
            return None
        with self._trace("mindbridge.recall", kind="stage") as span:
            question, k = context.prepared.text, context.limit
            plan = self._recall_plan(
                question,
                reference_at=context.reference,
                k=k,
                attempted=attempted,
            )
            program = execute(
                plan,
                reads,
                limit=k,
                # The digest the planner was already handed, recomputed only after a commit, so
                # the selectivity of every step is decided against one cached read per question
                # rather than a count per step.
                active_records=self._store.recall.recall_digest().records,
            )
            program = reads.mark_unread(program)
            span.set_attributes(
                {
                    RECALL_SHAPE: plan.shape,
                    RECALL_OPS: tuple(step.op for step in plan.steps),
                    RECALL_EXHAUSTIVE_ROWS: len(program.exhaustive),
                    RECALL_COMPLETE: program.complete,
                    RECALL_NON_SELECTIVE_STEPS: program.non_selective_steps,
                    RECALL_REPLAN: bool(attempted),
                    # Every way planning can fail resolves to this same plan, so equality with it
                    # is the only signal that says the planner did not decide this question --
                    # which is what a result claiming the feature ran has to be read against.
                    RECALL_FALLBACK: plan == fallback_plan(question, k=k),
                }
            )
            return program

    def _query_media_context(
        self, context: _RecallContext, reads: _RecallReads
    ) -> tuple[SearchHit, ...]:
        """Follow exact original query assets only from already scoped, hydrated ranked rows."""
        query_assets = {
            (asset.asset_id, asset.sha256)
            for asset in context.prepared.assets
            if asset.sha256 is not None
        }
        if not query_assets:
            return ()
        anchors = tuple(
            hit
            for hit in context.ranked[:RERANK_CANDIDATES]
            if any((asset.id, asset.sha256) in query_assets for asset in hit.assets)
        )[: context.limit]
        if not anchors:
            return ()
        with self._trace("mindbridge.expand.query_media", kind="stage") as span:
            neighbors = reads.neighbors(
                tuple(hit.id for hit in anchors),
                before=1,
                after=1,
                max_rows=self._settings.evidence_expansion_max_rows,
            ).rows
            # The generic neighbor primitive applies scope but carries no type predicate.
            # The caller's memory type still constrains this supplemental read.
            neighbors = tuple(
                hit
                for hit in neighbors
                if context.memory_type is None or hit.memory_type is context.memory_type
            )
            by_id = {hit.id: hit for hit in (*anchors, *neighbors)}
            span.set_attributes(
                {EXPANSION_ROWS: len(by_id), EXPANSION_EDGES: (f"query_asset:{len(anchors)}",)}
            )
            return tuple(by_id.values())

    def _ground_query_media_recall(
        self,
        program: RecallResult | None,
        reads: _RecallReads,
        context: _RecallContext,
        query_context: tuple[SearchHit, ...],
    ) -> tuple[tuple[SearchHit, ...], str | None] | None:
        """Give exact-source context budget before semantic additions and replaceable rank tail."""
        floor = grounding.grounding_hits(context.ranked, context.limit)
        floor_ids = {hit.id for hit in floor}
        routes = reads.semantic_routes() if self._settings.recall_query_expansion else ()
        exhaustive = program is not None and program.plan.exhaustive
        budget = self._settings.evidence_budget
        if exhaustive or any(routes):
            budget = (
                self._settings.recall_set_budget
                if budget is None
                else min(budget, self._settings.recall_set_budget)
            )
        note = None
        if program is not None and exhaustive:
            assert budget is not None
            required, omitted = grounding.budget_recall(
                replace(program, ranked=()),
                budget,
                media_limit=_RECALL_MEDIA_FACTOR * context.limit,
                max_rows=self._settings.recall_set_max_rows,
                required=floor,
            )
            note = recall_note(program, omitted=omitted)
        else:
            required = floor
        required_ids = {hit.id for hit in required}
        matched = {hit.id for hit in program.exhaustive} if program is not None else set()
        # Predicate rows keep their existing caps. All further supplemental media share what
        # is left; the original ranked floor is never charged against this supplemental cap.
        media_limit = max(
            0,
            _RECALL_MEDIA_FACTOR * context.limit
            - sum(bool(hit.assets) for hit in required if hit.id not in floor_ids),
        )
        hits, native_added = grounding.pack_grounding(
            required,
            tuple(hit for hit in query_context if hit.id not in matched or hit.id in required_ids),
            (),
            budget,
            media_limit=media_limit,
            max_added=self._settings.evidence_expansion_max_rows,
        )
        if not native_added:
            return None
        media_limit -= sum(bool(hit.assets) for hit in hits if hit.id not in required_ids)
        if native_added:
            native_note = (
                f"{native_added} supplemental records from exact query-media sources and their "
                "immediate corpus neighbors are included. Corpus adjacency does not establish "
                "a shared event or a complete answer set: do not state a total over them."
            )
            note = f"{note} {native_note}" if note else native_note
        offered = grounding.semantic_hits(routes, {hit.id for hit in hits} | matched, context.limit)
        before_semantic = {hit.id for hit in hits}
        hits, semantic_added = grounding.pack_grounding(
            hits, offered, (), budget, media_limit=media_limit
        )
        media_limit -= sum(bool(hit.assets) for hit in hits if hit.id not in before_semantic)
        if semantic_added:
            semantic_note = (
                f"{semantic_added} further records from bounded semantic queries are included. "
                "They are not a complete set: do not state a total over them."
            )
            note = f"{note} {semantic_note}" if note else semantic_note
        tail = context.ranked
        if program is not None and exhaustive:
            tail = grounding.recall_ranked_tail(
                program, tail, budgeted=self._settings.evidence_budget is not None
            )
        if budget is not None and (self._settings.evidence_budget is not None or exhaustive):
            hits = (*hits, *grounding.budgeted_hits(tail, hits, budget))
        remaining = self._settings.evidence_expansion_max_rows - native_added
        if remaining:
            expansion = self._expand_evidence(hits, context)
            expansion = replace(
                expansion,
                hits=tuple(
                    hit
                    for hit in expansion.hits
                    if hit.id not in matched
                    and (context.memory_type is None or hit.memory_type is context.memory_type)
                ),
            )
            hits, linked = grounding.pack_grounding(
                hits,
                expansion.hits,
                (),
                budget,
                media_limit=media_limit,
                max_added=remaining,
            )
            if linked:
                linked_note = grounding.expansion_note(
                    linked,
                    expansion,
                    chronological=program is not None and program.plan.shape == "sequence",
                )
                note = f"{note} {linked_note}" if note else linked_note
        return hits, note

    def _ground_semantic_recall(
        self,
        program: RecallResult,
        reads: _RecallReads,
        context: _RecallContext,
        required: tuple[SearchHit, ...],
        note: str | None,
    ) -> tuple[tuple[SearchHit, ...], str | None]:
        """Pack query-local evidence before the replaceable original ranked tail.

        Scores from distinct queries are not comparable. Each query can add at most one
        `limit`-sized window, after dedup; the character and media caps may narrow it further.
        The same limits apply to point and exhaustive plans, without licensing set totals.
        """
        routes = reads.semantic_routes()
        candidates = {hit.id for route in routes for hit in route}
        matched = {hit.id for hit in program.exhaustive}
        # Predicate rows keep their own row/media caps and completeness accounting.
        offered = grounding.semantic_hits(
            routes, {hit.id for hit in required} | matched, context.limit
        )
        budget = self._settings.recall_set_budget
        if self._settings.evidence_budget is not None:
            budget = min(budget, self._settings.evidence_budget)
        with self._trace("mindbridge.recall.ground", kind="stage") as span:
            hits, added = grounding.pack_grounding(
                required, offered, (), budget, media_limit=_RECALL_MEDIA_FACTOR * context.limit
            )
            required_ids = {hit.id for hit in required}
            novel = candidates - required_ids
            span.set_attributes(
                {
                    # Query counts are cumulative per ask; candidate/grounding counts describe
                    # this round. Counts only: no query, evidence or provider failure text.
                    RECALL_QUERY_COUNT: reads.queries,
                    RECALL_QUERY_FAILURES: reads.failures,
                    RECALL_QUERY_LIMITED: len(reads.limited),
                    RECALL_QUERY_CANDIDATES: len(candidates),
                    RECALL_QUERY_NEW_CANDIDATES: len(
                        candidates - {hit.id for hit in context.ranked}
                    ),
                    RECALL_QUERY_GROUNDED: added,
                    RECALL_QUERY_SELECTED: len(offered),
                    RECALL_QUERY_BUDGET_DROPPED: len(offered) - added,
                    RECALL_QUERY_OMITTED: len(novel) - added,
                }
            )
        if not added:
            return self._grounded_recall(program, context)
        semantic_note = (
            f"{added} further records from bounded semantic queries are included. "
            if program.plan.shape == "sequence"
            else f"{added} further records from bounded semantic queries follow the original evidence. "
        ) + "They are not a complete set: do not state a total over them."
        note = f"{note} {semantic_note}" if note else semantic_note
        tail = context.ranked
        if program.plan.exhaustive:
            tail = grounding.recall_ranked_tail(
                program, tail, budgeted=self._settings.evidence_budget is not None
            )
        if program.plan.exhaustive or self._settings.evidence_budget is not None:
            hits = (*hits, *grounding.budgeted_hits(tail, hits, budget))
        if self._settings.evidence_expansion:
            expansion = self._expand_evidence(hits, context)
            hits, linked = grounding.pack_grounding(
                hits, expansion.hits, (), budget, media_limit=_RECALL_MEDIA_FACTOR * context.limit
            )
            if linked:
                linked_note = grounding.expansion_note(
                    linked, expansion, chronological=program.plan.shape == "sequence"
                )
                note = f"{note} {linked_note}" if note else linked_note
        return hits, note

    def _grounded_recall(
        self,
        program: RecallResult | None,
        context: _RecallContext,
        *,
        defer_tail: bool = False,
    ) -> tuple[tuple[SearchHit, ...], str | None]:
        """Widen the grounding window by whatever the question's shape asks for beyond it.

        A ranking question keeps the window the ranking earned. A question that asked for every
        matching record adds that set to it, up to `recall_set_budget_chars`, and carries a line
        saying what the set is -- because a reader cannot see the predicate that produced its
        evidence, and a count over a silently truncated set is a wrong answer stated confidently.

        The set is added to the ranked window, never substituted for it. Grounding a plan on its
        exhaustive rows alone cost every question whose reads returned few rows the window it
        already had: measured on ATM-Hard, 12 grounded records down to 1, 2, 4 and 5, and one
        down to a refusal the unplanned path had answered. A plan says what else is relevant; it
        never says the ranking was wrong.
        """
        if program is not None:
            # A selective predicate must not strand spare budget or hide ranked follow-ups
            # that omit its literal terms. Query-local rows still pass semantic row/media caps.
            program = replace(
                program,
                ranked=(
                    ()
                    if defer_tail
                    else grounding.recall_ranked_tail(
                        program,
                        context.ranked,
                        budgeted=self._settings.evidence_budget is not None,
                    )
                ),
            )
        if program is None or not program.plan.exhaustive:
            if defer_tail:
                return grounding.grounding_hits(context.ranked, context.limit), None
            budget = self._settings.evidence_budget
            if not self._settings.evidence_expansion:
                return grounding.grounding_hits(
                    context.ranked, context.limit, budget_chars=budget
                ), None
            required = grounding.grounding_hits(context.ranked, context.limit)
            expansion = self._expand_evidence(required, context)
            hits, added = grounding.pack_grounding(
                required,
                expansion.hits,
                context.ranked,
                budget,
                media_limit=_RECALL_MEDIA_FACTOR * context.limit,
            )
            return hits, (
                grounding.expansion_note(
                    added,
                    expansion,
                    chronological=program is not None and program.plan.shape == "sequence",
                )
                if added
                else None
            )
        # Both settings constrain supplemental packing, after charging the mandatory ranked
        # window. A set plan may narrow that allowance, never widen it. Neither setting trims
        # the mandatory window or promises a provider input limit.
        budget = (
            self._settings.recall_set_budget
            if self._settings.evidence_budget is None
            else min(self._settings.recall_set_budget, self._settings.evidence_budget)
        )
        hits, omitted = grounding.budget_recall(
            program,
            budget,
            media_limit=_RECALL_MEDIA_FACTOR * context.limit,
            max_rows=self._settings.recall_set_max_rows,
            # Reserve the limit-sized window, including its modality floor, before additions.
            required=grounding.grounding_hits(context.ranked, context.limit),
        )
        note = recall_note(program, omitted=omitted)
        if self._settings.evidence_expansion and not defer_tail:
            # A plan already decided what else to read, so expansion follows its set rather than
            # competing with it: the rows go after everything the plan grounded, on whatever the
            # budget has left. Planning and expansion are separate mechanisms and this is the
            # only place they meet.
            expansion = self._expand_evidence(hits, context)
            packed, added = grounding.pack_grounding(
                hits,
                expansion.hits,
                (),
                budget,
                media_limit=_RECALL_MEDIA_FACTOR * context.limit,
            )
            if added:
                linked_note = grounding.expansion_note(
                    added, expansion, chronological=program.plan.shape == "sequence"
                )
                return packed, f"{note} {linked_note}"
        return hits, note

    def _expand_evidence(
        self, hits: Sequence[SearchHit], context: _RecallContext
    ) -> grounding.EvidenceExpansion:
        """Read the records this evidence is structurally linked to, at no model cost.

        Every linked row is offered, including one the ranking also found. Discarding those was
        tried, on the theory that a wider budget reaches them anyway, and it removed the whole
        benefit: with no character budget the candidate pool is `min(100, limit * 3)` while the
        window it grounds is `limit`, so a gold row at rank 20 is in the pool and will never be in
        the window. Gating on pool membership dropped exactly those rows -- complete support fell
        to the narrow window's own 0.7949 on ATM-Bench and the paired comparison went to 117 ties
        -- and kept the rows past rank 36, which are the ones that do not matter.

        What the measurement left is a simpler reading of this feature. Expansion does not reach
        evidence the ranking cannot; it reaches further down the same ranking by a different
        selector, and it is worth enabling where that selector is cheaper than cosine depth. The
        rows a wider window would have contained anyway are removed by `grounding.pack_grounding`'s own
        dedup, which is the only place that can know what the window will hold.
        """
        if not hits:
            return grounding.EvidenceExpansion(hits=(), edges={}, dropped=())
        with self._trace("mindbridge.expand", kind="stage") as span:
            expansion = _RecallReads(self._store, self._hydrator, context).related(
                tuple(hit.id for hit in hits),
                ceiling=grounding.expansion_ceiling(
                    context.limit,
                    self._store.recall.recall_digest().records,
                ),
                max_rows=self._settings.evidence_expansion_max_rows,
            )
            reachable = {hit.id for hit in context.ranked}
            span.set_attributes(
                {
                    EXPANSION_ROWS: len(expansion.hits),
                    # How many of the linked rows the ranking also proposed. Kept as a signal
                    # rather than a filter: gating on it was measured harmful, and the number is
                    # still what says whether a corpus's captures track its ranking.
                    EXPANSION_LINKED_ROWS: sum(1 for hit in expansion.hits if hit.id in reachable),
                    EXPANSION_EDGES: tuple(
                        f"{edge}:{count}" for edge, count in expansion.edges.items() if count
                    ),
                    EXPANSION_DROPPED_EDGES: expansion.dropped,
                }
            )
            return expansion

    def _replan_possible(self, context: _RecallContext, *, attempt: int) -> bool:
        """Whether a later round could replace this one, which is what makes it unstreamable.

        Another round needs an abstention. Experimental semantic recall may also investigate a
        strict refusal before releasing it; ordinary planning retains its best-effort-only rule.
        """
        return (
            self._settings.recall_planning
            and (context.answer_policy == "best_effort" or self._settings.recall_query_expansion)
            and attempt + 1 < self._settings.recall_rounds
        )

    def _recall_plan(
        self,
        question: str,
        *,
        reference_at: datetime,
        k: int,
        attempted: str = "",
    ) -> RecallPlan:
        """Ask the answerer how to read for this question, or keep today's single search.

        Every way this can go wrong lands on the same fallback -- a backend that cannot plan, a
        planner that fails, an empty completion, a plan the kernel will not run -- so the cost
        of a bad planner is one model call, never a different answer. Model output is data: it
        reaches `parse_recall_plan`, which decides what MindBridge does.
        """
        planner = self._backends.answerer
        if not isinstance(planner, RecallPlanningBackend) or not question.strip():
            return fallback_plan(question, k=k)
        model = getattr(planner, "generation_model", None)
        with self._model_trace(
            "generation",
            "plan",
            model=model if isinstance(model, str) else None,
            batch_size=1,
            modalities=(Modality.TEXT,),
        ):
            mark_model_requests(1)
            digest = self._corpus_digest()
            if self._settings.recall_query_expansion:
                digest += (
                    f"\n{SEMANTIC_QUERY_CAPABILITY}\n"
                    "similar.query executes "
                    "a text search for that query within the original scope and temporal intent. "
                    "At most two additional searches are allowed across all attempts; repeated "
                    "queries reuse results. Preserve the question's intent; use a focused "
                    "query for a missing fact rather than an invented answer. Semantic results "
                    "are bounded rankings and never establish completeness."
                )
            try:
                payload = planner.plan_recall(
                    question,
                    reference_at=reference_at,
                    corpus_digest=digest,
                    attempted=attempted,
                )
            except ModelError:
                # Planning is not answering. A provider that refused this call has said nothing
                # about whether it can answer, and the fallback plan is the one `ask` has always
                # run, so the question continues instead of failing on a hint.
                return fallback_plan(question, k=k)
        if payload is None:
            return fallback_plan(question, k=k)
        return parse_recall_plan(payload, reference_at=reference_at) or fallback_plan(question, k=k)

    def _corpus_digest(self) -> str:
        """Describe the active corpus in one line, so a plan cannot ask for what is not there."""
        digest = self._store.recall.recall_digest()
        if not digest.records:
            return "empty"
        span = (
            "no dated records"
            if digest.earliest is None or digest.latest is None
            else f"{datetime_text(digest.earliest)} to {datetime_text(digest.latest)}"
        )
        return (
            f"{digest.records} records; {span}; "
            f"modalities {', '.join(digest.modalities)}; "
            f"{digest.named_identities} named identities"
        )

    def _answer_chunks(  # noqa: C901 - streaming and non-streaming validation share one span
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answerer: GenerationBackend,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> Generator[str, None, AnswerResult]:
        """Yield answer deltas in provider order and return the validated grounded result.

        A backend without `stream_answer` yields its whole answer as one delta, so every caller
        sees the same shape and `ask_stream()` never requires a streaming provider.
        """
        model = getattr(answerer, "generation_model", None)
        with self._model_trace(
            "generation",
            "chat",
            model=model if isinstance(model, str) else None,
            batch_size=1,
            modalities=(
                modality
                for value in (question, *(ModelInput(hit.content, hit.assets) for hit in hits))
                for modality in value.modalities
            ),
        ):
            mark_model_requests(1)
            _record_generation_inputs(tuple(hits))
            buffered = False
            try:
                if isinstance(answerer, StreamingGenerationBackend):
                    started = perf_counter()
                    parts: builtins.list[str] = []
                    stream = answerer.stream_answer(
                        question, hits, answer_policy=answer_policy, exhaustive=exhaustive
                    )
                    result: object = None
                    try:
                        while True:
                            try:
                                part = next(stream)
                            except StopIteration as completed:
                                result = completed.value
                                break
                            if not isinstance(part, str):
                                raise ModelError(
                                    "generation model returned an invalid answer chunk",
                                    reason="response_invalid",
                                )
                            if part:
                                if not parts and current_model_request_count():
                                    trace.get_current_span().set_attribute(
                                        MODEL_TTFT, perf_counter() - started
                                    )
                                parts.append(part)
                                yield part
                    finally:
                        # This iterator owns the provider's streaming response. An abandoned
                        # `ask_stream()` unwinds to here; close explicitly so releasing the
                        # response does not depend on interpreter reference collection.
                        stream.close()
                    answer = "".join(parts)
                    if not answer.strip():
                        raise ModelError(
                            "generation model returned an invalid answer", reason="response_invalid"
                        )
                else:
                    result = answerer.answer(
                        question, hits, answer_policy=answer_policy, exhaustive=exhaustive
                    )
                    buffered = True
                if not isinstance(result, AnswerResult) or (
                    not buffered and answer_policy == "strict" and result.answer != answer
                ):
                    raise ModelError(
                        "generation model returned an invalid answer", reason="response_invalid"
                    )
            except MindBridgeError:
                raise
            except Exception as error:
                raise ModelError(
                    "failed to generate a grounded answer", reason="model_failed"
                ) from error
            if buffered:
                yield result.answer
            return result

    def _reinforce_answered(self, hits: Sequence[SearchHit]) -> None:
        """Count backend-declared evidence use after a non-abstained answer.

        The caller intersects the backend's result with the selected canonical hits. Declared
        use is not verified factual support. Bookkeeping failure must not discard the generated
        answer; disabling `reinforce_on_answer` keeps later rankings independent of this update.
        """
        if not hits or not self._settings.reinforce_on_answer:
            return
        with suppress(Exception), self._write_lock:
            self._store.records.reinforce_memories(
                tuple(hit.id for hit in hits),
                accessed_at=datetime.now(timezone.utc),
            )
