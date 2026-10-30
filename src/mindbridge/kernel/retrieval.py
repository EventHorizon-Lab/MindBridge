"""The retrieval plane: candidates from the index, truth from SQLite, one ranking."""

from __future__ import annotations

import builtins
import logging
import re
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, closing
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from itertools import chain

from opentelemetry.trace import Tracer

from mindbridge.exceptions import ModelError
from mindbridge.infrastructure.local.scoring import max_cosine_part_scores, max_cosine_scores
from mindbridge.infrastructure.local.store import StoredEmbedding, StoredMemory
from mindbridge.infrastructure.local.zvec_index import IndexHit
from mindbridge.kernel.content import PreparedContent
from mindbridge.kernel.contracts import Backends
from mindbridge.kernel.embedding import DOCUMENT_TASK, MAX_RETRIEVAL_KEYS, Embedding
from mindbridge.kernel.execution import Execution
from mindbridge.kernel.hydration import Hydrator
from mindbridge.kernel.lifecycle import Lifecycle, OperationAssets
from mindbridge.kernel.materialization import Materializer
from mindbridge.kernel.projection import Projection
from mindbridge.kernel.ranking import (
    LEXICAL_FULL_COVERAGE,
    LEXICAL_FULL_COVERAGE_RELEVANCE,
    MAX_LEXICAL_RERANK_BONUS,
    IndexCandidates,
    SearchOutcome,
    bounded_scale,
    extend_hydration_traces,
    extend_memory_type_traces,
    extend_missing_memory_traces,
    extend_ranked_traces,
    lexical_query_terms,
    lexical_relevance,
    merge_index_hits,
    merge_lexical_routes,
    merge_temporal_lexical_hits,
    parent_index_ids,
    parent_index_signals,
    qualified_candidates,
    ranking_signals,
    record_ranked_trace,
    retrieval_is_ambiguous,
    search_outcome,
)
from mindbridge.kernel.runtime import (
    Index,
    Storage,
    translate_index_errors,
    translate_storage_errors,
)
from mindbridge.kernel.settings import Settings
from mindbridge.kernel.speech import Speech
from mindbridge.kernel.temporal import (
    intersect_occurrence_range,
    occurrence_overlaps,
    query_temporal_range,
    resolved_reference_at,
    search_occurrence_range,
    temporal_context,
)
from mindbridge.kernel.tracing import Traced
from mindbridge.kernel.validation import (
    optional_memory_type,
    pushed_memory_types,
    validate_limit,
    validated_retrieval_scope,
)
from mindbridge.kernel.vision import Vision
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.types import (
    ContentInput,
    IndexQuantization,
    MemoryType,
    Modality,
    RetrievalCandidateTrace,
    RetrievalMode,
    RetrievalScope,
    SearchHit,
    TracedSearchResult,
    VisualRetrievalMatch,
)

RERANK_CANDIDATES = 100

_LOGGER = logging.getLogger(__name__)


# Everything a recall query's normalization drops before two of them are called near-equal:
# case, accents, punctuation, and whitespace runs. Deliberately lexical -- an embedding-based
# near-duplicate check would make the trigger depend on the model, so "did the user ask this
# again" would change meaning when the embedder was replaced.
_QUERY_NOISE = re.compile(r"[^\w\s]+", re.UNICODE)


_QUERY_SPACE = re.compile(r"\s+")


# Depth of every index route, deliberately not a function of the requested `limit`. Each route
# is truncated to it and a memory's dense relevance is the maximum over the routes that reached
# it, so a `limit`-derived depth made the score -- and the lexical term weights computed over the
# candidate pool -- properties of the request: at limit 20 against limit 100, 27-39 % of the
# candidates present in both runs reported a different dense relevance and 78-100 % a different
# lexical bonus, over adjacent-rank cosine gaps whose median is 0.0026-0.0077. Every public
# `limit` is capped at 100. Unquantized routes keep that fixed pool; quantized routes retain
# twice as many parents before authoritative FP32 scoring, without repeating that expansion
# in native grouping. The depth remains independent of the requested result count. Earlier
# widening to `limit * 3` cost 2.3x p50 on a common limit-8 query; the loop below widens further
# only when authoritative scope checks leave too few survivors.
_ROUTE_CANDIDATES = RERANK_CANDIDATES
_QUANTIZED_ROUTE_CANDIDATES = RERANK_CANDIDATES * 2


_MAX_QUERY_RETRIEVAL_KEYS = 7


class Retrieval(Traced):
    """One shared retrieval plane: index candidates, authoritative hydration, ranking."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        index: Index,
        backends: Backends,
        settings: Settings,
        lifecycle: Lifecycle,
        hydrator: Hydrator,
        materializer: Materializer,
        speech: Speech,
        vision: Vision,
        embedding: Embedding,
        projection: Projection,
        execution: Execution,
    ) -> None:
        super().__init__(tracer)
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._index = index
        self._backends = backends
        self._settings = settings
        self._lifecycle = lifecycle
        self._hydrator = hydrator
        self._materializer = materializer
        self._speech = speech
        self._vision = vision
        self._embedding = embedding
        self._projection = projection
        self._execution = execution

    def search(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> tuple[SearchHit, ...]:
        return self._search(
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
            capture_trace=False,
        ).hits

    def search_with_trace(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> TracedSearchResult:
        outcome = self._search(
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
            capture_trace=True,
        )
        assert outcome.trace is not None
        return TracedSearchResult(hits=outcome.hits, trace=outcome.trace)

    def _search(
        self,
        query: ContentInput,
        *,
        limit: int,
        memory_type: MemoryType | None,
        reference_at: datetime | None,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        scope: RetrievalScope | None,
        capture_trace: bool,
    ) -> SearchOutcome:
        with (
            self._trace("mindbridge.search", kind="operation"),
            self._lifecycle.operation(model_work=True) as assets,
        ):
            validate_limit(limit, maximum=100)
            occurred_from, occurred_until = search_occurrence_range(
                occurred_from,
                occurred_until,
            )
            scope = validated_retrieval_scope(scope)
            with self._trace("mindbridge.content.prepare", kind="stage"):
                prepared = self._materializer.prepare(query, assets)
            explicit_reference = resolved_reference_at(reference_at)
            reference, temporal_text = temporal_context(
                prepared.text,
                explicit_reference or datetime.now(timezone.utc),
                infer_reference=explicit_reference is None,
            )
            outcome = self.search_prepared(
                prepared,
                limit=limit,
                operation=assets,
                memory_types=pushed_memory_types(optional_memory_type(memory_type)),
                reference_at=reference,
                temporal_range=query_temporal_range(temporal_text, reference),
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                scope=scope,
                require_unambiguous=limit == 1,
                capture_trace=capture_trace,
            )
            self._speech.persist_transcripts(assets)
            self.note_query_failure(prepared.text, failed=not outcome.hits)
            return outcome

    def search_prepared(  # noqa: C901 - one shared retrieval plane
        self,
        prepared: PreparedContent,
        *,
        limit: int,
        operation: OperationAssets,
        memory_types: frozenset[MemoryType] | None,
        reference_at: datetime,
        temporal_range: tuple[datetime, datetime] | None,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        scope: RetrievalScope | None,
        require_unambiguous: bool,
        capture_trace: bool,
        capture_visual_matches: bool = False,
    ) -> SearchOutcome:
        trace_candidates: builtins.list[RetrievalCandidateTrace] | None = (
            [] if capture_trace else None
        )
        descriptions = (
            self._vision.query_descriptions(prepared, operation)
            if self._settings.query_vision
            else ()
        )
        vectors: tuple[tuple[float, ...], ...] = ()
        image_queries: tuple[tuple[str, tuple[float, ...]], ...] = ()
        if self._settings.retrieval_mode is RetrievalMode.LEXICAL:
            text_parts = tuple(value for kind, value in prepared.canonical_parts if kind == "text")
            focused_text = text_parts[0] if len(text_parts) > 1 else prepared.text
        else:
            # A lexical-only instance must not call a remote embedding service for its query.
            # Hybrid and dense preserve the existing multimodal preparation and query-key route.
            prepared = self._embedding.content(prepared, operation)
            text_parts = tuple(value for kind, value in prepared.canonical_parts if kind == "text")
            focused_text = text_parts[0] if len(text_parts) > 1 else prepared.text
            aggregate = self._embedding.route(prepared)
            focused = replace(
                prepared,
                text=focused_text,
                canonical_parts=(("text", focused_text),) if focused_text else (),
            )
            model_inputs = tuple(
                dict.fromkeys(
                    (
                        aggregate,
                        *self._embedding.inputs(
                            focused,
                            maximum_keys=_MAX_QUERY_RETRIEVAL_KEYS,
                        ),
                    )
                )
            )
            encoded = self._embedding.embed(model_inputs, task=EmbedTask.QUERY)
            vectors = tuple(dict.fromkeys(encoded))
            if capture_visual_matches:
                image_queries = tuple(
                    (value.assets[0].id, vector)
                    for value, vector in zip(model_inputs, encoded, strict=True)
                    if not value.text
                    and len(value.assets) == 1
                    and value.assets[0].modality is Modality.IMAGE
                )
            if descriptions and Modality.TEXT in self._backends.embedding_capabilities:
                try:
                    described_vectors = self._embedding.embed(
                        tuple(ModelInput(text=value) for value in descriptions),
                        task=EmbedTask.QUERY,
                    )
                except ModelError as error:
                    _LOGGER.warning("query description embedding failed: %s", error.reason)
                else:
                    vectors = tuple(dict.fromkeys((*vectors, *described_vectors)))
        lexical_queries = tuple(
            dict.fromkeys(
                value for value in (focused_text, *descriptions) if lexical_query_terms(value)
            )
        )
        if self._settings.retrieval_mode is RetrievalMode.DENSE:
            lexical_queries = ()
        # Quantized native order can hide a more relevant parent just below the ordinary
        # route window. Keep a wider, request-limit-independent pool for SQLite FP32 scoring.
        candidate_limit = (
            _ROUTE_CANDIDATES
            if self._settings.index_quantization is IndexQuantization.NONE
            else _QUANTIZED_ROUTE_CANDIDATES
        )
        candidate_ceiling = max(candidate_limit, limit * (MAX_RETRIEVAL_KEYS + 1))
        seen_index_ids: set[str] = set()
        # Both are static per document, so the index answers them itself instead of the loop
        # below widening its window until enough survivors of a post-filter turn up. That
        # widening is bounded, and for a one-person or one-room predicate the bound is reached
        # long before the matches are: the pushdown is what makes those scopes recall what the
        # store holds. SQLite still applies the same predicates below and decides.
        scope_place_id = None if scope is None else scope.place_id
        scope_identity_id = self._pushed_identity(scope)
        with self._write_lock:
            # A read asks for the current truth, so it closes an `add_stream` group that is still
            # open on this thread instead of skipping the drain with it: the consumer that searches
            # between two yields must see the item it was just handed.
            self._projection.drain(force=True)
        while True:
            with translate_index_errors("search memories"):
                if temporal_range is None:
                    candidates = self._index_candidates(
                        vectors,
                        lexical_queries=lexical_queries,
                        limit=candidate_limit,
                        memory_types=memory_types,
                        occurred_from=occurred_from,
                        occurred_until=occurred_until,
                        place_id=scope_place_id,
                        identity_id=scope_identity_id,
                        retrieval_mode=self._settings.retrieval_mode,
                    )
                else:
                    # Retrieve within the preferred window before truncating either route.
                    # Expanding global FTS until a rare window appears costs corpus-sized
                    # top-k results; native time filtering keeps the same bounded depth as
                    # dense retrieval. The global route preserves the soft time preference.
                    preferred_range = intersect_occurrence_range(
                        temporal_range,
                        occurred_from,
                        occurred_until,
                    )
                    preferred = (
                        IndexCandidates(dense=(), lexical=(), exhausted=True)
                        if preferred_range is None
                        else self._index_candidates(
                            vectors,
                            lexical_queries=lexical_queries,
                            limit=candidate_limit,
                            memory_types=memory_types,
                            occurred_from=preferred_range[0],
                            occurred_until=preferred_range[1],
                            place_id=scope_place_id,
                            identity_id=scope_identity_id,
                            retrieval_mode=self._settings.retrieval_mode,
                        )
                    )
                    fallback = self._index_candidates(
                        vectors,
                        lexical_queries=lexical_queries,
                        limit=candidate_limit,
                        memory_types=memory_types,
                        occurred_from=occurred_from,
                        occurred_until=occurred_until,
                        place_id=scope_place_id,
                        identity_id=scope_identity_id,
                        retrieval_mode=self._settings.retrieval_mode,
                    )
                    candidates = IndexCandidates(
                        dense=merge_index_hits(preferred.dense, fallback.dense),
                        lexical=merge_temporal_lexical_hits(preferred.lexical, fallback.lexical),
                        exhausted=preferred.exhausted and fallback.exhausted,
                    )
            index_ids = tuple(
                dict.fromkeys(hit.id for hit in (*candidates.dense, *candidates.lexical))
            )
            if not index_ids:
                return search_outcome(
                    (),
                    trace_candidates,
                    candidate_limit=candidate_limit,
                    exhaustive=candidates.exhausted,
                )
            with (
                self._trace("mindbridge.storage.hydrate", kind="stage"),
                translate_storage_errors("hydrate search candidates"),
            ):
                hydrated_documents = self._store.index.read_index_candidates(index_ids)
            documents = hydrated_documents
            if occurred_from is not None or occurred_until is not None:
                documents = tuple(
                    document
                    for document in documents
                    if occurrence_overlaps(
                        document.occurred_at,
                        document.occurred_end,
                        occurred_from,
                        occurred_until,
                    )
                )
            # Count scoped survivors only when the result can still widen the candidate window.
            if candidates.exhausted or candidate_limit >= candidate_ceiling:
                break
            current_index_ids = set(index_ids)
            if current_index_ids <= seen_index_ids:
                break
            candidate_parent_ids = tuple(
                dict.fromkeys(document.memory_id for document in documents)
            )
            with translate_storage_errors("apply search scope"):
                # Only the number of survivors decides whether to widen, so count them under
                # the same predicates instead of hydrating content, media assets and typed
                # context rows and calling `len()` on the records.
                active_count = self._store.records.count_memories(
                    candidate_parent_ids,
                    valid_at=None if scope is None else scope.valid_at,
                    known_at=None if scope is None else scope.known_at,
                    near=None if scope is None else scope.near,
                    radius_m=None if scope is None else scope.radius_m,
                    place_id=None if scope is None else scope.place_id,
                    identity_id=None if scope is None else scope.identity_id,
                    active_only=True,
                )
            if active_count >= limit:
                break
            seen_index_ids.update(current_index_ids)
            candidate_limit = min(candidate_limit * 2, candidate_ceiling)
        index_ids_by_memory = (
            parent_index_ids(hydrated_documents) if trace_candidates is not None else {}
        )
        extend_hydration_traces(
            trace_candidates,
            candidates,
            index_ids,
            hydrated_documents,
            documents,
            index_ids_by_memory,
            retrieval_mode=self._settings.retrieval_mode,
        )
        (
            dense_relevance,
            dense_confidence,
            lexical_relevance_by_rank,
            lexical_matches,
            matched_dense_index_ids,
        ) = parent_index_signals(candidates, documents)
        # `lexical_relevance_by_rank` rather than `lexical_matches`, which holds the same memory
        # ids in a set. Both mappings are filled in one pass over `documents`, so this preserves
        # native lexical-route insertion order instead of following the interpreter's string hash
        # seed. The visible effect is narrow -- `read_memories` does not
        # order by its argument, so the ranking and the ranked part of the trace never depended on
        # this, and the ranking sorts on `(-final_score, memory_id)` regardless -- but
        # `extend_missing_memory_traces` walks these ids directly, so a stale-index candidate's
        # position in `search_with_trace` did. A trace is a debugging surface; two runs of one
        # query on one library should print it in one order.
        parent_ids = tuple(dict.fromkeys((*dense_relevance, *lexical_relevance_by_rank)))
        if not parent_ids:
            return search_outcome(
                (),
                trace_candidates,
                candidate_limit=candidate_limit,
                exhaustive=candidates.exhausted,
            )
        with ExitStack() as candidate_lease, ExitStack() as scoring_snapshot:
            document_parts: Iterator[tuple[str, str, bytes]] | None = None
            document_vectors: Iterator[tuple[str, tuple[float, ...]]] | None = None
            with self._write_lock:
                with (
                    self._trace("mindbridge.storage.hydrate", kind="stage"),
                    translate_storage_errors("hydrate search results"),
                ):
                    memories = self._store.records.read_memories(
                        parent_ids,
                        valid_at=None if scope is None else scope.valid_at,
                        known_at=None if scope is None else scope.known_at,
                        near=None if scope is None else scope.near,
                        radius_m=None if scope is None else scope.radius_m,
                        place_id=None if scope is None else scope.place_id,
                        identity_id=None if scope is None else scope.identity_id,
                        active_only=True,
                    )
                kept = (
                    None
                    if memory_types is None
                    else frozenset(memory_type.value for memory_type in memory_types)
                )
                eligible_memories = tuple(
                    memory for memory in memories if kept is None or memory.memory_type in kept
                )
                image_matches = (
                    self._image_matches(image_queries, eligible_memories) if image_queries else ()
                )
                # Deletion must not separate authoritative hydration from media protection.
                # Candidates keep short leases; only winners retain them through the operation.
                candidate_lease.enter_context(
                    self._lifecycle.temporary_asset_lease(
                        tuple(asset for memory in eligible_memories for asset in memory.assets)
                    )
                )
                score_ids = tuple(
                    memory.memory_id
                    for memory in eligible_memories
                    if self._settings.index_quantization is not IndexQuantization.NONE
                    or (
                        memory.memory_id in lexical_matches
                        and memory.memory_id not in dense_relevance
                    )
                )
                if score_ids and vectors:
                    with translate_storage_errors("score search candidates"):
                        # BEGIN is deferred: consume the first row under the deletion lock to
                        # establish the WAL snapshot. Keep the stream bounded across SQL batches
                        # and release its reader immediately after scoring, before lexical work.
                        if self._settings.index_quantization is not IndexQuantization.NONE:
                            parts = scoring_snapshot.enter_context(
                                closing(
                                    self._store.index.iter_memory_embedding_parts(
                                        score_ids,
                                        space_id=self._backends.space_id,
                                        task=DOCUMENT_TASK,
                                    )
                                )
                            )
                            first_part = next(parts, None)
                            document_parts = (
                                iter(()) if first_part is None else chain((first_part,), parts)
                            )
                        else:
                            stored_vectors = scoring_snapshot.enter_context(
                                closing(
                                    self._store.index.iter_memory_embedding_vectors(
                                        score_ids,
                                        space_id=self._backends.space_id,
                                        task=DOCUMENT_TASK,
                                    )
                                )
                            )
                            first_vector = next(stored_vectors, None)
                            document_vectors = (
                                iter(())
                                if first_vector is None
                                else chain((first_vector,), stored_vectors)
                            )
            with self._trace("mindbridge.retrieval.rank", kind="stage"):
                extend_missing_memory_traces(
                    trace_candidates,
                    parent_ids,
                    memories,
                    index_ids_by_memory,
                    dense_relevance,
                    dense_confidence,
                    lexical_relevance_by_rank,
                    lexical_matches,
                    retrieval_mode=self._settings.retrieval_mode,
                )
                if kept is not None:
                    extend_memory_type_traces(
                        trace_candidates,
                        memories,
                        kept,
                        index_ids_by_memory,
                        dense_relevance,
                        dense_confidence,
                        lexical_relevance_by_rank,
                        lexical_matches,
                        retrieval_mode=self._settings.retrieval_mode,
                    )
                memories = eligible_memories
                if score_ids and vectors:
                    with (
                        self._trace("mindbridge.retrieval.score_completion", kind="stage"),
                        translate_storage_errors("score search candidates"),
                    ):
                        if document_parts is not None:
                            part_scores = max_cosine_part_scores(
                                vectors,
                                document_parts,
                                preferred_embeddings=matched_dense_index_ids,
                            )
                            for memory_id, (
                                embedding_id,
                                relevance,
                                confidence,
                            ) in part_scores.items():
                                dense_relevance[memory_id] = relevance
                                dense_confidence[memory_id] = confidence
                                matched_dense_index_ids[memory_id] = embedding_id
                        elif document_vectors is not None:
                            completed = max_cosine_scores(vectors, document_vectors)
                            for memory_id, (relevance, confidence) in completed.items():
                                dense_relevance[memory_id] = relevance
                                dense_confidence[memory_id] = confidence
                with translate_storage_errors("score search candidates"):
                    scoring_snapshot.close()
                lexical_coverage: dict[str, float] = {}
                if self._settings.retrieval_mode is RetrievalMode.HYBRID:
                    for query in lexical_queries:
                        for memory_id, strength in lexical_relevance(query, memories).items():
                            lexical_coverage[memory_id] = max(
                                lexical_coverage.get(memory_id, 0.0), strength
                            )
                ranked = []
                ranked_traces: dict[str, partial[RetrievalCandidateTrace]] | None = (
                    {} if trace_candidates is not None else None
                )
                for memory in memories:
                    memory_id = memory.memory_id
                    lexical_match = memory_id in lexical_matches
                    lexical_strength = (
                        lexical_coverage.get(memory_id, 0.0) if lexical_match else 0.0
                    )
                    lexical_score = (
                        LEXICAL_FULL_COVERAGE_RELEVANCE
                        * lexical_relevance_by_rank.get(memory_id, 0.0)
                        if lexical_strength >= LEXICAL_FULL_COVERAGE
                        else 0.0
                    )
                    if self._settings.retrieval_mode is RetrievalMode.LEXICAL:
                        # This diagnostic mode exposes Zvec's own normalized full-text route:
                        # stemmed and n-gram FTS, RRF-fused when both apply. Hybrid-only term
                        # coverage and dense fusion must not alter its admission or order.
                        relevance = lexical_relevance_by_rank.get(memory_id, 0.0)
                        base = relevance
                        lexical_score = relevance
                    else:
                        base = max(dense_relevance.get(memory_id, 0.0), lexical_score)
                        relevance = bounded_scale(
                            base,
                            1.0 + MAX_LEXICAL_RERANK_BONUS * lexical_strength,
                        )
                    lexical_rerank_bonus = relevance - base
                    (
                        final_score,
                        reinforcement_factor,
                        temporal_factor,
                        retention_factor,
                    ) = ranking_signals(
                        memory,
                        relevance,
                        reference_at=reference_at,
                        temporal_range=temporal_range,
                        decay_half_life=self._settings.decay_half_life,
                    )
                    # `minimum_relevance` gates evidence quality on the relevance scale: how well
                    # this memory matches, times how sure the observation itself was. It used to
                    # gate dense *confidence*, `(1 + cosine) / 2`, which made the 0.55 default
                    # admit cosine 0.15 and reject 0.05 while the caller read cosine back; and any
                    # full-text hit was handed a flat 0.6 there whatever its match strength, so one
                    # shared rare term carried a document at cosine -1.0 past the default floor and
                    # no floor above 0.6 could keep the lexical route at all. Both routes now
                    # contribute on this one scale and scale with match strength.
                    #
                    # The gate takes the signals the *query* asked about and leaves out the ones it
                    # did not. Temporal proximity is in: a caller who asks "in 2024" made the year
                    # part of the question, so overlapping it is evidence and missing it is not,
                    # and the boost is what lets an in-window full-text hit clear a floor its bare
                    # rank could not. Reinforcement and `decay_half_life_days` retention are out:
                    # neither is anything the query mentioned. They are bounded below by
                    # `RANK_FLOOR`, so with retention inside the gate a *perfectly* relevant
                    # memory decayed to 0.30 and to 0.09 once a window also missed it, under the
                    # 0.10 default — turning "prefer recent" into "hide old" for exactly the
                    # caller who enabled decay and then asked about last year. A long-lived
                    # personal memory may rank an old event last; it may not stop returning it.
                    #
                    # The cost is that `SearchHit.score` carries every factor and so can sit below
                    # the floor the caller set. `search_with_trace` reports the gated quantity as
                    # `gate_relevance`, beside the `retention_factor` that moved the score off it.
                    gate_relevance = relevance
                    if temporal_factor is not None:
                        gate_relevance = bounded_scale(gate_relevance, temporal_factor)
                    if memory.context is not None:
                        final_score *= memory.context.confidence
                        gate_relevance *= memory.context.confidence
                    ranked.append(
                        (
                            memory,
                            final_score,
                            gate_relevance,
                            lexical_match,
                        )
                    )
                    record_ranked_trace(
                        ranked_traces,
                        memory_id,
                        index_ids_by_memory,
                        dense_relevance,
                        dense_confidence,
                        lexical_relevance=lexical_score,
                        lexical_rerank_bonus=lexical_rerank_bonus,
                        lexical_match=lexical_match,
                        gate_relevance=gate_relevance,
                        base_relevance=relevance,
                        reinforcement_factor=reinforcement_factor,
                        temporal_factor=temporal_factor,
                        retention_factor=retention_factor,
                        final_score=final_score,
                    )
                ranked = qualified_candidates(
                    ranked,
                    minimum_relevance=self._settings.minimum_relevance,
                    trace_candidates=trace_candidates,
                    ranked_traces=ranked_traces,
                )
                ranked.sort(key=lambda item: (-item[1], item[0].memory_id))
                ambiguous = require_unambiguous and retrieval_is_ambiguous(
                    ranked,
                    margin=self._settings.ambiguity_margin,
                    temporal_range=temporal_range,
                )
                extend_ranked_traces(
                    trace_candidates,
                    ranked_traces,
                    ranked,
                    limit=limit,
                    ambiguous=ambiguous,
                )
                visible = () if ambiguous else ranked[:limit]
                self._lifecycle.lease_assets(
                    tuple(
                        asset
                        for memory, _score, _confidence, _lexical in visible
                        for asset in memory.assets
                    ),
                    operation.leased,
                )
                operation.persisted.update(
                    asset.asset_id
                    for memory, _score, _confidence, _lexical in visible
                    for asset in memory.assets
                )
        hits = tuple(
            self._hydrator.search_hit(memory, score)
            for memory, score, _confidence, _lexical in visible
        )
        return search_outcome(
            hits,
            trace_candidates,
            candidate_limit=candidate_limit,
            exhaustive=candidates.exhausted,
            ambiguous=ambiguous,
            matched_dense_index_ids={
                hit.id: matched_dense_index_ids[hit.id]
                for hit in hits
                if hit.id in matched_dense_index_ids
            },
            image_retrieval_matches=image_matches,
        )

    def _image_matches(
        self,
        queries: Sequence[tuple[str, tuple[float, ...]]],
        memories: Sequence[StoredMemory],
    ) -> tuple[VisualRetrievalMatch, ...]:
        """Preserve image-only provenance within the authoritative, scoped candidate pool."""
        visual = tuple(
            memory
            for memory in memories
            if any(asset.modality == "image" for asset in memory.assets)
        )
        if not visual:
            return ()
        with translate_storage_errors("read image retrieval keys"):
            documents = self._store.index.read_memory_index_documents(
                tuple(memory.memory_id for memory in visual)
            )
        by_memory: builtins.dict[str, builtins.list[StoredEmbedding]] = {}
        for document in documents:
            embedding = document.embedding
            if embedding.space_id == self._backends.space_id and embedding.task == DOCUMENT_TASK:
                by_memory.setdefault(embedding.memory_id, []).append(embedding)
        parts = tuple(
            (memory.memory_id, asset_id, embedding)
            for memory in visual
            for asset_id, embedding in self._embedding.image_parts(
                memory, by_memory.get(memory.memory_id, ())
            )
        )
        result: builtins.list[VisualRetrievalMatch] = []
        for query_id, query in queries:
            scores = max_cosine_scores(
                (query,), ((embedding.embedding_id, embedding.values) for _, _, embedding in parts)
            )
            ordered = sorted(
                parts,
                key=lambda part: (-scores[part[2].embedding_id][1], part[0], part[1]),
            )[:3]
            result.extend(
                VisualRetrievalMatch(
                    query_asset_id=query_id,
                    memory_id=memory_id,
                    source_asset_id=asset_id,
                    rank=rank,
                    cosine_similarity=2.0 * scores[embedding.embedding_id][1] - 1.0,
                )
                for rank, (memory_id, asset_id, embedding) in enumerate(ordered, 1)
            )
        return tuple(result)

    def _pushed_identity(self, scope: RetrievalScope | None) -> str | None:
        """Resolve a scoped identity to the ID the index projection stores, or `None`.

        A merge re-points every projected row onto the survivor, so a request naming a merged
        alias has to be resolved before it can be pushed down. An unknown identity is pushed
        unresolved: it matches nothing in the index, which is what SQLite decides too.
        """
        if scope is None or scope.identity_id is None:
            return None
        with translate_storage_errors("resolve a scoped identity"):
            return (
                self._store.identities.resolve_identity_id(scope.identity_id) or scope.identity_id
            )

    def _index_candidates(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        lexical_queries: Sequence[str],
        limit: int,
        memory_types: frozenset[MemoryType] | None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> IndexCandidates:
        # The index filter takes one type, so a set becomes one route per type rather than a
        # post-filter over a shared window. Each type then gets the full `limit` of depth, which
        # is what lets a request for a rare type reach records a common type outranks.
        type_values: tuple[str | None, ...] = (
            (None,)
            if memory_types is None
            else tuple(sorted(memory_type.value for memory_type in memory_types))
        )
        dense_calls = tuple(
            partial(
                self._index.search,
                vector,
                limit=limit,
                space_id=self._backends.space_id,
                task=DOCUMENT_TASK,
                memory_type=value,
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                place_id=place_id,
                identity_id=identity_id,
            )
            for value in type_values
            for vector in vectors
            if retrieval_mode is not RetrievalMode.LEXICAL
        )
        lexical_calls = (
            tuple(
                partial(
                    self._index.lexical_search,
                    query,
                    limit=limit,
                    space_id=self._backends.space_id,
                    task=DOCUMENT_TASK,
                    memory_type=value,
                    occurred_from=occurred_from,
                    occurred_until=occurred_until,
                    place_id=place_id,
                    identity_id=identity_id,
                )
                for value in type_values
                for query in lexical_queries
            )
            if retrieval_mode is not RetrievalMode.DENSE
            else ()
        )
        calls = (*dense_calls, *lexical_calls)
        routes: tuple[tuple[IndexHit, ...], ...]
        with self._trace("mindbridge.index.search", kind="stage") as span:
            span.set_attribute("mindbridge.index.route_count", len(calls))
            if not calls:
                return IndexCandidates(dense=(), lexical=(), exhausted=True)
            routes = self._execution.index_routes(calls)
        dense_routes = routes[: len(dense_calls)]
        lexical_routes = routes[len(dense_calls) :]
        return IndexCandidates(
            dense=merge_index_hits(*dense_routes),
            # One route keeps the index's own order. Several routes are re-ordered by native
            # relevance so the result does not reflect their route interleaving.
            lexical=merge_lexical_routes(lexical_routes),
            exhausted=all(len(route) < limit for route in routes),
        )

    def note_query_failure(self, text: str, *, failed: bool) -> None:
        """Record one empty recall, which is the whole of the QUERY_FAILURE signal.

        The stored query is the owner's own words about their own memory, in their own memory
        domain, and the table is capped on write so it stays a signal buffer rather than a
        growing query log. A non-text query records nothing: there is nothing to compare two of
        them by.
        """
        if not failed:
            return
        folded = unicodedata.normalize("NFKC", text).casefold()
        normalized = _QUERY_SPACE.sub(" ", _QUERY_NOISE.sub(" ", folded)).strip()
        if not normalized:
            return
        with self._write_lock, translate_storage_errors("record a query failure"):
            self._store.control.record_query_failure(
                text,
                normalized,
                failed_at=datetime.now(timezone.utc),
                keep=self._settings.query_failure_history,
            )
