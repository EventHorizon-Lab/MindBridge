"""The retrieval plane: candidates from the index, truth from SQLite, one ranking."""

from __future__ import annotations

import builtins
import re
import unicodedata
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from contextvars import copy_context
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial

from opentelemetry.trace import Tracer

from mindbridge.infrastructure.local.scoring import max_cosine_scores
from mindbridge.infrastructure.local.store import IndexCandidate
from mindbridge.infrastructure.local.zvec_index import IndexHit
from mindbridge.kernel.content import PreparedContent
from mindbridge.kernel.contracts import Backends
from mindbridge.kernel.embedding import DOCUMENT_TASK, MAX_RETRIEVAL_KEYS, Embedding
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
    overlaps_temporal_range,
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
from mindbridge.models.base import EmbedTask
from mindbridge.types import (
    ContentInput,
    MemoryType,
    RetrievalCandidateTrace,
    RetrievalMode,
    RetrievalScope,
    SearchHit,
    TracedSearchResult,
)

RERANK_CANDIDATES = 100


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
# `limit` is capped at 100, so one pool of `RERANK_CANDIDATES` serves them all and the rerank
# pool and the route depth are the same number. Widening it to the deepest previous pool
# (`limit * 3` at limit 100) instead would have cost 2.3x the p50 of a `limit` 8 search, which
# is the common one; a caller who needs more survivors than this pool holds still deepens it
# through the widening loop below.
_ROUTE_CANDIDATES = RERANK_CANDIDATES


_MAX_QUERY_RETRIEVAL_KEYS = 7


_MAX_INDEX_SEARCH_WORKERS = 4


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
        embedding: Embedding,
        projection: Projection,
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
        self._embedding = embedding
        self._projection = projection

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
            self._lifecycle.operation() as assets,
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
    ) -> SearchOutcome:
        trace_candidates: builtins.list[RetrievalCandidateTrace] | None = (
            [] if capture_trace else None
        )
        vectors: tuple[tuple[float, ...], ...] = ()
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
            vectors = tuple(
                dict.fromkeys(self._embedding.embed(model_inputs, task=EmbedTask.QUERY))
            )
        lexical_query = focused_text
        if not lexical_query_terms(lexical_query):
            lexical_query = ""
        if self._settings.retrieval_mode is RetrievalMode.DENSE:
            lexical_query = ""
        candidate_limit = _ROUTE_CANDIDATES
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
                        lexical_query=lexical_query,
                        limit=candidate_limit,
                        memory_types=memory_types,
                        occurred_from=occurred_from,
                        occurred_until=occurred_until,
                        place_id=scope_place_id,
                        identity_id=scope_identity_id,
                        retrieval_mode=self._settings.retrieval_mode,
                    )
                else:
                    # Zvec range-filtered FTS is unstable after dense queries; the global
                    # fallback below still contributes the same lexical route.
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
                            lexical_query="",
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
                        lexical_query=lexical_query,
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
                        lexical=merge_index_hits(preferred.lexical, fallback.lexical),
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
            with translate_index_errors("search memories"):
                candidates, hydrated_documents = self._deepen_temporal_lexical_candidates(
                    candidates,
                    hydrated_documents,
                    lexical_query=lexical_query,
                    temporal_range=temporal_range,
                    memory_types=memory_types,
                    route_limit=candidate_limit,
                    result_limit=limit,
                    place_id=scope_place_id,
                    identity_id=scope_identity_id,
                    retrieval_mode=self._settings.retrieval_mode,
                )
            index_ids = tuple(
                dict.fromkeys(hit.id for hit in (*candidates.dense, *candidates.lexical))
            )
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
            # The survivor count exists only to decide whether to widen, so neither it nor the
            # slate it counts is built where it can no longer change that decision. Every other
            # reason to stop is already known here, and asking first made a second scope pass
            # over the whole candidate window part of the price of every search that was never
            # going to widen -- which is every search whose routes came back short, and every
            # one that had already reached the ceiling.
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
                    # Passed here for consistency with every other scope axis, which all
                    # reach both reads. This one is the survivor count that drives candidate
                    # widening; no constructed corpus (30 or 120 memories) could make its
                    # absence change a result, so it is unproven rather than proven needed.
                    # Kept because omitting one axis at one of two sites is the anomaly a
                    # reader would have to explain, and because narrowing a count can only
                    # widen the search. The hydration site below is mutation-covered.
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
                if memory_types is not None:
                    kept = frozenset(memory_type.value for memory_type in memory_types)
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
                    memories = tuple(memory for memory in memories if memory.memory_type in kept)
                lexical_only_ids = tuple(
                    memory.memory_id
                    for memory in memories
                    if memory.memory_id in lexical_matches
                    and memory.memory_id not in dense_relevance
                )
                if lexical_only_ids and vectors:
                    with (
                        self._trace("mindbridge.retrieval.score_completion", kind="stage"),
                        translate_storage_errors("complete hybrid candidate scores"),
                        closing(
                            self._store.index.iter_memory_embedding_vectors(
                                lexical_only_ids,
                                space_id=self._backends.space_id,
                                task=DOCUMENT_TASK,
                            )
                        ) as document_vectors,
                    ):
                        completed = max_cosine_scores(vectors, document_vectors)
                    for memory_id, (relevance, confidence) in completed.items():
                        dense_relevance[memory_id] = relevance
                        dense_confidence[memory_id] = confidence
                lexical_coverage = (
                    lexical_relevance(lexical_query, memories)
                    if self._settings.retrieval_mode is RetrievalMode.HYBRID
                    else {}
                )
                ranked = []
                ranked_traces: dict[str, RetrievalCandidateTrace] | None = (
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
        )

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

    def _deepen_temporal_lexical_candidates(
        self,
        candidates: IndexCandidates,
        documents: tuple[IndexCandidate, ...],
        *,
        lexical_query: str,
        temporal_range: tuple[datetime, datetime] | None,
        memory_types: frozenset[MemoryType] | None,
        route_limit: int,
        result_limit: int,
        place_id: str | None = None,
        identity_id: str | None = None,
        retrieval_mode: RetrievalMode,
    ) -> tuple[IndexCandidates, tuple[IndexCandidate, ...]]:
        if temporal_range is None or not lexical_query or len(candidates.lexical) < route_limit:
            return candidates, documents
        lexical_by_id = {hit.id: hit for hit in candidates.lexical}
        # A heuristic proxy for "worth deepening for", NOT a bound on the gate. It cannot be one:
        # Hybrid applies a full-coverage demotion before gating lexical evidence; lexical mode
        # uses the native normalized FTS relevance directly. The threshold remains only a
        # deepening heuristic, not a final admission criterion.
        lexical_gate_scale = (
            1.0 if retrieval_mode is RetrievalMode.LEXICAL else LEXICAL_FULL_COVERAGE_RELEVANCE
        )
        qualified_parents = {
            document.memory_id
            for document in documents
            if (hit := lexical_by_id.get(document.embedding_id)) is not None
            and lexical_gate_scale * hit.relevance >= self._settings.minimum_relevance
            and overlaps_temporal_range(
                document.occurred_at,
                document.occurred_end,
                temporal_range,
            )
        }
        if len(qualified_parents) >= result_limit:
            return candidates, documents
        with (
            self._trace("mindbridge.storage.temporal_candidates", kind="stage"),
            translate_storage_errors("read temporal search candidates"),
        ):
            in_range, total = self._store.index.embedding_ids_in_range(
                *temporal_range,
                space_id=self._backends.space_id,
                task=DOCUMENT_TASK,
                # The whitelist only narrows what the deepening loop will consider, and the
                # ranking stage applies the full set afterwards, so one pushed type is a
                # sharpening and several are correctly left to it.
                memory_type=(
                    next(iter(memory_types)).value
                    if memory_types is not None and len(memory_types) == 1
                    else None
                ),
            )
        if not in_range:
            return candidates, documents
        required = min(result_limit, len(in_range))
        search_limit = route_limit
        lexical = candidates.lexical
        qualified = tuple(
            hit
            for hit in lexical
            if hit.id in in_range
            and lexical_gate_scale * hit.relevance >= self._settings.minimum_relevance
        )
        while len(qualified) < required and len(lexical) >= search_limit and search_limit < total:
            search_limit = min(search_limit * 2, total)
            lexical = self._index_candidates(
                (),
                lexical_query=lexical_query,
                limit=search_limit,
                memory_types=memory_types,
                place_id=place_id,
                identity_id=identity_id,
                retrieval_mode=retrieval_mode,
            ).lexical
            qualified = tuple(
                hit
                for hit in lexical
                if hit.id in in_range
                and lexical_gate_scale * hit.relevance >= self._settings.minimum_relevance
            )
        updated = IndexCandidates(
            dense=candidates.dense,
            lexical=merge_index_hits(candidates.lexical, qualified[:route_limit]),
            exhausted=candidates.exhausted,
        )
        added_ids = tuple(hit.id for hit in updated.lexical if hit.id not in lexical_by_id)
        if not added_ids:
            return updated, documents
        with (
            self._trace("mindbridge.storage.hydrate", kind="stage"),
            translate_storage_errors("hydrate temporal lexical candidates"),
        ):
            added = self._store.index.read_index_candidates(added_ids)
        by_id = {document.embedding_id: document for document in (*documents, *added)}
        return updated, tuple(by_id.values())

    def _index_candidates(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        lexical_query: str,
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
                    lexical_query,
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
            )
            if lexical_query and retrieval_mode is not RetrievalMode.DENSE
            else ()
        )
        calls = (*dense_calls, *lexical_calls)
        routes: tuple[tuple[IndexHit, ...], ...]
        with self._trace("mindbridge.index.search", kind="stage") as span:
            span.set_attribute("mindbridge.index.route_count", len(calls))
            if not calls:
                return IndexCandidates(dense=(), lexical=(), exhausted=True)
            if len(calls) == 1:
                routes = (calls[0](),)
            else:
                with ThreadPoolExecutor(
                    max_workers=min(_MAX_INDEX_SEARCH_WORKERS, len(calls))
                ) as executor:
                    futures = tuple(executor.submit(copy_context().run, call) for call in calls)
                    routes = tuple(future.result() for future in futures)
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
