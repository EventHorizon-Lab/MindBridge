"""The compile plane: closed, budgeted context over already-retrieved evidence.

Selection and structuring are pure and live in `mindbridge.context`; this module owns the
store-backed reads that close evidence over lineage, consent, and actors before that pure step
runs.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timezone
from functools import partial
from time import perf_counter

from opentelemetry.trace import Tracer

from mindbridge.context import EvidenceClosure, NamedActorLink, _rejection, compile_context
from mindbridge.exceptions import ValidationError
from mindbridge.kernel.content import PreparedContent, verified_context_excerpt
from mindbridge.kernel.contracts import Backends
from mindbridge.kernel.hydration import Hydrator
from mindbridge.kernel.lifecycle import Lifecycle
from mindbridge.kernel.materialization import Materializer
from mindbridge.kernel.retrieval import RERANK_CANDIDATES, Retrieval
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.speech import Speech
from mindbridge.kernel.temporal import query_temporal_range, resolved_reference_at, temporal_context
from mindbridge.kernel.tracing import Traced
from mindbridge.kernel.validation import (
    modality_names,
    pushed_memory_types,
    scope_description,
    validated_retrieval_scope,
)
from mindbridge.types import (
    ContentInput,
    ContextBudget,
    ContextBundle,
    ContextExcerpt,
    ContextUnknown,
    ContextUnknownKind,
    EvidenceBasis,
    MemoryKind,
    Modality,
    RetrievalScope,
    SearchHit,
)

# How much deeper `compile()` ranks when the budget carries a bound only the compiler can
# apply -- `min_confidence` or `freshness`. Those remove candidates after retrieval, so the
# window has to hold what they will remove for `candidates_exhausted` to mean the store ran
# out rather than the window did.
_POST_FILTER_WINDOW = 3


# Context compilation may complete an affirmative functional claim with low-relevance competing
# values.  This is an obligation lookup, never a second relevance route, so a finite cap must
# refuse rather than silently make a prefix look complete.
_LINEAGE_COMPLETION_REPRESENTATIVES = 8


def _functional_lineage(hit: SearchHit) -> tuple[str, str] | None:
    """Return the claim key whose competing values make delivery non-affirmative.

    This matches the compiler and slow-loop contract: relations and inferred traits can
    accumulate values, while STATE and a user-stated TRAIT have one functional value at a scoped
    point in time.
    """
    context = hit.context
    if (
        context is None
        or context.lineage_id is None
        or context.value is None
        or not (
            context.kind is MemoryKind.STATE
            or (context.kind is MemoryKind.TRAIT and context.basis is EvidenceBasis.USER_STATEMENT)
        )
    ):
        return None
    return context.lineage_id, context.value


def _functional_representatives(
    hits: Sequence[SearchHit],
) -> dict[str, dict[str, list[str]]]:
    """Group candidate IDs by functional lineage and value, preserving rank order."""
    values_by_lineage: dict[str, dict[str, list[str]]] = {}
    for hit in hits:
        functional = _functional_lineage(hit)
        if functional is None:
            continue
        lineage_id, value = functional
        values_by_lineage.setdefault(lineage_id, {}).setdefault(value, []).append(hit.id)
    return values_by_lineage


def _incomplete_functional_lineages(
    competing_values: Mapping[str, Mapping[str, Sequence[str]]],
    candidates: Mapping[str, SearchHit],
    closure_ids: Collection[str],
    budget: ContextBudget,
    reference_at: datetime,
    capped_lineages: Collection[str],
) -> set[str]:
    """Return lineages with a value that has no filter-eligible evidence closure."""
    incomplete = set(capped_lineages)
    for lineage_id, values in competing_values.items():
        if any(
            not any(
                memory_id in closure_ids
                and _rejection(candidates[memory_id], budget, reference_at) is None
                for memory_id in representatives
            )
            for representatives in values.values()
        ):
            incomplete.add(lineage_id)
    return incomplete


def _consent_withheld_required_closure(
    closures: Sequence[EvidenceClosure],
    consent_closure_ids: Collection[str],
    values_by_lineage: Mapping[str, Mapping[str, Sequence[str]]],
    candidates: Mapping[str, SearchHit],
    budget: ContextBudget,
    reference_at: datetime,
) -> bool:
    """Whether consent removed a closure without leaving a same-value alternative."""
    for closure in closures:
        if closure.anchor.id in consent_closure_ids:
            continue
        functional = _functional_lineage(closure.anchor)
        if functional is None:
            return True
        lineage_id, value = functional
        if not any(
            representative_id in consent_closure_ids
            and _rejection(candidates[representative_id], budget, reference_at) is None
            for representative_id in values_by_lineage[lineage_id][value]
        ):
            return True
    return False


def _deliverable_functional_representatives(
    hits: Sequence[SearchHit],
    closure_ids: Collection[str],
    budget: ContextBudget,
    reference_at: datetime,
) -> set[str]:
    """Choose the first deliverable candidate for every functional value."""
    selected_ids: set[str] = set()
    selected_values: set[tuple[str, str]] = set()
    for hit in hits:
        functional = _functional_lineage(hit)
        if functional is None or functional in selected_values:
            continue
        if hit.id not in closure_ids or _rejection(hit, budget, reference_at) is not None:
            continue
        selected_values.add(functional)
        selected_ids.add(hit.id)
    return selected_ids


def _closure_contains_functional_lineage(
    closure: EvidenceClosure,
    lineage_ids: Collection[str],
) -> bool:
    """Whether one closure carries a claim in a lineage this compilation withheld.

    Completion finds unsafe lineages only for ranked functional anchors.  A separately ranked
    derived assertion may still cite one as support, and closed selection would otherwise
    materialize that support through the derived closure.  This check deliberately makes no new
    discovery beneath arbitrary support: it only enforces the concrete obligations completion
    already found for this compilation.
    """
    return any(
        functional is not None and functional[0] in lineage_ids
        for member in closure.members
        if (functional := _functional_lineage(member)) is not None
    )


class Compilation(Traced):
    """Close evidence over lineage, consent, and actors, then compile a context bundle."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        backends: Backends,
        lifecycle: Lifecycle,
        hydrator: Hydrator,
        materializer: Materializer,
        speech: Speech,
        retrieval: Retrieval,
    ) -> None:
        super().__init__(tracer)
        self._store = storage.store
        self._backends = backends
        self._lifecycle = lifecycle
        self._hydrator = hydrator
        self._materializer = materializer
        self._speech = speech
        self._retrieval = retrieval

    def compile(
        self,
        goal: ContentInput,
        *,
        budget: ContextBudget | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        allow_partial_sources: bool = False,
    ) -> ContextBundle:
        with (
            self._trace("mindbridge.compile", kind="operation"),
            self._lifecycle.operation() as assets,
        ):
            # Taken before anything else, so `budget.max_latency_ms` bounds the compilation a
            # caller waits for rather than the selection pass alone.
            started_at = perf_counter()
            budget = ContextBudget() if budget is None else budget
            if not isinstance(budget, ContextBudget):
                raise ValidationError("budget must be a ContextBudget")
            if not isinstance(allow_partial_sources, bool):
                raise ValidationError("allow_partial_sources must be a boolean")
            scope = validated_retrieval_scope(scope)
            with self._trace("mindbridge.content.prepare", kind="stage"):
                prepared = self._materializer.prepare(goal, assets)
            explicit_reference = resolved_reference_at(reference_at)
            reference, temporal_text = temporal_context(
                prepared.text,
                explicit_reference or datetime.now(timezone.utc),
                infer_reference=explicit_reference is None,
            )
            # `validate_limit` bounds what a caller may ask a public search to return, not how deep the
            # kernel ranks, so `_search_prepared` has no ceiling of its own and a bundle may rank
            # past one hundred. Three candidates per slot is the same headroom `ask()` gives its
            # modality round robin -- widened when the budget carries a bound the index cannot
            # answer, because those are applied to the window after it comes back and a window
            # sized for the bundle alone would report exhaustion the store does not have.
            candidate_limit = max(RERANK_CANDIDATES, budget.max_items * 3) * (
                _POST_FILTER_WINDOW
                if budget.min_confidence > 0.0 or budget.freshness is not None
                else 1
            )
            outcome = self._retrieval.search_prepared(
                prepared,
                limit=candidate_limit,
                operation=assets,
                # `memory_types` is index-pushable one type at a time, so the kernel opens one
                # route per type. Without that a request for a rare type competes for the same
                # window as every common one and comes back empty while the store holds records.
                memory_types=pushed_memory_types(budget.memory_types),
                reference_at=reference,
                temporal_range=query_temporal_range(temporal_text, reference),
                occurred_from=None,
                occurred_until=None,
                scope=scope,
                require_unambiguous=False,
                capture_trace=False,
            )
            # The same transcript cache `search()` writes for a spoken query. It is a cache of
            # the query's own audio, not a memory: `compile()` stores nothing it retrieved.
            self._speech.persist_transcripts(assets)
            completion_hits, capped_lineages = self._competing_lineage_hits(
                outcome.hits,
                scope=scope,
            )
            candidate_by_id = {hit.id: hit for hit in outcome.hits}
            for hit in completion_hits:
                candidate_by_id.setdefault(hit.id, hit)
            candidates = tuple(candidate_by_id.values())
            closures, unavailable = self._evidence_closures(
                candidates,
                budget=budget,
                reference_at=reference,
                scope=scope,
                started_at=started_at,
            )
            closure_ids = frozenset(closure.anchor.id for closure in closures)
            values_by_lineage = _functional_representatives(candidates)
            competing_values = {
                lineage_id: values
                for lineage_id, values in values_by_lineage.items()
                if len(values) > 1
            }
            incomplete_lineages = _incomplete_functional_lineages(
                competing_values,
                candidate_by_id,
                closure_ids,
                budget,
                reference,
                capped_lineages,
            )
            if incomplete_lineages:
                # A functional assertion is never affirmative by itself once its indexed
                # lineage completion found an unrenderable competitor or reached its cap.  The
                # raw observations supporting it remain ordinary evidence, but every claim in
                # the unsafe lineage is withheld before the compiler can fall back to flat rank.
                withheld_ids = frozenset(
                    hit.id
                    for hit in candidates
                    if (functional := _functional_lineage(hit)) is not None
                    and functional[0] in incomplete_lineages
                )
                ranked_withheld = sum(1 for hit in outcome.hits if hit.id in withheld_ids)
                candidates = tuple(hit for hit in candidates if hit.id not in withheld_ids)
                closures = tuple(
                    closure
                    for closure in closures
                    if not _closure_contains_functional_lineage(closure, incomplete_lineages)
                )
                unavailable = (
                    *unavailable,
                    ContextUnknown(
                        kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                        detail=(
                            f"{ranked_withheld} ranked assertions required complete competing"
                            " evidence under the requested scope"
                        ),
                    ),
                )
            closure_hits = tuple(
                {
                    hit.id: hit
                    for hit in (
                        *candidates,
                        *(hit for closure in closures for hit in closure.members),
                    )
                }.values()
            )
            provisional = self._provisional_identities(closure_hits, scope=scope)
            permitted, named, withheld, restrained = self._consented_actors(
                closure_hits,
                self._named_actors(closure_hits, scope=scope),
                observed_identity_ids=frozenset(
                    identity_id
                    for identity_ids in provisional.values()
                    for identity_id in identity_ids
                ),
            )
            permitted_ids = {hit.id for hit in permitted}
            closures_before_consent = closures
            closures = tuple(
                closure
                for closure in closures
                if all(member.id in permitted_ids for member in closure.members)
            )
            hits = tuple(hit for hit in candidates if hit.id in permitted_ids)
            consent_closure_ids = frozenset(closure.anchor.id for closure in closures)
            withheld_closures = _consent_withheld_required_closure(
                closures_before_consent,
                consent_closure_ids,
                values_by_lineage,
                candidate_by_id,
                budget,
                reference,
            )
            consent_incomplete_lineages = {
                lineage_id
                for lineage_id, values in competing_values.items()
                if lineage_id not in incomplete_lineages
                if any(
                    not any(
                        memory_id in permitted_ids and memory_id in consent_closure_ids
                        for memory_id in representatives
                    )
                    for representatives in values.values()
                )
            }
            if consent_incomplete_lineages:
                # Consent cannot turn a previously known competitor into a silent agreement.  Do
                # not name the withheld identity or value here: the only safe presentation is
                # that the affirmative claim's competing evidence is unavailable.
                consent_withheld_ids = frozenset(
                    hit.id
                    for hit in hits
                    if (functional := _functional_lineage(hit)) is not None
                    and functional[0] in consent_incomplete_lineages
                )
                hits = tuple(hit for hit in hits if hit.id not in consent_withheld_ids)
                closures = tuple(
                    closure
                    for closure in closures
                    if not _closure_contains_functional_lineage(
                        closure, consent_incomplete_lineages
                    )
                )
                unavailable = (
                    *unavailable,
                    ContextUnknown(
                        kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                        detail="one or more ranked assertions required competing evidence withheld by consent",
                    ),
                )
            consent_closure_ids = frozenset(closure.anchor.id for closure in closures)
            deliverable_functional_ids = _deliverable_functional_representatives(
                hits, consent_closure_ids, budget, reference
            )
            hits = tuple(
                hit
                for hit in hits
                if (functional := _functional_lineage(hit)) is None
                or hit.id in deliverable_functional_ids
            )
            closures = tuple(
                closure
                for closure in closures
                if _functional_lineage(closure.anchor) is None
                or closure.anchor.id in deliverable_functional_ids
            )
            excerpts = (
                self._context_excerpts(
                    hits,
                    dict(outcome.matched_dense_index_ids),
                )
                if allow_partial_sources
                else {}
            )
            if withheld_closures:
                unavailable = (
                    *unavailable,
                    ContextUnknown(
                        kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                        detail="one or more ranked assertions required evidence withheld by consent",
                    ),
                )
            bundle = compile_context(
                prepared.text,
                hits,
                budget=budget,
                reference_at=reference,
                started_at=started_at,
                unknowns=(
                    *self._request_unknowns(prepared, scope, hits),
                    *withheld,
                    *unavailable,
                ),
                candidate_limit=candidate_limit,
                provisional={
                    memory_id: tuple(
                        identity_id for identity_id in identity_ids if identity_id not in restrained
                    )
                    for memory_id, identity_ids in self._provisional_identities(
                        permitted, scope=scope
                    ).items()
                    if any(identity_id not in restrained for identity_id in identity_ids)
                },
                named=named,
                co_derived_events=partial(self._co_derived_events, scope=scope),
                closures=closures,
                excerpts=excerpts,
            )
            # QUERY_FAILURE hook: a bundle with no evidence in it is the compiler reporting that
            # the goal found nothing, which is the same signal as an empty search.
            self._retrieval.note_query_failure(
                prepared.text,
                failed=not bundle.hits and not bundle.excerpts,
            )
            return bundle

    def _context_excerpts(
        self,
        hits: Sequence[SearchHit],
        matched_dense_index_ids: Mapping[str, str],
    ) -> dict[str, ContextExcerpt]:
        """Verify compiler-only excerpts for scoped, consented raw candidates."""
        eligible = tuple(
            hit
            for hit in hits
            if hit.id in matched_dense_index_ids
            and hit.modality is Modality.TEXT
            and not hit.assets
            and (hit.context is None or hit.context.kind is MemoryKind.OBSERVATION)
        )
        if not eligible:
            return {}
        embedding_ids = tuple(matched_dense_index_ids[hit.id] for hit in eligible)
        with translate_storage_errors("read text selectors"):
            selectors = self._store.index.read_text_selectors(embedding_ids)
        excerpts: dict[str, ContextExcerpt] = {}
        for hit in eligible:
            embedding_id = matched_dense_index_ids[hit.id]
            for stored in selectors.get(embedding_id, ()):
                excerpt = verified_context_excerpt(hit, embedding_id, stored)
                if excerpt is not None:
                    excerpts[hit.id] = excerpt
                    break
        return excerpts

    def _competing_lineage_hits(
        self,
        anchors: Sequence[SearchHit],
        *,
        scope: RetrievalScope | None,
    ) -> tuple[tuple[SearchHit, ...], frozenset[str]]:
        """Complete ranked functional claims with bounded, scope-correct competitors.

        Retrieval chooses the anchors.  This indexed SQLite read only asks whether a ranked
        functional lineage has a differently valued claim the context compiler must carry or
        withhold.  It assigns zero relevance to the completed rows, so they cannot become a new
        relevance route or displace an ordinary result except as an anchor's own obligation.
        """
        lineage_ids = tuple(
            dict.fromkeys(
                functional[0]
                for anchor in anchors
                if (functional := _functional_lineage(anchor)) is not None
            )
        )
        if not lineage_ids:
            return (), frozenset()
        with translate_storage_errors("complete competing lineage evidence"):
            memories, incomplete = self._store.semantics.read_functional_lineage_representatives(
                lineage_ids,
                per_lineage_limit=_LINEAGE_COMPLETION_REPRESENTATIVES,
                valid_at=None if scope is None else scope.valid_at,
                known_at=None if scope is None else scope.known_at,
                near=None if scope is None else scope.near,
                radius_m=None if scope is None else scope.radius_m,
                place_id=None if scope is None else scope.place_id,
                identity_id=None if scope is None else scope.identity_id,
            )
        return tuple(self._hydrator.search_hit(memory, 0.0) for memory in memories), incomplete

    def _evidence_closures(  # noqa: C901 - bounded graph traversal keeps scope checks in one place
        self,
        anchors: Sequence[SearchHit],
        *,
        budget: ContextBudget,
        reference_at: datetime,
        scope: RetrievalScope | None,
        started_at: float | None = None,
    ) -> tuple[tuple[EvidenceClosure, ...], tuple[ContextUnknown, ...]]:
        """Hydrate finite, scoped transitive provenance for ranked compilation anchors."""
        started_at = perf_counter() if started_at is None else started_at
        by_id = {anchor.id: anchor for anchor in anchors}
        frontier = tuple(
            source_id
            for anchor in anchors
            if anchor.context is not None
            for source_id in anchor.context.evidence_ids
        )
        expanded: set[str] = set()
        bounded: set[str] = set()
        deadline = False
        max_nodes = max(1, len(anchors) * budget.max_items)
        while frontier:
            if (
                budget.max_latency_ms is not None
                and (perf_counter() - started_at) * 1_000 > budget.max_latency_ms
            ):
                deadline = True
                break
            current = tuple(dict.fromkeys(frontier))
            wanted = tuple(source_id for source_id in current if source_id not in by_id)
            selected: tuple[str, ...] = ()
            if wanted:
                remaining = max_nodes - len(by_id)
                if remaining <= 0:
                    bounded.update(wanted)
                    break
                selected, rejected = wanted[:remaining], wanted[remaining:]
                bounded.update(rejected)
                with translate_storage_errors("hydrate compilation evidence"):
                    hydrated = self._store.records.read_memories(
                        selected,
                        valid_at=None if scope is None else scope.valid_at,
                        known_at=None if scope is None else scope.known_at,
                        near=None if scope is None else scope.near,
                        radius_m=None if scope is None else scope.radius_m,
                        place_id=None if scope is None else scope.place_id,
                        identity_id=None if scope is None else scope.identity_id,
                        active_only=True,
                    )
                for memory in hydrated:
                    by_id[memory.memory_id] = self._hydrator.search_hit(memory, 0.0)
            frontier = tuple(
                child
                for source_id in current
                if source_id not in expanded
                and (source := by_id.get(source_id)) is not None
                and source.context is not None
                for child in source.context.evidence_ids
            )
            expanded.update(current)
        closures: list[EvidenceClosure] = []
        failed: dict[str, str] = {}
        max_depth = min(budget.max_items, 256)

        def traverse(  # noqa: C901 - tri-colour validation retains all hard eligibility checks
            anchor: SearchHit,
        ) -> tuple[tuple[str, ...], str | None]:
            states: dict[str, int] = {}
            ordered: list[str] = []

            def visit(memory_id: str, depth: int) -> str | None:
                if depth > max_depth:
                    return "bounded"
                if memory_id in bounded:
                    return "bounded"
                hit = by_id.get(memory_id)
                if hit is None:
                    return (
                        "unavailable before the compilation deadline"
                        if deadline
                        else "unavailable under the requested scope"
                    )
                if _rejection(hit, budget, reference_at) is not None:
                    return "excluded by the requested context bounds"
                state = states.get(memory_id, 0)
                if state == 1:
                    return "cyclic"
                if state == 2:
                    return None
                states[memory_id] = 1
                context = hit.context
                if context is not None:
                    for source_id in context.evidence_ids:
                        if (reason := visit(source_id, depth + 1)) is not None:
                            return reason
                states[memory_id] = 2
                ordered.append(memory_id)
                return None

            reason = visit(anchor.id, 0)
            return tuple(ordered), reason

        for anchor in anchors:
            ordered, reason = traverse(anchor)
            if reason is not None:
                failed[anchor.id] = reason
                continue
            # `visit` is postorder; the anchor remains query-ranked and leads its own closure.
            closures.append(
                EvidenceClosure(
                    anchor,
                    (
                        anchor,
                        *(by_id[memory_id] for memory_id in ordered if memory_id != anchor.id),
                    ),
                )
            )
        reasons: dict[str, int] = {}
        for reason in failed.values():
            reasons[reason] = reasons.get(reason, 0) + 1
        unknowns = tuple(
            ContextUnknown(
                kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                detail=f"{count} ranked assertions required evidence {reason}",
            )
            for reason, count in sorted(reasons.items())
        )
        return tuple(closures), unknowns

    def _co_derived_events(
        self,
        cues: Sequence[str],
        *,
        scope: RetrievalScope | None,
    ) -> dict[str, tuple[str, ...]]:
        """Resolve, for the selected affect entries, the events their observations also formed.

        One batched read per compilation, under the same bitemporal and spatial scope the hits
        were hydrated with, so the hop cannot surface an event the search itself would have
        dropped. The compiler calls this after selection, so the read is bounded by the budget
        rather than by the candidate window. The edge is shared evidence -- co-occurrence inside
        one capture -- and never a cause.
        """
        if not cues:
            return {}
        with translate_storage_errors("read co-derived events"):
            return self._store.identities.co_derived_events(
                cues,
                valid_at=None if scope is None else scope.valid_at,
                known_at=None if scope is None else scope.known_at,
                near=None if scope is None else scope.near,
                radius_m=None if scope is None else scope.radius_m,
                place_id=None if scope is None else scope.place_id,
                identity_id=None if scope is None else scope.identity_id,
            )

    def _consented_actors(
        self,
        hits: Sequence[SearchHit],
        named: Mapping[str, tuple[NamedActorLink, ...]],
        *,
        observed_identity_ids: frozenset[str] = frozenset(),
    ) -> tuple[
        tuple[SearchHit, ...],
        dict[str, tuple[NamedActorLink, ...]],
        tuple[ContextUnknown, ...],
        frozenset[str],
    ]:
        """Drop the named actors whose subject withheld or withdrew consent, and say so.

        Only the naming assertion is dropped, not the person's memories: consent governs being
        treated as a recognized person, not whether the evening happened. The bundle then says
        a person is missing from `actors` rather than silently compiling a scene with a hole in
        it, which is what makes the omission auditable instead of a retrieval mystery.

        Both routes to an actor are filtered, because both publish the same person. A bound
        `ENTITY` hit renders as its own line, and a resolved identity edge on any other memory
        -- the face in a photo, the speaker in a clip -- becomes a `NamedActor` whether or not
        the assertion that names them ranked at all.
        """
        bound = {
            hit.context.identity_id
            for hit in hits
            if hit.context is not None
            and hit.context.identity_id is not None
            and hit.context.kind is MemoryKind.ENTITY
        }
        # Every identity the enrichment could name, not only the ones a retrieved `ENTITY` hit
        # carried: a bundle that reached a person's clip but not their naming assertion used to
        # find nothing bound here, consult no consent at all, and then name them from the clip.
        bound.update(link[0] for links in named.values() for link in links)
        bound.update(observed_identity_ids)
        if not bound:
            return tuple(hits), dict(named), (), frozenset()
        with translate_storage_errors("read identity consent"):
            restrained = self._store.identities.restrained_identities() & bound
        if not restrained:
            return tuple(hits), dict(named), (), frozenset()
        kept = tuple(
            hit
            for hit in hits
            if hit.context is None
            or hit.context.kind is not MemoryKind.ENTITY
            or hit.context.identity_id not in restrained
        )
        permitted = {
            memory_id: kept_links
            for memory_id, links in named.items()
            if (kept_links := tuple(link for link in links if link[0] not in restrained))
        }
        return (
            kept,
            permitted,
            (
                ContextUnknown(
                    kind=ContextUnknownKind.CONSENT_WITHHELD,
                    detail=(
                        f"{len(restrained)} recognized "
                        f"{'person is' if len(restrained) == 1 else 'people are'} withheld from"
                        " the actors section by their own recorded consent"
                    ),
                ),
            ),
            restrained,
        )

    def _provisional_identities(
        self,
        hits: Sequence[SearchHit],
        *,
        scope: RetrievalScope | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Resolve which people the candidate evidence observed but nobody has named.

        Deterministic kernel policy, like every other identity decision: a person is
        provisional exactly while no visible naming assertion names them.
        """
        if not hits:
            return {}
        with translate_storage_errors("read provisional identities"):
            return self._store.identities.provisional_identities(
                tuple(hit.id for hit in hits),
                valid_at=None if scope is None else scope.valid_at,
                known_at=None if scope is None else scope.known_at,
            )

    def _named_actors(
        self,
        hits: Sequence[SearchHit],
        *,
        scope: RetrievalScope | None = None,
    ) -> dict[str, tuple[NamedActorLink, ...]]:
        """Resolve which people the candidate evidence's identity edge already names.

        Deterministic kernel policy, the positive counterpart of `_provisional_identities`: a
        person is named exactly while a visible naming assertion names them, whether or not
        that assertion itself ranked into the bundle.
        """
        if not hits:
            return {}
        with translate_storage_errors("read named actors"):
            return self._store.identities.named_actors(
                tuple(hit.id for hit in hits),
                valid_at=None if scope is None else scope.valid_at,
                known_at=None if scope is None else scope.known_at,
            )

    def _request_unknowns(
        self,
        prepared: PreparedContent,
        scope: RetrievalScope | None,
        hits: Sequence[SearchHit],
    ) -> tuple[ContextUnknown, ...]:
        """Name what the request asked for that the compiler cannot see in the hits alone."""
        unknowns: list[ContextUnknown] = []
        unsupported = sorted(
            {
                asset.modality
                for asset in prepared.assets
                if Modality(asset.modality) not in self._backends.embedding_capabilities
            }
        )
        if unsupported:
            unknowns.append(
                ContextUnknown(
                    kind=ContextUnknownKind.MODALITY_UNSUPPORTED,
                    detail=(
                        f"the goal's {', '.join(unsupported)} was not embedded natively;"
                        f" embedding accepts {modality_names(self._backends.embedding_capabilities)},"
                        " so only derived text could have matched"
                    ),
                )
            )
        if not hits and (described := scope_description(scope)) is not None:
            unknowns.append(ContextUnknown(kind=ContextUnknownKind.SCOPE_EMPTY, detail=described))
        return tuple(unknowns)
