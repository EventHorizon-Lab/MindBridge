"""Pure context selection behind `Memory.compile`.

The compiler selects and structures already-retrieved evidence. It calls no model, resolves no
conflict, and writes nothing itself: every value here is a deterministic function of the ranked
hits, the budget, the reference clock, and -- for the deadline alone -- the wall clock.
Authoritative visibility and scope were applied by the retrieval path that produced the hits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from time import perf_counter
from types import MappingProxyType
from typing import TypeAlias

from mindbridge.types import (
    KIND_MEMORY_TYPES,
    ContextBudget,
    ContextBundle,
    ContextConflict,
    ContextUnknown,
    ContextUnknownKind,
    MemoryKind,
    MemoryType,
    Modality,
    NamedActor,
    ProvisionalActor,
    SearchHit,
    render_hit_line,
    render_named_actor_line,
    render_provisional_actor_line,
)

# One (identity_id, name, naming_assertion_id) edge a memory carries to a currently named
# identity. `naming_assertion_id` is `None` only when a caller cannot cite it cheaply.
NamedActorLink: TypeAlias = tuple[str, str, str | None]

# Text-equivalent cost of one grounded media part, at the usual four characters per token. The
# modalities are an order of magnitude apart in what a model charges for them: an image part
# measured near five hundred tokens against this stack where a ten-second video part measured
# near three thousand, so one flat number would either starve text or overrun on video. These
# are coarse by design; the budget is the caller's knob, not these constants.
_ASSET_EVIDENCE_CHARS: Mapping[Modality, int] = MappingProxyType(
    {
        Modality.IMAGE: 2_000,
        Modality.AUDIO: 4_000,
        Modality.VIDEO: 12_000,
    }
)
_DEFAULT_ASSET_EVIDENCE_CHARS = 4_000
# Bundle sections, keyed by the `ContextBundle` field they fill.
_SECTIONS: tuple[str, ...] = (
    "actors",
    "relationships",
    "scene",
    "facts",
    "episodes",
    "procedures",
    "affect",
    "traits",
)
# A typed kind decides the section first; an untyped or generic record falls back to its type.
_KIND_SECTIONS: Mapping[MemoryKind, str] = MappingProxyType(
    {
        MemoryKind.ENTITY: "actors",
        MemoryKind.RELATION: "relationships",
        MemoryKind.STATE: "scene",
        MemoryKind.AFFECT: "affect",
        MemoryKind.TRAIT: "traits",
    }
)
_TYPE_SECTIONS: Mapping[MemoryType, str] = MappingProxyType(
    {
        MemoryType.SEMANTIC: "facts",
        MemoryType.EPISODIC: "episodes",
        MemoryType.PROCEDURAL: "procedures",
    }
)
# Which memory types can reach each section at all. A kind-keyed section is fed by whatever type
# a formed record of that kind is stored as, a type-keyed one by its own type. This is what lets a
# `memory_types` filter name the sections it emptied instead of only the types it dropped: kind
# `affect` is stored `episodic`, so a `{semantic}` request cannot fill `affect` however it ranks.
_SECTION_MEMORY_TYPES: Mapping[str, frozenset[MemoryType]] = MappingProxyType(
    {
        name: frozenset(
            {
                KIND_MEMORY_TYPES.get(kind, MemoryType.SEMANTIC)
                for kind, section in _KIND_SECTIONS.items()
                if section == name
            }
            | {memory_type for memory_type, section in _TYPE_SECTIONS.items() if section == name}
        )
        for name in _SECTIONS
    }
)
# Kinds that assert one value about one subject, so two of them in a lineage can disagree.
_CONFLICT_KINDS = frozenset({MemoryKind.STATE, MemoryKind.RELATION, MemoryKind.TRAIT})
# Why a candidate the retrieval kernel returned never became a bundle line. Reported as counts
# under `unknowns` so a caller reading a thin bundle knows which bound produced it.
_EXCLUSIONS: Mapping[str, str] = MappingProxyType(
    {
        "memory_types": "outside the requested memory types",
        "min_confidence": "below the requested minimum confidence",
        "freshness": "older than the requested freshness window",
    }
)
# One asserted value: the value, the memory asserting it, and that memory's subject and predicate.
_Claim: TypeAlias = tuple[str, str, str | None, str | None]


def evidence_cost(hit: SearchHit) -> int:
    """Return what grounding one hit costs, in characters of text-equivalent evidence."""
    return len(hit.content) + _asset_cost(hit)


def _asset_cost(hit: SearchHit) -> int:
    # `AssetRef.modality` is optional on the public type; an unresolved asset is charged the
    # default rather than being treated as free.
    return sum(
        _DEFAULT_ASSET_EVIDENCE_CHARS
        if asset.modality is None
        else _ASSET_EVIDENCE_CHARS.get(asset.modality, _DEFAULT_ASSET_EVIDENCE_CHARS)
        for asset in hit.assets
    )


def bundle_cost(hit: SearchHit) -> int:
    """Return what one hit costs a bundle: its rendered line plus its media parts.

    The rendered line, not the record text: the `- [id]` frame and the confidence suffix are
    characters the consumer pays for, so charging only `len(content)` would hand a caller a
    budget that `render()` overruns. Media parts are charged their text equivalent, which is far
    more than the zero characters they render as, so the number is an upper bound on the text.
    """
    return len(render_hit_line(hit)) + 1 + _asset_cost(hit)


def named_actor_cost(actor: NamedActor) -> int:
    """Return what one named-actor line costs, priced exactly as `render_named_actor_line`
    renders it, so the price and the text can never drift apart.
    """
    return len(render_named_actor_line(actor)) + 1


def provisional_actor_cost(actor: ProvisionalActor) -> int:
    """Return what one provisional-actor line costs, priced exactly as
    `render_provisional_actor_line` renders it.
    """
    return len(render_provisional_actor_line(actor)) + 1


def _actor_cost(entry: NamedActor | ProvisionalActor) -> int:
    if isinstance(entry, NamedActor):
        return named_actor_cost(entry)
    return provisional_actor_cost(entry)


def _frame_cost(goal: str, reference_at: datetime, budget: ContextBudget) -> int:
    """Return an upper bound on the fixed header `render()` writes above the first section."""
    return (
        len(f"# Context: {goal}")
        + len("Each line is one memory: [id] content (confidence; validity).")
        + len(f"Reference time: {reference_at.isoformat()}")
        # `chars` cannot have more digits than `max_chars`, nor `included` than `max_items`.
        + len(
            f"Budget: {budget.max_chars}/{budget.max_chars} chars,"
            f" {budget.max_items}/{budget.max_items} items"
        )
        + 4
    )


def _heading_cost(section: str) -> int:
    """Return what one section heading costs: a blank line, `## `, the name, two newlines."""
    return len(section) + 5


def compile_context(
    goal: str,
    hits: Sequence[SearchHit],
    *,
    budget: ContextBudget,
    reference_at: datetime,
    started_at: float | None = None,
    unknowns: Sequence[ContextUnknown] = (),
    candidate_limit: int | None = None,
    provisional: Mapping[str, Sequence[str]] = MappingProxyType({}),
    named: Mapping[str, Sequence[NamedActorLink]] = MappingProxyType({}),
) -> ContextBundle:
    """Partition, filter, and budget ranked hits into one bundle.

    `started_at` is a `time.perf_counter()` reading taken before retrieval, so `budget
    .max_latency_ms` bounds the whole compilation rather than this function. `unknowns` carries
    what the caller already knows and the compiler cannot see, such as an empty spatial scope.
    `candidate_limit` is the depth retrieval was asked to rank, so a bundle that lost evidence
    and filled that window can say the ranking may continue past it.

    `provisional` maps a memory ID to the recognized people it observed whom no visible naming
    assertion names. The kernel resolves that, deterministically, before calling; the compiler
    only keeps the entries whose evidence actually made it into the bundle, so the bundle never
    reports a person the reader cannot see the evidence for.

    `named` maps a memory ID to the identities its identity edge resolves to that a currently
    visible naming assertion names -- the positive counterpart of `provisional`, resolved the
    same way by the kernel before calling. An identity `named` reports is never also reported
    as provisional, even if a stale `provisional` entry still names it. Both a named and a
    provisional actor render a line, and that line is priced and fit into whatever `max_chars`
    has left after the ranked hits: it is included when it fits and dropped, not appended for
    free, when it does not.
    """
    started_at = perf_counter() if started_at is None else started_at
    excluded: dict[str, int] = {}
    candidates: list[SearchHit] = []
    for hit in hits:
        reason = _rejection(hit, budget, reference_at)
        if reason is None:
            candidates.append(hit)
        else:
            excluded[reason] = excluded.get(reason, 0) + 1
    overhead = _frame_cost(goal, reference_at, budget)
    sections = _select(candidates, budget, overhead)
    # Rank order, not section order: `_lineage_conflict` pairs each value with the highest-ranked
    # memory asserting it, so a lineage whose claims land in different sections must still be read
    # by score. `ContextBundle.hits` sorts the same way.
    included = tuple(
        sorted(
            (hit for name in _SECTIONS for hit in sections[name]),
            key=lambda hit: (-hit.score, hit.id),
        )
    )
    omitted = len(candidates) - len(included)
    # The deadline is checked here, between section assembly and the optional enrichment that
    # follows it. Nothing already computed is discarded and no stage is cut in half, so a bundle
    # under a deadline is a prefix of the one without it, never a different one.
    skipped = _past_deadline(budget, started_at)
    # Conflict detection reads every candidate the filters kept, not only what the budget bought,
    # so dropping one side of a disagreement for want of a slot cannot make it disappear.
    conflicts = (
        ()
        if skipped
        else _conflicts(
            sorted(candidates, key=lambda hit: (-hit.score, hit.id)),
            frozenset(hit.id for hit in included),
        )
    )
    occurred_from, occurred_until = _occurred_range(included)
    elapsed_ms = _elapsed_ms(started_at)
    # Actor lines are priced and fit in after the ranked hits, into whatever `_select` left of
    # `max_chars`: `_bundle_chars` already bounds that at or below `max_chars`, so this can
    # never push the total over it. A named identity is never also reported provisional, even
    # if a stale `provisional` entry still names it.
    base_chars = _bundle_chars(overhead, sections)
    named_actors = _named_actors(included, named)
    provisional_actors = _provisional_actors(
        included,
        provisional,
        exclude=frozenset(actor.identity_id for actor in named_actors),
    )
    fitted_actors, actor_chars, actor_excluded = _fit_actors(
        (*named_actors, *provisional_actors),
        budget.max_chars - base_chars,
    )
    return ContextBundle(
        goal=goal,
        reference_at=reference_at,
        budget=budget,
        actors=(*sections["actors"], *fitted_actors),
        relationships=sections["relationships"],
        scene=sections["scene"],
        episodes=sections["episodes"],
        facts=sections["facts"],
        procedures=sections["procedures"],
        affect=sections["affect"],
        traits=sections["traits"],
        conflicts=conflicts,
        unknowns=_unknowns(
            unknowns,
            budget,
            sections,
            excluded,
            omitted,
            skipped=skipped,
            exhausted=(
                candidate_limit is not None
                and len(hits) >= candidate_limit
                and bool(excluded or omitted)
            ),
            candidate_limit=candidate_limit,
            actor_excluded=actor_excluded,
        ),
        occurred_from=occurred_from,
        occurred_until=occurred_until,
        frames=_frames(included),
        places=tuple(sorted({hit.place_id for hit in included if hit.place_id})),
        omitted=omitted,
        chars=base_chars + actor_chars,
        elapsed_ms=elapsed_ms,
        deadline_exceeded=(
            budget.max_latency_ms is not None and elapsed_ms > budget.max_latency_ms
        ),
    )


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


def _past_deadline(budget: ContextBudget, started_at: float) -> bool:
    """Whether the declared deadline had already passed at this checkpoint."""
    return budget.max_latency_ms is not None and _elapsed_ms(started_at) > budget.max_latency_ms


def _rejection(hit: SearchHit, budget: ContextBudget, reference_at: datetime) -> str | None:
    """Return which budget bound excludes this hit, or `None` when it is a candidate."""
    if budget.memory_types is not None and hit.memory_type not in budget.memory_types:
        return "memory_types"
    # A record with no typed context is evidence in its own right, not a low-confidence inference.
    confidence = 1.0 if hit.context is None else hit.context.confidence
    if confidence < budget.min_confidence:
        return "min_confidence"
    if budget.freshness is not None and reference_at - _anchor(hit) > budget.freshness:
        return "freshness"
    return None


def _anchor(hit: SearchHit) -> datetime:
    """Freshness anchors on event end, then event start, then when the record was written."""
    return hit.occurred_end or hit.occurred_at or hit.created_at


def _select(
    candidates: Sequence[SearchHit],
    budget: ContextBudget,
    overhead: int,
) -> dict[str, tuple[SearchHit, ...]]:
    """Rank decides the top half of `max_items`; the bottom half seats the sections it missed.

    Section diversity must not cost rank readability. A floor slot per section at a small budget
    lets a rank-fifty trait evict a rank-two episode, and a bundle that no longer reads by score
    is worse than one missing a section. So the floor round runs only when the bottom half of
    `max_items` can seat every section the candidates span; below that, rank decides all of it.
    """
    sections: dict[str, list[SearchHit]] = {name: [] for name in _SECTIONS}
    taken: set[str] = set()
    used = overhead
    media = 0
    head = -(-budget.max_items // 2)
    spanned = {_section(hit) for hit in candidates}
    rounds: tuple[tuple[int, bool], ...] = (
        ((head, False), (budget.max_items, True), (budget.max_items, False))
        if budget.max_items - head >= len(spanned)
        else ((budget.max_items, False),)
    )
    for ceiling, floor in rounds:
        for hit in candidates:
            if len(taken) >= ceiling:
                break
            section = _section(hit)
            if hit.id in taken or (floor and sections[section]):
                continue
            if budget.max_media_items is not None and (
                media + len(hit.assets) > budget.max_media_items
            ):
                continue
            cost = bundle_cost(hit) + (0 if sections[section] else _heading_cost(section))
            # One oversized hit does not close the bundle: a cheaper lower-ranked candidate can
            # still fit, and `omitted` reports everything the budget could not buy.
            if used + cost > budget.max_chars:
                continue
            sections[section].append(hit)
            taken.add(hit.id)
            used += cost
            media += len(hit.assets)
    return {name: tuple(section) for name, section in sections.items()}


def _bundle_chars(overhead: int, sections: Mapping[str, tuple[SearchHit, ...]]) -> int:
    """Return what this bundle charged against `max_chars`, priced exactly as `_select` did."""
    return overhead + sum(
        _heading_cost(name) + sum(bundle_cost(hit) for hit in section)
        for name, section in sections.items()
        if section
    )


def _section(hit: SearchHit) -> str:
    context = hit.context
    if context is not None and context.kind in _KIND_SECTIONS:
        return _KIND_SECTIONS[context.kind]
    return _TYPE_SECTIONS[hit.memory_type]


def _unknowns(
    supplied: Sequence[ContextUnknown],
    budget: ContextBudget,
    sections: Mapping[str, tuple[SearchHit, ...]],
    excluded: Mapping[str, int],
    omitted: int,
    *,
    skipped: bool,
    exhausted: bool = False,
    candidate_limit: int | None = None,
    actor_excluded: int = 0,
) -> tuple[ContextUnknown, ...]:
    """Name what the request implied and this bundle does not carry, in one stable order."""
    found = list(supplied)
    found.extend(
        ContextUnknown(
            kind=ContextUnknownKind.BUDGET_EXCLUDED,
            detail=f"{excluded[reason]} candidates {text}",
        )
        for reason, text in _EXCLUSIONS.items()
        if excluded.get(reason)
    )
    if omitted:
        found.append(
            ContextUnknown(
                kind=ContextUnknownKind.BUDGET_EXCLUDED,
                detail=(
                    f"{omitted} candidates did not fit {budget.max_items} items"
                    f" and {budget.max_chars} chars"
                ),
            )
        )
    if actor_excluded:
        found.append(
            ContextUnknown(
                kind=ContextUnknownKind.BUDGET_EXCLUDED,
                detail=f"{actor_excluded} actor lines did not fit {budget.max_chars} chars",
            )
        )
    if exhausted:
        found.append(
            ContextUnknown(
                kind=ContextUnknownKind.CANDIDATES_EXHAUSTED,
                detail=(
                    f"retrieval filled its {candidate_limit} candidate window, so the counts"
                    " above bound what this window held and not what the store holds"
                ),
            )
        )
    found.extend(_section_empty(budget, sections))
    if skipped:
        found.append(
            ContextUnknown(
                kind=ContextUnknownKind.STAGE_SKIPPED,
                detail=(
                    "conflict detection was skipped after the"
                    f" {budget.max_latency_ms} ms deadline passed"
                ),
            )
        )
    return tuple(sorted(found, key=lambda item: (item.kind.value, item.detail)))


def _section_empty(
    budget: ContextBudget,
    sections: Mapping[str, tuple[SearchHit, ...]],
) -> tuple[ContextUnknown, ...]:
    """Name each section a `memory_types` request left empty, and why it is empty.

    Only a request that named types gets these: without one an empty section means the store
    holds nothing for it, which is not a statement about this request. With one, an empty
    section is either a section the filter can never fill -- its records carry a type the
    request excluded -- or one the filter admits and no record of that type reached the bundle
    at all (`feeds.isdisjoint(present)`). The second variant is deliberately narrower than "this
    section is empty": a kind-keyed section (`actors`, `scene`, `affect`, `traits`) that is empty
    while a *different* section already carries a record of the same underlying `MemoryType`
    stays silent, because flagging it would be noise in most bundles -- the type was not missing,
    it just formed into another kind.
    """
    requested = budget.memory_types
    if requested is None:
        return ()
    present = frozenset(hit.memory_type for section in sections.values() for hit in section)
    found: list[ContextUnknown] = []
    for name in _SECTIONS:
        if sections[name]:
            continue
        feeds = _SECTION_MEMORY_TYPES[name]
        if feeds.isdisjoint(requested):
            detail = (
                f"the {name} section is empty: memory_types kept only"
                f" {_type_names(requested)}, and {name} carries only"
                f" {_type_names(feeds)} memory"
            )
        elif feeds <= requested and feeds.isdisjoint(present):
            detail = f"the {name} section is empty: no {_type_names(feeds)} memory was included"
        else:
            continue
        found.append(ContextUnknown(kind=ContextUnknownKind.SECTION_EMPTY, detail=detail))
    return tuple(found)


def _type_names(memory_types: frozenset[MemoryType]) -> str:
    return ", ".join(memory_type.value for memory_type in sorted(memory_types))


def _conflicts(
    hits: Sequence[SearchHit],
    included_ids: frozenset[str],
) -> tuple[ContextConflict, ...]:
    lineages: dict[str, list[_Claim]] = {}
    for hit in hits:
        context = hit.context
        if context is None or context.kind not in _CONFLICT_KINDS:
            continue
        lineage_id, value = context.lineage_id, context.value
        if lineage_id is None or value is None:
            continue
        lineages.setdefault(lineage_id, []).append(
            (value, hit.id, context.subject, context.predicate)
        )
    # A lineage no included memory takes part in is not this bundle's disagreement to report.
    return tuple(
        conflict
        for lineage_id, claims in lineages.items()
        if any(memory_id in included_ids for _value, memory_id, _subject, _predicate in claims)
        and (conflict := _lineage_conflict(lineage_id, claims)) is not None
    )


def _lineage_conflict(lineage_id: str, claims: Sequence[_Claim]) -> ContextConflict | None:
    claimed: dict[str, str] = {}
    for value, memory_id, _subject, _predicate in claims:
        claimed.setdefault(value, memory_id)
    if len(claimed) < 2:
        return None
    _value, _memory_id, subject, predicate = claims[0]
    return ContextConflict(
        lineage_id=lineage_id,
        subject=subject,
        predicate=predicate,
        values=tuple(claimed),
        memory_ids=tuple(claimed.values()),
    )


def _occurred_range(hits: Sequence[SearchHit]) -> tuple[datetime | None, datetime | None]:
    spans = tuple(
        (hit.occurred_at, hit.occurred_end or hit.occurred_at)
        for hit in hits
        if hit.occurred_at is not None
    )
    if not spans:
        return None, None
    return min(start for start, _end in spans), max(end for _start, end in spans)


def _frames(hits: Sequence[SearchHit]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                hit.context.spatial.frame_id
                for hit in hits
                if hit.context is not None and hit.context.spatial is not None
            }
        )
    )


def _provisional_actors(
    hits: Sequence[SearchHit],
    provisional: Mapping[str, Sequence[str]],
    *,
    exclude: frozenset[str] = frozenset(),
) -> tuple[ProvisionalActor, ...]:
    """Name every unnamed person the included evidence observed, in identity order.

    These are appended to the ranked actors rather than competing with them for an item slot:
    an unnamed person earned no hit slot, and dropping them would leave an agent unable to say
    that somebody it does not recognize is in the room. `exclude` drops an identity `named`
    already reported, so a stale `provisional` entry can never list the same identity twice.
    """
    if not provisional:
        return ()
    observed: dict[str, list[str]] = {}
    for hit in hits:
        for identity_id in provisional.get(hit.id, ()):
            if identity_id in exclude:
                continue
            memory_ids = observed.setdefault(identity_id, [])
            if hit.id not in memory_ids:
                memory_ids.append(hit.id)
    return tuple(
        ProvisionalActor(identity_id=identity_id, memory_ids=tuple(memory_ids))
        for identity_id, memory_ids in sorted(observed.items())
    )


def _named_actors(
    hits: Sequence[SearchHit],
    named: Mapping[str, Sequence[NamedActorLink]],
) -> tuple[NamedActor, ...]:
    """Name every identity the included evidence's identity edge names, in identity order.

    Aggregated across every included memory that carries the edge, the same way
    `_provisional_actors` aggregates an unnamed person's observing memories, and dropped
    entirely when the evidence that carried the edge did not make it into the bundle.
    """
    if not named:
        return ()
    names: dict[str, str] = {}
    assertions: dict[str, str | None] = {}
    observed: dict[str, list[str]] = {}
    for hit in hits:
        for identity_id, name, naming_assertion_id in named.get(hit.id, ()):
            names[identity_id] = name
            assertions[identity_id] = naming_assertion_id
            memory_ids = observed.setdefault(identity_id, [])
            if hit.id not in memory_ids:
                memory_ids.append(hit.id)
    return tuple(
        NamedActor(
            identity_id=identity_id,
            name=names[identity_id],
            memory_ids=tuple(memory_ids),
            naming_assertion_id=assertions[identity_id],
        )
        for identity_id, memory_ids in sorted(observed.items())
    )


def _fit_actors(
    candidates: Sequence[NamedActor | ProvisionalActor],
    remaining: int,
) -> tuple[tuple[NamedActor | ProvisionalActor, ...], int, int]:
    """Fit actor lines into what is left of `max_chars` after the ranked hits.

    Priced and greedy exactly like `_select` prices a hit: one oversized entry does not close
    the bundle to the rest, so a cheaper actor can still fit. Returns the entries that fit,
    what they cost, and how many did not fit.
    """
    fitted: list[NamedActor | ProvisionalActor] = []
    spent = 0
    excluded = 0
    for entry in candidates:
        cost = _actor_cost(entry)
        if cost <= remaining - spent:
            fitted.append(entry)
            spent += cost
        else:
            excluded += 1
    return tuple(fitted), spent, excluded
