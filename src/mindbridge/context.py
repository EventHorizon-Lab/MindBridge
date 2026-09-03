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
    ContextBudget,
    ContextBundle,
    ContextConflict,
    ContextUnknown,
    ContextUnknownKind,
    MemoryKind,
    MemoryType,
    Modality,
    SearchHit,
)

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
    # `AssetRef.modality` is optional on the public type; an unresolved asset is charged the
    # default rather than being treated as free.
    charges = (
        _DEFAULT_ASSET_EVIDENCE_CHARS
        if asset.modality is None
        else _ASSET_EVIDENCE_CHARS.get(asset.modality, _DEFAULT_ASSET_EVIDENCE_CHARS)
        for asset in hit.assets
    )
    return len(hit.content) + sum(charges)


def compile_context(
    goal: str,
    hits: Sequence[SearchHit],
    *,
    budget: ContextBudget,
    reference_at: datetime,
    started_at: float | None = None,
    unknowns: Sequence[ContextUnknown] = (),
    candidate_limit: int | None = None,
) -> ContextBundle:
    """Partition, filter, and budget ranked hits into one bundle.

    `started_at` is a `time.perf_counter()` reading taken before retrieval, so `budget
    .max_latency_ms` bounds the whole compilation rather than this function. `unknowns` carries
    what the caller already knows and the compiler cannot see, such as an empty spatial scope.
    `candidate_limit` is the depth retrieval was asked to rank, so a bundle that lost evidence
    and filled that window can say the ranking may continue past it.
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
    sections = _select(candidates, budget)
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
    return ContextBundle(
        goal=goal,
        reference_at=reference_at,
        budget=budget,
        actors=sections["actors"],
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
            included,
            excluded,
            omitted,
            skipped=skipped,
            exhausted=(
                candidate_limit is not None
                and len(hits) >= candidate_limit
                and bool(excluded or omitted)
            ),
            candidate_limit=candidate_limit,
        ),
        occurred_from=occurred_from,
        occurred_until=occurred_until,
        frames=_frames(included),
        places=tuple(sorted({hit.place_id for hit in included if hit.place_id})),
        omitted=omitted,
        chars=sum(evidence_cost(hit) for hit in included),
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
) -> dict[str, tuple[SearchHit, ...]]:
    """Fill the sections in rank order, one guaranteed slot each before any second slot."""
    sections: dict[str, list[SearchHit]] = {name: [] for name in _SECTIONS}
    taken: set[str] = set()
    used = 0
    for guaranteed in (True, False):
        for hit in candidates:
            if len(taken) >= budget.max_items:
                break
            section = _section(hit)
            if hit.id in taken or (guaranteed and sections[section]):
                continue
            cost = evidence_cost(hit)
            # One oversized hit does not close the bundle: a cheaper lower-ranked candidate can
            # still fit, and `omitted` reports everything the budget could not buy.
            if used + cost > budget.max_chars:
                continue
            sections[section].append(hit)
            taken.add(hit.id)
            used += cost
    return {name: tuple(section) for name, section in sections.items()}


def _section(hit: SearchHit) -> str:
    context = hit.context
    if context is not None and context.kind in _KIND_SECTIONS:
        return _KIND_SECTIONS[context.kind]
    return _TYPE_SECTIONS[hit.memory_type]


def _unknowns(
    supplied: Sequence[ContextUnknown],
    budget: ContextBudget,
    included: Sequence[SearchHit],
    excluded: Mapping[str, int],
    omitted: int,
    *,
    skipped: bool,
    exhausted: bool = False,
    candidate_limit: int | None = None,
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
    if budget.memory_types is not None:
        present = {hit.memory_type for hit in included}
        found.extend(
            ContextUnknown(
                kind=ContextUnknownKind.SECTION_EMPTY,
                detail=f"no {memory_type.value} memory was included",
            )
            for memory_type in sorted(budget.memory_types)
            if memory_type not in present
        )
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
