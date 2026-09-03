"""Pure context selection behind `Memory.compile`.

The compiler selects and structures already-retrieved evidence. It calls no model, writes nothing,
and never resolves a conflict: every value here is a deterministic function of the ranked hits, the
budget, and the reference clock. Authoritative visibility and scope were applied by the retrieval
path that produced the hits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import TypeAlias

from mindbridge.types import (
    ContextBudget,
    ContextBundle,
    ContextConflict,
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
_SECTIONS: tuple[str, ...] = ("actors", "facts", "episodes", "procedures", "affect", "traits")
# A typed kind decides the section first; an untyped or generic record falls back to its type.
_KIND_SECTIONS: Mapping[MemoryKind, str] = MappingProxyType(
    {
        MemoryKind.ENTITY: "actors",
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
) -> ContextBundle:
    """Partition, filter, and budget ranked hits into one bundle."""
    candidates = tuple(hit for hit in hits if _admits(hit, budget, reference_at))
    sections = _select(candidates, budget)
    included = tuple(hit for name in _SECTIONS for hit in sections[name])
    occurred_from, occurred_until = _occurred_range(included)
    return ContextBundle(
        goal=goal,
        reference_at=reference_at,
        budget=budget,
        actors=sections["actors"],
        episodes=sections["episodes"],
        facts=sections["facts"],
        procedures=sections["procedures"],
        affect=sections["affect"],
        traits=sections["traits"],
        conflicts=_conflicts(included),
        occurred_from=occurred_from,
        occurred_until=occurred_until,
        frames=_frames(included),
        omitted=len(candidates) - len(included),
        chars=sum(evidence_cost(hit) for hit in included),
    )


def _admits(hit: SearchHit, budget: ContextBudget, reference_at: datetime) -> bool:
    if budget.memory_types is not None and hit.memory_type not in budget.memory_types:
        return False
    # A record with no typed context is evidence in its own right, not a low-confidence inference.
    confidence = 1.0 if hit.context is None else hit.context.confidence
    if confidence < budget.min_confidence:
        return False
    if budget.freshness is None:
        return True
    return reference_at - _anchor(hit) <= budget.freshness


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


def _conflicts(hits: Sequence[SearchHit]) -> tuple[ContextConflict, ...]:
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
    return tuple(
        conflict
        for lineage_id, claims in lineages.items()
        if (conflict := _lineage_conflict(lineage_id, claims)) is not None
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
