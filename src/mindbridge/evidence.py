"""Compact, source-qualified evidence for answer generation.

Labels identify records within one answer context, not independent witnesses.
Counts and overlaps describe omitted supporting records without serializing an
unbounded provenance graph into the model's answer context.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from mindbridge.types import MemoryContext, SearchHit

_MAX_EXACT_OMITTED_SOURCES = 64


def answer_evidence_payloads(
    hits: Sequence[SearchHit],
    *,
    omitted_media: Mapping[str, Mapping[str, int]] | None = None,
) -> list[dict[str, object]]:
    """Project selected records without retrieving or inventing source content.

    Selection order is preserved. Event time, validity and recording time retain
    their distinct meanings; neither record order nor a missing source label
    establishes chronology, corroboration or source visibility. The adapter may
    identify media excluded from the actual request; those counts describe missing
    input, not missing information in the record's retained text.
    """
    qualified = any(hit.context is not None for hit in hits)
    selected: dict[str, str] = {}
    for hit in hits:
        if hit.id not in selected:
            selected[hit.id] = f"E{len(selected) + 1}"
    external_labels: dict[str, str] = {}

    def source_label(source_id: str) -> str:
        if source_id in selected:
            return selected[source_id]
        if source_id not in external_labels:
            external_labels[source_id] = f"S{len(external_labels) + 1}"
        return external_labels[source_id]

    omitted_sources = {
        hit.id: frozenset(sid for sid in hit.context.evidence_ids if sid not in selected)
        for hit in hits
        if hit.context is not None
    }
    exact_sources = _small_source_set(tuple(omitted_sources.values()))
    shared_sources = {} if exact_sources else _shared_sources(omitted_sources, selected)

    payloads: list[dict[str, object]] = []
    for hit in hits:
        payload = _record_payload(hit, selected[hit.id] if qualified else None)
        if omitted_media is not None and (media_counts := omitted_media.get(hit.id)):
            payload["media_omitted"] = dict(media_counts)
        if hit.context is not None:
            details = _context_payload(hit.context)
            details.update(
                _source_payload(
                    hit.context,
                    selected,
                    source_label,
                    exact_sources=exact_sources,
                    omitted_count=len(omitted_sources[hit.id]),
                    shared=shared_sources.get(hit.id, ()),
                )
            )
            payload["context"] = details
        payloads.append(payload)
    return payloads


def _record_payload(hit: SearchHit, label: str | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "content": hit.content,
        "memory_type": hit.memory_type.value,
    }
    if hit.occurred_at is not None:
        payload["occurred_at"] = hit.occurred_at.isoformat()
    else:
        payload["created_at"] = hit.created_at.isoformat()
    if hit.occurred_end is not None:
        payload["occurred_end"] = hit.occurred_end.isoformat()
    payload["metadata"] = dict(hit.metadata)
    if label is not None:
        payload["evidence_label"] = label
    if hit.place_id is not None:
        payload["place_id"] = hit.place_id
    return payload


def _context_payload(context: MemoryContext) -> dict[str, object]:
    details: dict[str, object] = {
        "kind": context.kind.value,
        "basis": context.basis.value,
        "confidence": context.confidence,
        "recorded_at": context.recorded_at.isoformat(),
    }
    for key, time_value in (
        ("valid_from", context.valid_from),
        ("valid_until", context.valid_until),
    ):
        if time_value is not None:
            details[key] = time_value.isoformat()
    for key, value in (
        ("subject", context.subject),
        ("predicate", context.predicate),
        ("value", context.value),
    ):
        if value is not None:
            details[key] = value
    return details


def _source_payload(
    context: MemoryContext,
    selected: Mapping[str, str],
    source_label: Callable[[str], str],
    *,
    exact_sources: bool,
    omitted_count: int,
    shared: Sequence[dict[str, object]],
) -> dict[str, object]:
    # A source_id describes provenance, whereas supersedes_id describes a
    # correction. Do not combine correction edges with supporting ones.
    details: dict[str, object] = {}
    details.update(
        _support_payload(
            context.evidence_ids,
            selected,
            source_label,
            exact_sources=exact_sources,
            omitted_count=omitted_count,
            shared=shared,
        )
    )
    if context.source_id is not None:
        details["source_label"] = source_label(context.source_id)
        if context.source_id not in selected:
            details["source_not_in_context"] = True
    if context.supersedes_id is not None:
        details["supersedes_label"] = source_label(context.supersedes_id)
    return details


def _support_payload(
    evidence_ids: Sequence[str],
    selected: Mapping[str, str],
    source_label: Callable[[str], str],
    *,
    exact_sources: bool,
    omitted_count: int,
    shared: Sequence[dict[str, object]],
) -> dict[str, object]:
    if exact_sources:
        source_ids = tuple(dict.fromkeys(evidence_ids))
        if not source_ids:
            return {}
        details: dict[str, object] = {
            "evidence_labels": [source_label(sid) for sid in source_ids],
            "supporting_record_count": len(source_ids),
        }
        absent = [source_label(sid) for sid in source_ids if sid not in selected]
        if absent:
            details["sources_not_in_context"] = absent
        return details
    details = {}
    visible_sources = tuple(dict.fromkeys(sid for sid in evidence_ids if sid in selected))
    if visible_sources or omitted_count:
        details["supporting_record_count"] = len(visible_sources) + omitted_count
    if visible_sources:
        details["evidence_labels"] = [selected[sid] for sid in visible_sources]
    if omitted_count:
        details["sources_not_in_context_count"] = omitted_count
    if shared:
        details["shared_sources_not_in_context"] = list(shared)
    return details


def _small_source_set(source_sets: Sequence[frozenset[str]]) -> bool:
    seen: set[str] = set()
    for source_ids in source_sets:
        for source_id in source_ids:
            seen.add(source_id)
            if len(seen) > _MAX_EXACT_OMITTED_SOURCES:
                return False
    return True


def _shared_sources(
    omitted: Mapping[str, frozenset[str]],
    selected: Mapping[str, str],
) -> dict[str, list[dict[str, object]]]:
    """Bound the rendered provenance by selected records, not lifetime fanout.

    Pairwise overlap does not determine higher-order unions or prove independent
    observations. Computation still depends on the sizes of the stored source sets.
    """
    shared: dict[str, list[dict[str, object]]] = {}
    items = tuple(omitted.items())
    for position, (memory_id, source_ids) in enumerate(items):
        if not source_ids:
            continue
        for other_id, other_sources in items[position + 1 :]:
            overlap = len(source_ids & other_sources)
            if overlap:
                shared.setdefault(memory_id, []).append(
                    {"evidence_label": selected[other_id], "count": overlap}
                )
                shared.setdefault(other_id, []).append(
                    {"evidence_label": selected[memory_id], "count": overlap}
                )
    return shared
