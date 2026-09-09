from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import cast

from mindbridge.evidence import answer_evidence_payloads
from mindbridge.models import openai_sdk as openai_backend
from mindbridge.types import EvidenceBasis, MemoryContext, MemoryKind, SearchHit

NOW = datetime(2026, 9, 7, tzinfo=timezone.utc)


def _hit(
    memory_id: str,
    *,
    evidence_ids: tuple[str, ...] = (),
    source_id: str | None = None,
    supersedes_id: str | None = None,
) -> SearchHit:
    return SearchHit(
        id=memory_id,
        content=f"content for {memory_id}",
        score=0.8,
        created_at=NOW,
        context=MemoryContext(
            kind=MemoryKind.EVENT,
            basis=EvidenceBasis.MODEL_INFERENCE,
            confidence=0.7,
            valid_from=None,
            valid_until=None,
            recorded_at=NOW,
            evidence_ids=evidence_ids,
            source_id=source_id,
            supersedes_id=supersedes_id,
        ),
    )


def test_source_origin_stays_separate_from_supporting_evidence() -> None:
    hits = (
        _hit(
            "derived",
            evidence_ids=("selected-source", "omitted-support"),
            source_id="provenance-origin",
        ),
        SearchHit(
            id="selected-source",
            content="selected source",
            score=0.7,
            created_at=NOW,
        ),
    )

    context = answer_evidence_payloads(hits)[0]["context"]

    assert context == {
        "kind": "event",
        "basis": "model_inference",
        "confidence": 0.7,
        "recorded_at": NOW.isoformat(),
        "evidence_labels": ["E2", "S1"],
        "supporting_record_count": 2,
        "sources_not_in_context": ["S1"],
        "source_label": "S2",
        "source_not_in_context": True,
    }


def test_support_count_is_unique_across_selected_and_omitted_sources() -> None:
    """Repeated lineage edges remain one supporting record in the answer context."""
    hits = (
        _hit(
            "derived",
            evidence_ids=(
                "selected-source",
                "omitted-support",
                "selected-source",
                "omitted-support",
            ),
        ),
        SearchHit(
            id="selected-source",
            content="selected source",
            score=0.7,
            created_at=NOW,
        ),
    )

    context = cast(Mapping[str, object], answer_evidence_payloads(hits)[0]["context"])

    assert context["supporting_record_count"] == 2
    assert context["evidence_labels"] == ["E2", "S1"]
    assert context["sources_not_in_context"] == ["S1"]


def test_support_count_is_invariant_when_omitted_sources_switch_to_aggregate() -> None:
    """The compact projection keeps each record's unique support cardinality."""
    for omitted_count, compact in ((64, False), (65, True)):
        omitted = tuple(f"source-{index}" for index in range(omitted_count))
        hits = (
            _hit("derived", evidence_ids=("selected-source", *omitted, omitted[0])),
            SearchHit(
                id="selected-source",
                content="selected source",
                score=0.7,
                created_at=NOW,
            ),
        )

        context = cast(Mapping[str, object], answer_evidence_payloads(hits)[0]["context"])

        assert context["supporting_record_count"] == omitted_count + 1
        assert ("sources_not_in_context_count" in context) is compact


def test_origin_and_correction_do_not_increase_support_count() -> None:
    hits = (
        _hit(
            "derived",
            evidence_ids=("support",),
            source_id="origin-only",
            supersedes_id="correction-only",
        ),
    )

    context = cast(Mapping[str, object], answer_evidence_payloads(hits)[0]["context"])

    assert context["supporting_record_count"] == 1
    assert context["evidence_labels"] == ["S1"]
    assert context["source_label"] == "S2"
    assert context["supersedes_label"] == "S3"


def test_raw_payload_keeps_the_frozen_unqualified_shape() -> None:
    hit = SearchHit(
        id="raw",
        content="unchanged raw memory",
        score=0.8,
        created_at=NOW,
        metadata={"capture": "sdk"},
    )

    assert json.dumps(answer_evidence_payloads((hit,)), separators=(",", ":")) == (
        '[{"content":"unchanged raw memory","memory_type":"semantic",'
        f'"created_at":"{NOW.isoformat()}","metadata":{{"capture":"sdk"}}}}]'
    )


def _large_payload(source_count: int) -> tuple[tuple[SearchHit, ...], list[dict[str, object]]]:
    common = tuple(f"private-source-{index:05d}" for index in range(source_count))
    hits = (
        _hit(
            "first-derived",
            evidence_ids=("selected-source", *common),
            source_id="origin-first",
            supersedes_id="corrected-first",
        ),
        _hit(
            "second-derived",
            evidence_ids=(*common[:5], "second-only-source"),
            source_id="origin-second",
        ),
        SearchHit(
            id="selected-source",
            content="selected source",
            score=0.6,
            created_at=NOW,
        ),
    )
    return hits, answer_evidence_payloads(hits)


def test_large_omitted_fanout_is_counted_with_bounded_shared_overlap() -> None:
    hits, payloads = _large_payload(10_000)
    first = cast(Mapping[str, object], payloads[0]["context"])
    second = cast(Mapping[str, object], payloads[1]["context"])

    assert first["evidence_labels"] == ["E3"]
    assert first["supporting_record_count"] == 10_001
    assert first["sources_not_in_context_count"] == 10_000
    assert first["shared_sources_not_in_context"] == [{"evidence_label": "E2", "count": 5}]
    assert first["source_label"] == "S1"
    assert first["source_not_in_context"] is True
    assert first["supersedes_label"] == "S2"
    assert second["sources_not_in_context_count"] == 6
    assert second["supporting_record_count"] == 6
    assert second["shared_sources_not_in_context"] == [{"evidence_label": "E1", "count": 5}]

    rendered = json.dumps(payloads)
    assert "private-source-" not in rendered
    assert len(rendered) - len(json.dumps(_large_payload(100)[1])) < 10
    assert openai_backend._answer_system_prompt(hits, {}, payloads) == (
        openai_backend._GROUNDED_SYSTEM_PROMPT
        + openai_backend._QUALIFIED_EVIDENCE_PROMPT
        + openai_backend._COMPACT_PROVENANCE_PROMPT
    )
