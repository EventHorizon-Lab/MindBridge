from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from mindbridge import SearchHit
from mindbridge.benchmarks.complementary import parse_plan, select_evidence, selector_payload


def _hit(content: str, identifier: str = "memory") -> SearchHit:
    return SearchHit(
        identifier,
        content,
        0.5,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        metadata={"reference_answer": "FORBIDDEN", "target_character_ids": ["SECRET"]},
    )


def _plan() -> str:
    return json.dumps(
        {
            "facets": ["first event", "exception"],
            "records": [
                {"index": 0, "relevance": 3, "covers": [0]},
                {"index": 1, "relevance": 3, "covers": [0]},
                {"index": 2, "relevance": 2, "covers": [1]},
            ],
        }
    )


def test_coverage_spends_second_slot_on_complementary_evidence() -> None:
    hits = [_hit("first"), _hit("similar"), _hit("exception")]
    plan = parse_plan(_plan(), 3)
    assert [h.content for h in select_evidence(hits, plan, arm="relevance", rows=2, chars=30)] == [
        "first",
        "similar",
    ]
    assert [h.content for h in select_evidence(hits, plan, arm="coverage", rows=2, chars=30)] == [
        "first",
        "exception",
    ]


def test_selection_caps_and_duplicate_invariance() -> None:
    hits = [_hit("first"), _hit("first", "copy"), _hit("last")]
    selected = select_evidence(hits, None, arm="rank", rows=3, chars=9)
    assert [h.content for h in selected] == ["first", "last"]
    assert select_evidence(hits, None, arm="rank", rows=3, chars=4) == (hits[2],)


def test_selector_cannot_observe_metadata_or_storage_ids() -> None:
    payload = selector_payload("Who?", [_hit("Observed action", "OPAQUE_ID")])
    assert "Observed action" in payload
    assert all(value not in payload for value in ["SECRET", "FORBIDDEN", "OPAQUE_ID"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "facet", "extra"])
def test_invalid_selector_output_is_rejected(mutation: str) -> None:
    data = json.loads(_plan())
    if mutation == "missing":
        data["records"].pop()
    elif mutation == "duplicate":
        data["records"][2]["index"] = 0
    elif mutation == "facet":
        data["records"][2]["covers"] = [9]
    else:
        data["answer"] = "guessed answer"
    with pytest.raises(ValueError):
        parse_plan(json.dumps(data), 3)
