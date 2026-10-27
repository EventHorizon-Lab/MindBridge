"""Temporal candidate discovery must not reset lexical evidence strength."""

from __future__ import annotations

import pytest

from mindbridge.infrastructure.local.zvec_index import IndexHit
from mindbridge.kernel.ranking import merge_index_hits, merge_temporal_lexical_hits


def test_temporal_merge_preserves_known_global_scores() -> None:
    global_hit = IndexHit("weak", relevance=0.08, lexical_match=True)
    preferred = IndexHit("weak", relevance=1.0, lexical_match=True)
    assert merge_temporal_lexical_hits((preferred,), (global_hit,)) == (global_hit,)


def test_temporal_bm25_normalization_uses_each_global_type_route() -> None:
    fallback = (
        IndexHit(
            "semantic-best", 1.0, lexical_match=True, lexical_score=4.0, lexical_route="semantic"
        ),
        IndexHit(
            "episodic-best", 1.0, lexical_match=True, lexical_score=1.0, lexical_route="episodic"
        ),
    )
    preferred = (
        IndexHit(
            "semantic-new", 1.0, lexical_match=True, lexical_score=2.0, lexical_route="semantic"
        ),
        IndexHit(
            "episodic-new", 1.0, lexical_match=True, lexical_score=0.75, lexical_route="episodic"
        ),
    )
    merged = merge_temporal_lexical_hits(preferred, fallback)
    assert {hit.id: hit.relevance for hit in merged} == {
        "semantic-best": 1.0,
        "episodic-best": 1.0,
        "semantic-new": 0.5,
        "episodic-new": 0.75,
    }


def test_temporal_rank_fusion_without_global_ranks_stays_below_global_tail() -> None:
    preferred = (
        IndexHit("new-first", 1.0, lexical_match=True),
        IndexHit("new-second", 0.5, lexical_match=True),
    )
    fallback = (
        IndexHit("best", 1.0, lexical_match=True),
        IndexHit("tail", 0.2, lexical_match=True),
    )
    merged = merge_temporal_lexical_hits(preferred, fallback)
    assert merged[:2] == fallback
    assert 0.0 < merged[3].relevance < merged[2].relevance < 0.2
    assert all(hit.relevance == 0.0 for hit in merge_temporal_lexical_hits(preferred, ()))


def test_route_merge_preserves_native_lexical_score_metadata() -> None:
    hit = IndexHit("shared", 0.8, lexical_match=True, lexical_score=2.0, lexical_route="semantic")
    merged = merge_index_hits((hit,), (hit,))
    assert merged == (hit,)


@pytest.mark.parametrize("score", (float("nan"), float("inf"), -1.0, True))
def test_native_lexical_score_rejects_invalid_strength(score: float) -> None:
    with pytest.raises(ValueError, match="lexical_score"):
        IndexHit("bad", 1.0, lexical_match=True, lexical_score=score)
