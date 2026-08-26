"""Which round's answer a reflecting recall reports, and what one failed wave costs.

`_answer_within_reflection_budget` used to keep whatever the last round returned. A reflection
round re-ranks rather than extends -- the fused ranking and the visible page are each bounded by a
limit -- so a follow-up wave can push the memory that supported an answer out of the set the next
round is handed, and that round's abstention then replaced a supported answer with no answer at
all. Reflection could make a recall strictly worse than not reflecting, which is the input shape
`answer_from_evidence` v13 makes routine by instructing the model to return a supported but
incomplete answer *together with* follow-up queries.
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from recall_doubles import (
    NOW,
    FixedEmbedder,
    RecallStore,
    ScriptedAnswerer,
    SimilarityIndex,
    memory,
)

from mindbridge.application.capabilities import Embedder
from mindbridge.application.ports import (
    Answerer,
    EmbeddingIndex,
    EmbeddingMatch,
    EmbeddingSearch,
    GeneratedAnswer,
    MediaUrlSigner,
    MemoryStore,
    OccurrenceVerifier,
)
from mindbridge.application.recall import RecallMemories
from mindbridge.contracts import RecallQuery, RecallRequest
from mindbridge.core import ModelOutputError

_SUPPORTING = memory("memory_supporting", "The blue toolbox is on the workbench.")
_DISTRACTION = memory("memory_distraction", "The screen fades to black.", minutes=1)
_MEMORIES = (_SUPPORTING, _DISTRACTION)

_ANSWERED = GeneratedAnswer(
    answer="on the workbench",
    confidence=0.7,
    retrieval_queries=("where was the toolbox last seen",),
)
_ABSTAINED = GeneratedAnswer(answer=None, confidence=0.0)


async def test_a_later_abstention_does_not_erase_an_answer_it_no_longer_holds() -> None:
    """Round 1 answers and asks a follow-up; the follow-up displaces the memory and abstains.

    `limit=1` is the smallest page on which the displacement is visible, and it is the same
    truncation that does it at any size: the follow-up ranking is fused with the current one and
    the page is cut to `limit`, so the memory the answer was made from can fall off the end.
    """
    store = RecallStore(_MEMORIES)
    index = SimilarityIndex({"memory_supporting": 0.9}, {"memory_distraction": 0.9})
    answerer = ScriptedAnswerer((_ANSWERED, _ABSTAINED))

    result = await _recall(store, index, answerer).run(_request(limit=1))

    assert len(answerer.rounds) == 2, "reflection did not run, so this proves nothing"
    assert [item.memory_id for item in answerer.rounds[0]] == ["memory_supporting"]
    assert [item.memory_id for item in answerer.rounds[1]] == ["memory_distraction"]
    assert result.answer == "on the workbench"
    assert result.confidence == 0.7
    # The memories reported are the ones that answer was made from, not the later round's.
    assert [item.memory_id for item in result.memories] == ["memory_supporting"]


async def test_an_abstention_that_still_held_the_evidence_replaces_the_answer() -> None:
    """A round that saw everything the answering round saw is the later, better-informed verdict.

    Nothing was displaced here, so the abstention is a judgement about the same evidence rather
    than about its absence, and it is the one the recall reports.
    """
    store = RecallStore(_MEMORIES)
    index = SimilarityIndex({"memory_supporting": 0.9})
    answerer = ScriptedAnswerer((_ANSWERED, _ABSTAINED))

    result = await _recall(store, index, answerer).run(_request(limit=1))

    assert len(answerer.rounds) == 2, "reflection did not run, so this proves nothing"
    assert [item.memory_id for item in answerer.rounds[1]] == ["memory_supporting"]
    assert result.answer is None
    assert result.confidence == 0.0


async def test_a_failed_followup_wave_costs_its_own_opinion_and_nothing_else() -> None:
    """One retrieval wave raising must not throw away the answer the round before produced."""
    store = RecallStore(_MEMORIES)
    index = _FailingIndex({"memory_supporting": 0.9})
    answerer = ScriptedAnswerer((_ANSWERED, GeneratedAnswer(answer="still here", confidence=0.5)))

    result = await _recall(store, index, answerer).run(_request())

    assert index.failures, "the follow-up wave did not fail, so this proves nothing"
    assert len(answerer.rounds) == 2, "the reflection round did not run"
    assert result.answer == "still here"


async def test_a_cancelled_followup_wave_still_propagates() -> None:
    """`return_exceptions` must not turn a cancellation of the whole recall into a partial one."""
    store = RecallStore(_MEMORIES)
    index = _FailingIndex({"memory_supporting": 0.9}, error=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await _recall(store, index, ScriptedAnswerer((_ANSWERED, _ABSTAINED))).run(_request())


async def test_the_reordered_round_reuses_the_evidence_the_first_round_read() -> None:
    """A temporal reorder changes the order of the candidates, never which spans they cite.

    Re-resolving them was three pooled queries and one presign per distinct object, paid again
    for every answer round and once more for the response.
    """
    store = RecallStore(_MEMORIES)
    index = SimilarityIndex({"memory_supporting": 0.9, "memory_distraction": 0.8})
    answerer = ScriptedAnswerer(
        (
            GeneratedAnswer(answer="the newest one", confidence=0.6, temporal_order="newest"),
            GeneratedAnswer(answer="the newest one", confidence=0.8, temporal_order="newest"),
        )
    )

    await _recall(store, index, answerer).run(_request())

    assert len(answerer.rounds) == 2, "no reorder round ran, so this proves nothing"
    assert [item.memory_id for item in answerer.rounds[1]] == [
        "memory_distraction",
        "memory_supporting",
    ]
    assert store.evidence_reads == [("evidence_memory_supporting", "evidence_memory_distraction")]
    # A reused read is the tuple a re-read would have returned, order included: the spans follow
    # the memories asking for them, which is the whole point of reordering the candidates.
    assert [
        tuple(item.evidence_span.evidence_id for item in round_evidence)
        for round_evidence in answerer.evidence_rounds
    ] == [
        ("evidence_memory_supporting", "evidence_memory_distraction"),
        ("evidence_memory_distraction", "evidence_memory_supporting"),
    ]
    assert sorted(store.presigns) == [
        "media_evidence_memory_distraction",
        "media_evidence_memory_supporting",
    ]


class _FailingIndex(SimilarityIndex):
    """Serve the first retrieval wave and raise on every later one, as a follow-up would."""

    def __init__(
        self,
        similarities: dict[str, float],
        *,
        error: type[BaseException] = ModelOutputError,
    ) -> None:
        super().__init__(similarities)
        self.error = error
        self.failures = 0

    async def search_embeddings(self, search: EmbeddingSearch) -> tuple[EmbeddingMatch, ...]:
        if self.wave_index > 0:
            self.failures += 1
            raise self.error("follow-up retrieval wave failed")
        return await super().search_embeddings(search)


def _request(*, limit: int = 12) -> RecallRequest:
    return RecallRequest(
        tenant_id="tenant_recall",
        query=RecallQuery(text="where is the blue toolbox"),
        limit=limit,
    )


def _recall(
    store: RecallStore,
    index: SimilarityIndex,
    answerer: ScriptedAnswerer,
) -> RecallMemories:
    return RecallMemories(
        cast(MemoryStore, store),
        cast(Answerer, answerer),
        cast(OccurrenceVerifier, None),
        embedding_index=cast(EmbeddingIndex, index),
        media_url_signer=cast(MediaUrlSigner, store),
        embedder=cast(Embedder, FixedEmbedder()),
        minimum_embedding_similarity=0.0,
        clock=lambda: NOW,
    )
