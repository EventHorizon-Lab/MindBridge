"""What `MINDBRIDGE_MINIMUM_EMBEDDING_SIMILARITY` does, and where it stops doing it.

Measured on MM-Lifelong (2026-08-24, `.benchmarks/e2e0824/reports/mm-lifelong.md` §2.2): with the
floor at its default 0.0, recall returned exactly `--recall-limit` memories on every single row --
`mean_mem = 12.0` for abstained rows and 12.0 for answered rows on both splits -- so the system had
no way to say "I hold nothing about this" and instead handed the answer model twelve captions from
tens of thousands of seconds away. The populated arm scored 0.000 and 0.017 against a blind floor of
0.090, with gold evidence present for 100 of 100 questions: not a coverage failure.

These tests are about the *capability*, not about ranking. `mm-lifelong.md` §2.2(b) measured that a
retrieval hit does not predict answering (Fisher p = 1.0, the third corpus on which that association
failed), so nothing here should be read as an argument that a floor raises hit@k.
"""

from __future__ import annotations

from typing import cast

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
    GeneratedAnswer,
    MediaUrlSigner,
    MemoryStore,
    OccurrenceVerifier,
)
from mindbridge.application.recall import RecallMemories
from mindbridge.contracts import RecallQuery, RecallRequest
from mindbridge.core import EmbeddedObjectType

# One memory the query is genuinely about and two it is only faintly near. 0.62 is not a tuned
# threshold: it is simply between the two, so a floor placed there separates them.
_SIMILARITIES = {"memory_near": 0.90, "memory_far": 0.31, "memory_farther": 0.12}
_MEMORIES = (
    memory("memory_near", "The blue toolbox is on the workbench."),
    memory("memory_far", "Someone laughs with their mouth open.", minutes=1),
    memory("memory_farther", "The screen fades to black.", minutes=2),
)


async def test_the_configured_floor_reaches_every_embedding_search() -> None:
    """The knob is only a capability if every retrieval channel is actually given it."""
    store = RecallStore(_MEMORIES)
    index = SimilarityIndex(_SIMILARITIES)
    await _recall(store, index, floor=0.62).run(_request())

    assert index.searches, "no embedding search ran, so the assertion below proves nothing"
    assert {search.minimum_similarity for search in index.searches} == {0.62}
    assert {EmbeddedObjectType.MEMORY_RECORD} <= {
        object_type for search in index.searches for object_type in search.object_types
    }


async def test_a_floor_lets_recall_return_fewer_memories_than_the_limit() -> None:
    """Nothing between the index and the response re-pads the ranking back up to `limit`."""
    store = RecallStore(_MEMORIES)
    index = SimilarityIndex(_SIMILARITIES)

    unfloored = await _recall(store, index, floor=0.0).run(_request())
    floored = await _recall(store, index, floor=0.62).run(_request())

    assert [item.memory_id for item in unfloored.memories] == [
        "memory_near",
        "memory_far",
        "memory_farther",
    ]
    assert [item.memory_id for item in floored.memories] == ["memory_near"]


async def test_a_floor_that_excludes_everything_returns_an_empty_recall_not_an_error() -> None:
    """The point of the knob: "I hold nothing relevant" has to be a result, not a failure.

    It also has to be a *cheap* result. An empty ranking must not reach the evidence reader or
    the object store at all, which is what the two emptiness assertions below pin down.
    """
    store = RecallStore(_MEMORIES)
    # A model that answers anyway from an empty page: with nothing grounded and nothing asked
    # with, that answer came from the question alone and recall refuses to pass it off as recall.
    answerer = ScriptedAnswerer((GeneratedAnswer(answer="probably the workbench", confidence=0.8),))
    result = await _recall(
        store, SimilarityIndex(_SIMILARITIES), floor=0.95, answerer=answerer
    ).run(_request())

    assert result.memories == ()
    assert result.evidence == ()
    assert result.answer is None
    assert result.confidence == 0.0
    # The answer stage is still asked, because with no candidates its one useful output is the
    # follow-up queries that drive reflection -- but it is asked with nothing attached.
    assert answerer.rounds == [()]
    assert store.evidence_reads == []
    assert store.presigns == []


async def test_the_full_text_channel_has_no_floor_of_its_own() -> None:
    """The floor binds the dense channel only, and the two are fused as equals.

    `search_memories` matches on `to_tsvector(summary) @@ tsquery` with the query's lexemes ORed
    together, plus a substring test -- neither of which has any similarity to compare against a
    floor. So a floored recall returns what the dense channel admitted *plus* whatever shares a
    word with the question, and an operator setting the knob has to know that. There is no
    principled composition available inside recall: `ts_rank_cd` and cosine are not on one scale.
    """
    store = RecallStore(_MEMORIES, lexical=frozenset({"memory_farther"}))
    floored = await _recall(store, SimilarityIndex(_SIMILARITIES), floor=0.62).run(_request())

    assert sorted(item.memory_id for item in floored.memories) == [
        "memory_farther",
        "memory_near",
    ]


async def test_a_media_only_query_is_floored_end_to_end() -> None:
    """With no query text the full-text channel is not consulted, so the floor is the only gate.

    Same corpus and same floor as the test above, where the full-text channel put `memory_farther`
    back. This is the shape in which the knob delivers what it promises today.
    """
    store = RecallStore(_MEMORIES, lexical=frozenset({"memory_farther"}))
    recall = _recall(store, SimilarityIndex(_SIMILARITIES), floor=0.62)

    result = await recall.run(
        RecallRequest(
            tenant_id="tenant_recall",
            query=RecallQuery(media_object_ids=("media_asked_with",)),
            limit=12,
        )
    )

    assert [item.memory_id for item in result.memories] == ["memory_near"]


def _request() -> RecallRequest:
    return RecallRequest(
        tenant_id="tenant_recall",
        query=RecallQuery(text="where is the blue toolbox"),
        limit=12,
    )


def _recall(
    store: RecallStore,
    index: SimilarityIndex,
    *,
    floor: float,
    answerer: ScriptedAnswerer | None = None,
) -> RecallMemories:
    return RecallMemories(
        cast(MemoryStore, store),
        cast(Answerer, answerer or ScriptedAnswerer()),
        cast(OccurrenceVerifier, None),
        embedding_index=cast(EmbeddingIndex, index),
        media_url_signer=cast(MediaUrlSigner, store),
        embedder=cast(Embedder, FixedEmbedder()),
        minimum_embedding_similarity=floor,
        clock=lambda: NOW,
    )
