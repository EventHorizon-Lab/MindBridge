"""Failed recall must not become positive evidence for future ranking."""

from collections.abc import Generator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from mindbridge import (
    AbstentionReason,
    AnswerPolicy,
    AnswerResult,
    Memory,
    Modality,
    ModelInput,
    SearchHit,
)


class _Answerer:
    generation_capabilities = frozenset({Modality.TEXT})
    generation_model = "answer-feedback-fixture:v1"

    def __init__(self, reason: AbstentionReason) -> None:
        self.reason: AbstentionReason | None = reason

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        return AnswerResult(
            answer="Insufficient evidence." if self.reason is not None else "The kettle is red.",
            hits=tuple(hits),
            abstained=self.reason is not None,
            abstention_reason=self.reason,
        )

    def stream_answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> Generator[str, None, AnswerResult]:
        result = self.answer(question, hits, answer_policy=answer_policy, exhaustive=exhaustive)
        yield result.answer
        return result

    def close(self) -> None:
        pass


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("reason", tuple(AbstentionReason))
def test_abstention_preserves_ranking_and_only_success_reinforces(
    tmp_path: Path, streaming: bool, reason: AbstentionReason
) -> None:
    answerer = _Answerer(reason)
    started = datetime.now(timezone.utc)
    reference = started + timedelta(days=1)
    with Memory(
        tmp_path, embedder=TinyEmbedder(), answerer=answerer, minimum_relevance=0
    ) as memory:
        records = memory.add_many(("The kettle is red.", "The crate is blue."))
        before = memory.search("kettle", limit=2, reference_at=reference)
        for _attempt in range(3):
            if streaming:
                chunks = tuple(memory.ask_stream("kettle", limit=2, reference_at=reference))
                result = chunks[-1].result
                assert result is not None
                assert "".join(chunk.text for chunk in chunks) == result.answer
            else:
                result = memory.ask("kettle", limit=2, reference_at=reference)
            assert result.abstained and result.hits

        for record in records:
            stored = memory._store.records.read_memory(record.id)
            assert stored is not None
            assert stored.access_count == 0
            assert stored.last_accessed_at is None
        assert memory.search("kettle", limit=2, reference_at=reference) == before
        assert memory._store.control.read_repeated_query_failures(
            limit=10, since=started, minimum=3
        )

        answerer.reason = None
        successful = memory.ask("kettle", limit=2, reference_at=reference)
        assert not successful.abstained
        for hit in successful.hits:
            reinforced = memory._store.records.read_memory(hit.id)
            assert reinforced is not None and reinforced.access_count == 1
