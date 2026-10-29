"""query-vision-v1: fixed OCR anchors recover support outside native candidate pools.

These synthetic models establish retrieval and delivery mechanics, not real OCR quality.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge import (
    AnswerPolicy,
    AnswerResult,
    Blob,
    EmbedTask,
    IndexQuantization,
    Memory,
    Modality,
    ModelInput,
    RetrievalMode,
    SearchHit,
)


class _Models:
    embedding_model = "query-vision-v1"
    embedding_space = "query-vision-v1:4"
    embedding_dimension = 4
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
    generation_model = "query-vision-reader-v1"
    generation_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
    vision_model = "query-vision-ocr-v1"
    vision_space = "query-vision-ocr-v1"
    vision_capabilities = frozenset({Modality.IMAGE})

    def __init__(self) -> None:
        self.description_calls = 0
        self.delivered: tuple[SearchHit, ...] = ()

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (0.0, 1.0, 0.0, 0.0)
            if "ZX93" in value.text
            else (1.0, 0.0, 0.0, 0.0)
            if task is EmbedTask.QUERY
            else (0.9, 0.0, math.sqrt(0.19), 0.0)
            for value in inputs
        )

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        self.description_calls += 1
        return tuple("Text: ZX93" for _ in inputs)

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        del question, answer_policy, exhaustive
        self.delivered = tuple(hits)
        return AnswerResult(
            answer="Orion Hall" if any("ZX93" in hit.content for hit in hits) else "unknown",
            hits=tuple(hits),
        )

    def close(self) -> None:
        pass


@pytest.mark.parametrize("seed", (20261027, 20261028, 20261029))
@pytest.mark.parametrize("quantization", (IndexQuantization.NONE, IndexQuantization.INT8))
def test_query_vision_delivers_support_beyond_native_candidates(
    tmp_path: Path, seed: int, quantization: IndexQuantization
) -> None:
    for enabled in (False, True):
        models = _Models()
        with Memory(
            tmp_path / str(enabled),
            embedder=models,
            answerer=models,
            vision_describer=models,
            query_vision=enabled,
            retrieval_mode=RetrievalMode.DENSE,
            index_quantization=quantization,
            evidence_budget_chars=None,
            ambiguity_margin=0,
            reinforce_on_answer=False,
        ) as memory:
            texts = [f"unrelated corridor {index}" for index in range(230)]
            texts.append("ZX93 is the registration desk in Orion Hall")
            random.Random(seed).shuffle(texts)
            records = memory.add_many(texts)
            support = next(record for record in records if "ZX93" in record.content)
            question = ("What is its name?", Blob(b"query-vision-v1", "image/png"))
            discovered = memory.search_with_trace(question, limit=100)
            candidates = {row.memory_id for row in discovered.trace.candidates}
            assert (support.id in candidates) is enabled
            result = memory.ask(
                question, limit=1, reference_at=datetime(2026, 10, 28, tzinfo=timezone.utc)
            )
            assert (support.id in {hit.id for hit in models.delivered}) is enabled
            assert (result.answer == "Orion Hall") is enabled
            assert models.description_calls == (2 if enabled else 0)
