"""Persisted FP32 buffers must preserve exact scores and evidence-part selection."""

from __future__ import annotations

import math
import random
import struct
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge.infrastructure.local import LocalStore, StoredEmbedding, StoredMemory
from mindbridge.infrastructure.local.scoring import max_cosine_part_scores


@pytest.mark.parametrize("query_count", (1, 3, 4, 8))
@pytest.mark.parametrize("dimension", (64, 1024))
def test_fp32_buffers_preserve_scores_and_winning_parts_across_batches(
    query_count: int, dimension: int
) -> None:
    rng = random.Random(20261027)
    codec = struct.Struct(f"<{dimension}f")
    queries = tuple(tuple(rng.uniform(-1, 1) for _ in range(dimension)) for _ in range(query_count))
    buffers = tuple(
        (
            f"parent-{index % 13}",
            f"part-{index}",
            codec.pack(*(rng.uniform(-1, 1) * scale for _ in range(dimension))),
        )
        for index, scale in enumerate((1e-38, 1.0, 1e38) * 86)
    )
    # Repeat a part after multiple batches to exercise stable parent reduction.
    buffers = (*buffers, (buffers[0][0], "preferred", buffers[0][2]))
    preferred = {buffers[0][0]: "preferred"}
    decoded = tuple((parent, part, codec.unpack(vector)) for parent, part, vector in buffers)
    expected = max_cosine_part_scores(queries, decoded, preferred_embeddings=preferred)

    assert max_cosine_part_scores(queries, buffers, preferred_embeddings=preferred) == expected


@pytest.mark.parametrize("query_count", (1, 4))
@pytest.mark.parametrize("invalid", ("dimension", "zero", "nan", "inf"))
def test_fp32_buffers_reject_malformed_and_invalid_vectors(query_count: int, invalid: str) -> None:
    dimension = 64
    queries = ((1.0,) + (0.0,) * (dimension - 1),) * query_count
    valid = struct.pack(f"<{dimension}f", *queries[0])
    vector = {
        "dimension": valid[:-1],
        "zero": bytes(dimension * 4),
        "nan": struct.pack(f"<{dimension}f", math.nan, *((1.0,) * (dimension - 1))),
        "inf": struct.pack(f"<{dimension}f", math.inf, *((1.0,) * (dimension - 1))),
    }[invalid]
    parts = (
        ("invalid", "part", vector),
        *((str(index), str(index), valid) for index in range(127)),
    )

    with pytest.raises(ValueError, match="Cosine scoring requires"):
        max_cosine_part_scores(queries, parts)


@pytest.mark.parametrize("query_count", (1, 4))
def test_fp32_buffers_preserve_preferred_ties_across_batches(query_count: int) -> None:
    axis = (1.0,) + (0.0,) * 63
    orthogonal = (0.0, 1.0) + (0.0,) * 62
    best = struct.pack("<64f", *axis)
    other = struct.pack("<64f", *orthogonal)
    parts = (
        ("parent", "first", best),
        *(("parent", str(index), other) for index in range(127)),
        ("parent", "preferred", best),
    )
    assert max_cosine_part_scores(
        (axis,) * query_count, parts, preferred_embeddings={"parent": "preferred"}
    ) == {"parent": ("preferred", 1.0, 1.0)}


def test_embedding_part_stream_retains_exact_fp32_bytes_and_decoded_vector_reads(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    memory = StoredMemory(
        memory_id="parent",
        content="source",
        metadata_json="{}",
        created_at=now,
        updated_at=now,
    )
    embedding = StoredEmbedding(
        embedding_id="part",
        memory_id="parent",
        values=(0.6, 0.8),
        normalized=True,
        model_id="buffer-fixture-v1",
        space_id="buffer-fixture-v1:2",
        task="retrieval.document",
        created_at=now,
    )
    encoded = struct.pack("<2f", *embedding.values)
    with LocalStore(tmp_path) as store:
        store.records.write_memory(memory, (embedding,))
        assert tuple(
            store.index.iter_memory_embedding_parts(
                (memory.memory_id,), space_id=embedding.space_id, task=embedding.task
            )
        ) == ((memory.memory_id, embedding.embedding_id, encoded),)
        assert tuple(
            store.index.iter_memory_embedding_vectors(
                (memory.memory_id,), space_id=embedding.space_id, task=embedding.task
            )
        ) == ((memory.memory_id, struct.unpack("<2f", encoded)),)
