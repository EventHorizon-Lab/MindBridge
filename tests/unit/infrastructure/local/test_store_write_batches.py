"""Batching must bound redundant reads without weakening selector validation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from mindbridge.infrastructure.local import LocalStore, StoredEmbedding, StoredMemory
from mindbridge.infrastructure.local.store.rows import StoredTextSelector, StoredTextSpanPiece

_NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)


def _memory(number: int, *, content: str | None = None) -> StoredMemory:
    return StoredMemory(
        memory_id=f"memory-{number}",
        content=f"标题🙂 source {number}" if content is None else content,
        metadata_json="{}",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _embedding(memory: StoredMemory, part: int) -> StoredEmbedding:
    digest = sha256(memory.content.encode()).hexdigest()
    return StoredEmbedding(
        embedding_id=f"{memory.memory_id}-part-{part}",
        memory_id=memory.memory_id,
        object_part=part,
        values=(0.6, 0.8),
        normalized=True,
        model_id="batch-fixture-v1",
        space_id="batch-fixture-v1:2",
        task="retrieval.document",
        created_at=_NOW,
        text_selectors=(
            StoredTextSelector(
                parent_content_sha256=digest,
                embedding_input_sha256=digest,
                recipe_version="batch-fixture-v1",
                pieces=(StoredTextSpanPiece("body", 0, len(memory.content), digest),),
            ),
        ),
    )


@pytest.mark.parametrize("replacement", (False, True))
def test_batch_selector_reads_scale_with_parents_instead_of_embedding_keys(
    tmp_path: Path, replacement: bool
) -> None:
    memories = tuple(_memory(number) for number in range(4))
    embeddings = tuple(_embedding(memory, part) for memory in memories for part in range(6))
    statements: list[str] = []
    with LocalStore(tmp_path) as store:
        if replacement:
            store.records.write_memories(memories)
        with store._connections.connection() as connection:
            connection.set_trace_callback(statements.append)
        if replacement:
            store.records.replace_memory_embeddings(memories, embeddings)
        else:
            assert store.records.write_memories(memories, embeddings) == (True,) * 4
        content_reads = [
            statement
            for statement in statements
            if " ".join(statement.upper().split()).startswith(
                (
                    "SELECT CONTENT FROM MEMORY_RECORDS ",
                    "SELECT MEMORY_ID, CONTENT FROM MEMORY_RECORDS ",
                )
            )
        ]
        assert len(content_reads) <= len(memories)
        stored = store.index.read_text_selectors(tuple(row.embedding_id for row in embeddings))
        assert stored == {row.embedding_id: row.text_selectors for row in embeddings}
        assert {row.embedding_id for row in store.index.pending_index_operations()} == {
            row.embedding_id for row in embeddings
        }


@pytest.mark.parametrize("replacement", (False, True))
@pytest.mark.parametrize("invalid_field", ("parent", "piece", "input"))
def test_invalid_selector_rolls_back_the_batch_and_its_outbox(
    tmp_path: Path, replacement: bool, invalid_field: str
) -> None:
    memories = (_memory(0), _memory(1))
    first, second = (_embedding(memory, 0) for memory in memories)
    selector = second.text_selectors[0]
    wrong_digest = "0" * 64
    if invalid_field == "parent":
        selector = replace(selector, parent_content_sha256=wrong_digest)
    elif invalid_field == "piece":
        selector = replace(
            selector, pieces=(replace(selector.pieces[0], piece_sha256=wrong_digest),)
        )
    else:
        selector = replace(selector, embedding_input_sha256=wrong_digest)
    second = replace(second, text_selectors=(selector,))
    with LocalStore(tmp_path) as store:
        if replacement:
            old = (_memory(0, content="old first"), _memory(1, content="old second"))
            store.records.write_memories(old, tuple(_embedding(memory, 0) for memory in old))
        ids = tuple(memory.memory_id for memory in memories)
        embedding_ids = (first.embedding_id, second.embedding_id)
        previous_memories = store.records.read_memories(ids)
        previous_selectors = store.index.read_text_selectors(embedding_ids)
        previous_operations = store.index.pending_index_operations()
        with pytest.raises(ValueError, match="text selector"):
            if replacement:
                store.records.replace_memory_embeddings(memories, (first, second))
            else:
                store.records.write_memories(memories, (first, second))
        assert store.records.read_memories(ids) == previous_memories
        assert store.index.read_text_selectors(embedding_ids) == previous_selectors
        assert store.index.pending_index_operations() == previous_operations


def test_selector_source_follows_interleaved_parent_keys(tmp_path: Path) -> None:
    first, second = _memory(0), _memory(1)
    embeddings = (_embedding(first, 0), _embedding(second, 0), _embedding(first, 1))
    with LocalStore(tmp_path) as store:
        store.records.write_memories((first, second), embeddings)
        assert store.index.read_text_selectors(tuple(row.embedding_id for row in embeddings)) == {
            row.embedding_id: row.text_selectors for row in embeddings
        }


def test_selector_source_does_not_outlive_a_write_batch(tmp_path: Path) -> None:
    memory = _memory(0)
    embedding = _embedding(memory, 0)
    with LocalStore(tmp_path) as store:
        store.records.write_memory(memory, (embedding,))
        store.records.write_memory(replace(memory, content="changed source"))
        with pytest.raises(ValueError, match="text selector parent digest"):
            store.index.write_embedding(_embedding(memory, 1))
        assert store.index.read_embedding(f"{memory.memory_id}-part-1") is None


def test_replacing_embeddings_removes_old_selectors_and_queues_both_index_actions(
    tmp_path: Path,
) -> None:
    memory = _memory(0)
    old = (_embedding(memory, 0), _embedding(memory, 1))
    changed = replace(memory, content="新的🙂 source")
    new = (_embedding(changed, 0), _embedding(changed, 2))
    with LocalStore(tmp_path) as store:
        store.records.write_memory(memory, old)
        pending = store.index.pending_index_operations()
        assert store.index.acknowledge_index_operations(pending) == len(pending)
        store.records.replace_memory_embeddings((changed,), new)
        assert store.records.read_memory(memory.memory_id) == changed
        assert store.index.read_embedding(old[1].embedding_id) is None
        ids = tuple(row.embedding_id for row in (*old, new[1]))
        assert store.index.read_text_selectors(ids) == {
            new[0].embedding_id: new[0].text_selectors,
            old[1].embedding_id: (),
            new[1].embedding_id: new[1].text_selectors,
        }
        operations = store.index.pending_index_operations()
        assert len(operations) == 4
        assert {(row.embedding_id, row.action) for row in operations} == {
            (old[0].embedding_id, "delete"),
            (old[1].embedding_id, "delete"),
            (new[0].embedding_id, "upsert"),
            (new[1].embedding_id, "upsert"),
        }
