"""Row accessors and the text, time, and vector encodings every table shares."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Literal

from mindbridge.infrastructure.local.store.rows import (
    IndexDocument,
    StoredAsset,
    StoredEmbedding,
    StoredMemory,
    require_aware,
)
from mindbridge.types import MemoryContext

SQLITE_PARAMETER_BATCH = 900


def prepare_write_batch(
    memories: Iterable[StoredMemory],
    embeddings: Iterable[StoredEmbedding],
) -> tuple[
    tuple[StoredMemory, ...],
    tuple[StoredEmbedding, ...],
    dict[str, set[str]],
]:
    supplied_memories = tuple(memories)
    supplied_embeddings = tuple(embeddings)
    if not supplied_memories and supplied_embeddings:
        raise ValueError("embeddings require at least one written memory")
    memory_ids = [memory.memory_id for memory in supplied_memories]
    if len(set(memory_ids)) != len(memory_ids):
        raise ValueError("memory IDs must be unique")
    memory_id_set = set(memory_ids)
    if any(embedding.memory_id not in memory_id_set for embedding in supplied_embeddings):
        raise ValueError("all embeddings must belong to a written memory")
    embedding_ids = [embedding.embedding_id for embedding in supplied_embeddings]
    if len(set(embedding_ids)) != len(embedding_ids):
        raise ValueError("embedding IDs must be unique")
    supplied_by_memory: dict[str, set[str]] = {memory_id: set() for memory_id in memory_ids}
    for embedding in supplied_embeddings:
        supplied_by_memory[embedding.memory_id].add(embedding.embedding_id)
    return supplied_memories, supplied_embeddings, supplied_by_memory


def canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def memory_from_row(
    row: sqlite3.Row,
    *,
    assets: tuple[StoredAsset, ...] = (),
    context: MemoryContext | None = None,
) -> StoredMemory:
    return StoredMemory(
        memory_id=row_text(row, "memory_id"),
        content=row_text(row, "content"),
        modality=row_text(row, "modality"),
        memory_type=row_text(row, "memory_type"),
        place_id=optional_row_text(row, "place_id"),
        assets=assets,
        metadata_json=row_text(row, "metadata_json"),
        occurred_at=optional_datetime_from_row(row, "occurred_at"),
        occurred_end=optional_datetime_from_row(row, "occurred_end"),
        last_accessed_at=optional_datetime_from_row(row, "last_accessed_at"),
        access_count=int(row["access_count"]),
        created_at=parse_datetime(row_text(row, "created_at")),
        updated_at=parse_datetime(row_text(row, "updated_at")),
        forgotten_at=optional_datetime_from_row(row, "forgotten_at"),
        context=context,
    )


def asset_from_row(row: sqlite3.Row) -> StoredAsset:
    return StoredAsset(
        asset_id=row_text(row, "asset_id"),
        modality=row_text(row, "modality"),
        mime_type=row_text(row, "mime_type"),
        size_bytes=int(row["size_bytes"]),
        sha256=row_text(row, "sha256"),
        relative_path=row_text(row, "relative_path"),
        name=None if row["name"] is None else row_text(row, "name"),
        transcript=(None if row["transcript"] is None else row_text(row, "transcript")),
        created_at=parse_datetime(row_text(row, "created_at")),
    )


def embedding_from_row(row: sqlite3.Row) -> StoredEmbedding:
    dimension = int(row["dimension"])
    vector = row["vector"]
    if not isinstance(vector, bytes):
        raise RuntimeError("stored embedding vector is not a BLOB")
    return StoredEmbedding(
        embedding_id=row_text(row, "embedding_id"),
        memory_id=row_text(row, "memory_id"),
        object_part=int(row["object_part"]),
        model_id=row_text(row, "model_id"),
        space_id=row_text(row, "space_id"),
        task=row_text(row, "task"),
        values=unpack_vector(vector, dimension),
        normalized=bool(int(row["normalized"])),
        created_at=parse_datetime(row_text(row, "created_at")),
    )


def index_document_from_row(
    row: sqlite3.Row,
    identity_ids: Mapping[str, tuple[str, ...]],
) -> IndexDocument:
    return IndexDocument(
        embedding=embedding_from_row(row),
        content=row_text(row, "content"),
        metadata_json=row_text(row, "metadata_json"),
        memory_type=row_text(row, "memory_type"),
        occurred_at=optional_datetime_from_row(row, "occurred_at"),
        occurred_end=optional_datetime_from_row(row, "occurred_end"),
        place_id=optional_row_text(row, "place_id"),
        identity_ids=identity_ids.get(row_text(row, "memory_id"), ()),
    )


def index_action(row: sqlite3.Row) -> Literal["upsert", "delete"]:
    action = row_text(row, "action")
    if action == "upsert":
        return "upsert"
    if action == "delete":
        return "delete"
    raise RuntimeError(f"invalid queued index action: {action}")


def row_text(row: sqlite3.Row, column: str) -> str:
    value = row[column]
    if not isinstance(value, str):
        raise RuntimeError(f"stored {column} is not text")
    return value


def optional_row_text(row: sqlite3.Row, column: str) -> str | None:
    return None if row[column] is None else row_text(row, column)


def row_blob(row: sqlite3.Row, column: str) -> bytes:
    value = row[column]
    if not isinstance(value, bytes):
        raise RuntimeError(f"stored {column} is not a BLOB")
    return value


def require_storable_vector(values: tuple[float, ...], *, normalized: bool) -> None:
    """Check vector content once, at the boundary where it becomes authoritative."""
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("embedding values must be finite and non-empty")
    if normalized and not math.isclose(
        math.sqrt(sum(value * value for value in values)),
        1.0,
        rel_tol=1e-4,
        abs_tol=1e-6,
    ):
        raise ValueError("normalized embedding must have unit length")


def pack_vector(values: tuple[float, ...]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(value: bytes, dimension: int) -> tuple[float, ...]:
    if dimension <= 0 or len(value) != dimension * 4:
        raise RuntimeError("stored embedding dimension does not match its FP32 BLOB")
    return struct.unpack(f"<{dimension}f", value)


def normalized_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized or any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain finite values")
    magnitude = math.sqrt(math.fsum(value * value for value in normalized))
    if magnitude == 0.0:
        raise ValueError(f"{name} must not be a zero vector")
    return tuple(value / magnitude for value in normalized)


def datetime_text(value: datetime) -> str:
    require_aware(value, "datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def optional_datetime_text(value: datetime | None) -> str | None:
    return None if value is None else datetime_text(value)


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require_aware(parsed, "stored datetime")
    return parsed


def optional_datetime_from_row(row: sqlite3.Row, column: str) -> datetime | None:
    value = row[column]
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"stored {column} is not text")
    return parse_datetime(value)
