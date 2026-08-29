"""Checks for the small public values and exception boundary."""

import pickle
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge.exceptions import (
    IndexUnavailableError,
    MemoryNotFoundError,
    MindBridgeError,
    ModelError,
    StorageError,
    ValidationError,
)
from mindbridge.types import (
    AnswerResult,
    AssetRef,
    Blob,
    MemoryRecord,
    MemoryType,
    Modality,
    Page,
    SearchHit,
)

NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


def test_memory_record_metadata_is_detached_and_serializable() -> None:
    metadata = {"source": "camera"}
    memory = MemoryRecord(
        id="memory_1", content="A red screwdriver.", created_at=NOW, metadata=metadata
    )

    metadata["source"] = "microphone"

    assert memory.metadata == {"source": "camera"}
    assert memory.memory_type is MemoryType.SEMANTIC
    memory.metadata["source"] = "local edit"  # type: ignore[index]
    assert asdict(memory)["metadata"] == {"source": "local edit"}
    assert pickle.loads(pickle.dumps(memory)) == memory
    with pytest.raises(FrozenInstanceError):
        memory.content = "changed"  # type: ignore[misc]


def test_search_hit_is_flat_and_rejects_invalid_scores() -> None:
    hit = SearchHit(
        id="memory_1",
        content="A red screwdriver.",
        score=0.9,
        created_at=NOW,
        memory_type=MemoryType.EPISODIC,
    )

    assert hit.id == "memory_1"
    assert hit.content == "A red screwdriver."
    assert hit.memory_type is MemoryType.EPISODIC
    assert not hasattr(hit, "memory")
    with pytest.raises(ValidationError, match="between zero and one"):
        SearchHit(id="memory_1", content="A red screwdriver.", score=float("nan"), created_at=NOW)
    with pytest.raises(ValidationError, match="between zero and one"):
        SearchHit(id="memory_1", content="A red screwdriver.", score=1.1, created_at=NOW)
    with pytest.raises(ValidationError, match="memory_type"):
        MemoryRecord(
            id="memory_1",
            content="A red screwdriver.",
            created_at=NOW,
            memory_type="episodic",  # type: ignore[arg-type]
        )


def test_media_inputs_are_explicit_immutable_values(tmp_path: Path) -> None:
    blob = Blob(b"image", "image/png", name="cat.png")
    opaque = AssetRef(id="asset_1", modality=Modality.IMAGE)
    resolved = _asset(tmp_path, "asset_1", Modality.IMAGE, "image/png")

    assert blob.media_type == "image/png"
    assert "data=" not in repr(blob)
    assert not opaque.is_resolved
    assert resolved.is_resolved
    assert str(resolved.path) not in repr(resolved)
    assert pickle.loads(pickle.dumps(resolved)) == resolved
    with pytest.raises(ValidationError, match="media_type"):
        Blob(b"image", "image/*")


def test_records_preserve_persisted_media_modality_and_allow_empty_derived_text(
    tmp_path: Path,
) -> None:
    image = _asset(tmp_path, "image_1", Modality.IMAGE, "image/png")
    audio = _asset(tmp_path, "audio_1", Modality.AUDIO, "audio/wav")

    image_record = MemoryRecord(
        id="memory_image",
        content="",
        created_at=NOW,
        assets=(image,),
        modality=Modality.IMAGE,
    )
    omni_hit = SearchHit(
        id="memory_omni",
        content="derived description",
        score=0.9,
        created_at=NOW,
        assets=(image, audio),
        modality=Modality.OMNI,
    )

    assert image_record.modality is Modality.IMAGE
    assert omni_hit.modality is Modality.OMNI
    with pytest.raises(ValidationError, match="content or media"):
        MemoryRecord(id="memory_1", content=" ", created_at=NOW)
    with pytest.raises(ValidationError, match="resolved"):
        MemoryRecord(
            id="memory_1",
            content="image",
            created_at=NOW,
            assets=(AssetRef("asset_1"),),
            modality=Modality.IMAGE,
        )


def test_public_values_reject_naive_times() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        MemoryRecord(id="memory_1", content="memory", created_at=NOW.replace(tzinfo=None))


def test_answer_and_page_reuse_the_public_values() -> None:
    memory = MemoryRecord(id="memory_1", content="A red screwdriver.", created_at=NOW)
    hit = SearchHit(id=memory.id, content=memory.content, score=0.9, created_at=memory.created_at)

    assert AnswerResult(answer="It is in the toolbox.", hits=(hit,)).hits == (hit,)
    assert Page(items=(memory,), next_cursor="cursor_1").items == (memory,)


def test_public_exceptions_have_stable_categories() -> None:
    assert issubclass(ValidationError, MindBridgeError)
    assert issubclass(ValidationError, ValueError)
    assert issubclass(MemoryNotFoundError, MindBridgeError)
    assert issubclass(MemoryNotFoundError, LookupError)
    assert issubclass(IndexUnavailableError, StorageError)
    assert {
        ValidationError.code,
        MemoryNotFoundError.code,
        ModelError.code,
        StorageError.code,
        IndexUnavailableError.code,
    } == {
        "validation_error",
        "memory_not_found",
        "model_error",
        "storage_error",
        "index_unavailable",
    }


def _asset(
    directory: Path,
    asset_id: str,
    modality: Modality,
    media_type: str,
) -> AssetRef:
    return AssetRef(
        id=asset_id,
        modality=modality,
        media_type=media_type,
        size_bytes=5,
        sha256="a" * 64,
        name=f"{asset_id}.bin",
        path=directory / asset_id,
    )
