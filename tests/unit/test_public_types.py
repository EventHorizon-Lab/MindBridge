"""Checks for the small public values and exception boundary."""

import pickle
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindbridge import IndexQuantization, PrefetchResult, StreamInput
from mindbridge.exceptions import (
    RETRYABLE_REASONS,
    IndexUnavailableError,
    MemoryNotFoundError,
    MindBridgeError,
    ModelError,
    ModelOutputTruncatedError,
    SpeakerNotFoundError,
    StorageError,
    ValidationError,
)
from mindbridge.types import (
    AbstentionReason,
    AnswerResult,
    AssetRef,
    Blob,
    MemoryRecord,
    MemoryType,
    Modality,
    Page,
    RetrievalCandidateTrace,
    RetrievalRejection,
    RetrievalTrace,
    SearchHit,
    TracedSearchResult,
)

NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


def test_index_quantization_is_a_stable_string_enum() -> None:
    assert tuple(mode.value for mode in IndexQuantization) == (
        "none",
        "fp16",
        "int8",
        "rabitq",
    )


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


def test_stream_input_snapshots_omni_parts_and_metadata() -> None:
    parts: list[str | Blob] = ["observation", Blob(b"image", "image/png")]
    metadata = {"source": "camera"}
    item = StreamInput(
        parts,
        occurred_at=NOW,
        occurred_end=NOW + timedelta(seconds=30),
        metadata=metadata,
        memory_type=MemoryType.EPISODIC,
    )

    parts[0] = "changed"
    metadata["source"] = "changed"

    assert item.content == ("observation", Blob(b"image", "image/png"))
    assert item.metadata == {"source": "camera"}
    assert pickle.loads(pickle.dumps(item)) == item
    with pytest.raises(ValidationError, match="timezone"):
        StreamInput("clip", occurred_at=NOW.replace(tzinfo=None))
    with pytest.raises(ValidationError, match="timezone"):
        StreamInput("clip", occurred_at="2026-08-31T09:00:00Z")  # type: ignore[arg-type]


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

    with pytest.raises(ValidationError, match="positive integer"):
        PrefetchResult(revision=0, hits=(hit,))


def test_retrieval_trace_values_are_immutable_and_bounded() -> None:
    hit = SearchHit(id="memory_1", content="memory", score=0.9, created_at=NOW)
    candidate = RetrievalCandidateTrace(
        memory_id=hit.id,
        index_ids=("embedding_1",),
        dense_relevance=0.9,
        dense_confidence=0.8,
        lexical_relevance=0.6,
        lexical_rerank_bonus=0.2,
        gate_relevance=0.8,
        base_relevance=0.9,
        reinforcement_factor=1.0,
        final_score=hit.score,
        rank=1,
    )
    result = TracedSearchResult(
        hits=(hit,),
        trace=RetrievalTrace(
            candidates=(candidate,),
            candidate_limit=50,
            exhaustive=True,
        ),
    )

    assert pickle.loads(pickle.dumps(result)) == result
    assert tuple(reason.value for reason in RetrievalRejection) == (
        "stale_index",
        "occurrence_range",
        "missing_memory",
        "memory_type",
        "minimum_relevance",
        "ambiguity",
        "limit",
    )
    with pytest.raises(ValidationError, match="must not exceed one"):
        RetrievalCandidateTrace(
            memory_id=hit.id,
            index_ids=("embedding_1",),
            final_score=1.1,
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
    with pytest.raises(ValidationError, match="later than occurred_at"):
        MemoryRecord(
            id="memory_1",
            content="memory",
            created_at=NOW,
            occurred_at=NOW,
            occurred_end=NOW,
        )
    record = MemoryRecord(
        id="memory_1",
        content="memory",
        created_at=NOW,
        occurred_at=NOW,
        occurred_end=NOW + timedelta(minutes=5),
    )
    assert record.occurred_end == NOW + timedelta(minutes=5)


def test_answer_and_page_reuse_the_public_values() -> None:
    memory = MemoryRecord(id="memory_1", content="A red screwdriver.", created_at=NOW)
    hit = SearchHit(id=memory.id, content=memory.content, score=0.9, created_at=memory.created_at)

    assert AnswerResult(answer="It is in the toolbox.", hits=(hit,)).hits == (hit,)
    abstention = AnswerResult(
        answer="unknown",
        abstained=True,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
    )
    assert abstention.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    with pytest.raises(ValidationError, match="must agree"):
        AnswerResult(answer="unknown", abstained=True)
    assert Page(items=(memory,), next_cursor="cursor_1").items == (memory,)


def test_public_exceptions_have_stable_categories() -> None:
    assert issubclass(ValidationError, MindBridgeError)
    assert issubclass(ValidationError, ValueError)
    assert issubclass(MemoryNotFoundError, MindBridgeError)
    assert issubclass(MemoryNotFoundError, LookupError)
    assert issubclass(IndexUnavailableError, StorageError)
    assert issubclass(ModelOutputTruncatedError, ModelError)
    assert {
        ValidationError.code,
        MemoryNotFoundError.code,
        ModelError.code,
        ModelOutputTruncatedError.code,
        StorageError.code,
        IndexUnavailableError.code,
    } == {
        "validation_error",
        "memory_not_found",
        "model_error",
        "model_output_truncated",
        "storage_error",
        "index_unavailable",
    }


def test_public_errors_carry_optional_reason_stage_and_subject() -> None:
    plain = ModelError("embedding request failed")
    classified = ModelError(
        "embedding request failed",
        reason="rate_limited",
        stage="embed",
        subject="asset_image",
    )

    assert (plain.reason, plain.stage, plain.subject, plain.retryable) == (None, None, None, False)
    assert classified.reason == "rate_limited"
    assert classified.stage == "embed"
    assert classified.subject == "asset_image"
    assert classified.retryable is True
    assert str(classified) == "embedding request failed"


def test_retryability_is_a_lookup_on_reason_and_defaults_to_false() -> None:
    for reason in RETRYABLE_REASONS:
        assert ModelError("failed", reason=reason).retryable is True
    for reason in ("auth_failed", "request_rejected", "schema_unsupported", "invented_reason"):
        assert ModelError("failed", reason=reason).retryable is False


def test_single_condition_errors_default_their_own_reason() -> None:
    assert MemoryNotFoundError("missing").reason == "memory_not_found"
    assert SpeakerNotFoundError("missing").reason == "speaker_not_found"
    assert ValidationError("bad").reason == "input_invalid"
    assert ModelOutputTruncatedError("cut off").reason == "output_truncated"
    assert ModelOutputTruncatedError("cut off").retryable is False


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
