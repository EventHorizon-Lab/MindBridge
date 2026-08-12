"""Tests for contracts shared by Python, REST, and MCP entry points."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from mindbridge.contracts import (
    FeedbackRequest,
    IdentityObservationInput,
    MediaObjectInput,
    ObserveRequest,
    RecallFilters,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import FeedbackType, IdentityKind, MediaKind, MemoryType, SensorKind

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_contracts_reject_unknown_fields() -> None:
    """Typos cannot silently change a public request."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RecallQuery(text="red screwdriver", typo=True)  # type: ignore[call-arg]


def test_recall_query_accepts_media_without_text() -> None:
    """Recall remains multimodal-first rather than requiring language."""
    query = RecallQuery(media_object_ids=("media_01",))

    assert query.text is None


def test_recall_query_requires_one_modality() -> None:
    """An empty query cannot trigger an unbounded search."""
    with pytest.raises(ValidationError, match="requires text or media_object_ids"):
        RecallQuery()


def test_feedback_contract_requires_typed_targets_and_corrections() -> None:
    with pytest.raises(ValidationError, match="memory_id"):
        FeedbackRequest(tenant_id="tenant_01", feedback_type=FeedbackType.WRONG)
    with pytest.raises(ValidationError, match="correction_summary"):
        FeedbackRequest(
            tenant_id="tenant_01",
            feedback_type=FeedbackType.CORRECTION,
            memory_id="memory_01",
        )
    with pytest.raises(ValidationError, match="recall_trace_id"):
        FeedbackRequest(tenant_id="tenant_01", feedback_type=FeedbackType.MISSING)


def test_recall_query_rejects_whitespace_text() -> None:
    """Whitespace cannot masquerade as a query."""
    with pytest.raises(ValidationError, match="at least 1 character"):
        RecallQuery(text=" ")


def test_recall_query_rejects_duplicate_media() -> None:
    """One physical query object is encoded at most once."""
    with pytest.raises(ValidationError, match="must be unique"):
        RecallQuery(media_object_ids=("media_01", "media_01"))


def test_recall_defaults_to_returning_evidence() -> None:
    """Evidence is the default product behavior, not an opt-in debug field."""
    request = RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="toolbox"))

    assert request.include_evidence is True


def test_recall_rejects_duplicate_explicit_memory_context() -> None:
    with pytest.raises(ValidationError, match="memory_ids must be unique"):
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="What happened next?"),
            memory_ids=("memory_01", "memory_01"),
        )


def test_recall_filters_reject_reversed_time_range() -> None:
    """Structured time filtering rejects impossible intervals."""
    with pytest.raises(ValidationError, match="occurred_before"):
        RecallFilters(
            occurred_after=NOW,
            occurred_before=datetime(2026, 8, 11, 11, 0, tzinfo=timezone.utc),
        )


def test_observe_requires_timezone_aware_timestamps() -> None:
    """Device timestamps must identify a real instant before ingestion."""
    with pytest.raises(ValidationError, match="timezone info"):
        _observe_request(observed_at=datetime(2026, 8, 11, 12, 0))  # noqa: DTZ001


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("sequence", 2**63),
        ("clock_offset_ms", 2**31),
        ("clock_offset_ms", -(2**31) - 1),
    ),
)
def test_observe_rejects_values_outside_storage_integer_ranges(
    field_name: str,
    value: int,
) -> None:
    payload = _observe_request().model_dump()
    payload[field_name] = value

    with pytest.raises(ValidationError):
        ObserveRequest.model_validate(payload)


def test_media_hash_is_canonicalized_for_content_deduplication() -> None:
    media = _observe_request().media_objects[0].model_copy(update={"sha256": "A" * 64})

    canonical = MediaObjectInput.model_validate(media.model_dump())

    assert canonical.sha256 == "a" * 64


@pytest.mark.parametrize("field_name", ("size_bytes", "duration_ms"))
def test_media_rejects_values_outside_storage_integer_range(field_name: str) -> None:
    payload = _observe_request().media_objects[0].model_dump()
    payload[field_name] = 2**63

    with pytest.raises(ValidationError):
        MediaObjectInput.model_validate(payload)


def test_write_contracts_reject_inconsistent_ranges_and_references() -> None:
    media = _observe_request().media_objects[0]
    with pytest.raises(ValidationError, match="ended_at must not precede occurred_at"):
        _observe_request(ended_at=NOW - timedelta(milliseconds=1))
    with pytest.raises(ValidationError, match="duplicate IDs"):
        _observe_request(media_objects=(media, media))
    with pytest.raises(ValidationError, match="media duration exceeds source observation"):
        _observe_request(media_objects=(media.model_copy(update={"duration_ms": 1}),))
    with pytest.raises(ValidationError, match="ended_at must not precede occurred_at"):
        RememberRequest(
            tenant_id="tenant_01",
            summary="Remember this",
            memory_type=MemoryType.SEMANTIC,
            occurred_at=NOW,
            ended_at=NOW - timedelta(milliseconds=1),
        )
    with pytest.raises(ValidationError, match="evidence_ids must not contain duplicates"):
        RememberRequest(
            tenant_id="tenant_01",
            summary="Remember this",
            memory_type=MemoryType.SEMANTIC,
            occurred_at=NOW,
            evidence_ids=("evidence_01", "evidence_01"),
        )


def test_observe_accepts_only_bounded_anonymous_identity_metadata() -> None:
    identity = IdentityObservationInput(
        identity_id="person_device_01",
        kind=IdentityKind.FACE,
        start_ms=0,
        end_ms=1,
        confidence=0.9,
        model_id="insightface/buffalo_l",
        model_revision="1.0.1",
    )

    with pytest.raises(ValidationError, match="exceeds source duration"):
        _observe_request(identity_observations=(identity,))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        IdentityObservationInput(
            **identity.model_dump(),
            embedding=[1.0, 0.0],  # type: ignore[call-arg]
        )


def test_request_collections_reject_unbounded_fanout() -> None:
    media = _observe_request().media_objects[0]
    identity = IdentityObservationInput(
        identity_id="person_device_01",
        kind=IdentityKind.FACE,
        start_ms=0,
        end_ms=0,
        confidence=0.9,
        model_id="insightface/buffalo_l",
        model_revision="1.0.1",
    )

    with pytest.raises(ValidationError, match="at most 8 items"):
        _observe_request(
            media_objects=tuple(
                media.model_copy(update={"media_object_id": f"media_{index}"}) for index in range(9)
            )
        )
    with pytest.raises(ValidationError, match="at most 256 items"):
        _observe_request(identity_observations=(identity,) * 257)
    with pytest.raises(ValidationError, match="at most 100 items"):
        RememberRequest(
            tenant_id="tenant_01",
            summary="Remember this",
            memory_type=MemoryType.SEMANTIC,
            occurred_at=NOW,
            evidence_ids=tuple(f"evidence_{index}" for index in range(101)),
        )
    with pytest.raises(ValidationError, match="at most 100 items"):
        RecallFilters(person_ids=tuple(f"person_{index}" for index in range(101)))


def _observe_request(
    *,
    observed_at: datetime = NOW,
    ended_at: datetime = NOW,
    media_objects: tuple[MediaObjectInput, ...] | None = None,
    identity_observations: tuple[IdentityObservationInput, ...] = (),
) -> ObserveRequest:
    return ObserveRequest(
        tenant_id="tenant_01",
        device_id="device_01",
        boot_id="boot_01",
        sequence=1,
        sensor=SensorKind.CAMERA,
        media_objects=media_objects
        or (
            MediaObjectInput(
                media_object_id="media_01",
                kind=MediaKind.VIDEO,
                uri="s3://memories/video.mp4",
                sha256="a" * 64,
                size_bytes=100,
                created_at=NOW,
            ),
        ),
        occurred_at=NOW,
        ended_at=ended_at,
        observed_at=observed_at,
        identity_observations=identity_observations,
    )
