"""Tests for contracts shared by Python, REST, and MCP entry points."""

from datetime import datetime, timezone

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
)
from mindbridge.core import FeedbackType, IdentityKind, MediaKind, SensorKind

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


def _observe_request(
    *,
    observed_at: datetime = NOW,
    identity_observations: tuple[IdentityObservationInput, ...] = (),
) -> ObserveRequest:
    return ObserveRequest(
        tenant_id="tenant_01",
        device_id="device_01",
        boot_id="boot_01",
        sequence=1,
        sensor=SensorKind.CAMERA,
        media_objects=(
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
        ended_at=NOW,
        observed_at=observed_at,
        identity_observations=identity_observations,
    )
