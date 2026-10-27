"""The public formation input preserves old constructors and validates source timestamps."""

from datetime import datetime, timedelta, timezone

import pytest

from mindbridge import FormationInput, ModelInput, ObservationContext, ValidationError

OBSERVED_AT = datetime(2026, 3, 2, 12, tzinfo=timezone.utc)


def test_formation_input_old_positional_constructor_has_no_implicit_clock() -> None:
    target = FormationInput("source", ModelInput(text="yesterday"), ObservationContext(), (), False)

    assert target.occurred_at is None
    assert target.occurred_end is None
    assert target.created_at is None


def test_formation_input_preserves_aware_timestamps() -> None:
    start = OBSERVED_AT.astimezone(timezone(timedelta(hours=8)))
    target = FormationInput(
        "source",
        ModelInput(text="yesterday"),
        ObservationContext(),
        occurred_at=start,
        occurred_end=start + timedelta(minutes=1),
        created_at=OBSERVED_AT + timedelta(days=1),
    )

    assert target.occurred_at == OBSERVED_AT
    assert target.occurred_at.utcoffset() == timedelta(hours=8)
    assert target.occurred_end == OBSERVED_AT + timedelta(minutes=1)
    assert target.created_at == datetime(2026, 3, 3, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize("field", ["occurred_at", "occurred_end", "created_at"])
def test_formation_input_rejects_naive_timestamps(field: str) -> None:
    with pytest.raises(ValidationError, match="timezone"):
        FormationInput(
            "source",
            ModelInput(text="yesterday"),
            ObservationContext(),
            **{field: OBSERVED_AT.replace(tzinfo=None)},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("start", [None, OBSERVED_AT, OBSERVED_AT + timedelta(seconds=1)])
def test_formation_input_rejects_end_without_a_positive_interval(start: datetime | None) -> None:
    with pytest.raises(ValidationError, match="occurred_end must be later than occurred_at"):
        FormationInput(
            "source",
            ModelInput(text="yesterday"),
            ObservationContext(),
            occurred_at=start,
            occurred_end=OBSERVED_AT,
        )
