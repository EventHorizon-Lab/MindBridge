"""Shared validation for framework-independent domain records."""

import math
from datetime import datetime, timezone

from mindbridge.core.errors import DomainInvariantError


def utc_now() -> datetime:
    """The single aware-clock reading every default timestamp uses."""
    return datetime.now(timezone.utc)


def require_similarity(value: float, field_name: str) -> None:
    """Reject a cosine bound outside the only range a normalized comparison can produce."""
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise DomainInvariantError(f"{field_name} must be between -1 and 1")


def require_probability(value: float, field_name: str) -> None:
    """Reject a confidence or probability outside the unit interval."""
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise DomainInvariantError(f"{field_name} must be between 0 and 1")


def require_aware_datetime(value: datetime, field_name: str) -> None:
    """Reject timestamps whose real-world instant is ambiguous."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainInvariantError(f"{field_name} must include timezone information")


def require_non_empty(value: str, field_name: str) -> None:
    """Reject blank domain identifiers and required text."""
    if not value.strip():
        raise DomainInvariantError(f"{field_name} must not be empty")
