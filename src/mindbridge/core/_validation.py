"""Shared validation for framework-independent domain records."""

from datetime import datetime, timezone

from mindbridge.core.errors import DomainInvariantError


def utc_now() -> datetime:
    """The single aware-clock reading every default timestamp uses."""
    return datetime.now(timezone.utc)


def require_aware_datetime(value: datetime, field_name: str) -> None:
    """Reject timestamps whose real-world instant is ambiguous."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainInvariantError(f"{field_name} must include timezone information")


def require_non_empty(value: str, field_name: str) -> None:
    """Reject blank domain identifiers and required text."""
    if not value.strip():
        raise DomainInvariantError(f"{field_name} must not be empty")
