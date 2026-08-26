"""Shared validation for framework-independent domain records."""

import math
from collections.abc import Container
from datetime import datetime, timezone

from mindbridge.core.errors import DomainInvariantError


def utc_now() -> datetime:
    """The single aware-clock reading every default timestamp uses."""
    return datetime.now(timezone.utc)


def require_similarity(value: float, field_name: str) -> None:
    """Reject a cosine bound outside the only range a normalized comparison can produce."""
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise DomainInvariantError(f"{field_name} must be between -1 and 1")


def require_bounded_count(
    value: int,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
) -> None:
    """Reject a page size, window, or budget outside the range its sweep allows.

    Every consolidation kind bounds the same shape of knob, and each one had written the
    comparison and the message out by hand.
    """
    if not minimum <= value <= maximum:
        raise DomainInvariantError(f"{field_name} must be between {minimum} and {maximum}")


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


# Migration 0021 dropped `model_revision` from every derived record and `space_revision`
# from every vector. Nothing writes either name any more, but three readers still meet
# values that were written before that migration and validate them against models whose
# config is `extra="forbid"`: an operator's `*_CONFIG_JSON` object, an edge device's
# spooled `ObserveRequest` payloads, and -- during a rolling upgrade, where the API goes
# first -- `/v1/observe` bodies from a device still on the previous release. For a field
# that was removed, refusing the request is the wrong answer: the value is not wanted, so
# ignoring it is what a forward-compatible reader does. Dropping only this closed list
# keeps `extra="forbid"` doing its real job of catching a typo.
RETIRED_FIELD_NAMES = frozenset(
    {
        "model_revision",
        "space_revision",
        "association_model_revision",
    }
)


def without_retired_fields(data: object, accepted: Container[str] = frozenset()) -> object:
    """Drop keys retired by migration 0021 so an older writer's payload still validates.

    `accepted` is the names the reader still declares, and a retired name in it is kept.
    One of these names went on meaning something after 0021: the local Jina embedder's
    `model_revision` is a loader argument that selects which weights and which remote code
    get executed, not a record of what already ran. Dropping it would silently replace an
    operator's pin with the default, which is worse than the strictness this helper exists
    to relax -- so the rule is to ignore a retired name only where nothing reads it.
    """
    if not isinstance(data, dict):
        return data
    retired = {key for key in RETIRED_FIELD_NAMES if key in data and key not in accepted}
    if not retired:
        return data
    return {key: value for key, value in data.items() if key not in retired}
