"""Validation of caller-supplied arguments at the public boundary."""

from __future__ import annotations

import base64
import json
import math
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import get_args

from mindbridge.exceptions import ValidationError
from mindbridge.infrastructure.local.store import StoredMemory, datetime_text
from mindbridge.types import AnswerPolicy, MemoryType, Modality, RetrievalMode, RetrievalScope

MAX_TEXT_CHARACTERS = 65_536


def positive_seconds(value: object, name: str) -> timedelta:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValidationError(f"{name} must be a positive finite number")
    return timedelta(seconds=float(value))


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{name} must be a positive integer")
    return value


def unit_interval(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= value <= 1.0
    ):
        raise ValidationError(f"{name} must be between zero and one")
    return float(value)


def validated_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be non-empty text")
    normalized = unicodedata.normalize("NFC", value.strip())
    if len(normalized) > MAX_TEXT_CHARACTERS:
        raise ValidationError(f"{name} must not exceed {MAX_TEXT_CHARACTERS} characters")
    return normalized


def modality_names(modalities: Iterable[Modality]) -> str:
    return ", ".join(sorted(modality.value for modality in modalities)) or "nothing"


def scope_description(scope: RetrievalScope | None) -> str | None:
    """Describe a requested scope that matched nothing, or `None` when none was requested."""
    if scope is None:
        return None
    bounds = []
    if scope.place_id is not None:
        bounds.append(f"place {scope.place_id}")
    if scope.identity_id is not None:
        bounds.append(f"identity {scope.identity_id}")
    if scope.near is not None:
        radius = "" if scope.radius_m is None else f" within {scope.radius_m} m"
        bounds.append(f"frame {scope.near.frame_id}{radius}")
    if scope.valid_at is not None:
        bounds.append(f"valid at {scope.valid_at.isoformat()}")
    if scope.known_at is not None:
        bounds.append(f"known at {scope.known_at.isoformat()}")
    if not bounds:
        return None
    return f"no memory matched the requested scope: {', '.join(bounds)}"


def validated_retrieval_scope(value: RetrievalScope | None) -> RetrievalScope | None:
    if value is not None and not isinstance(value, RetrievalScope):
        raise ValidationError("scope must be a RetrievalScope")
    return value


def validated_retrieval_mode(value: RetrievalMode) -> RetrievalMode:
    if not isinstance(value, RetrievalMode):
        raise ValidationError("retrieval_mode must be a RetrievalMode")
    return value


def validated_memory_type(value: object) -> MemoryType:
    if not isinstance(value, MemoryType):
        raise ValidationError("memory_type must be a MemoryType value")
    return value


def optional_memory_type(value: object) -> MemoryType | None:
    return None if value is None else validated_memory_type(value)


def pushed_memory_types(
    requested: MemoryType | frozenset[MemoryType] | None,
) -> frozenset[MemoryType] | None:
    """Return the type filter to push into the index, or `None` when it would filter nothing.

    A request naming every type is the unfiltered request written out, and pushing it would buy
    one index route per type for a filter that removes nothing.
    """
    if requested is None:
        return None
    pushed = frozenset({requested}) if isinstance(requested, MemoryType) else requested
    return None if pushed >= frozenset(MemoryType) else pushed


def validated_evidence_budget(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError("evidence_budget_chars must be a positive integer")
    return value


def validated_decay_half_life(days: float | None) -> timedelta | None:
    if days is None:
        return None
    if (
        isinstance(days, bool)
        or not isinstance(days, int | float)
        or not math.isfinite(days)
        or days <= 0
    ):
        raise ValidationError("decay_half_life_days must be a positive finite number")
    try:
        return timedelta(days=days)
    except OverflowError:
        raise ValidationError("decay_half_life_days is too large") from None


def validated_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{name} must be non-empty and trimmed")
    return value


def strict_bool(value: object, name: str) -> bool:
    """Read a switch as the boolean it is: a truthy value is a mistake, not a silent yes."""
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be a boolean")
    return value


def capture_flag(capture: object) -> bool:
    """Read `capture` as the mode it is, so a truthy value is a mistake and not a silent yes."""
    if not isinstance(capture, bool):
        raise ValidationError("capture must be a boolean")
    return capture


def memory_id_filter(memory_ids: Sequence[str] | None) -> tuple[str, ...] | None:
    """Normalize an optional `memory_ids` filter the way `forget()` normalizes its argument."""
    if memory_ids is None:
        return None
    if isinstance(memory_ids, (str, bytes)):
        raise ValidationError("memory_ids must be a sequence of memory IDs")
    try:
        return tuple(validated_identifier(memory_id, "memory_id") for memory_id in memory_ids)
    except TypeError:
        raise ValidationError("memory_ids must be a sequence of memory IDs") from None


def validated_identity_relationship(value: object) -> str:
    relationship = validated_text(value, "identity relationship")
    if len(relationship) > 255 or not relationship.isprintable():
        raise ValidationError("identity relationship must be at most 255 printable characters")
    return relationship


def validated_identity_name(value: object) -> str:
    name = validated_text(value, "identity name")
    if len(name) > 255 or not name.isprintable():
        raise ValidationError("identity name must be at most 255 printable characters")
    return name


def validate_limit(value: object, *, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValidationError(f"limit must be between 1 and {maximum}")


def validate_answer_policy(value: object) -> None:
    if value not in get_args(AnswerPolicy):
        raise ValidationError("answer_policy must be 'strict' or 'best_effort'")


def encode_cursor(memory: StoredMemory) -> str:
    payload = json.dumps(
        ["v1", datetime_text(memory.created_at), memory.memory_id],
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(cursor: object) -> tuple[datetime, str]:
    if not isinstance(cursor, str) or not cursor.strip() or cursor != cursor.strip():
        raise ValidationError("cursor is invalid")
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        decoded: object = json.loads(payload)
        if (
            not isinstance(decoded, list)
            or len(decoded) != 3
            or decoded[0] != "v1"
            or not isinstance(decoded[1], str)
            or not isinstance(decoded[2], str)
        ):
            raise ValueError
        created_at = datetime.fromisoformat(decoded[1].replace("Z", "+00:00"))
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError
        memory_id = validated_identifier(decoded[2], "cursor memory_id")
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise ValidationError("cursor is invalid") from None
    return created_at, memory_id
