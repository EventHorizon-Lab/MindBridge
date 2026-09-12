"""Validated row values the store reads and writes."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Literal, NoReturn

from mindbridge.types import MemoryContext

MEMORY_MODALITIES = frozenset({"text", "image", "video", "audio", "omni"})


MEMORY_TYPES = frozenset({"semantic", "episodic", "procedural"})


_ASSET_MODALITIES = frozenset({"image", "video", "audio"})


_SHA256_HEX_LENGTH = 64


_MEDIA_TYPE = re.compile(r"[!#$&^_.+0-9A-Za-z-]+/[!#$&^_.+0-9A-Za-z-]+\Z")


@dataclass(frozen=True, slots=True)
class RecallDigest:
    """What the active corpus looks like, for a caller that plans a read over it.

    Small enough to send to a planner in one line, and cheap enough to answer per question: a
    plan that asks for records outside the span, or for a modality nothing carries, is a plan
    that reads nothing.
    """

    records: int
    earliest: datetime | None
    latest: datetime | None
    modalities: tuple[str, ...]
    named_identities: int


class RecallRead(tuple["StoredMemory", ...]):
    """The rows one recall primitive returned, carrying how many IDs its predicate selected.

    A tuple of memories, so every caller reads it as the row sequence it always was. `selected`
    is the count `max_rows` was applied to in SQL, and it exceeds the rows themselves whenever a
    bitemporal, spatial or metric scope drops one during hydration -- a scope that has exactly
    one implementation, in `read_memories`, and so cannot be pushed into the selecting statement.

    That difference is the only thing that can say whether rows beyond the bound exist. Deciding
    it from the row count instead reports a read the bound truncated as a read that returned
    everything, which is how a caller states a total over a set it does not hold.
    """

    selected: int

    def __new__(
        cls,
        memories: Sequence[StoredMemory] = (),
        *,
        selected: int | None = None,
    ) -> RecallRead:
        read = super().__new__(cls, memories)
        read.selected = len(read) if selected is None else selected
        return read


@dataclass(frozen=True, slots=True)
class StoredAsset:
    """Immutable metadata for one content-addressed local media asset."""

    asset_id: str
    modality: str
    mime_type: str
    size_bytes: int
    sha256: str
    relative_path: str
    created_at: datetime
    name: str | None = None
    transcript: str | None = None

    def __post_init__(self) -> None:
        digest = validated_sha256(self.sha256)
        if self.asset_id != digest:
            raise ValueError("asset_id must equal the asset sha256")
        object.__setattr__(self, "sha256", digest)
        modality = validated_modality(self.modality, asset=True)
        object.__setattr__(self, "modality", modality)
        mime_type = validated_mime_type(self.mime_type, modality)
        object.__setattr__(self, "mime_type", mime_type)
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValueError("size_bytes must be a positive integer")
        expected_path = _asset_relative_path(digest)
        if self.relative_path != expected_path:
            raise ValueError(f"relative_path must be {expected_path!r}")
        if self.name is not None:
            object.__setattr__(self, "name", validate_asset_name(self.name))
        if self.transcript is not None and not isinstance(self.transcript, str):
            raise ValueError("transcript must be text or None")
        require_aware(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class StoredMemory:
    """The authoritative local representation of one memory."""

    memory_id: str
    content: str
    metadata_json: str
    created_at: datetime
    updated_at: datetime
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None
    modality: str = "text"
    memory_type: str = "semantic"
    assets: tuple[StoredAsset, ...] = ()
    last_accessed_at: datetime | None = None
    access_count: int = 0
    forgotten_at: datetime | None = None
    context: MemoryContext | None = None
    place_id: str | None = None

    def __post_init__(self) -> None:
        require_identifier(self.memory_id, "memory_id")
        require_optional_identifier(self.place_id, "place_id")
        object.__setattr__(self, "modality", validated_modality(self.modality, asset=False))
        if self.memory_type not in MEMORY_TYPES:
            raise ValueError("memory_type must be semantic, episodic, or procedural")
        if not isinstance(self.assets, tuple) or not all(
            isinstance(asset, StoredAsset) for asset in self.assets
        ):
            raise ValueError("assets must be a tuple of StoredAsset values")
        asset_modalities = {asset.modality for asset in self.assets}
        if self.modality == "text" and self.assets:
            raise ValueError("text memories cannot contain media assets")
        if self.modality in _ASSET_MODALITIES and asset_modalities != {self.modality}:
            raise ValueError(f"{self.modality} memories require only {self.modality} assets")
        if self.modality == "omni" and len(asset_modalities) < 2:
            raise ValueError("omni memories require at least two media modalities")
        if not self.content.strip() and not self.assets:
            raise ValueError("a memory must contain text or at least one asset")
        object.__setattr__(self, "metadata_json", canonical_object_json(self.metadata_json))
        require_aware(self.created_at, "created_at")
        require_aware(self.updated_at, "updated_at")
        _require_interval(self.occurred_at, self.occurred_end)
        if self.last_accessed_at is not None:
            require_aware(self.last_accessed_at, "last_accessed_at")
        _access_count(self.access_count)
        if self.context is not None and not isinstance(self.context, MemoryContext):
            raise ValueError("context must be a MemoryContext or None")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")


@dataclass(frozen=True, slots=True)
class StoredTextSpanPiece:
    """One exact, digest-bound source range used to build a text embedding key."""

    role: Literal["context", "body"]
    start_codepoint: int
    end_codepoint: int
    piece_sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"context", "body"}:
            raise ValueError("text span role must be context or body")
        if self.start_codepoint < 0 or self.end_codepoint <= self.start_codepoint:
            raise ValueError("text span offsets must be a non-empty half-open interval")
        validated_sha256(self.piece_sha256)


@dataclass(frozen=True, slots=True)
class StoredTextSelector:
    """One exact source-piece recipe for an embedding input."""

    parent_content_sha256: str
    embedding_input_sha256: str
    recipe_version: str
    pieces: tuple[StoredTextSpanPiece, ...]

    def __post_init__(self) -> None:
        validated_sha256(self.parent_content_sha256)
        validated_sha256(self.embedding_input_sha256)
        require_identifier(self.recipe_version, "text selector recipe version")
        if not self.pieces:
            raise ValueError("a text selector must contain at least one source piece")
        if any(not isinstance(piece, StoredTextSpanPiece) for piece in self.pieces):
            raise ValueError("text selector pieces must be StoredTextSpanPiece values")
        positions = tuple(
            (piece.start_codepoint, piece.end_codepoint, piece.role) for piece in self.pieces
        )
        if len(positions) != len(set(positions)):
            raise ValueError("text selector pieces must be unique")
        roles = tuple(piece.role for piece in self.pieces)
        if roles not in {("body",), ("context", "body")}:
            raise ValueError("text selector pieces must be one body or one context then one body")


@dataclass(frozen=True, slots=True)
class StoredEmbedding:
    """An FP32 vector retained in SQLite so the search index is rebuildable.

    Vector *content* is checked once, where a vector enters SQLite (`write_embedding`), and not
    here. Hydrating a search candidate rebuilds this value ~100 times per search while the search
    path reads only `embedding_id`, `memory_id` and `object_part` from it -- the vector itself is
    consumed solely by `ZvecIndex.upsert`, which validates what it consumes. Re-checking finiteness
    and unit length in `__post_init__` therefore cost two O(dimension) Python loops per candidate
    per search and bought nobody anything: 4.91 ms to hydrate 100 candidates of 1024 dimensions, of
    which 3.39 ms was those loops.
    """

    embedding_id: str
    memory_id: str
    values: tuple[float, ...]
    model_id: str
    space_id: str
    task: str
    created_at: datetime
    object_part: int = 0
    normalized: bool = False
    text_selectors: tuple[StoredTextSelector, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.embedding_id, "embedding_id")
        require_identifier(self.memory_id, "memory_id")
        require_identifier(self.model_id, "model_id")
        require_identifier(self.space_id, "space_id")
        require_identifier(self.task, "task")
        require_aware(self.created_at, "created_at")
        if self.object_part < 0:
            raise ValueError("object_part must not be negative")
        if any(not isinstance(value, StoredTextSelector) for value in self.text_selectors):
            raise ValueError("text_selectors must contain StoredTextSelector values")


@dataclass(frozen=True, slots=True, kw_only=True)
class StoredEvidenceClauseChange:
    """One reversible clause mutation captured in an operation's durable effects."""

    memory_id: str
    clause_id: str
    previous_active: bool
    previous_confidence: float | None
    applied_confidence: float
    applied_recorded_at: datetime
    applied_version: int

    def __post_init__(self) -> None:
        require_identifier(self.memory_id, "memory_id")
        validated_sha256(self.clause_id)
        for confidence in (self.previous_confidence, self.applied_confidence):
            if confidence is not None and (
                not math.isfinite(confidence) or not 0 <= confidence <= 1
            ):
                raise ValueError("clause confidence must be between zero and one")
        require_aware(self.applied_recorded_at, "applied_recorded_at")
        if self.applied_version <= 0:
            raise ValueError("applied clause version must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class StoredOperation:
    """One append-only control-plane operation-log row.

    The same value describes a pending write and a logged read: a caller supplies everything but
    `operation_id`, and the store returns a copy carrying the assigned id and the effects it
    actually applied.
    """

    operation_key: str
    intent: str
    trigger: str
    operation_json: str
    applied_at: datetime
    model_id: str | None = None
    recipe: str | None = None
    operation_id: int = 0
    created_ids: tuple[str, ...] = ()
    changed_ids: tuple[str, ...] = ()
    # Existing deterministic records whose retired version this operation restated. Rollback
    # retires this operation's new version without pretending the physical record was created.
    activated_ids: tuple[str, ...] = ()
    # Records this operation moved out of ordinary recall, whether as a FORGET intent or as the
    # consolidation forgetting a CONSOLIDATE carried. Rollback clears exactly these.
    forgotten_ids: tuple[str, ...] = ()
    # Evidence rows this operation actually inserted. Rollback retires exactly these, so a link
    # that predated the operation survives it.
    linked: tuple[tuple[str, str], ...] = ()
    # Complete support clauses this operation inserted or reactivated. Unlike `linked`, this
    # preserves a conjunction for rollback rather than treating each member as an alternative.
    # Kept for logs written by the incomplete schema-17 development build; new rows use the
    # before/after snapshots below.
    linked_clauses: tuple[tuple[str, str], ...] = ()
    clause_changes: tuple[StoredEvidenceClauseChange, ...] = ()
    # `(memory_id, version)` pairs this operation's own lineage rule superseded: the current
    # versions of the other records in the derived record's lineage whose validity it overlapped.
    # Rollback restores exactly these, so a supersession the backend never named is reversible.
    superseded: tuple[tuple[str, int], ...] = ()
    rolled_back_at: datetime | None = None
    # Post-hoc judgement of this operation, written only by `record_operation_outcome`.
    outcome: str | None = None
    outcome_note: str | None = None

    def __post_init__(self) -> None:  # noqa: C901 - validates durable operation effects
        require_identifier(self.operation_key, "operation_key")
        require_identifier(self.intent, "intent")
        require_identifier(self.trigger, "trigger")
        if not self.operation_json.strip():
            raise ValueError("operation_json must not be blank")
        require_aware(self.applied_at, "applied_at")
        if self.rolled_back_at is not None:
            require_aware(self.rolled_back_at, "rolled_back_at")
        if self.operation_id < 0:
            raise ValueError("operation_id must not be negative")
        for name in ("created_ids", "changed_ids", "activated_ids", "forgotten_ids"):
            for memory_id in getattr(self, name):
                require_identifier(memory_id, name)
        for memory_id, source_memory_id in self.linked:
            require_identifier(memory_id, "memory_id")
            require_identifier(source_memory_id, "source_memory_id")
        for memory_id, clause_id in self.linked_clauses:
            require_identifier(memory_id, "memory_id")
            validated_sha256(clause_id)
        if not all(
            isinstance(change, StoredEvidenceClauseChange) for change in self.clause_changes
        ):
            raise ValueError("clause_changes must contain StoredEvidenceClauseChange values")
        for memory_id, version in self.superseded:
            require_identifier(memory_id, "memory_id")
            if version <= 0:
                raise ValueError("superseded version must be positive")
        require_optional_identifier(self.outcome, "outcome")
        if self.outcome is None and self.outcome_note is not None:
            raise ValueError("an outcome note requires an outcome")


@dataclass(frozen=True, slots=True)
class StoredQueryFailure:
    """One group of near-equal recalls that came back empty, newest failure last."""

    query: str
    normalized: str
    failures: int
    failed_at: datetime

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.normalized:
            raise ValueError("a query failure must carry its query text")
        if self.failures <= 0:
            raise ValueError("failures must be positive")
        require_aware(self.failed_at, "failed_at")


@dataclass(frozen=True, slots=True)
class StoredCandidate:
    """One unit of deliberation work derived from state the store already keeps."""

    trigger: str
    memory_ids: tuple[str, ...]
    evidence_count: int

    def __post_init__(self) -> None:
        require_identifier(self.trigger, "trigger")
        if not self.memory_ids:
            raise ValueError("a candidate must name at least one memory")
        for memory_id in self.memory_ids:
            require_identifier(memory_id, "memory_id")
        if self.evidence_count <= 0:
            raise ValueError("evidence_count must be positive")


@dataclass(frozen=True, slots=True)
class IndexOperation:
    """One durable mutation awaiting a successful search-index flush."""

    operation_id: int
    embedding_id: str
    action: Literal["upsert", "delete"]


@dataclass(frozen=True, slots=True)
class IndexDocument:
    """Current SQLite payload for an embedding queued for indexing."""

    embedding: StoredEmbedding
    content: str
    metadata_json: str
    memory_type: str = "semantic"
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None
    place_id: str | None = None
    # Every identity this memory is about: its semantic subject, plus everyone observed in its
    # media. Projected so the search index can filter on it; SQLite stays authoritative.
    identity_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.memory_type not in MEMORY_TYPES:
            raise ValueError("memory_type must be semantic, episodic, or procedural")
        _require_interval(self.occurred_at, self.occurred_end)
        require_optional_identifier(self.place_id, "place_id")
        for identity_id in self.identity_ids:
            require_identifier(identity_id, "identity_id")


@dataclass(frozen=True, slots=True)
class IndexCandidate:
    """Ranking-path projection of an indexed embedding.

    Retrieval ranks on index scores and event times; it never reads the stored vector or the
    memory content. Hydrating a full ``IndexDocument`` for that would unpack and revalidate one
    FP32 vector per candidate, which measured at twenty-one times the cost of the query that
    produced the row. This projection reads the four columns ranking uses.
    """

    embedding_id: str
    memory_id: str
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None

    def __post_init__(self) -> None:
        require_identifier(self.embedding_id, "embedding_id")
        require_identifier(self.memory_id, "memory_id")
        _require_interval(self.occurred_at, self.occurred_end)


@dataclass(frozen=True, slots=True)
class IdentityLink:
    """A validated face-and-voice merge that can be committed atomically."""

    target_id: str
    source_id: str
    name: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class IdentityExemplarState:
    modality: Literal["face", "voice"]
    position: int
    model_id: str
    space_id: str
    dimension: int
    vector: bytes
    created_at: str


@dataclass(frozen=True, slots=True)
class IdentityState:
    identity_id: str
    name: str | None
    created_at: str
    updated_at: str
    exemplars: tuple[IdentityExemplarState, ...]


@dataclass(frozen=True, slots=True)
class IdentityChange:
    identity_id: str
    previous: IdentityState | None


@dataclass(frozen=True, slots=True)
class SpeechRollback:
    """Internal undo token for an unreferenced speech analysis."""

    asset_id: str
    identities: tuple[IdentityChange, ...]


def _access_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 20:
        raise ValueError("access_count must be between zero and twenty")
    return value


def validated_identity_modality(value: object) -> Literal["face", "voice"]:
    if value == "face" or value == "voice":
        return value
    raise RuntimeError(f"invalid stored identity modality: {value!r}")


def require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _require_interval(start: datetime | None, end: datetime | None) -> None:
    if start is not None:
        require_aware(start, "occurred_at")
    if end is not None:
        require_aware(end, "occurred_end")
        if start is None or end <= start:
            raise ValueError("occurred_end must be later than occurred_at")


def require_identifier(value: str, name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")


def require_optional_identifier(value: str | None, name: str) -> None:
    if value is not None:
        require_identifier(value, name)


def validated_identity_name(value: str, field: str = "name") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"identity {field} must be non-empty text")
    normalized = value.strip()
    if len(normalized) > 255 or not normalized.isprintable():
        raise ValueError(f"identity {field} must be at most 255 printable characters")
    return normalized


def validated_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
    return value


def validated_modality(value: str, *, asset: bool) -> str:
    choices = _ASSET_MODALITIES if asset else MEMORY_MODALITIES
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"modality must be one of: {allowed}")
    return value


def validated_mime_type(value: str, modality: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("mime_type must be non-empty and trimmed")
    canonical = value.casefold()
    if _MEDIA_TYPE.fullmatch(canonical) is None or canonical.split("/", 1)[0] != modality:
        raise ValueError(f"mime_type must be a canonical {modality} media type")
    return canonical


def _asset_relative_path(digest: str) -> str:
    return f"assets/{digest[:2]}/{digest}"


def validate_asset_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or len(value.encode("utf-8")) > 255
    ):
        raise ValueError("asset name must be a safe filename of at most 255 bytes")
    return value


# Hydration re-canonicalizes metadata the store itself wrote, so the round trip is a no-op that a
# search pays ~140 times; the result is a pure function of the text, and invalid JSON is not cached.
# Process-wide, so only small payloads are cached: 1024 entries x 4 KiB bounds it near 8 MiB.
_CANONICAL_JSON_CACHE_CHARS = 4_096


def canonical_object_json(value: str) -> str:
    if len(value) > _CANONICAL_JSON_CACHE_CHARS:
        return _canonical_object_json_uncached(value)
    return _canonical_object_json_cached(value)


@lru_cache(maxsize=1024)
def _canonical_object_json_cached(value: str) -> str:
    return _canonical_object_json_uncached(value)


def _canonical_object_json_uncached(value: str) -> str:
    try:
        decoded: object = json.loads(value, parse_constant=_reject_json_constant)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata_json must be valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("metadata_json must encode an object")
    return json.dumps(
        decoded, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is not supported: {value}")
