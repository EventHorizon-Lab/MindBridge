"""What the caller gave: content preparation, identity, and bounded text keys.

The memory ID is a digest of canonical ordered content, so everything that contributes to it
lives here and nowhere else. Derived text a model produced is the neighbouring `derived`
module; this one never depends on it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from mindbridge.exceptions import StorageError, ValidationError
from mindbridge.infrastructure.local.store import (
    StoredAsset,
    StoredMemory,
    StoredTextSelector,
    StoredTextSpanPiece,
    datetime_text,
    optional_datetime_text,
)
from mindbridge.kernel.temporal import validated_occurred_at, validated_occurred_end
from mindbridge.kernel.validation import validated_memory_type
from mindbridge.models.base import ModelInput
from mindbridge.types import (
    AssetRef,
    Blob,
    ContentAtom,
    ContentInput,
    ContextExcerpt,
    MemoryContext,
    MemoryKind,
    MemoryRecord,
    MemoryType,
    Modality,
    ObservationContext,
    SearchHit,
    TextSpanPiece,
    TextSpanSelector,
)

_MAX_CONTENT_PARTS = 128


_MAX_METADATA_BYTES = 262_144


_TEXT_KEY_CHARACTERS = 2_048


_TEXT_KEY_OVERLAP = 256


_TEXT_KEY_CONTEXT = 256


_MEDIA_TYPES = {
    ".aac": "audio/aac",
    ".avi": "video/x-msvideo",
    ".avif": "image/avif",
    ".bmp": "image/bmp",
    ".flac": "audio/flac",
    ".gif": "image/gif",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".m4a": "audio/mp4",
    ".m4v": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".ogv": "video/ogg",
    ".opus": "audio/ogg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".wav": "audio/wav",
    ".weba": "audio/webm",
    ".webm": "video/webm",
    ".webp": "image/webp",
}


@dataclass(frozen=True, slots=True)
class PreparedContent:
    text: str
    assets: tuple[StoredAsset, ...]
    modality: Modality
    canonical_parts: tuple[tuple[str, str], ...]
    audio_transcript: bool = False
    visual_description: bool = False


@dataclass(frozen=True, slots=True)
class PreparedMemory:
    memory_id: str
    content: PreparedContent
    metadata_json: str
    occurred_at: datetime | None
    occurred_end: datetime | None
    memory_type: MemoryType
    context: ObservationContext | MemoryContext | None = None
    # The record column, carried separately from `context` because both write paths need it and
    # only an `ObservationContext` has one: a formed record's `MemoryContext` deliberately has no
    # place, so formation sets this from the observation it was formed from.
    place_id: str | None = None
    # Releases before place-scoped identity included `place_id` in the content address. Kept only
    # long enough to reuse an existing row from such a store; newly written rows always use the
    # current ID, so two otherwise equal observations in different places remain distinct.
    legacy_memory_id: str | None = None


@dataclass(frozen=True, slots=True)
class _TextKeySpan:
    role: Literal["context", "body"]
    start_codepoint: int
    end_codepoint: int


@dataclass(frozen=True, slots=True)
class _TextKeyDescriptor:
    key: str
    spans: tuple[_TextKeySpan, ...]


def content_atoms(content: ContentInput) -> tuple[ContentAtom, ...]:
    if isinstance(content, (str, Path, Blob, AssetRef)):
        return (content,)
    if isinstance(content, bytes) or not isinstance(content, Sequence):
        raise ValidationError("content must be text, media, or an ordered sequence of them")
    atoms = tuple(content)
    if not atoms:
        raise ValidationError("content must not be empty")
    if len(atoms) > _MAX_CONTENT_PARTS:
        raise ValidationError(f"content must not exceed {_MAX_CONTENT_PARTS} parts")
    if any(not isinstance(atom, (str, Path, Blob, AssetRef)) for atom in atoms):
        raise ValidationError("content contains an unsupported input value")
    return atoms


def declared_atom_modality(atom: ContentAtom) -> Modality | None:
    if isinstance(atom, str):
        return Modality.TEXT
    if isinstance(atom, Blob):
        return media_hint(atom.name, atom.media_type)[0]
    if isinstance(atom, Path):
        return media_hint(atom.name, None)[0]
    if atom.modality is not None:
        return atom.modality
    if atom.media_type is not None:
        return media_hint(atom.name, atom.media_type)[0]
    return None


def snapshot_content(content: ContentInput) -> ContentInput:
    atoms = content_atoms(content)
    if any(isinstance(atom, Path) for atom in atoms):
        raise ValidationError("prefetch paths are mutable; use Blob or AssetRef")
    return atoms[0] if len(atoms) == 1 else atoms


def prepare_memory(
    content: PreparedContent,
    *,
    occurred_at: datetime | None,
    occurred_end: datetime | None,
    metadata: Mapping[str, object] | None,
    memory_type: MemoryType,
    context: ObservationContext | None = None,
) -> PreparedMemory:
    normalized_occurred_at = validated_occurred_at(occurred_at)
    normalized_occurred_end = validated_occurred_end(normalized_occurred_at, occurred_end)
    normalized_memory_type = validated_memory_type(memory_type)
    metadata_json = encode_metadata(metadata)
    identity: dict[str, object] = {
        "parts": content.canonical_parts,
        "metadata": json.loads(metadata_json),
        "occurred_at": optional_datetime_text(normalized_occurred_at),
    }
    if normalized_occurred_end is not None:
        identity["occurred_end"] = datetime_text(normalized_occurred_end)
    if normalized_memory_type is not MemoryType.SEMANTIC:
        identity["memory_type"] = normalized_memory_type.value
    legacy_memory_id: str | None = None
    if context is not None:
        if not isinstance(context, ObservationContext):
            raise ValidationError("context must be an ObservationContext")
        context_identity = observation_context_identity(context)
        identity["context"] = context_identity
        if context.place_id is not None:
            legacy_identity = dict(identity)
            legacy_identity["context"] = {
                key: value for key, value in context_identity.items() if key != "place_id"
            }
            legacy_memory_id = _observation_memory_id(legacy_identity)
    memory_id = _observation_memory_id(identity)
    return PreparedMemory(
        memory_id=memory_id,
        content=content,
        metadata_json=metadata_json,
        occurred_at=normalized_occurred_at,
        occurred_end=normalized_occurred_end,
        memory_type=normalized_memory_type,
        context=context,
        place_id=None if context is None else context.place_id,
        legacy_memory_id=(legacy_memory_id if legacy_memory_id != memory_id else None),
    )


def _observation_memory_id(identity: Mapping[str, object]) -> str:
    payload = json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_prepared_memory_ids(
    prepared: Sequence[PreparedMemory],
    candidates: Sequence[StoredMemory],
) -> tuple[PreparedMemory, ...]:
    """Prefer current IDs, falling back only to an old row from the same explicit place."""
    existing = {memory.memory_id: memory for memory in candidates}
    return tuple(
        replace(memory, memory_id=legacy_id, legacy_memory_id=None)
        if memory.memory_id not in existing
        and (legacy_id := memory.legacy_memory_id) is not None
        and (legacy := existing.get(legacy_id)) is not None
        and legacy.place_id == memory.place_id
        else memory
        for memory in prepared
    )


def observation_context_identity(context: ObservationContext) -> dict[str, object]:
    spatial = context.spatial
    value: dict[str, object] = {
        "basis": context.basis.value,
        "source_id": context.source_id,
        "confidence": context.confidence,
        "valid_from": optional_datetime_text(context.valid_from),
        "valid_until": optional_datetime_text(context.valid_until),
    }
    if context.place_id is not None:
        value["place_id"] = context.place_id
    if spatial is not None:
        value["spatial"] = {
            "frame_id": spatial.frame_id,
            "anchor": spatial.anchor.value,
            "position_m": (spatial.x, spatial.y, spatial.z),
            "orientation_xyzw": spatial.orientation_xyzw,
            "position_uncertainty_m": spatial.position_uncertainty_m,
        }
    return value


def stored_memory_context(
    memory: PreparedMemory,
    *,
    recorded_at: datetime,
) -> MemoryContext | None:
    context = memory.context
    if context is None or isinstance(context, MemoryContext):
        return context
    return MemoryContext(
        kind=MemoryKind.OBSERVATION,
        basis=context.basis,
        confidence=context.confidence,
        valid_from=context.valid_from,
        valid_until=context.valid_until,
        recorded_at=recorded_at,
        lineage_id=memory.memory_id,
        source_id=context.source_id,
        spatial=context.spatial,
    )


def observation_from_record(record: MemoryRecord) -> ObservationContext:
    context = record.context
    if context is None:
        return ObservationContext()
    return ObservationContext(
        basis=context.basis,
        source_id=context.source_id,
        confidence=context.confidence,
        valid_from=context.valid_from,
        valid_until=context.valid_until,
        spatial=context.spatial,
    )


def media_hint(name: str | None, media_type: str | None) -> tuple[Modality, str]:
    resolved = media_type
    if resolved is None and name:
        resolved = _MEDIA_TYPES.get(Path(name).suffix.casefold())
    if resolved is None:
        raise ValidationError(
            "media type could not be inferred; provide a known file suffix or media_type"
        )
    normalized = resolved.strip().lower()
    top_level = normalized.split("/", 1)[0]
    try:
        modality = Modality(top_level)
    except ValueError:
        raise ValidationError("media_type must be image, video, or audio") from None
    if modality not in {Modality.IMAGE, Modality.VIDEO, Modality.AUDIO}:
        raise ValidationError("media_type must be image, video, or audio")
    return modality, normalized


def memory_modality(assets: Sequence[StoredAsset]) -> Modality:
    kinds = {asset.modality for asset in assets}
    if not kinds:
        return Modality.TEXT
    if len(kinds) == 1:
        return Modality(next(iter(kinds)))
    return Modality.OMNI


def text_content(text: str) -> PreparedContent:
    return PreparedContent(
        text=text,
        assets=(),
        modality=Modality.TEXT,
        canonical_parts=(("text", text),),
    )


def contextual_text_keys(text: str) -> tuple[str, ...]:
    if len(text) <= _TEXT_KEY_CHARACTERS:
        return (text,)
    context = text.splitlines()[0][:_TEXT_KEY_CONTEXT].strip()
    step = _TEXT_KEY_CHARACTERS - _TEXT_KEY_OVERLAP
    keys = []
    for start in range(0, len(text), step):
        chunk = text[start : start + _TEXT_KEY_CHARACTERS].strip()
        if not chunk:
            continue
        if context and not chunk.startswith(context):
            chunk = f"{context}\n\n{chunk}"
        keys.append(chunk)
        if start + _TEXT_KEY_CHARACTERS >= len(text):
            break
    return tuple(keys)


_TEXT_SELECTOR_RECIPE = "mindbridge-contextual-text-key-v1"


def _stripped_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Return the exact source interval retained by `text[start:end].strip()`."""
    source = text[start:end]
    stripped_left = source.lstrip()
    if not stripped_left:
        return None
    left = len(source) - len(stripped_left)
    stripped = stripped_left.rstrip()
    return start + left, start + left + len(stripped)


def contextual_text_key_descriptors(text: str) -> tuple[_TextKeyDescriptor, ...]:
    """Mirror `contextual_text_keys` while retaining exact source ranges.

    `contextual_text_keys` remains the byte-for-byte authority for embedding inputs. Tests compare
    these descriptors with it, and callers join a descriptor to an embedding only by exact
    `ModelInput` equality.
    """
    if len(text) <= _TEXT_KEY_CHARACTERS:
        return (
            _TextKeyDescriptor(
                key=text,
                spans=(_TextKeySpan("body", 0, len(text)),),
            ),
        )
    first_line_end = len(text.splitlines()[0])
    context_interval = _stripped_span(text, 0, min(first_line_end, _TEXT_KEY_CONTEXT))
    context = "" if context_interval is None else text[context_interval[0] : context_interval[1]]
    step = _TEXT_KEY_CHARACTERS - _TEXT_KEY_OVERLAP
    descriptors = []
    for start in range(0, len(text), step):
        interval = _stripped_span(text, start, min(len(text), start + _TEXT_KEY_CHARACTERS))
        if interval is None:
            continue
        chunk = text[interval[0] : interval[1]]
        spans: tuple[_TextKeySpan, ...] = (_TextKeySpan("body", *interval),)
        key = chunk
        if context and not chunk.startswith(context):
            if context_interval is None:  # pragma: no cover - `context` proves this unreachable
                raise AssertionError("context interval is missing")
            key = f"{context}\n\n{chunk}"
            spans = (_TextKeySpan("context", *context_interval), *spans)
        descriptors.append(_TextKeyDescriptor(key=key, spans=spans))
        if start + _TEXT_KEY_CHARACTERS >= len(text):
            break
    return tuple(descriptors)


def text_selector_map(
    memory: PreparedMemory,
    *,
    raw_observation: bool,
) -> dict[ModelInput, tuple[StoredTextSelector, ...]]:
    """Return selectors only when a new raw key is an exact projection of stored text."""
    content = memory.content
    if (
        not raw_observation
        or content.modality is not Modality.TEXT
        or content.assets
        or content.audio_transcript
        or content.visual_description
        or content.canonical_parts != (("text", content.text),)
        or not content.text
    ):
        return {}
    parent_digest = hashlib.sha256(content.text.encode("utf-8")).hexdigest()
    aggregate = memory.content
    aggregate_input = ModelInput(text=aggregate.text, assets=())
    grouped: dict[ModelInput, list[StoredTextSelector]] = {}
    for descriptor in contextual_text_key_descriptors(content.text):
        model_input = ModelInput(text=descriptor.key)
        # `_embedding_inputs` removes atomic keys equal to its aggregate. Preserve that exact
        # behavior and never attach a selector to the unsliced aggregate route.
        if model_input == aggregate_input:
            continue
        selector = StoredTextSelector(
            parent_content_sha256=parent_digest,
            embedding_input_sha256=hashlib.sha256(descriptor.key.encode("utf-8")).hexdigest(),
            recipe_version=_TEXT_SELECTOR_RECIPE,
            pieces=tuple(
                StoredTextSpanPiece(
                    role=span.role,
                    start_codepoint=span.start_codepoint,
                    end_codepoint=span.end_codepoint,
                    piece_sha256=hashlib.sha256(
                        content.text[span.start_codepoint : span.end_codepoint].encode("utf-8")
                    ).hexdigest(),
                )
                for span in descriptor.spans
            ),
        )
        selectors = grouped.setdefault(model_input, [])
        if selector not in selectors:
            selectors.append(selector)
    return {model_input: tuple(selectors) for model_input, selectors in grouped.items()}


def verified_context_excerpt(
    hit: SearchHit,
    embedding_id: str,
    stored: StoredTextSelector,
) -> ContextExcerpt | None:
    """Return a public excerpt only when every durable selector claim still verifies."""
    if stored.recipe_version != _TEXT_SELECTOR_RECIPE:
        return None
    parent_digest = hashlib.sha256(hit.content.encode("utf-8")).hexdigest()
    if parent_digest != stored.parent_content_sha256:
        return None
    pieces = []
    for stored_piece in stored.pieces:
        if stored_piece.end_codepoint > len(hit.content):
            return None
        source_text = hit.content[stored_piece.start_codepoint : stored_piece.end_codepoint]
        if hashlib.sha256(source_text.encode("utf-8")).hexdigest() != stored_piece.piece_sha256:
            return None
        pieces.append(
            TextSpanPiece(
                role=stored_piece.role,
                start_codepoint=stored_piece.start_codepoint,
                end_codepoint=stored_piece.end_codepoint,
                source_text=source_text,
                sha256=stored_piece.piece_sha256,
            )
        )
    context_pieces = tuple(piece.source_text for piece in pieces if piece.role == "context")
    body_pieces = tuple(piece.source_text for piece in pieces if piece.role == "body")
    if len(body_pieces) != 1 or len(context_pieces) > 1:
        return None
    embedding_input = (
        body_pieces[0] if not context_pieces else f"{context_pieces[0]}\n\n{body_pieces[0]}"
    )
    if hashlib.sha256(embedding_input.encode("utf-8")).hexdigest() != stored.embedding_input_sha256:
        return None
    selector = TextSpanSelector(
        parent_content_sha256=stored.parent_content_sha256,
        embedding_input_sha256=stored.embedding_input_sha256,
        recipe_version=stored.recipe_version,
        pieces=tuple(pieces),
    )
    return ContextExcerpt(
        source_memory_id=hit.id,
        matched_index_id=embedding_id,
        content=_excerpt_content(hit.content, pieces),
        selector=selector,
        score=hit.score,
        created_at=hit.created_at,
        occurred_at=hit.occurred_at,
        occurred_end=hit.occurred_end,
        memory_type=hit.memory_type,
        context=hit.context,
        place_id=hit.place_id,
    )


def _excerpt_content(parent: str, pieces: Sequence[TextSpanPiece]) -> str:
    """Render ordered exact pieces with an explicit marker for every omitted range."""
    intervals = sorted((piece.start_codepoint, piece.end_codepoint) for piece in pieces)
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    omission = "[… omitted source text …]"
    rendered = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            rendered.append(omission)
        rendered.append(parent[start:end])
        cursor = end
    if cursor < len(parent):
        rendered.append(omission)
    return " ".join(rendered)


def asset_content(asset: StoredAsset) -> PreparedContent:
    return PreparedContent(
        text="",
        assets=(asset,),
        modality=Modality(asset.modality),
        canonical_parts=(("asset", asset.sha256),),
    )


def prepared_modalities(prepared: PreparedContent) -> frozenset[Modality]:
    values = {Modality(asset.modality) for asset in prepared.assets}
    if prepared.text:
        values.add(Modality.TEXT)
    return frozenset(values)


def encode_metadata(metadata: Mapping[str, object] | None) -> str:
    if metadata is None:
        return "{}"
    if not isinstance(metadata, Mapping):
        raise ValidationError("metadata must be a mapping")
    copied = dict(metadata)
    if any(not isinstance(key, str) or not key.strip() for key in copied):
        raise ValidationError("metadata keys must be non-empty strings")
    try:
        serialized = json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (RecursionError, TypeError, ValueError):
        raise ValidationError("metadata must contain JSON-compatible values") from None
    if len(serialized.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValidationError(f"metadata must not exceed {_MAX_METADATA_BYTES} UTF-8 bytes")
    return serialized


def decode_metadata(value: str) -> Mapping[str, object]:
    try:
        decoded: object = json.loads(value)
    except ValueError as error:
        raise StorageError("stored memory metadata is invalid", reason="unexpected") from error
    if not isinstance(decoded, dict) or any(not isinstance(key, str) for key in decoded):
        raise StorageError("stored memory metadata is not an object", reason="unexpected")
    return cast(dict[str, object], decoded)


def embedding_row_id(memory_id: str, object_part: int) -> str:
    if object_part == 0:
        return memory_id
    return hashlib.sha256(f"mindbridge-embedding:{memory_id}:{object_part}".encode()).hexdigest()
