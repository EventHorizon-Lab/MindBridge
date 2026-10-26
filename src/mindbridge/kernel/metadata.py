"""Pure store-marker validation and index upgrade policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from mindbridge.exceptions import StorageError, ValidationError
from mindbridge.types import IndexQuantization

_INDEX_RECIPE_PREFIX = (
    "zvec-0.8:hnsw-cosine-m50-efc500:fts-bigram-fused:grouped-range:context-keys-v12"
)

# These projections can be rebuilt from authoritative FP32 embeddings. Version 11 is excluded:
# two development branches used that number for different schemas, so neither can be trusted.
_REINDEXABLE_INDEX_RECIPES = frozenset(
    f"zvec-0.7:hnsw-cosine-m50-efc500:{full_text}:grouped-range:"
    f"context-keys-v{version}:quantization-{mode.value}"
    for full_text, version in (
        ("fts-dual-language", 9),
        ("fts-stemmed-plus-bigram", 10),
        ("fts-bigram-fused", 12),
    )
    for mode in IndexQuantization
)

# Earlier recipes also changed how content was keyed, so their vectors must be regenerated.
_LEGACY_INDEX_RECIPES = frozenset(
    {
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:single-vector-v2",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:type-time-filters:single-vector-v3",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:type-time-filters:multi-vector-v4",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:interval-filters:multi-vector-v5",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:interval-filters:context-keys-v6",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:grouped-range:context-keys-v7",
        *(
            "zvec-0.7:hnsw-cosine-m50-efc500:fts-dual-language:grouped-range:"
            f"context-keys-v8:quantization-{mode.value}"
            for mode in IndexQuantization
        ),
    }
)

STORE_METADATA_KEYS = {
    "model": "embedding.model_id",
    "space": "embedding.space_id",
    "transcription": "transcription.space_id",
    "face": "face.space_id",
    "face_analysis": "face.analysis_space_id",
    "dimension": "embedding.dimension",
    "index": "index.recipe",
}
EVIDENCE_RECIPE_KEY = "evidence.projection_recipe"
MIGRATION_TARGET_KEY = "embedding.migration_target"


def validated_index_quantization(value: object) -> IndexQuantization:
    if not isinstance(value, IndexQuantization):
        raise ValidationError("index_quantization must be an IndexQuantization value")
    return value


def index_recipe_for(quantization: IndexQuantization) -> str:
    return f"{_INDEX_RECIPE_PREFIX}:quantization-{quantization.value}"


def _known_metadata_upgrade(
    key: str, stored: str, legacy_embedding_spaces: frozenset[str]
) -> bool | None:
    if key == STORE_METADATA_KEYS["space"] and stored in legacy_embedding_spaces:
        return True
    if key == STORE_METADATA_KEYS["index"] and stored in (
        _LEGACY_INDEX_RECIPES
        | _REINDEXABLE_INDEX_RECIPES
        | {index_recipe_for(mode) for mode in IndexQuantization}
    ):
        return stored in _LEGACY_INDEX_RECIPES
    return None


@dataclass(frozen=True, slots=True)
class MetadataPlan:
    updates: Mapping[str, str]
    rebuild_index: bool
    rebuild_embeddings: bool
    migration_target: str


def plan_metadata(
    expected: Mapping[str, str],
    stored: Mapping[str, str | None],
    *,
    has_records: bool,
    index_exists: bool,
    legacy_embedding_spaces: frozenset[str],
) -> MetadataPlan:
    """Validate every marker before returning changes that recovery may commit."""
    migration_target = json.dumps(dict(expected), sort_keys=True, separators=(",", ":"))
    pending = stored.get(MIGRATION_TARGET_KEY)
    if pending is not None and pending != migration_target:
        raise StorageError(
            "an unfinished embedding migration targets different settings; "
            "reopen with the migration target to finish it"
        )
    updates: dict[str, str] = {}
    rebuild_index = pending is not None
    rebuild_embeddings = pending is not None
    for key, value in expected.items():
        previous = stored[key]
        if key == EVIDENCE_RECIPE_KEY and previous is None and has_records:
            previous = "legacy-v1"
        if previous == value:
            # Record the inferred recipe once, without rewriting an existing equal marker.
            if stored[key] is None:
                updates[key] = value
        elif previous is None:
            if key == STORE_METADATA_KEYS["index"] and index_exists:
                rebuild_index = rebuild_embeddings = True
            else:
                updates[key] = value
        elif (
            reembed := _known_metadata_upgrade(key, previous, legacy_embedding_spaces)
        ) is not None:
            rebuild_index = True
            rebuild_embeddings = rebuild_embeddings or reembed
        else:
            raise StorageError(
                f"local store metadata mismatch for {key}: expected {value!r}, found {previous!r}"
            )
    return MetadataPlan(updates, rebuild_index, rebuild_embeddings, migration_target)
