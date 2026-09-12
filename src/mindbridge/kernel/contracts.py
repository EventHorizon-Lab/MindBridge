"""Backend contracts and capability routing.

A backend declares atomic modalities and a space; this module validates the declaration once at
wiring time, folds it into an immutable `Backends` value, and answers the routing questions the
planes ask against declared capability rather than provider names.
"""

from __future__ import annotations

import builtins
import logging
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.infrastructure.local.zvec_index import validate_index_configuration
from mindbridge.kernel.content import PreparedContent, prepared_modalities
from mindbridge.kernel.validation import positive_int
from mindbridge.models.base import (
    ConsolidationBackend,
    EmbeddingBackend,
    FaceBackend,
    FormationBackend,
    GenerationBackend,
    SpeechBackend,
    StreamingGenerationBackend,
    TranscriptionBackend,
    VisionDescriptionBackend,
    _modalities,
)
from mindbridge.types import IndexQuantization, MemoryCapabilities, Modality

_LOGGER = logging.getLogger(__name__)


_INDEX_RECIPE_PREFIX = (
    "zvec-0.7:hnsw-cosine-m50-efc500:fts-bigram-fused:grouped-range:context-keys-v12"
)


# Recipes whose stored embeddings are still correct, so the index is rebuilt from SQLite without
# paying to embed the content again. A full-text tokenizer change belongs here and not below: it
# rewrites the derived documents and leaves every vector untouched. Version 9 named its CJK field
# `fts-dual-language` because it ran a Chinese segmenter, which returned nothing for Japanese or
# Korean; version 10 tokenized that field into script-agnostic character bigrams but admitted
# only documents containing a listed script and answered a query from one field or the other by
# detecting the query's script. Version 12 makes two changes at once, which is why it skips 11:
# it decides what enters the bigram field by what the stemmed field already answers rather than
# by script, and fuses both routes on any query with something for each (what makes a
# mixed-script query able to reach a Latin-only memory); and it adds the `place_id` and
# `identity_ids` filter fields, which are projections of rows SQLite already holds. Two unmerged
# branches each shipped a different `v11`, so that number is burned rather than reused: a store
# either of them built now fails loudly on open instead of matching a schema it does not have.
_REINDEXABLE_INDEX_RECIPES = frozenset(
    f"zvec-0.7:hnsw-cosine-m50-efc500:{full_text}:grouped-range:"
    f"context-keys-v{version}:quantization-{mode.value}"
    for full_text, version in (("fts-dual-language", 9), ("fts-stemmed-plus-bigram", 10))
    for mode in IndexQuantization
)


# Recipes that also invalidate the stored embeddings, so reopening pays for a full re-embed.
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


_NO_TRANSCRIPTION_SPACE = "none:asr-v1"


_NO_FACE_SPACE = "none:face-v1"


STORE_METADATA_KEYS = {
    "model": "embedding.model_id",
    "space": "embedding.space_id",
    "transcription": "transcription.space_id",
    "face": "face.space_id",
    "face_analysis": "face.analysis_space_id",
    "dimension": "embedding.dimension",
    "index": "index.recipe",
}


class Closable(Protocol):
    def close(self) -> None: ...


def fallback_unsupported(
    prepared: PreparedContent,
    supported: frozenset[Modality],
    operation: str,
    *,
    rescuable: frozenset[Modality] = frozenset(),
) -> frozenset[Modality]:
    unsupported = prepared_modalities(prepared) - supported
    fallback = {Modality.AUDIO}
    if prepared.visual_description:
        fallback.update((Modality.IMAGE, Modality.VIDEO))
    # This runs before any derived text exists, so it cannot be seen yet -- only its possibility.
    # `rescuable` is the modality set a transcript or a visual description could still rescue,
    # which is why a video reaching a video-less embedder is no longer fatal when the transcriber
    # can read it. Empty by default, so a caller that does not pass it keeps the strict behaviour.
    fallback.update(rescuable)
    fatal = set(unsupported - fallback)
    if Modality.TEXT not in supported:
        fatal.update(unsupported & fallback)
    if fatal:
        names = ", ".join(sorted(modality.value for modality in fatal))
        raise ModelError(
            f"configured {operation} model does not support: {names}",
            reason="unsupported_modality",
        )
    if unsupported:
        # Not fatal, but the whole capability of this modality now rides on a transcript or a
        # description that may never arrive. Silent fallback is how media encoding has died
        # unnoticed here before.
        _LOGGER.warning(
            "configured %s model does not support %s; falling back to derived text",
            operation,
            ", ".join(sorted(modality.value for modality in unsupported)),
        )
    return unsupported


def require_audio_transcription(capabilities: frozenset[Modality]) -> None:
    if Modality.AUDIO not in capabilities:
        raise ModelError(
            "audio fallback requires a transcription model with audio capability",
            reason="unsupported_modality",
        )


def _transcription_contract(
    transcriber: SpeechBackend | TranscriptionBackend | None,
) -> tuple[frozenset[Modality], str]:
    if transcriber is None:
        return frozenset(), _NO_TRANSCRIPTION_SPACE
    return _modalities(transcriber.transcription_capabilities, "transcription"), _model_text(
        transcriber.transcription_space,
        "transcription space",
    )


def _generation_contract(answerer: GenerationBackend | None) -> frozenset[Modality]:
    return (
        frozenset()
        if answerer is None
        else _modalities(answerer.generation_capabilities, "generation")
    )


def _vision_contract(
    describer: VisionDescriptionBackend | None,
) -> tuple[frozenset[Modality], str, str]:
    if describer is None:
        return frozenset(), "none", "none"
    capabilities = _modalities(describer.vision_capabilities, "vision")
    if not capabilities or capabilities - {Modality.IMAGE, Modality.VIDEO}:
        raise ValidationError("vision capabilities must contain image or video")
    return (
        capabilities,
        _model_text(describer.vision_model, "vision model"),
        _model_text(describer.vision_space, "vision space"),
    )


def _formation_contract(
    former: FormationBackend | None,
) -> tuple[frozenset[Modality], str, str]:
    if former is None:
        return frozenset(), "none", "none"
    capabilities = _modalities(former.formation_capabilities, "formation")
    if not capabilities:
        raise ValidationError("formation capabilities must not be empty")
    return (
        capabilities,
        _model_text(former.formation_model, "formation model"),
        _model_text(former.formation_space, "formation space"),
    )


def _consolidation_contract(
    consolidator: ConsolidationBackend | None,
) -> tuple[str, str]:
    if consolidator is None:
        return "none", "none"
    return (
        _model_text(consolidator.consolidation_model, "consolidation model"),
        _model_text(consolidator.consolidation_recipe, "consolidation recipe"),
    )


def _face_contract(
    analyzer: FaceBackend | None,
) -> tuple[frozenset[Modality], str, str, str]:
    if analyzer is None:
        return frozenset(), "none", _NO_FACE_SPACE, _NO_FACE_SPACE
    capabilities = _modalities(analyzer.face_capabilities, "face")
    if not capabilities or capabilities - {Modality.IMAGE, Modality.VIDEO}:
        raise ValidationError("face capabilities must contain image or video")
    return (
        capabilities,
        _model_text(analyzer.face_model, "face model"),
        _embedding_space(analyzer.face_space),
        _embedding_space(analyzer.face_analysis_space),
    )


def _embedding_contract(
    embedder: EmbeddingBackend,
) -> tuple[frozenset[Modality], str, str, int]:
    return (
        _modalities(embedder.embedding_capabilities, "embedding"),
        _model_text(embedder.embedding_model, "embedding model"),
        _embedding_space(embedder.embedding_space),
        positive_int(embedder.embedding_dimension, "embedder.embedding_dimension"),
    )


def _embedding_space(value: object) -> str:
    space = _model_text(value, "embedding space")
    if "'" in space or "\\" in space or any(ord(character) < 32 for character in space):
        raise ValidationError("embedding space contains characters unsupported by Zvec")
    return space


def _model_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be non-empty text")
    return value.strip()


def unique_resources(resources: Sequence[Closable]) -> tuple[Closable, ...]:
    seen: set[int] = set()
    unique: builtins.list[Closable] = []
    for resource in resources:
        identity = id(resource)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(resource)
    return tuple(unique)


def present_resources(*resources: Closable | None) -> tuple[Closable, ...]:
    return tuple(resource for resource in resources if resource is not None)


def close_quietly(*resources: Closable | None) -> None:
    """Close each distinct present resource once, swallowing whatever `close()` raises."""
    for resource in unique_resources(present_resources(*resources)):
        with suppress(Exception):
            resource.close()


def declared_capabilities(
    *,
    embedder: EmbeddingBackend,
    answerer: GenerationBackend | None = None,
    transcriber: SpeechBackend | TranscriptionBackend | None = None,
    vision_describer: VisionDescriptionBackend | None = None,
    face_analyzer: FaceBackend | None = None,
    former: FormationBackend | None = None,
    consolidator: ConsolidationBackend | None = None,
) -> MemoryCapabilities:
    """Declare what one set of backends can do, without opening a store.

    `Memory.capabilities` reads it, and so does `mindbridge doctor`, which probes recipes without
    owning a `data_dir`. Both therefore publish one document rather than two descriptions.
    """
    embedding, embedding_model, space_id, dimension = _embedding_contract(embedder)
    transcription, transcription_space = _transcription_contract(transcriber)
    vision, vision_model, _vision_space = _vision_contract(vision_describer)
    face, face_model, _face_space, _face_analysis_space = _face_contract(face_analyzer)
    formation, formation_model, _formation_space = _formation_contract(former)
    consolidation_model, _consolidation_recipe = _consolidation_contract(consolidator)
    return MemoryCapabilities(
        embedding=embedding,
        embedding_model=embedding_model,
        embedding_space=space_id,
        embedding_dimension=dimension,
        generation=_generation_contract(answerer),
        transcription=transcription,
        vision=vision,
        face=face,
        formation=formation,
        # The contracts substitute a `"none"` sentinel for an absent backend because the space
        # and recipe digests hash these strings. That sentinel is an implementation detail, so
        # the published value is absent when the backend is, keyed on the backend itself rather
        # than on the string -- a real model could be named "none".
        generation_model=getattr(answerer, "generation_model", None),
        transcription_space=None if transcriber is None else transcription_space,
        vision_model=None if vision_describer is None else vision_model,
        face_model=None if face_analyzer is None else face_model,
        formation_model=None if former is None else formation_model,
        consolidation_model=None if consolidator is None else consolidation_model,
        speaker_recognition=isinstance(transcriber, SpeechBackend),
        streaming_generation=isinstance(answerer, StreamingGenerationBackend),
    )


def validated_index_quantization(value: object) -> IndexQuantization:
    if not isinstance(value, IndexQuantization):
        raise ValidationError("index_quantization must be an IndexQuantization value")
    return value


def index_recipe_for(quantization: IndexQuantization) -> str:
    return f"{_INDEX_RECIPE_PREFIX}:quantization-{quantization.value}"


def known_metadata_upgrade(
    key: str,
    stored: str,
    legacy_embedding_spaces: frozenset[str],
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
class Backends:
    """The injected model backends and the contracts they declared, validated once."""

    embedder: EmbeddingBackend
    answerer: GenerationBackend | None
    transcriber: SpeechBackend | TranscriptionBackend | None
    vision_describer: VisionDescriptionBackend | None
    face_analyzer: FaceBackend | None
    former: FormationBackend | None
    consolidator: ConsolidationBackend | None
    embedding_capabilities: frozenset[Modality]
    embedding_model: str
    space_id: str
    embedding_dimension: int
    generation_capabilities: frozenset[Modality]
    transcription_capabilities: frozenset[Modality]
    transcription_space: str
    vision_capabilities: frozenset[Modality]
    vision_model: str
    vision_space: str
    face_capabilities: frozenset[Modality]
    face_model: str
    face_space: str
    face_analysis_space: str
    formation_capabilities: frozenset[Modality]
    formation_model: str
    formation_space: str
    consolidation_model: str
    consolidation_recipe: str

    def resources(self) -> tuple[Closable, ...]:
        """Every distinct backend that may hold a client, in closing order."""
        return unique_resources(
            present_resources(
                self.embedder,
                self.transcriber,
                self.vision_describer,
                self.face_analyzer,
                self.former,
                self.consolidator,
                self.answerer,
            )
        )


def resolve_backends(
    *,
    index_quantization: IndexQuantization,
    embedder: EmbeddingBackend,
    answerer: GenerationBackend | None,
    transcriber: SpeechBackend | TranscriptionBackend | None,
    vision_describer: VisionDescriptionBackend | None,
    face_analyzer: FaceBackend | None,
    former: FormationBackend | None,
    consolidator: ConsolidationBackend | None,
) -> Backends:
    """Validate each backend's declared contract and fold the results into one value."""
    embedding_capabilities, embedding_model, space_id, embedding_dimension = _embedding_contract(
        embedder
    )
    try:
        validate_index_configuration(embedding_dimension, index_quantization)
    except ValueError as error:
        raise ValidationError(str(error)) from None
    generation_capabilities = _generation_contract(answerer)
    transcription_capabilities, transcription_space = _transcription_contract(transcriber)
    vision_capabilities, vision_model, vision_space = _vision_contract(vision_describer)
    face_capabilities, face_model, face_space, face_analysis_space = _face_contract(face_analyzer)
    formation_capabilities, formation_model, formation_space = _formation_contract(former)
    consolidation_model, consolidation_recipe = _consolidation_contract(consolidator)
    return Backends(
        embedder=embedder,
        answerer=answerer,
        transcriber=transcriber,
        vision_describer=vision_describer,
        face_analyzer=face_analyzer,
        former=former,
        consolidator=consolidator,
        embedding_capabilities=embedding_capabilities,
        embedding_model=embedding_model,
        space_id=space_id,
        embedding_dimension=embedding_dimension,
        generation_capabilities=generation_capabilities,
        transcription_capabilities=transcription_capabilities,
        transcription_space=transcription_space,
        vision_capabilities=vision_capabilities,
        vision_model=vision_model,
        vision_space=vision_space,
        face_capabilities=face_capabilities,
        face_model=face_model,
        face_space=face_space,
        face_analysis_space=face_analysis_space,
        formation_capabilities=formation_capabilities,
        formation_model=formation_model,
        formation_space=formation_space,
        consolidation_model=consolidation_model,
        consolidation_recipe=consolidation_recipe,
    )
