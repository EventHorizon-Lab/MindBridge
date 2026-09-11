"""Minimal model values shared by local and remote backends."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from mindbridge.exceptions import ValidationError
from mindbridge.types import (
    AnswerPolicy,
    AnswerResult,
    AssetRef,
    FormationProposal,
    MemoryOperation,
    MemoryRecord,
    MemoryTrigger,
    Modality,
    ObservationContext,
    SearchHit,
)


class EmbedTask(str, Enum):
    """Retrieval semantics for asymmetric embedding models."""

    QUERY = "retrieval.query"
    DOCUMENT = "retrieval.document"


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Aggregated text and ordered, resolved media assets for one model call."""

    text: str = ""
    assets: tuple[AssetRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValidationError("model input text must be text")
        assets = tuple(self.assets)
        if not self.text.strip() and not assets:
            raise ValidationError("model input must contain text or media")
        if any(not isinstance(asset, AssetRef) or not asset.is_resolved for asset in assets):
            raise ValidationError("model input assets must be resolved")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "assets", assets)

    @property
    def modality(self) -> Modality:
        """Return text, one media kind, or omni for two or more media kinds."""
        kinds = {asset.modality for asset in self.assets}
        if not kinds:
            return Modality.TEXT
        if len(kinds) == 1:
            kind = next(iter(kinds))
            if kind is not None:
                return kind
        return Modality.OMNI

    @property
    def modalities(self) -> frozenset[Modality]:
        """Return every atomic modality present in this input."""
        values = {asset.modality for asset in self.assets}
        if self.text:
            values.add(Modality.TEXT)
        return frozenset(value for value in values if value is not None)


@dataclass(frozen=True, slots=True)
class FormationInput:
    """One committed observation presented to an automatic formation backend."""

    memory_id: str
    content: ModelInput
    context: ObservationContext

    def __post_init__(self) -> None:
        if not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise ValidationError("formation input memory_id must not be blank")
        if not isinstance(self.content, ModelInput):
            raise ValidationError("formation input content must be a ModelInput")
        if not isinstance(self.context, ObservationContext):
            raise ValidationError("formation input context must be an ObservationContext")


def _modalities(
    values: Iterable[Modality],
    name: str,
) -> frozenset[Modality]:
    try:
        normalized = frozenset(
            value if isinstance(value, Modality) else Modality(value) for value in values
        )
    except (TypeError, ValueError):
        raise ValidationError(f"{name} capabilities contain an invalid modality") from None
    if Modality.OMNI in normalized:
        raise ValidationError(f"{name} capabilities must contain only atomic modalities")
    return normalized


@runtime_checkable
class EmbeddingBackend(Protocol):
    """One thread-safe embedding model with a stable vector-space recipe."""

    @property
    def embedding_capabilities(self) -> frozenset[Modality]: ...

    @property
    def embedding_model(self) -> str: ...

    @property
    def embedding_space(self) -> str: ...

    @property
    def embedding_dimension(self) -> int: ...

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class FormationBackend(Protocol):
    """One thread-safe model that proposes typed semantics without writing storage."""

    @property
    def formation_capabilities(self) -> frozenset[Modality]: ...

    @property
    def formation_model(self) -> str: ...

    @property
    def formation_space(self) -> str: ...

    def form(
        self,
        inputs: Sequence[FormationInput],
    ) -> tuple[tuple[FormationProposal, ...], ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class ConsolidationBackend(Protocol):
    """One thread-safe model that proposes memory operations over a bounded evidence set."""

    @property
    def consolidation_model(self) -> str: ...

    @property
    def consolidation_recipe(self) -> str: ...

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SpeechTurn:
    """One timed local-speaker turn returned by a speech backend."""

    start_ms: int
    end_ms: int
    text: str
    speaker_label: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_ms, bool)
            or not isinstance(self.start_ms, int)
            or isinstance(self.end_ms, bool)
            or not isinstance(self.end_ms, int)
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
        ):
            raise ValidationError("speech turn must have a positive time range")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValidationError("speech turn text must not be blank")
        object.__setattr__(self, "text", self.text.strip())
        if self.speaker_label is not None:
            if not isinstance(self.speaker_label, str) or not self.speaker_label.strip():
                raise ValidationError("speech turn speaker_label must not be blank")
            object.__setattr__(self, "speaker_label", self.speaker_label.strip())


@dataclass(frozen=True, slots=True)
class SpeakerEmbedding:
    """One backend-local speaker label and its voice identity exemplar."""

    speaker_label: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_label, str) or not self.speaker_label.strip():
            raise ValidationError("speaker embedding label must not be blank")
        values = tuple(self.values)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValidationError("speaker embedding must contain finite values")
        object.__setattr__(self, "speaker_label", self.speaker_label.strip())
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class SpeechAnalysis:
    """Timed transcript and voice identity exemplars for one media asset."""

    turns: tuple[SpeechTurn, ...]
    speakers: tuple[SpeakerEmbedding, ...]

    def __post_init__(self) -> None:
        turns, speakers = tuple(self.turns), tuple(self.speakers)
        if any(not isinstance(turn, SpeechTurn) for turn in turns) or any(
            not isinstance(speaker, SpeakerEmbedding) for speaker in speakers
        ):
            raise ValidationError("speech analysis contains invalid values")
        object.__setattr__(self, "turns", turns)
        object.__setattr__(self, "speakers", speakers)


@runtime_checkable
class SpeechBackend(Protocol):
    """One thread-safe speech analyzer with a stable recognition recipe."""

    @property
    def transcription_capabilities(self) -> frozenset[Modality]: ...

    @property
    def transcription_model(self) -> str: ...

    @property
    def transcription_space(self) -> str: ...

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    """One detected face and its backend-local identity exemplar."""

    face_label: str
    values: tuple[float, ...]
    bounding_box: tuple[float, float, float, float]
    observed_at_ms: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.face_label, str) or not self.face_label.strip():
            raise ValidationError("face embedding label must not be blank")
        values = tuple(self.values)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValidationError("face embedding must contain finite values")
        box = tuple(self.bounding_box)
        if len(box) != 4 or any(not math.isfinite(value) for value in box):
            raise ValidationError("face bounding_box must contain four finite values")
        x, y, width, height = box
        if (
            x < 0.0
            or y < 0.0
            or width <= 0.0
            or height <= 0.0
            or x + width > 1.0
            or y + height > 1.0
        ):
            raise ValidationError("face bounding_box must be normalized within the frame")
        if self.observed_at_ms is not None and (
            isinstance(self.observed_at_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or self.observed_at_ms < 0
        ):
            raise ValidationError("face observed_at_ms must be a non-negative integer")
        object.__setattr__(self, "face_label", self.face_label.strip())
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "bounding_box", box)


@dataclass(frozen=True, slots=True)
class FaceAnalysis:
    """Detected face exemplars for one image or sampled video."""

    faces: tuple[FaceEmbedding, ...]

    def __post_init__(self) -> None:
        faces = tuple(self.faces)
        if any(not isinstance(face, FaceEmbedding) for face in faces):
            raise ValidationError("face analysis contains invalid values")
        labels = {face.face_label for face in faces}
        if len(labels) != len(faces):
            raise ValidationError("face analysis labels must be unique")
        object.__setattr__(self, "faces", faces)


@runtime_checkable
class FaceBackend(Protocol):
    """One thread-safe local face analyzer with a stable embedding recipe."""

    @property
    def face_capabilities(self) -> frozenset[Modality]: ...

    @property
    def face_model(self) -> str: ...

    @property
    def face_space(self) -> str: ...

    @property
    def face_analysis_space(self) -> str: ...

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class GenerationBackend(Protocol):
    """One thread-safe grounded-answer adapter over a provider SDK.

    Both keyword arguments are defaulted and sent only when they are not their default, so a
    backend written against the two-argument signature keeps answering. `exhaustive` says the
    hits are every record a recall program's predicate matched, in time order, rather than a
    ranking -- the one thing a grounded prompt cannot infer from the hits themselves and gets
    wrong by describing their order as rank.
    """

    @property
    def generation_capabilities(self) -> frozenset[Modality]: ...

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult: ...

    def close(self) -> None: ...


@runtime_checkable
class StreamingGenerationBackend(Protocol):
    """Optional generation capability that yields answer text in provider order."""

    def stream_answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> Iterator[str]: ...


@runtime_checkable
class RecallPlanningBackend(Protocol):
    """Optional generation capability that plans how to retrieve for one question.

    The return value is the model's own JSON text, which the kernel validates: a backend that
    cannot plan, or one whose plan is unusable, costs the caller a fallback to plain similarity
    search and nothing else. `corpus_digest` is one line describing what is in the store, so a
    plan cannot ask for a time span or a modality that does not exist. `attempted` describes what
    an earlier round of the same question already read and why it was not enough; it is empty on
    the first round and sent only when it is not, so a backend written against the shorter
    signature keeps planning.
    """

    def plan_recall(
        self,
        question: str,
        *,
        reference_at: datetime,
        corpus_digest: str,
        attempted: str = "",
    ) -> str | None: ...


@runtime_checkable
class TranscriptionBackend(Protocol):
    """One thread-safe plain-transcription adapter over a provider SDK."""

    @property
    def transcription_capabilities(self) -> frozenset[Modality]: ...

    @property
    def transcription_model(self) -> str: ...

    @property
    def transcription_space(self) -> str: ...

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class VisionDescriptionBackend(Protocol):
    """One thread-safe visual-description adapter for text embedding fallback.

    ``vision_space`` identifies the whole caption recipe -- the model *and* the prompt or protocol
    version -- so a store can cache one caption per asset without a later recipe silently serving
    an earlier one. It is not the model name: an adapter that edits its prompt must return a
    different space, because the caption lands inside a searchable document where a stale one is
    invisible.
    """

    @property
    def vision_capabilities(self) -> frozenset[Modality]: ...

    @property
    def vision_model(self) -> str: ...

    @property
    def vision_space(self) -> str: ...

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        """Return one caption per input, in order -- never more, never fewer, never empty.

        `inputs[i].text`, when set, is derived transcript context this write already knows about
        the visual -- the clip's own words under the `speaker_N` labels the index prints, not
        anything the caller wrote. It is context to read, not content to echo back: repeating it
        verbatim in the caption duplicates text the document already carries under its own
        section.

        A line starting with the literal prefix ``Fact:`` is reserved syntax, not prose. The
        kernel splits it out of the visible caption into its own indexed `[facts:<asset>]`
        section (`_split_description`), so a fact must never depend on surrounding sentences to
        be read correctly. One fact shape is more than indexed text: a line reading exactly
        ``speaker_N is called <name>`` (case-insensitive, tolerant of `speaker N`/`speaker-N`)
        binds the diarised label `speaker_N` to a stated name and performs an automatic
        `IDENTIFY` write the caller can see through `operations()` and reverse through
        `rollback()` -- never a label this clip's own recognizer did not produce, and never over
        a name the identity already carries.
        """
        ...

    def close(self) -> None: ...


def _is_generation_abort_rejection(error: object) -> bool:
    """Recognize a gateway that aborted its own generation and reported it as a 400.

    Measured live on one inner-prism gateway: 'Model output became abnormal while generating a
    JSON response for response_format. The generation was aborted because the partial output may
    be incomplete or invalid JSON. Please retry the request or adjust your prompt or JSON
    schema.' An identical request seconds later routinely succeeds, so this is a fact about the
    provider at that moment, not about the request. Most 400s -- a malformed prompt, an
    unsupported image -- are permanent, hence a narrow message match rather than a status rule.
    Duck-typed on the SDK exception's own ``status_code`` and parsed ``body``, so the product
    needs no import of the optional adapter that raised it.
    """
    if getattr(error, "status_code", None) != 400:
        return False
    body = getattr(error, "body", None)
    message = body.get("message") if isinstance(body, Mapping) else getattr(body, "message", None)
    normalized = message.casefold() if isinstance(message, str) else ""
    return "model output became abnormal" in normalized and "generation was aborted" in normalized
