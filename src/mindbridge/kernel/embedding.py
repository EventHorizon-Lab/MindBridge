"""The embedding plane: routing prepared content into the embedder and keying rows."""

from __future__ import annotations

import builtins
import math
from collections.abc import (
    Sequence,
)
from dataclasses import replace
from datetime import datetime, timezone

from opentelemetry import trace
from opentelemetry.trace import Tracer

from mindbridge._telemetry import EMBEDDING_PARTS_ELIDED, EMBEDDING_TASK, mark_model_requests
from mindbridge.exceptions import MindBridgeError, ModelError
from mindbridge.infrastructure.local.store import StoredEmbedding, StoredMemory
from mindbridge.kernel.content import (
    PreparedContent,
    PreparedMemory,
    asset_content,
    contextual_text_keys,
    embedding_row_id,
    prepared_modalities,
    text_content,
)
from mindbridge.kernel.contracts import Backends, fallback_unsupported, require_audio_transcription
from mindbridge.kernel.derived import (
    derived_text,
    has_stream_description,
    has_stream_transcript,
    retrieval_content,
    speech_identity_text,
    stored_canonical_parts,
    without_speech_identities,
)
from mindbridge.kernel.hydration import Hydrator
from mindbridge.kernel.lifecycle import OperationAssets
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.settings import Settings
from mindbridge.kernel.speech import Speech
from mindbridge.kernel.tracing import Traced
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.types import (
    Modality,
    SpeakerSegment,
)

DOCUMENT_TASK = EmbedTask.DOCUMENT.value


_REEMBED_PAGE_SIZE = 32


MAX_RETRIEVAL_KEYS = 128


# The reason an embedding backend reports when media does not fit one inline request.
_PAYLOAD_TOO_LARGE = "payload_too_large"


def _record_elided_parts(count: int) -> None:
    """Publish how many retrieval keys the embedding model could not carry, so no loss is silent."""
    if count <= 0:
        return
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(EMBEDDING_PARTS_ELIDED, count)


def _normalized_vector(values: Sequence[float], dimension: int) -> tuple[float, ...]:
    if len(values) != dimension or any(
        isinstance(value, bool) or not isinstance(value, int | float) for value in values
    ):
        raise ModelError(f"embedding model must return {dimension} numeric values")
    vector = tuple(float(value) for value in values)
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0.0:
        raise ModelError(
            "embedding model returned a non-finite or zero vector", reason="response_invalid"
        )
    if math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-12):
        return vector
    return tuple(value / norm for value in vector)


class Embedding(Traced):
    """Route prepared content into the embedder and key each stored row's vectors."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        backends: Backends,
        settings: Settings,
        hydrator: Hydrator,
        speech: Speech,
    ) -> None:
        super().__init__(tracer)
        self._store = storage.store
        self._backends = backends
        self._settings = settings
        self._hydrator = hydrator
        self._speech = speech

    def embed(
        self,
        inputs: Sequence[ModelInput],
        *,
        task: EmbedTask,
    ) -> tuple[tuple[float, ...], ...]:
        with self._model_trace(
            "embedding",
            "embeddings",
            model=self._backends.embedding_model,
            batch_size=len(inputs),
            modalities=(modality for value in inputs for modality in value.modalities),
        ) as span:
            span.set_attribute(EMBEDDING_TASK, task.value)
            mark_model_requests(1)
            try:
                vectors = self._backends.embedder.embed(inputs, task=task)
            except MindBridgeError:
                raise
            except Exception as error:
                raise ModelError("failed to embed memory input", reason="model_failed") from error
            if len(vectors) != len(inputs):
                raise ModelError(
                    "embedding model returned the wrong number of vectors",
                    reason="response_invalid",
                )
            return tuple(
                _normalized_vector(vector, self._backends.embedding_dimension) for vector in vectors
            )

    def embed_document_parts(
        self,
        parts: Sequence[tuple[PreparedMemory, int, ModelInput]],
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[PreparedMemory, int, ModelInput], ...]]:
        """Embed one write's retrieval keys, degrading a key the model cannot carry.

        A memory whose media exceeds what the embedding model accepts inline used to fail the
        whole write, which discards the memory rather than the route that could not take it.
        `_embedding_inputs` already produces one key per atomic part, so the oversized key can be
        dropped while the rest of the memory is stored and stays retrievable. A memory left with
        no key at all still fails, because then nothing would find it.
        """
        parts = tuple(parts)
        try:
            return self.embed(
                tuple(model_input for _memory, _part, model_input in parts),
                task=EmbedTask.DOCUMENT,
            ), parts
        except ModelError as error:
            if error.reason != _PAYLOAD_TOO_LARGE or len(parts) < 2:
                raise
        vectors: builtins.list[tuple[float, ...]] = []
        kept: builtins.list[tuple[PreparedMemory, int, ModelInput]] = []
        for entry in parts:
            try:
                vectors.extend(self.embed((entry[2],), task=EmbedTask.DOCUMENT))
            except ModelError as error:
                if error.reason != _PAYLOAD_TOO_LARGE:
                    raise
                continue
            kept.append(entry)
        # The index carries a memory's full-text document on its `object_part == 0` row alone,
        # and part 0 is the aggregate key -- the one holding every asset, so the one an oversized
        # asset elides. Leaving the gap stores a memory with no lexical document at all, which is
        # a silent retrieval hole rather than the degradation this promises. Renumbering restores
        # the invariant; the part index orders a memory's keys and is not a handle on any input.
        renumbered: builtins.list[tuple[PreparedMemory, int, ModelInput]] = []
        next_part: dict[str, int] = {}
        for memory, _dropped_part, model_input in kept:
            position = next_part.get(memory.memory_id, 0)
            next_part[memory.memory_id] = position + 1
            renumbered.append((memory, position, model_input))
        kept = renumbered
        embedded = {entry[0].memory_id for entry in kept}
        unreachable = tuple(
            entry[0].memory_id for entry in parts if entry[0].memory_id not in embedded
        )
        if unreachable:
            raise ModelError(
                "every retrieval key for this memory exceeds what the embedding model accepts "
                "inline; supply smaller media or an embedding backend that uploads it",
                reason=_PAYLOAD_TOO_LARGE,
                subject=unreachable[0],
            )
        _record_elided_parts(len(parts) - len(kept))
        return tuple(vectors), tuple(kept)

    def route(self, prepared: PreparedContent) -> ModelInput:
        value = self._hydrator.model_input(prepared)
        missing = value.modalities - self._backends.embedding_capabilities
        if not missing:
            return value
        if (
            missing == {Modality.TEXT}
            and (prepared.audio_transcript or prepared.visual_description)
            and value.modalities - {Modality.TEXT} <= self._backends.embedding_capabilities
        ):
            return ModelInput(assets=value.assets)
        fallback = {Modality.AUDIO}
        if prepared.visual_description:
            fallback.update((Modality.IMAGE, Modality.VIDEO))
        if any(asset.transcript for asset in prepared.assets):
            # By here the transcript is attached to the asset, so the speech can stand in for a
            # video the embedder refuses. `fallback_unsupported` already agreed a route exists on
            # the strength of it being derivable; this is the same decision once the text is in
            # hand, and without it the write raises after paying for the transcription. Read from
            # the assets rather than `prepared.audio_transcript`, which the transcript-derivation
            # path does not set -- the audio route never needed it, since AUDIO is unconditionally
            # in the fallback set above, and `derived_text` reads the assets too.
            fallback.add(Modality.VIDEO)
        if missing <= fallback:
            text = derived_text(value.text, prepared.assets)
            assets = tuple(asset for asset in value.assets if asset.modality not in missing)
            if text or assets:
                routed = ModelInput(text=text, assets=assets)
                missing = routed.modalities - self._backends.embedding_capabilities
                if not missing:
                    return routed
        names = ", ".join(sorted(modality.value for modality in missing))
        raise ModelError(
            f"configured embedding model does not support: {names}",
            reason="unsupported_modality",
        )

    def content(
        self,
        prepared: PreparedContent,
        operation: OperationAssets,
    ) -> PreparedContent:
        media = prepared_modalities(prepared) - {Modality.TEXT}
        if (
            (prepared.audio_transcript or prepared.visual_description)
            and Modality.TEXT not in self._backends.embedding_capabilities
            and media
            and media <= self._backends.embedding_capabilities
        ):
            return prepared
        rescues = self._speech.transcript_fallback(prepared.assets)
        unsupported = fallback_unsupported(
            prepared,
            self._backends.embedding_capabilities,
            "embedding",
            rescuable=rescues,
        )
        if Modality.AUDIO in unsupported:
            if prepared.audio_transcript:
                return prepared
            require_audio_transcription(self._backends.transcription_capabilities)
        elif Modality.VIDEO in unsupported and Modality.VIDEO in rescues:
            # The embedder cannot take the video but the transcriber can read it, so the speech
            # becomes the embedding key. The frames are not embedded in this composition -- the
            # honest cost of the route -- while the asset stays stored and on the record.
            if prepared.audio_transcript:
                return prepared
        elif unsupported & {Modality.IMAGE, Modality.VIDEO} or not self._speech.derives_transcripts(
            prepared.assets
        ):
            return prepared
        return self._speech.with_audio_transcripts(prepared, operation)

    def inputs(
        self,
        prepared: PreparedContent,
        *,
        maximum_keys: int = MAX_RETRIEVAL_KEYS,
    ) -> tuple[ModelInput, ...]:
        prepared = retrieval_content(prepared)
        aggregate = self.route(prepared)
        text_parts = tuple(
            value
            for kind, value in prepared.canonical_parts
            if kind in {"text", "audio_transcript", "visual_description"}
        )
        if prepared.text and prepared.text != "\n\n".join(text_parts):
            text_parts = (*text_parts, prepared.text)
        if (
            prepared.audio_transcript or prepared.visual_description
        ) and Modality.TEXT not in self._backends.embedding_capabilities:
            text_parts = ()
        text_keys = tuple(key for text in text_parts for key in contextual_text_keys(text))
        atomic = [
            *(self.route(text_content(text)) for text in text_keys),
            *(
                self.route(asset_content(asset))
                for asset in prepared.assets
                if not (
                    (
                        prepared.audio_transcript
                        and asset.modality == Modality.AUDIO.value
                        and Modality.AUDIO not in self._backends.embedding_capabilities
                        and asset.transcript is None
                    )
                    or (
                        prepared.visual_description
                        and asset.modality in {Modality.IMAGE.value, Modality.VIDEO.value}
                        and Modality(asset.modality) not in self._backends.embedding_capabilities
                    )
                )
            ),
        ]
        unique = tuple(dict.fromkeys(value for value in atomic if value != aggregate))
        if len(unique) > maximum_keys:
            unique = tuple(
                unique[round(index * (len(unique) - 1) / (maximum_keys - 1))]
                for index in range(maximum_keys)
            )
        return (aggregate, *unique)

    def prepare(
        self,
        memory: PreparedMemory,
        operation: OperationAssets,
    ) -> PreparedMemory:
        content = memory.content
        if self._settings.index_speech and self._speech.answer_speech_assets(content.assets):
            content = self._speech.with_speech_identities(content, operation)
        return replace(
            memory,
            content=self.content(content, operation),
        )

    def refresh_speaker_memories(
        self,
        memories: Sequence[StoredMemory],
        *,
        speaker_id: str,
        speaker_name: str | None,
        operation: OperationAssets,
        previous_speaker_id: str | None = None,
        update_operation: bool = True,
    ) -> tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]]:
        prepared: list[tuple[StoredMemory, PreparedContent]] = []
        for memory in memories:
            segments_by_asset: dict[str, tuple[SpeakerSegment, ...]] = {}
            speech_assets = tuple(
                asset for asset in memory.assets if asset.modality in {"audio", "video"}
            )
            with translate_storage_errors("read cached speaker recognition"):
                for asset in speech_assets:
                    segments = self._store.media.read_speech(
                        asset.asset_id,
                        space_id=self._backends.transcription_space,
                    )
                    if segments is None:
                        continue
                    refreshed = tuple(
                        replace(
                            segment,
                            speaker_id=speaker_id,
                            speaker_name=speaker_name,
                        )
                        if segment.speaker_id in {speaker_id, previous_speaker_id}
                        else segment
                        for segment in segments
                    )
                    segments_by_asset[asset.asset_id] = refreshed
                    if update_operation:
                        operation.speech_segments[asset.asset_id] = refreshed
                        operation.transcripts[asset.asset_id] = "\n".join(
                            segment.text for segment in refreshed
                        )
            base = without_speech_identities(memory.content, tuple(segments_by_asset))
            content = PreparedContent(
                text=speech_identity_text(
                    base,
                    tuple(asset for asset in speech_assets if asset.asset_id in segments_by_asset),
                    segments_by_asset,
                ),
                assets=memory.assets,
                modality=Modality(memory.modality),
                # Recovered section by section, not handed over as one string: the derived
                # sections were embedded as their own atomic keys on the way in, and merging
                # them here would key 2048-character windows of the whole document instead --
                # on exactly the memories a stated name reindexes.
                canonical_parts=stored_canonical_parts(base, memory.assets),
                audio_transcript=has_stream_transcript(base, memory.assets),
                visual_description=has_stream_description(base, memory.assets),
            )
            prepared.append((memory, self.content(content, operation)))

        parts = tuple(
            (memory, content, object_part, model_input)
            for memory, content in prepared
            for object_part, model_input in enumerate(self.inputs(content))
        )
        vectors = self.embed(
            tuple(model_input for _memory, _content, _part, model_input in parts),
            task=EmbedTask.DOCUMENT,
        )
        now = datetime.now(timezone.utc)
        updated = tuple(
            replace(memory, content=content.text, updated_at=now) for memory, content in prepared
        )
        embeddings = tuple(
            StoredEmbedding(
                embedding_id=embedding_row_id(memory.memory_id, object_part),
                memory_id=memory.memory_id,
                values=vector,
                model_id=self._backends.embedding_model,
                space_id=self._backends.space_id,
                task=DOCUMENT_TASK,
                created_at=now,
                object_part=object_part,
                normalized=True,
            )
            for (memory, _content, object_part, _model_input), vector in zip(
                parts,
                vectors,
                strict=True,
            )
        )
        return updated, embeddings

    def reembed_memories(self) -> None:
        after: tuple[datetime, str] | None = None
        while True:
            with translate_storage_errors("read memories for embedding migration"):
                memories = self._store.records.list_memories(limit=_REEMBED_PAGE_SIZE, after=after)
            if not memories:
                return
            parts = tuple(
                (memory, object_part, model_input)
                for memory in memories
                for object_part, model_input in enumerate(
                    self.inputs(
                        PreparedContent(
                            text=memory.content,
                            assets=memory.assets,
                            modality=Modality(memory.modality),
                            # Same reason as `_refresh_speaker_memories`: an embedder swap has to
                            # re-key each stored row the way `add()` keyed it, section by section.
                            canonical_parts=stored_canonical_parts(
                                memory.content,
                                memory.assets,
                            ),
                            audio_transcript=has_stream_transcript(
                                memory.content,
                                memory.assets,
                            ),
                            visual_description=has_stream_description(
                                memory.content,
                                memory.assets,
                            ),
                        )
                    )
                )
            )
            vectors = self.embed(
                tuple(model_input for _memory, _part, model_input in parts),
                task=EmbedTask.DOCUMENT,
            )
            now = datetime.now(timezone.utc)
            embeddings = tuple(
                StoredEmbedding(
                    embedding_id=embedding_row_id(memory.memory_id, object_part),
                    memory_id=memory.memory_id,
                    values=vector,
                    model_id=self._backends.embedding_model,
                    space_id=self._backends.space_id,
                    task=DOCUMENT_TASK,
                    created_at=now,
                    object_part=object_part,
                    normalized=True,
                )
                for (memory, object_part, _model_input), vector in zip(
                    parts,
                    vectors,
                    strict=True,
                )
            )
            with translate_storage_errors("replace migrated embeddings"):
                self._store.records.replace_memory_embeddings(memories, embeddings)
            last = memories[-1]
            after = (last.created_at, last.memory_id)
