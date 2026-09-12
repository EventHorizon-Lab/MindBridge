"""Vision perception: descriptions, faces, and the identities they link."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from time import sleep
from typing import cast

from opentelemetry.trace import Tracer

from mindbridge._telemetry import (
    IDENTITY_EVIDENCE_ASSETS,
    IDENTITY_EVIDENCE_REQUIRED,
    IDENTITY_LINKED,
    VISION_BATCHES_FAILED,
    VISION_BATCHES_RETRIED,
    mark_model_requests,
)
from mindbridge.control import dump_operation, operation_key
from mindbridge.exceptions import MindBridgeError, ModelError, ValidationError
from mindbridge.infrastructure.local.store import (
    IdentityLink,
    StoredAsset,
    StoredEmbedding,
    StoredMemory,
    StoredOperation,
)
from mindbridge.kernel.content import PreparedContent, asset_content
from mindbridge.kernel.contracts import Backends
from mindbridge.kernel.derived import (
    face_identity_text,
    has_indexed_speech,
    has_stream_description,
    speaker_prose,
    truncated_describe_context,
    validated_descriptions,
)
from mindbridge.kernel.embedding import Embedding
from mindbridge.kernel.formation import IDENTITY_LINK_RECIPE
from mindbridge.kernel.hydration import Hydrator
from mindbridge.kernel.lifecycle import Lifecycle, OperationAssets
from mindbridge.kernel.materialization import Materializer
from mindbridge.kernel.projection import Projection
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.settings import Settings
from mindbridge.kernel.speech import Speech
from mindbridge.kernel.tracing import Traced
from mindbridge.kernel.validation import MAX_TEXT_CHARACTERS
from mindbridge.models.base import (
    FaceAnalysis,
    FaceBackend,
    ModelInput,
    _is_generation_abort_rejection,
)
from mindbridge.types import (
    AssetRef,
    Blob,
    IdentityChange,
    MemoryIntent,
    MemoryOperation,
    MemoryTrigger,
    Modality,
)

_LOGGER = logging.getLogger(__name__)


# Seconds to wait before describing a throttled or overloaded batch again. Coarse and short: the
# SDK client has already spent its own `max_retries` budget on this batch by the time one of
# these failures reaches us, and what is left to wait out is a provider throttling a whole ingest
# rather than one request. Module-level so a test can shorten it.
VISION_RETRY_BACKOFF = (1.0, 4.0, 16.0)


def _transient_vision_failure(error: ModelError) -> bool:
    """Whether describing the same batch again could plausibly work.

    `retryable` is the closed public vocabulary -- a 429, a timeout, a dropped connection -- and
    REST, MCP, and the CLI all publish it, so widening it here would change what those surfaces
    say about every other operation. A 5xx is deliberately not in it and is the other answer a
    loaded endpoint gives a write burst, so it is read off the provider exception's own status
    code: duck-typed, because the SDK that raised it is an optional adapter dependency.

    One inner-prism gateway also answers a 400 -- `request_rejected`, not in `RETRYABLE_REASONS`
    -- when it aborts JSON generation mid-response rather than rejecting the request itself, and
    an identical retry 5s later succeeds; `_is_generation_abort_rejection` recognizes it by the
    provider's own words. `answer` and `formation` never call this function, so their 400s are
    unchanged here; the benchmark harness applies the same recognition to its own wait-out.
    """
    if error.retryable:
        return True
    cause = error.__cause__
    status = getattr(cause, "status_code", None)
    if isinstance(status, int) and 500 <= status < 600:
        return True
    return _is_generation_abort_rejection(cause)


class Vision(Traced):
    """Visual descriptions, face recognition, and the cross-modal identity links they justify."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        backends: Backends,
        settings: Settings,
        lifecycle: Lifecycle,
        hydrator: Hydrator,
        materializer: Materializer,
        speech: Speech,
        embedding: Embedding,
        projection: Projection,
    ) -> None:
        super().__init__(tracer)
        self._formation_lock = storage.formation_lock
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._backends = backends
        self._settings = settings
        self._lifecycle = lifecycle
        self._hydrator = hydrator
        self._materializer = materializer
        self._speech = speech
        self._embedding = embedding
        self._projection = projection

    def pending_descriptions(
        self,
        contents: Sequence[PreparedContent],
        operation: OperationAssets,
    ) -> dict[str, str]:
        """Describe every yet-undescribed visual asset in one write, in one model call.

        Deriving the text is a paid call, so it follows from `vision_describer` being configured
        and from nothing else. Whether it is *indexed* is not a capability question -- see
        `_with_visual_descriptions`. Only the write path calls this: `_embedding_content` is
        shared with the query path, where describing an image query would buy a call per search.

        An asset already described in this vision space is read back from SQLite instead of
        described again, so two ingests of one corpus build identical documents and the second
        spends nothing. The measured endpoint returns a different caption for the same image on
        every call even at temperature 0 with a fixed seed, so without this a re-ingest -- or a
        re-derive after a crash -- silently changes what a memory's full-text document says.
        Freshly derived captions are staged on the operation and persisted after the asset rows
        exist, exactly as transcripts are.
        """
        if self._backends.vision_describer is None:
            return {}
        assets = tuple(
            {
                asset.asset_id: asset
                for content in contents
                for asset in content.assets
                if Modality(asset.modality) in self._backends.vision_capabilities
                # Either derived section standing in the document means this asset is described:
                # a caption that was all facts leaves no description marker behind, and asking
                # for it again would buy the same text twice and append a duplicate section.
                and not has_stream_description(content.text, (asset,))
            }.values()
        )
        if not assets:
            return {}
        with translate_storage_errors("read cached visual descriptions"):
            cached = self._store.media.read_visual_descriptions(
                tuple(asset.asset_id for asset in assets),
                space_id=self._backends.vision_space,
            )
        pending = tuple(asset for asset in assets if asset.asset_id not in cached)
        if not pending:
            # No request is made, so the vision span and its token counters never open: a run that
            # re-ingests a described corpus reports zero vision cost because it paid none.
            return dict(cached)
        try:
            descriptions = self._described_batch(pending, operation)
        except ModelError as error:
            _LOGGER.warning(
                "vision description failed for %d asset(s); storing them without a caption: %s",
                len(pending),
                error,
            )
            # A description is derived convenience, and the caller handed us an observation to
            # store: losing the caption must never lose the memory. One malformed reply from the
            # describer used to fail the whole `add`, which an ingesting caller then reports as
            # unwritten memories -- a far larger loss than the empty document this leaves behind.
            # The batch is counted on the vision span as `mindbridge.vision.failed_batches`, so
            # how much derived text a run did not get is measurable rather than silent. A failed
            # batch is never cached, so a later ingest retries it.
            return dict(cached)
        fresh = dict(zip((asset.asset_id for asset in pending), descriptions, strict=True))
        operation.descriptions.update(fresh)
        return {**cached, **fresh}

    def _described_batch(
        self,
        pending: Sequence[StoredAsset],
        operation: OperationAssets,
    ) -> tuple[str, ...]:
        """Describe one batch of visuals, waiting out a throttled or overloaded endpoint first.

        The SDK client retries a 429 or a 5xx on its own `max_retries` budget, which is seconds.
        A provider throttling a whole ingest throttles it for minutes, and the caller's fail-open
        then stores every asset in the burst without a caption, behind one log line -- a round of
        write-path measurement lost an entire arm to exactly that. Three bounded waits, then the
        batch fails open as it did before, uncached, so a later ingest retries it.

        ponytail: the wait is taken under the write lock when speech is indexed, the same lock
        the describe call itself already holds for seconds. Move it outside `_speech_index_guard`
        if concurrent writers ever matter more than the caption does.
        """
        inputs = tuple(
            self._hydrator.model_input(
                replace(
                    asset_content(asset),
                    # Whatever this write already knows the clip says, under the same `speaker_N`
                    # labels the index projection prints. A durable fact about a person is in the
                    # words, not the pixels: four stills say two people are at a table, and only
                    # the dialogue says which one is called Lily. Empty for an image and for any
                    # composition with no speech backend, which then describes pixels as before.
                    # Capped: an unbounded transcript would let one long clip's words dominate the
                    # token budget every visual in it pays.
                    text=truncated_describe_context(
                        speaker_prose(
                            asset.asset_id,
                            operation.speech_segments.get(asset.asset_id, ()),
                        )
                        or ""
                    ),
                )
            )
            for asset in pending
        )
        for wait in VISION_RETRY_BACKOFF:
            try:
                return self._vision_descriptions(inputs, final=False)
            except ModelError as error:
                if not _transient_vision_failure(error):
                    raise
                _LOGGER.warning(
                    "describing %d visual(s) was refused as %s; retrying in %.0fs",
                    len(inputs),
                    error.reason or "a provider failure",
                    wait,
                )
                sleep(wait)
        return self._vision_descriptions(inputs)

    def persist_descriptions(self, operation: OperationAssets) -> None:
        """Cache captions derived in this operation, once their assets are stored.

        Captions are derived before the `media_assets` rows exist -- they can rescue media the
        embedder cannot take, so they have to precede the write -- which is why they are staged on
        the operation and written here instead of where they are computed. An asset that did not
        reach storage is skipped rather than failing the write: the caption is derived convenience
        and its foreign key is the asset.
        """
        if not operation.descriptions:
            return
        pending = {
            asset_id: description
            for asset_id, description in operation.descriptions.items()
            if asset_id in operation.persisted and description.strip()
        }
        operation.descriptions.clear()
        if not pending:
            return
        with (
            self._trace("mindbridge.storage.write", kind="stage"),
            self._write_lock,
            translate_storage_errors("cache visual descriptions"),
        ):
            self._store.media.write_visual_descriptions(
                pending,
                model_id=self._backends.vision_model,
                space_id=self._backends.vision_space,
            )

    def describe(self, images: Sequence[Blob]) -> tuple[str, ...]:
        """Caption live frames through the cache the write path fills, not around it.

        A scene the stream describes and a later `add()` of the same frame must not pay two
        calls: the endpoint returns a different caption for the same image on every call, so the
        second would also change what the memory's document says. The caption this derives is
        cached when the memory carrying the frame is written -- the row's foreign key is the
        asset, which does not exist until then.
        """
        if self._backends.vision_describer is None:
            raise ModelError(
                "vision description backend is not configured",
                reason="backend_not_configured",
            )
        if not images:
            raise ValidationError("vision description requires at least one image")
        with self._lifecycle.operation() as operation:
            prepared = tuple(self._materializer.prepare(image, operation) for image in images)
            # One blob in, one asset out, so the cached captions line up with the inputs.
            asset_ids = tuple(content.assets[0].asset_id for content in prepared)
            with translate_storage_errors("read cached visual descriptions"):
                described = dict(
                    self._store.media.read_visual_descriptions(
                        asset_ids, space_id=self._backends.vision_space
                    )
                )
            pending = tuple(
                {
                    asset_id: index
                    for index, asset_id in enumerate(asset_ids)
                    if asset_id not in described
                }.items()
            )
            if pending:
                fresh = self._vision_descriptions(
                    tuple(self._hydrator.model_input(prepared[index]) for _id, index in pending)
                )
                described.update(
                    zip((asset_id for asset_id, _index in pending), fresh, strict=True)
                )
            return tuple(described[asset_id] for asset_id in asset_ids)

    def _vision_descriptions(
        self, inputs: Sequence[ModelInput], *, final: bool = True
    ) -> tuple[str, ...]:
        """Describe one batch, and account a failure as lost or merely retried.

        A caller still inside its retry budget passes ``final=False``: a transient failure there
        is going to be attempted again, so it is counted on `VISION_BATCHES_RETRIED` rather than
        `VISION_BATCHES_FAILED`. Anything else -- the last attempt, or a failure no retry could
        fix -- is final regardless of what the caller passed, because no further attempt follows
        it either way.
        """
        if self._backends.vision_describer is None:
            raise ModelError(
                "vision description backend is not configured",
                reason="backend_not_configured",
            )
        inputs = tuple(inputs)
        # Media only: an input's text is the context this write derived for the visual, and a
        # describer's capability set is narrowed to image and video at construction, so counting
        # text as required would refuse every described clip that arrived with a transcript.
        unsupported = frozenset(
            asset.modality
            for value in inputs
            for asset in value.assets
            if asset.modality is not None
            and asset.modality not in self._backends.vision_capabilities
        )
        if unsupported:
            names = ", ".join(sorted(modality.value for modality in unsupported))
            raise ModelError(
                f"configured vision model does not support: {names}",
                reason="unsupported_modality",
            )
        with self._model_trace(
            "vision",
            "vision.description",
            model=self._backends.vision_model,
            batch_size=len(inputs),
            modalities=(modality for value in inputs for modality in value.modalities),
        ) as span:
            mark_model_requests(1)
            # Validated inside the span so that a reply the model did return but that cannot be
            # used is counted as a failed batch too, and so that the count lands on a span the
            # evaluation telemetry aggregates -- it reads counters only from model spans.
            try:
                return validated_descriptions(
                    self._backends.vision_describer.describe(inputs), inputs
                )
            except Exception as error:
                if not final and isinstance(error, ModelError) and _transient_vision_failure(error):
                    span.set_attribute(VISION_BATCHES_RETRIED, 1)
                else:
                    span.set_attribute(VISION_BATCHES_FAILED, 1)
                if isinstance(error, MindBridgeError):
                    raise
                raise ModelError(
                    "failed to describe vision input", reason="model_failed"
                ) from error

    def recognize(
        self,
        assets: Sequence[StoredAsset],
        operation: OperationAssets,
        *,
        link_identities: bool = True,
    ) -> None:
        if not isinstance(self._backends.face_analyzer, FaceBackend):
            raise ModelError("no face backend is configured", reason="backend_not_configured")
        face_assets = tuple(
            {
                asset.asset_id: asset
                for asset in assets
                if asset.modality in {"image", "video"}
                and asset.asset_id not in operation.face_observations
            }.values()
        )
        speech_assets = self._speech.answer_speech_assets(
            tuple(asset for asset in face_assets if asset.modality == "video")
        )
        if speech_assets:
            self._speech.recognize(speech_assets, operation)
        missing = []
        with (
            self._trace("mindbridge.storage.lookup", kind="stage"),
            translate_storage_errors("read cached face recognition"),
        ):
            for asset in face_assets:
                observations = self._store.media.read_faces(
                    asset.asset_id,
                    space_id=self._backends.face_analysis_space,
                )
                if observations is None:
                    missing.append(asset)
                else:
                    operation.face_observations[asset.asset_id] = observations
        if missing:
            analyses = self._analyze_faces(
                tuple(self._hydrator.asset_ref(asset) for asset in missing)
            )
            with (
                self._trace("mindbridge.storage.write", kind="stage"),
                self._write_lock,
                translate_storage_errors("persist face recognition"),
            ):
                for asset, analysis in zip(missing, analyses, strict=True):
                    self._store.media.write_asset(asset)
                    operation.persisted.add(asset.asset_id)
                    # Deliberately no preferred_identity. That shortcut adopted the asset's
                    # lone voice identity for its lone face with no corroboration at all, a
                    # second cross-modal door that bypassed the evidence gate in
                    # _link_asset_identity. Cross-modal binding now has exactly one entrance.
                    operation.face_observations[asset.asset_id] = self._store.media.write_faces(
                        asset.asset_id,
                        analysis,
                        model_id=self._backends.face_model,
                        space_id=self._backends.face_space,
                        analysis_space_id=self._backends.face_analysis_space,
                        minimum_similarity=self._settings.face_similarity,
                        minimum_margin=self._settings.face_margin,
                    )
        analyzed = {asset.asset_id for asset in missing}
        if link_identities:
            # `_formation_lock` before `_write_lock`, the same order `consolidate()`'s apply
            # phase and `add()`'s formation use: a corroborated MERGE must not land while a
            # consolidate apply pass is in progress, and taking the two locks in a different
            # order here would deadlock against that path.
            with (
                self._formation_lock,
                self._write_lock,
                translate_storage_errors("link face and voice identities"),
            ):
                for asset in face_assets:
                    self._link_asset_identity(asset.asset_id, operation)
        # Linking re-points these observations to the surviving identity, but every counted
        # value here is merge-invariant: one face identity maps to one surviving identity, and
        # `identity_score` is carried through unchanged, so the counts do not depend on whether
        # this runs before or after the link. The link decision has its own span.
        for asset in face_assets:
            self._trace_identity_yield(
                "mindbridge.identity.faces",
                tuple(
                    (observation.identity_id, observation.identity_score)
                    for observation in operation.face_observations[asset.asset_id]
                ),
                cached=asset.asset_id not in analyzed,
            )

    def _identity_link_is_corroborated(
        self,
        speaker_id: str,
        face_id: str,
        asset_id: str,
    ) -> bool:
        """Record this asset's voice-and-face co-occurrence and report whether it is enough.

        One asset's co-occurrence is not evidence that one person produced both. Egocentric
        capture is the adversarial case: the wearer speaks while a different person's face
        fills the frame, so a single clip would bind the listener's face to the wearer's voice
        and nothing downstream could tell.

        Counting assets raises the price of that mistake but does not prevent it. A wearer
        talks to the same person across many clips, so the wrong pair accumulates as fast as a
        genuine speaker's, and measurement on synthetic egocentric traffic confirms it: at the
        default of two assets the wearer still binds to an interlocutor's face under every
        ingestion order tried. What the count does buy is that a face seen once is never
        bound, and `_link_asset_identity` keeps the damage to that one bind by refusing to let
        an identity holding both modalities absorb anything further.
        """
        observed = self._store.identities.record_identity_link_evidence(
            speaker_id, face_id, asset_id
        )
        corroborated = observed >= self._settings.identity_link_min_assets
        with self._trace("mindbridge.identity.link", kind="stage") as span:
            span.set_attribute(IDENTITY_EVIDENCE_ASSETS, observed)
            span.set_attribute(IDENTITY_EVIDENCE_REQUIRED, self._settings.identity_link_min_assets)
            span.set_attribute(IDENTITY_LINKED, corroborated)
        return corroborated

    def _relabelled_speaker_index(
        self,
        plan: IdentityLink,
        speaker_id: str,
        operation: OperationAssets,
    ) -> tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]]:
        """Re-embed the merged speaker's indexed memories so the merge stays atomic."""
        if not (self._settings.index_speech and plan.source_id == speaker_id):
            return (), ()
        memory_ids = self._store.identities.speaker_memory_ids(plan.source_id)
        if not memory_ids:
            return (), ()
        indexed = tuple(
            memory
            for memory in self._store.records.read_memories(memory_ids)
            if has_indexed_speech(memory)
        )
        if not indexed:
            return (), ()
        return self._embedding.refresh_speaker_memories(
            indexed,
            speaker_id=plan.target_id,
            speaker_name=plan.name,
            previous_speaker_id=plan.source_id,
            update_operation=False,
            operation=operation,
        )

    def _identity_link_operation(self, plan: IdentityLink) -> StoredOperation:
        """Build the MERGE log row one corroborated cross-modal bind commits with."""
        proposed = MemoryOperation(
            intent=MemoryIntent.MERGE,
            identity=IdentityChange(
                identity_id=plan.target_id,
                moved_ids=(plan.source_id,),
            ),
            rationale=(
                f"one voice and one face co-occurred in at least "
                f"{self._settings.identity_link_min_assets} assets"
            ),
        )
        return StoredOperation(
            operation_key=operation_key(proposed, recipe=IDENTITY_LINK_RECIPE),
            intent=proposed.intent.value,
            # Independent evidence accumulated until it corroborated the pair, which is exactly
            # what this trigger names -- no clock and no host request is involved.
            trigger=MemoryTrigger.EVIDENCE.value,
            model_id=self._backends.face_model,
            recipe=IDENTITY_LINK_RECIPE,
            operation_json=dump_operation(proposed),
            applied_at=datetime.now(timezone.utc),
        )

    def _link_asset_identity(self, asset_id: str, operation: OperationAssets) -> None:
        speaker_ids = {
            segment.speaker_id
            for segment in operation.speech_segments.get(asset_id, ())
            if segment.speaker_id is not None
        }
        face_ids = {
            observation.identity_id for observation in operation.face_observations.get(asset_id, ())
        }
        if len(speaker_ids) != 1 or len(face_ids) != 1:
            return
        speaker_id, face_id = next(iter(speaker_ids)), next(iter(face_ids))
        # Withheld or withdrawn consent stops the merge on either side. Fusing a voice and a
        # face is a new claim about a person -- that these two templates are one human -- so it
        # is exactly the processing a refusal refuses, and unlike enrolment it is not something
        # answering a question needs.
        with translate_storage_errors("read identity consent"):
            restrained = self._store.identities.restrained_identities()
        if {speaker_id, face_id} & restrained:
            return
        if speaker_id == face_id:
            # Already one identity. Recording this would store an identity co-occurring with
            # itself, once per asset forever, and no such pair can ever yield a plan.
            return
        if not self._identity_link_is_corroborated(speaker_id, face_id, asset_id):
            return
        # Only a voice-only and a face-only identity may fuse here. Letting a fragment rejoin
        # an identity that already holds its modality is what turns one wrong cross-modal bind
        # into a cascade: once a wearer's voice owns one interlocutor's face, that identity
        # holds both modalities, and every later fragment (the interlocutor's own voice, then
        # the next interlocutor's face) is a fragment rejoining it. Measured on synthetic
        # egocentric traffic (one off-camera wearer, three interlocutors, random ingestion
        # order), permitting it collapsed all four people into one identity every time;
        # refusing it capped the damage at the single unavoidable first bind and raised
        # correct merges from 0/3 to 2/3. `LocalStore` still offers the wider merge to a
        # caller that has established the claim some other way.
        plan = self._store.identities.identity_link_plan(speaker_id, face_id)
        if plan is None:
            return
        memories, embeddings = self._relabelled_speaker_index(plan, speaker_id, operation)
        if (
            self._store.identities.link_identities(
                speaker_id,
                face_id,
                expected=plan,
                memories=memories,
                embeddings=embeddings,
                # Model reasoning proposes; the kernel authorizes and commits. The recognizers
                # produced the templates and the corroboration rule above authorized the bind,
                # so the bind commits with its own log row in the same transaction: it is
                # visible through `operations()` and reversible through `rollback()`, which
                # splits the two identities apart again.
                operation=self._identity_link_operation(plan),
            )
            is None
        ):
            return
        if embeddings:
            self._projection.drain()
        linked_ids = {plan.target_id, plan.source_id}
        for cached_asset, segments in tuple(operation.speech_segments.items()):
            operation.speech_segments[cached_asset] = tuple(
                replace(
                    segment,
                    speaker_id=plan.target_id,
                    speaker_name=plan.name,
                )
                if segment.speaker_id in linked_ids
                else segment
                for segment in segments
            )
        for cached_asset, observations in tuple(operation.face_observations.items()):
            operation.face_observations[cached_asset] = tuple(
                replace(
                    observation,
                    identity_id=plan.target_id,
                    identity_name=plan.name,
                )
                if observation.identity_id in linked_ids
                else observation
                for observation in observations
            )

    def _analyze_faces(
        self,
        assets: Sequence[AssetRef],
    ) -> tuple[FaceAnalysis, ...]:
        if not isinstance(self._backends.face_analyzer, FaceBackend):
            raise ModelError("no face backend is configured", reason="backend_not_configured")
        with self._model_trace(
            "face",
            "face_recognition",
            model=self._backends.face_model,
            batch_size=len(assets),
            modalities=(cast(Modality, asset.modality) for asset in assets),
        ):
            try:
                analyses = self._backends.face_analyzer.analyze(assets)
            except MindBridgeError:
                raise
            except Exception as error:
                raise ModelError("failed to analyze face input", reason="model_failed") from error
            if len(analyses) != len(assets) or any(
                not isinstance(analysis, FaceAnalysis) for analysis in analyses
            ):
                raise ModelError("face model returned invalid output", reason="response_invalid")
            return tuple(analyses)

    def deferred_rescue(self, assets: Sequence[StoredAsset]) -> frozenset[Modality]:
        """Modalities `settle()` can still rescue for an embedder that cannot take them.

        `capture()` commits before any model runs, so it cannot see the transcript or the visual
        description that will exist by the time the record is embedded -- only which of them the
        configured composition will derive.
        """
        rescued = self._speech.transcript_fallback(assets)
        if self._backends.vision_describer is not None:
            rescued |= self._backends.vision_capabilities
        return rescued

    def answer_face_assets(
        self,
        assets: Sequence[StoredAsset],
    ) -> tuple[StoredAsset, ...]:
        if not isinstance(self._backends.face_analyzer, FaceBackend):
            return ()
        supported = {modality.value for modality in self._backends.face_capabilities}
        return tuple(
            {
                asset.asset_id: asset
                for asset in assets
                if asset.modality in {"image", "video"} and asset.modality in supported
            }.values()
        )

    def with_face_identities(
        self,
        prepared: PreparedContent,
        operation: OperationAssets,
        *,
        link_identities: bool = True,
    ) -> PreparedContent:
        face_assets = self.answer_face_assets(prepared.assets)
        self.recognize(face_assets, operation, link_identities=link_identities)
        text = face_identity_text(prepared.text, face_assets, operation.face_observations)
        if len(text) > MAX_TEXT_CHARACTERS:
            raise ModelError(
                "face identity evidence exceeded the supported text length",
                reason="payload_too_large",
            )
        return replace(prepared, text=text)
