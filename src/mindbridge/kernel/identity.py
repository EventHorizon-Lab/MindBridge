"""Identity governance: naming, consent, linking, unlinking, and erasure."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone

from opentelemetry.trace import Tracer

from mindbridge._telemetry import IDENTITY_NAMES_BOUND, IDENTITY_NAMES_REFUSED
from mindbridge.control import dump_operation, operation_key
from mindbridge.exceptions import (
    IdentityNotFoundError,
    MemoryNotFoundError,
    ModelError,
    SpeakerNotFoundError,
    ValidationError,
)
from mindbridge.infrastructure.local.store import StoredEmbedding, StoredMemory, StoredOperation
from mindbridge.kernel.content import prepare_memory
from mindbridge.kernel.contracts import Backends
from mindbridge.kernel.derived import has_indexed_speech
from mindbridge.kernel.embedding import Embedding
from mindbridge.kernel.formation import (
    CONSENT_RECIPE,
    IDENTITY_LINK_RECIPE,
    NAMING_RECIPE,
    Formation,
    consent_proposal,
    formation_context,
    formation_memory_id,
    formation_memory_type,
    naming_proposal,
)
from mindbridge.kernel.hydration import operation_record
from mindbridge.kernel.lifecycle import Lifecycle, OperationAssets
from mindbridge.kernel.materialization import Materializer
from mindbridge.kernel.projection import Projection
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.speech import Speech
from mindbridge.kernel.tracing import Traced
from mindbridge.kernel.validation import (
    validated_identifier,
    validated_identity_name,
    validated_identity_relationship,
)
from mindbridge.kernel.vision import Vision
from mindbridge.models.base import FaceBackend, SpeechBackend
from mindbridge.types import (
    ConsentClaim,
    ConsentState,
    EvidenceBasis,
    FaceObservation,
    IdentityChange,
    IdentityClaim,
    IdentityErasure,
    IdentityProfile,
    MemoryIntent,
    MemoryOperation,
    MemoryOperationRecord,
    MemoryTrigger,
    Modality,
    SpeakerSegment,
)

_LOGGER = logging.getLogger(__name__)


class Identities(Traced):
    """Naming, consent, linking, unlinking, and erasing the people memories are about."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        backends: Backends,
        lifecycle: Lifecycle,
        materializer: Materializer,
        speech: Speech,
        embedding: Embedding,
        projection: Projection,
        vision: Vision,
        formation: Formation,
    ) -> None:
        super().__init__(tracer)
        self._formation_lock = storage.formation_lock
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._backends = backends
        self._lifecycle = lifecycle
        self._materializer = materializer
        self._speech = speech
        self._embedding = embedding
        self._projection = projection
        self._vision = vision
        self._formation = formation

    def speech(self, memory_id: str) -> tuple[SpeakerSegment, ...]:
        with (
            self._trace("mindbridge.speech", kind="operation"),
            self._lifecycle.operation() as operation,
        ):
            normalized_id = validated_identifier(memory_id, "memory_id")
            with self._write_lock, translate_storage_errors("read speech memory"):
                memory = self._store.records.read_memory(normalized_id)
            if memory is None:
                raise MemoryNotFoundError(f"memory does not exist: {normalized_id}")
            speech_assets = tuple(
                {
                    asset.asset_id: asset
                    for asset in memory.assets
                    if asset.modality in {"audio", "video"}
                }.values()
            )
            if not speech_assets:
                return ()
            if not isinstance(self._backends.transcriber, SpeechBackend):
                raise ModelError(
                    "configured transcription backend does not provide speaker recognition",
                    reason="backend_not_configured",
                )
            self._lifecycle.lease_assets(speech_assets, operation.leased)
            operation.persisted.update(asset.asset_id for asset in speech_assets)
            self._speech.recognize(speech_assets, operation)
            return tuple(
                segment
                for asset in speech_assets
                for segment in operation.speech_segments[asset.asset_id]
            )

    def faces(self, memory_id: str) -> tuple[FaceObservation, ...]:
        with (
            self._trace("mindbridge.faces", kind="operation"),
            self._lifecycle.operation() as operation,
        ):
            normalized_id = validated_identifier(memory_id, "memory_id")
            with self._write_lock, translate_storage_errors("read face memory"):
                memory = self._store.records.read_memory(normalized_id)
            if memory is None:
                raise MemoryNotFoundError(f"memory does not exist: {normalized_id}")
            visual_assets = tuple(
                {
                    asset.asset_id: asset
                    for asset in memory.assets
                    if asset.modality in {"image", "video"}
                }.values()
            )
            if not visual_assets:
                return ()
            if not isinstance(self._backends.face_analyzer, FaceBackend):
                raise ModelError("no face backend is configured", reason="backend_not_configured")
            unsupported = {
                Modality(asset.modality)
                for asset in visual_assets
                if Modality(asset.modality) not in self._backends.face_capabilities
            }
            if unsupported:
                names = ", ".join(sorted(modality.value for modality in unsupported))
                raise ModelError(
                    f"configured face backend does not support: {names}",
                    reason="unsupported_modality",
                )
            self._lifecycle.lease_assets(visual_assets, operation.leased)
            operation.persisted.update(asset.asset_id for asset in visual_assets)
            self._vision.recognize(visual_assets, operation)
            return tuple(
                observation
                for asset in visual_assets
                for observation in operation.face_observations[asset.asset_id]
            )

    def register_speaker(
        self,
        speaker_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        self._register_identity(speaker_id, name, relationship=relationship, speaker=True)

    def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        self._register_identity(identity_id, name, relationship=relationship, speaker=False)

    def identity(self, identity_id: str) -> IdentityProfile | None:
        with (
            self._trace("mindbridge.identity", kind="operation"),
            self._lifecycle.operation(),
            self._write_lock,
        ):
            requested_id = validated_identifier(identity_id, "identity_id")
            with translate_storage_errors("read identity profile"):
                return self._store.identities.identity_profile(requested_id)

    def record_consent(
        self,
        identity_id: str,
        state: ConsentState,
        *,
        note: str | None = None,
    ) -> MemoryOperationRecord | None:
        requested_id = validated_identifier(identity_id, "identity_id")
        if not isinstance(state, ConsentState):
            raise ValidationError("state must be a ConsentState")
        claim = ConsentClaim(identity_id=requested_id, state=state, note=note)
        with (
            self._trace("mindbridge.record_consent", kind="operation"),
            self._lifecycle.operation() as operation,
            # `_formation_lock` before `_write_lock`, the same order `consolidate()`'s apply
            # phase and `add()`'s formation use: `_assert_consent` commits through the same
            # `_commit_formation` path as naming, so a consent statement must not land while a
            # consolidate apply pass is in progress either.
            self._formation_lock,
            self._write_lock,
        ):
            with translate_storage_errors("read identity memories"):
                normalized_id = self._store.identities.resolve_identity_id(requested_id)
                known = normalized_id is not None and (
                    self._store.identities.identity_memory_ids(normalized_id) is not None
                )
            if normalized_id is None or not known:
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            return self._assert_consent(
                replace(claim, identity_id=normalized_id),
                operation=operation,
            )

    def consent(self, identity_id: str) -> ConsentState | None:
        with (
            self._trace("mindbridge.consent", kind="operation"),
            self._lifecycle.operation(),
            self._write_lock,
        ):
            requested_id = validated_identifier(identity_id, "identity_id")
            with translate_storage_errors("read identity consent"):
                return self._store.identities.identity_consent(requested_id)

    def _assert_consent(
        self,
        claim: ConsentClaim,
        *,
        operation: OperationAssets,
    ) -> MemoryOperationRecord | None:
        """Commit one consent statement as a bound STATE assertion with its own log row.

        Structurally the twin of `_assert_identity_name`: the assertion is the record, the
        projection reads it, and the log row is what makes it auditable and reversible. A STATE
        rather than an ENTITY kind because that is what it is -- how this person may be
        processed, right now -- and because it keeps consent out of the naming lineage the
        registry projects.
        """
        proposal = consent_proposal(claim)
        now = datetime.now(timezone.utc)
        context = replace(
            formation_context(
                None,
                proposal,
                model_id=None,
                recipe=CONSENT_RECIPE,
                recorded_at=now,
                identity_id=claim.identity_id,
            ),
            # No validity interval, unlike an ordinary STATE. A state of the world holds over a
            # period and a later one splits it; a consent statement simply stands until the
            # subject makes another, so the lineage rule retires the old one outright instead of
            # carrying a past-covering version that would go on projecting beside the new one.
            valid_from=None,
        )
        prepared = replace(
            prepare_memory(
                self._materializer.prepare(proposal.content, operation),
                occurred_at=None,
                occurred_end=None,
                metadata=None,
                memory_type=formation_memory_type(proposal.kind),
            ),
            memory_id=formation_memory_id(
                claim.identity_id,
                proposal,
                recipe=CONSENT_RECIPE,
                context=context,
            ),
            context=context,
        )
        proposed = MemoryOperation(
            intent=MemoryIntent.CONSENT,
            consent=claim,
            rationale="the data subject stated how they may be processed",
        )
        key = operation_key(proposed, recipe=CONSENT_RECIPE)
        with translate_storage_errors("check a consent operation"):
            if self._store.control.read_operations(operation_key=key):
                return None
        logged = self._formation.commit(
            ((prepared, None, proposal.confidence),),
            (),
            completed_at=now,
            recipe=CONSENT_RECIPE,
            operation=StoredOperation(
                operation_key=key,
                intent=proposed.intent.value,
                trigger=MemoryTrigger.MANUAL.value,
                recipe=CONSENT_RECIPE,
                operation_json=dump_operation(proposed),
                applied_at=now,
            ),
        )
        return None if logged is None else operation_record(logged)

    def unlink_identity(self, alias_id: str) -> str | None:
        with (
            self._trace("mindbridge.unlink_identity", kind="operation"),
            self._lifecycle.operation() as operation,
            # `_formation_lock` before `_write_lock`, the same order `consolidate()`'s apply
            # phase and `add()`'s formation use: a split must not land while a consolidate apply
            # pass is in progress, and taking the two locks in a different order here would
            # deadlock against that path.
            self._formation_lock,
            self._write_lock,
        ):
            requested_id = validated_identifier(alias_id, "alias_id")
            memories, embeddings = self._unlinked_speaker_index(requested_id, operation)
            with translate_storage_errors("unlink identity"):
                restored = self._store.identities.unlink_identity(
                    requested_id,
                    memories=memories,
                    embeddings=embeddings,
                    operation=self._identity_split_operation(requested_id),
                )
            self._projection.drain()
            return restored

    def _identity_split_operation(self, alias_id: str) -> StoredOperation | None:
        """Build the CORRECT log row one split commits with, or None when there is no merge.

        An ID that is not a merge alias resolves to itself, and the store refuses the split
        anyway, so there is nothing to log and no row is built.
        """
        with translate_storage_errors("read identity alias"):
            survivor = self._store.identities.resolve_identity_id(alias_id)
        if survivor is None or survivor == alias_id:
            return None
        proposed = MemoryOperation(
            intent=MemoryIntent.CORRECT,
            identity=IdentityChange(identity_id=survivor, moved_ids=(alias_id,)),
            rationale="the host split this face-and-voice merge",
        )
        return StoredOperation(
            operation_key=operation_key(proposed, recipe=IDENTITY_LINK_RECIPE),
            intent=proposed.intent.value,
            trigger=MemoryTrigger.MANUAL.value,
            recipe=IDENTITY_LINK_RECIPE,
            operation_json=dump_operation(proposed),
            applied_at=datetime.now(timezone.utc),
        )

    def _unlinked_speaker_index(
        self,
        alias_id: str,
        operation: OperationAssets,
    ) -> tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]]:
        """Rebuild the indexed speech text that a pending unlink is about to invalidate.

        Reversing a merge moves every segment of the contributed modality back, so text
        indexed under the surviving identity's name would go on quoting a name that is no
        longer the speaker's, and answer to it in search. Only a voice contribution matters
        here: a face name is projected at answer time and never written into stored content.
        """
        with translate_storage_errors("read identity memories"):
            if self._store.identities.identity_alias_modality(alias_id) != "voice":
                return (), ()
            target_id = self._store.identities.resolve_identity_id(alias_id)
        if target_id is None:
            return (), ()
        # Every one of the target's segments moves to the restored identity, and a merge keeps
        # one profile which the target keeps, so no naming assertion is bound to the identity
        # coming back: its projection is a nameless speaker. Asking the store would resolve the
        # alias straight back to the target and answer with the wrong person's name.
        return self.reindex_speech(
            alias_id,
            speaker_name=None,
            operation=operation,
            previous_speaker_id=target_id,
        )

    def deleted_naming_index(
        self,
        memory_id: str,
        operation: OperationAssets,
    ) -> tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]]:
        """Rebuild indexed speech text for the projection a pending delete will leave behind.

        Deleting the record that names somebody is the ordinary delete path applied to an
        extraordinary record. The name it projected has to stop answering to search in the same
        commit that removes the assertion, so the text is rebuilt from the assertion that will
        still be visible afterwards -- usually none, which reindexes the speaker as nameless.
        Deleting a memory an assertion cites as its only evidence takes the assertion with it,
        so the store answers for those too rather than only for the assertion itself.
        """
        with translate_storage_errors("read identity naming assertion"):
            removed, projection = self._store.identities.naming_projection_after_delete(memory_id)
        memories: tuple[StoredMemory, ...] = ()
        embeddings: tuple[StoredEmbedding, ...] = ()
        for identity_id, projected in projection:
            reindexed, revectored = self.reindex_speech(
                identity_id,
                speaker_name=projected,
                operation=operation,
            )
            # The deleted record can be one of the speaker's own memories. Handing it back as a
            # replacement would ask the same transaction to reindex a row it is removing.
            memories += tuple(memory for memory in reindexed if memory.memory_id not in removed)
            embeddings += tuple(
                embedding for embedding in revectored if embedding.memory_id not in removed
            )
        return memories, embeddings

    def projected_identity(self, identity_id: str) -> tuple[str | None, str | None]:
        with translate_storage_errors("read identity naming assertion"):
            return self._store.identities.projected_identity_name(identity_id)

    def reindex_speech(
        self,
        identity_id: str,
        *,
        speaker_name: str | None,
        operation: OperationAssets,
        previous_speaker_id: str | None = None,
    ) -> tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]]:
        """Rebuild every indexed memory whose speech projection names one identity.

        This always runs beside a projection recompute, so the text a search matches and the
        name `identities` reports are written from the same assertion in one commit.
        """
        with translate_storage_errors("read identity memories"):
            memory_ids = self._store.identities.speaker_memory_ids(
                previous_speaker_id or identity_id
            )
            if not memory_ids:
                return (), ()
            stored = self._store.records.read_memories(memory_ids)
        indexed = tuple(memory for memory in stored if has_indexed_speech(memory))
        if not indexed:
            return (), ()
        return self._embedding.refresh_speaker_memories(
            indexed,
            speaker_id=identity_id,
            speaker_name=speaker_name,
            previous_speaker_id=previous_speaker_id,
            update_operation=previous_speaker_id is None,
            operation=operation,
        )

    def _register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
        speaker: bool,
    ) -> None:
        id_label = "speaker_id" if speaker else "identity_id"
        requested_id = validated_identifier(identity_id, id_label)
        normalized_name = validated_identity_name(name)
        normalized_relationship = (
            None if relationship is None else validated_identity_relationship(relationship)
        )
        operation_name = "register_speaker" if speaker else "register_identity"
        with (
            self._trace(f"mindbridge.{operation_name}", kind="operation"),
            self._lifecycle.operation() as operation,
            # `_formation_lock` before `_write_lock`, the same order `consolidate()`'s apply
            # phase and `add()`'s formation use: the IDENTIFY naming commit below must not land
            # while a consolidate apply pass is in progress, and taking the two locks in a
            # different order here would deadlock against that path.
            self._formation_lock,
            self._write_lock,
        ):
            with translate_storage_errors("read identity memories"):
                normalized_id = self._store.identities.resolve_identity_id(requested_id)
                if normalized_id is None:
                    identity_exists = False
                    memory_ids = None
                elif speaker:
                    memory_ids = self._store.identities.speaker_memory_ids(normalized_id)
                    identity_exists = memory_ids is not None
                else:
                    identity_exists = (
                        self._store.identities.identity_memory_ids(normalized_id) is not None
                    )
                    speaker_memory_ids = self._store.identities.speaker_memory_ids(normalized_id)
                    memory_ids = () if speaker_memory_ids is None else speaker_memory_ids
            if not identity_exists:
                if speaker:
                    raise SpeakerNotFoundError(f"speaker does not exist: {requested_id}")
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            assert normalized_id is not None
            assert memory_ids is not None
            # `_assert_identity_name` prepares the affected speech documents before it commits,
            # then writes the assertion, registry projection, documents, vectors, and log in one
            # SQLite transaction.
            self._assert_identity_name(
                normalized_id,
                normalized_name,
                relationship=(
                    normalized_relationship
                    if normalized_relationship is not None
                    else self._recorded_relationship(normalized_id)
                ),
                operation=operation,
            )

    def _recorded_relationship(self, identity_id: str) -> str | None:
        _name, relationship = self.projected_identity(identity_id)
        return relationship

    def _assert_identity_name(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None,
        operation: OperationAssets,
    ) -> str:
        """Record naming one person as a typed ENTITY claim bound to that identity.

        This is what makes a name retrievable knowledge instead of a label on a row, and it is
        the versioned thing `identities.name` projects. It rests on the host's authority rather
        than on a model, so it needs neither a former nor a consolidator to be configured, and
        it carries no evidence: nothing was observed, somebody said so.

        It is logged like any other control-plane operation, so naming is auditable through
        `operations()` and reversible through `rollback()`. The logged intent is `IDENTIFY` and
        the claim travels on the operation itself, so `operations()` never has to pass an
        identity ID off as a memory ID in `evidence_ids`. A host cites nothing: nothing was
        observed, somebody said so, so the evidence set is empty.

        Re-asserting the same name and relationship is idempotent, because the derived memory ID
        is a function of the claim; a different name supersedes through the shared lineage.
        """
        proposal = naming_proposal(name, relationship, basis=EvidenceBasis.USER_STATEMENT)
        now = datetime.now(timezone.utc)
        context = formation_context(
            None,
            proposal,
            model_id=None,
            recipe=NAMING_RECIPE,
            recorded_at=now,
            identity_id=identity_id,
        )
        prepared = replace(
            prepare_memory(
                self._materializer.prepare(proposal.content, operation),
                occurred_at=None,
                occurred_end=None,
                metadata=None,
                memory_type=formation_memory_type(proposal.kind),
            ),
            memory_id=formation_memory_id(
                identity_id,
                proposal,
                recipe=NAMING_RECIPE,
                context=context,
            ),
            context=context,
        )
        proposed = MemoryOperation(
            intent=MemoryIntent.IDENTIFY,
            claim=IdentityClaim(identity_id=identity_id, name=name, relationship=relationship),
            rationale="the host registered this name",
        )
        base_key = operation_key(proposed, recipe=NAMING_RECIPE)
        with translate_storage_errors("check a naming operation"):
            key = self._store.control.naming_operation_key(base_key, prepared.memory_id)
        # `_register_identity` holds `_formation_lock` for this whole call (the same lock
        # `add()`'s automatic formation holds across its own `_commit_formation`, see
        # `_form_sources`), so this IDENTIFY commit cannot land while a consolidate apply pass
        # is in progress either.
        self._formation.commit(
            ((prepared, None, proposal.confidence),),
            (),
            completed_at=now,
            recipe=NAMING_RECIPE,
            # Re-asserting a standing name changes nothing, so it logs nothing: the operation
            # key is already active and the log's unique index would refuse it anyway.
            operation=(
                None
                if key is None
                else StoredOperation(
                    operation_key=key,
                    intent=proposed.intent.value,
                    trigger=MemoryTrigger.MANUAL.value,
                    recipe=NAMING_RECIPE,
                    operation_json=dump_operation(proposed),
                    applied_at=now,
                )
            ),
            projection_identity_id=identity_id,
            projection_factory=lambda: self.reindex_speech(
                identity_id,
                speaker_name=name,
                operation=operation,
            ),
        )
        return prepared.memory_id

    def forget_identity(self, identity_id: str) -> IdentityErasure:
        requested_id = validated_identifier(identity_id, "identity_id")
        with (
            self._trace("mindbridge.forget_identity", kind="operation"),
            self._lifecycle.operation() as operation,
            # `_formation_lock` before `_write_lock`, the same order `consolidate()`'s apply
            # phase and `add()`'s formation use: an irreversible erasure must not land while a
            # consolidate apply pass is in progress, and taking the two locks in a different
            # order here would deadlock against that path.
            self._formation_lock,
            self._write_lock,
        ):
            with translate_storage_errors("read identity memories"):
                normalized_id = self._store.identities.resolve_identity_id(requested_id)
                known = normalized_id is not None and (
                    self._store.identities.identity_memory_ids(normalized_id) is not None
                )
            if normalized_id is None or not known:
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            # `speaker_name=None` is the projection of a person with no naming assertion left,
            # which is what erasure makes them. The rebuilt documents are handed to the store so
            # the erasure and the reindex commit in one transaction: a crash between them would
            # leave the name searchable for a person whose template was already gone.
            memories, embeddings = self.reindex_speech(
                normalized_id,
                speaker_name=None,
                operation=operation,
            )
            with translate_storage_errors("forget identity"):
                erased = self._store.identities.forget_identity(
                    normalized_id,
                    memories=memories,
                    embeddings=embeddings,
                    operation=self._identity_erasure_operation(normalized_id),
                )
            if erased is None:
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            erasure, orphaned = erased
            self._lifecycle.queue_asset_cleanup(orphaned)
            self._projection.drain()
            return erasure

    def _identity_erasure_operation(self, identity_id: str) -> StoredOperation:
        """Build the irreversible log row one identity erasure commits with.

        Erasure is physical forgetting of a person, so the row is audit history and nothing
        else: it carries the identity, its aliases, and the naming assertions the erasure
        deleted -- ids and counts, never content -- and `rollback()` refuses it. That is the
        marker that separates this from the cognitive `FORGET` a row with `target_ids` records.
        """
        with translate_storage_errors("read identity aliases"):
            members = self._store.identities.identity_equivalence_class(identity_id)
        proposed = MemoryOperation(
            intent=MemoryIntent.FORGET,
            identity=IdentityChange(
                identity_id=identity_id,
                moved_ids=() if members is None else members[1:],
            ),
            rationale="the data subject asked to be erased",
        )
        return StoredOperation(
            operation_key=operation_key(proposed, recipe=IDENTITY_LINK_RECIPE),
            intent=proposed.intent.value,
            trigger=MemoryTrigger.MANUAL.value,
            recipe=IDENTITY_LINK_RECIPE,
            operation_json=dump_operation(proposed),
            applied_at=datetime.now(timezone.utc),
        )

    def bind_speaker_names(self, operation: OperationAssets) -> None:
        """Register the staged names, now that the memory carrying the voices is readable.

        `register_identity` in all but name -- it lands the same `IDENTIFY` assertion through the
        same commit, so it is auditable through `operations()` and reversible through
        `rollback()` -- but it resolves and checks the person itself rather than raising on one
        who has since been merged away, and it holds the existing operation instead of nesting a
        second one inside this write.

        A different standing name is never overwritten. The name came from a model reading a
        transcript, and one mishearing would rewrite every document that person appears in; the
        host's own `register_identity` stays the only thing that can replace a name. Nothing here
        links identities either: `identity_link_min_assets` governs who is the same person, and a
        name is not evidence about that.
        """
        bindings = dict(operation.speaker_names)
        operation.speaker_names.clear()
        bound = 0
        refused = 0
        for identity_id, name in bindings.items():
            with translate_storage_errors("read identity profile"):
                profile = self._store.identities.identity_profile(identity_id)
            try:
                proposed = None if profile is None else validated_identity_name(name)
            except ValidationError:
                proposed = None
            if profile is None or proposed is None:
                # The identity a staged name resolved to has since vanished (merged or deleted),
                # or the stated name failed validation. Either way the fact named nobody, which
                # is worth counting apart from a silent `continue` even though there is nobody
                # left to warn about by name.
                operation.speaker_names_refused += 1
                continue
            if profile.name == proposed:
                continue
            if profile.name is not None:
                refused += 1
                _LOGGER.warning(
                    "a distilled fact called identity %s %r, which is already called %r; "
                    "keeping the registered name",
                    profile.identity_id,
                    proposed,
                    profile.name,
                )
                continue
            with self._formation_lock, self._write_lock:
                self._assert_identity_name(
                    profile.identity_id,
                    proposed,
                    relationship=profile.relationship,
                    operation=operation,
                )
            bound += 1
        refused += operation.speaker_names_refused
        operation.speaker_names_refused = 0
        if bound or refused:
            with self._trace("mindbridge.identity.names", kind="stage") as span:
                span.set_attribute(IDENTITY_NAMES_BOUND, bound)
                span.set_attribute(IDENTITY_NAMES_REFUSED, refused)
