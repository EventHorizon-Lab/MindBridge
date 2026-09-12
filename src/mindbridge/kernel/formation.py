"""Formation: grounding typed proposals in committed observations.

The pure helpers decide identity, lineage, refusal, and inheritance; `Formation` runs the
formation backend and commits its accepted proposals with evidence, versions, and vectors in
one store transaction.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import cast

from opentelemetry.trace import Tracer

from mindbridge._telemetry import mark_model_requests, record_formation_refusals
from mindbridge.exceptions import MindBridgeError, ModelError
from mindbridge.infrastructure.local.store import (
    CONSENT_PREDICATE,
    StoredEmbedding,
    StoredMemory,
    StoredOperation,
    canonical_subject,
    datetime_text,
)
from mindbridge.kernel.content import (
    PreparedMemory,
    embedding_row_id,
    encode_metadata,
    observation_from_record,
    prepare_memory,
)
from mindbridge.kernel.contracts import Backends
from mindbridge.kernel.embedding import DOCUMENT_TASK, Embedding
from mindbridge.kernel.lifecycle import OperationAssets
from mindbridge.kernel.materialization import Materializer
from mindbridge.kernel.projection import Projection
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.tracing import Traced
from mindbridge.models.base import EmbedTask, FormationInput, ModelInput
from mindbridge.types import (
    KIND_MEMORY_TYPES,
    ConsentClaim,
    EvidenceBasis,
    FormationProposal,
    MemoryContext,
    MemoryIntent,
    MemoryKind,
    MemoryOperation,
    MemoryRecord,
    MemoryType,
)

_LOGGER = logging.getLogger(__name__)


_EMPTY_METADATA_JSON = "{}"


# Naming a person is a claim the host asserts, so its recipe names the kernel rule that produced
# it rather than a model space. That is what lets `register_identity` work with no former and no
# consolidator configured.
NAMING_RECIPE = "mindbridge-identity-naming-v1"


NAMING_PREDICATE = "identity"


# Recording consent rests on the same authority as naming -- somebody said so -- so it is the
# same kind of assertion under its own recipe and its own predicate, which keeps the two in
# separate lineages: consenting never renames anybody and renaming never re-opens consent.
CONSENT_RECIPE = "mindbridge-identity-consent-v1"


# Binding one voice to one face is the kernel's own corroboration rule over co-occurrence
# evidence it counted, not a model's proposal, so the recipe names that rule. It is what makes
# the MERGE row's lineage honest: the recognizer models produced the templates, this rule
# decided the bind, and `rollback()` reverses it.
IDENTITY_LINK_RECIPE = "mindbridge-identity-link-v1"


def formation_memory_type(kind: MemoryKind) -> MemoryType:
    return KIND_MEMORY_TYPES.get(kind, MemoryType.SEMANTIC)


def formation_refusal(
    proposal: FormationProposal,
    source: FormationInput,
    *,
    host_authored: bool = False,
) -> str | None:
    """Say why the kernel refuses to ground this proposal in this source, or `None` to keep it.

    A refusal costs the proposal and nothing else, on both paths that ask: formation drops it,
    and consolidation moves on to the next cited source.
    """
    if proposal.kind is MemoryKind.RESPONSE_POLICY and not host_authored:
        # How the system should behave toward somebody is a grant, not an inference: one
        # observed cue must not become standing guidance. Keyed on where the proposal came
        # from, never on the basis it claims -- a formation or consolidation backend builds
        # its own `FormationProposal` and could name any basis -- so the only path that
        # authorizes one is the host's own `apply()`.
        return "response_policy formation requires explicit host authorization"
    if proposal.kind is MemoryKind.AFFECT and (
        proposal.cue_modality is None or proposal.cue_modality not in source.content.modalities
    ):
        return "affect formation must name a modality present in its source"
    if proposal.spatial is not None:
        observed = source.context.spatial
        if (
            observed is None
            or proposal.spatial.frame_id != observed.frame_id
            or proposal.spatial.anchor is not observed.anchor
        ):
            return "spatial formation must use the source observation frame and anchor"
    return None


class RejectedOperation(Exception):
    """Internal signal that kernel policy refused one proposed operation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def naming_proposal(
    name: str,
    relationship: str | None,
    *,
    basis: EvidenceBasis,
) -> FormationProposal:
    """Build the ENTITY proposal one naming claim asserts, whoever claimed it."""
    return FormationProposal(
        kind=MemoryKind.ENTITY,
        content=(
            f"{name} is a recognized person."
            if relationship is None
            else f"{name} is a recognized person, {relationship}."
        ),
        basis=basis,
        subject=name,
        predicate=NAMING_PREDICATE,
        value=relationship,
        confidence=1.0,
    )


def consent_proposal(claim: ConsentClaim) -> FormationProposal:
    """Build the STATE proposal one consent statement asserts."""
    stated = f"Processing consent for {claim.identity_id} is {claim.state.value}."
    return FormationProposal(
        kind=MemoryKind.STATE,
        content=stated if claim.note is None else f"{stated} {claim.note}",
        # The one basis a consent statement can have. Nothing was observed and no model may
        # infer it: the person said so.
        basis=EvidenceBasis.USER_STATEMENT,
        # The identity itself, not the name it currently projects: a person may be unnamed, and
        # renaming one must not fork their consent history into a second subject.
        subject=claim.identity_id,
        predicate=CONSENT_PREDICATE,
        value=claim.state.value,
        confidence=1.0,
    )


def is_consent_assertion(
    memories: Mapping[str, StoredMemory],
    memory_id: str,
) -> bool:
    """Report whether one record is a standing consent statement about a person."""
    memory = memories.get(memory_id)
    context = None if memory is None else memory.context
    return (
        context is not None
        and context.identity_id is not None
        and context.predicate == CONSENT_PREDICATE
    )


def is_bound_naming_assertion(
    memories: Mapping[str, StoredMemory],
    memory_id: str,
) -> bool:
    memory = memories.get(memory_id)
    context = None if memory is None else memory.context
    return (
        context is not None
        and context.kind is MemoryKind.ENTITY
        and context.identity_id is not None
    )


def is_derived(memories: Mapping[str, StoredMemory], memory_id: str) -> bool:
    memory = memories.get(memory_id)
    context = None if memory is None else memory.context
    return context is not None and context.kind is not MemoryKind.OBSERVATION


def consolidation_primary(
    proposal: FormationProposal,
    sources: Sequence[MemoryRecord],
    *,
    host_authored: bool = False,
) -> MemoryRecord | None:
    """Return the newest cited source the proposal is grounded in, or `None`.

    Formation binds one proposal to one source. A consolidation cites several, so the kernel
    requires the AFFECT cue modality and the spatial frame to match at least one of them and
    inherits validity from the newest such source, which keeps the derived ID stable when the
    same evidence set is proposed again.
    """
    ranked = sorted(
        sources,
        key=lambda source: (
            source.occurred_end or source.occurred_at or source.created_at,
            source.id,
        ),
    )
    for source in reversed(ranked):
        refusal = formation_refusal(
            proposal,
            FormationInput(
                memory_id=source.id,
                content=ModelInput(text=source.content, assets=source.assets),
                context=observation_from_record(source),
            ),
            host_authored=host_authored,
        )
        if refusal is None:
            return source
    return None


def retiring_targets(operation: MemoryOperation) -> set[str]:
    """IDs this operation would take out of ordinary recall or retire the current version of.

    A `REINFORCE` target is strengthened rather than retired, so it is not one of these; a
    `CONSOLIDATE` target is consolidation forgetting and is.
    """
    if operation.intent is MemoryIntent.REINFORCE:
        return set()
    return set(operation.target_ids)


def _grounded_formation_pairs(
    pairs: Sequence[tuple[PreparedMemory, str | None, float]],
) -> tuple[tuple[tuple[PreparedMemory, str | None, float], ...], int]:
    """Keep the first of each conflicting proposal from one source; return the rest as refused.

    Two proposals of one record that disagree, and two contradictory states in one lineage, are
    grounding faults inside a single model response -- not damage to the envelope. The source is
    already committed when formation runs, so failing the write would report a stored observation
    as unwritten and every retry would re-run the model and fail identically.
    """
    kept: builtins.list[tuple[PreparedMemory, str | None, float]] = []
    refused = 0
    seen: dict[tuple[str, str | None], PreparedMemory] = {}
    states: dict[tuple[str | None, str], builtins.list[MemoryContext]] = {}
    for pair in pairs:
        prepared, source_id, _confidence = pair
        if seen.setdefault((prepared.memory_id, source_id), prepared) != prepared:
            refused += 1
            continue
        context = prepared.context
        if isinstance(context, MemoryContext) and context.kind is MemoryKind.STATE:
            lineage = states.setdefault(
                (source_id, context.lineage_id or prepared.memory_id),
                [],
            )
            if any(
                standing.value != context.value
                and _valid_intervals_overlap(
                    standing.valid_from,
                    standing.valid_until,
                    context.valid_from,
                    context.valid_until,
                )
                for standing in lineage
            ):
                refused += 1
                continue
            lineage.append(context)
        kept.append(pair)
    return tuple(kept), refused


def _valid_intervals_overlap(
    left_from: datetime | None,
    left_until: datetime | None,
    right_from: datetime | None,
    right_until: datetime | None,
) -> bool:
    return not (
        (left_until is not None and right_from is not None and left_until <= right_from)
        or (right_until is not None and left_from is not None and right_until <= left_from)
    )


def agreed_inheritance(
    sources: Sequence[MemoryRecord],
) -> tuple[str | None, Mapping[str, object] | None]:
    """The place and the metadata every cited source agrees on, or nothing.

    A place is a hard retrieval filter, so disagreement inherits nothing rather than a majority.
    """
    places = {source.place_id for source in sources}
    tags = {encode_metadata(source.metadata) for source in sources}
    return (
        sources[0].place_id if len(places) == 1 else None,
        sources[0].metadata if len(tags) == 1 else None,
    )


def _agreed_columns(
    values: Sequence[PreparedMemory],
    stored: StoredMemory | None,
) -> PreparedMemory:
    """Return the first proposal of one derived record carrying only agreed inherited columns.

    Same rule as `agreed_inheritance`, applied where the sources arrive one write at a time: the
    place and the metadata survive only while every proposal of this record -- and the row already
    written for it -- says the same thing.
    """
    places = {value.place_id for value in values}
    tags = {value.metadata_json for value in values}
    if stored is not None:
        places.add(stored.place_id)
        tags.add(stored.metadata_json)
    return replace(
        values[0],
        place_id=values[0].place_id if len(places) == 1 else None,
        metadata_json=values[0].metadata_json if len(tags) == 1 else _EMPTY_METADATA_JSON,
    )


def formation_context(
    source: MemoryRecord | None,
    proposal: FormationProposal,
    *,
    model_id: str | None,
    recipe: str,
    recorded_at: datetime,
    identity_id: str | None = None,
) -> MemoryContext:
    """Build the typed context for one proposal, optionally derived from a source memory.

    A naming assertion has no source memory and no model behind it: the household owner said
    so. It passes `source=None` and carries the identity binding instead.
    """
    source_context = None if source is None else observation_from_record(source)
    valid_from = proposal.valid_from
    valid_until = proposal.valid_until
    spatial = proposal.spatial
    if source is not None and source_context is not None:
        valid_from = valid_from or source_context.valid_from or source.occurred_at
        valid_until = valid_until or source_context.valid_until
        spatial = spatial or source_context.spatial
    if valid_from is None and proposal.kind in {MemoryKind.STATE, MemoryKind.TRAIT}:
        valid_from = recorded_at
    return MemoryContext(
        kind=proposal.kind,
        # How the system should behave toward somebody is a grant: the kernel refuses every
        # model-proposed `RESPONSE_POLICY`, so one that reaches persistence was authorized by
        # the host calling `apply()` itself, and the stored record names that authorization
        # rather than the default the proposal carried. Stamped here, on the persisted context
        # alone, and never on the proposal: the log keeps the proposal it was handed and
        # `formation_memory_id` keys on that basis, so replaying a logged row mints the same
        # ID and is recognised as the duplicate it is.
        basis=(
            EvidenceBasis.RESPONSE_FEEDBACK
            if proposal.kind is MemoryKind.RESPONSE_POLICY
            else proposal.basis
        ),
        confidence=proposal.confidence,
        valid_from=valid_from,
        valid_until=valid_until,
        recorded_at=recorded_at,
        lineage_id=_formation_lineage_id(proposal, spatial=spatial, identity_id=identity_id),
        source_id=None if source_context is None else source_context.source_id,
        subject=proposal.subject,
        predicate=proposal.predicate,
        value=proposal.value,
        evidence_ids=() if source is None else (source.id,),
        model_id=model_id,
        recipe=recipe,
        identity_id=identity_id,
        spatial=spatial,
        cue_modality=proposal.cue_modality,
        valence=proposal.valence,
        arousal=proposal.arousal,
    )


def _formation_lineage_id(
    proposal: FormationProposal,
    *,
    spatial: object = None,
    identity_id: str | None = None,
) -> str:
    frame_id = getattr(spatial, "frame_id", None)
    anchor = getattr(getattr(spatial, "anchor", None), "value", None)
    # A bound claim keys on the person, not on how the subject happened to be spelled, so every
    # naming assertion about one identity lands in one lineage and a rename supersedes rather
    # than forks. An unbound claim keeps the original payload byte for byte, so lineage IDs
    # already on disk stay valid.
    binding: dict[str, object] = (
        {"subject": canonical_subject(proposal.subject)}
        if identity_id is None
        else {"subject": None, "identity_id": identity_id}
    )
    payload = json.dumps(
        {
            "kind": proposal.kind.value,
            "predicate": canonical_subject(proposal.predicate),
            "frame_id": frame_id,
            "anchor": anchor,
            **binding,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"mindbridge-lineage-v1:{payload}".encode()).hexdigest()


def formation_memory_id(
    source_id: str,
    proposal: FormationProposal,
    *,
    recipe: str,
    context: MemoryContext,
) -> str:
    episode_source = (
        source_id
        if (
            proposal.kind in {MemoryKind.EVENT, MemoryKind.AFFECT, MemoryKind.STATE}
            or (
                proposal.kind is MemoryKind.TRAIT and proposal.basis is EvidenceBasis.USER_STATEMENT
            )
        )
        else None
    )
    spatial = context.spatial
    payload = json.dumps(
        {
            "recipe": recipe,
            "kind": proposal.kind.value,
            # Only present once something is bound, so two identities that share a name stay two
            # records while every ID minted before the binding existed keeps its value.
            **({} if context.identity_id is None else {"identity_id": context.identity_id}),
            "subject": canonical_subject(proposal.subject),
            "predicate": canonical_subject(proposal.predicate),
            "value": canonical_subject(proposal.value),
            "assertion_basis": (
                proposal.basis.value
                if proposal.basis is not EvidenceBasis.MODEL_INFERENCE
                else None
            ),
            "cue_modality": (
                None if proposal.cue_modality is None else proposal.cue_modality.value
            ),
            "episode_source": episode_source,
            "content": proposal.content if proposal.kind is MemoryKind.EVENT else None,
            "valid_from": (
                datetime_text(context.valid_from)
                if episode_source is not None and context.valid_from is not None
                else None
            ),
            "valid_until": (
                datetime_text(context.valid_until)
                if episode_source is not None and context.valid_until is not None
                else None
            ),
            "spatial": (
                None
                if spatial is None
                else {
                    "frame_id": spatial.frame_id,
                    "anchor": spatial.anchor.value,
                    "position_m": (spatial.x, spatial.y, spatial.z),
                    "orientation_xyzw": spatial.orientation_xyzw,
                    "position_uncertainty_m": spatial.position_uncertainty_m,
                }
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"mindbridge-formation-v1:{payload}".encode()).hexdigest()


class Formation(Traced):
    """Run the formation backend over committed sources and commit accepted proposals."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        backends: Backends,
        materializer: Materializer,
        embedding: Embedding,
        projection: Projection,
    ) -> None:
        super().__init__(tracer)
        self._formation_lock = storage.formation_lock
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._backends = backends
        self._materializer = materializer
        self._embedding = embedding
        self._projection = projection

    def form_sources(
        self,
        sources: Sequence[MemoryRecord],
        *,
        operation: OperationAssets,
    ) -> None:
        if self._backends.former is None or not sources:
            return
        with self._formation_lock:
            self._form_sources_locked(sources, operation=operation)

    def _form_sources_locked(
        self,
        sources: Sequence[MemoryRecord],
        *,
        operation: OperationAssets,
    ) -> None:
        assert self._backends.former is not None
        pending = tuple({source.id: source for source in sources}.values())
        pending = tuple(
            source
            for source in pending
            if not self._store.semantics.formation_completed(
                source.id, self._backends.formation_space
            )
        )
        if not pending:
            return
        all_inputs = tuple(
            FormationInput(
                memory_id=source.id,
                content=ModelInput(text=source.content, assets=source.assets),
                context=observation_from_record(source),
            )
            for source in pending
        )
        inputs = tuple(
            value
            for value in all_inputs
            if value.content.modalities <= self._backends.formation_capabilities
        )
        if not inputs:
            return
        try:
            with self._model_trace(
                "formation",
                "form",
                model=self._backends.formation_model,
                batch_size=len(inputs),
                modalities=(modality for value in inputs for modality in value.content.modalities),
            ):
                mark_model_requests(1)
                proposals_by_source = self._backends.former.form(inputs)
        except MindBridgeError:
            raise
        except Exception as error:
            raise ModelError(
                "automatic memory formation failed",
                reason="model_failed",
                stage="form",
            ) from error
        if (
            not isinstance(proposals_by_source, tuple)
            or len(proposals_by_source) != len(inputs)
            or any(
                not isinstance(values, tuple)
                or any(not isinstance(value, FormationProposal) for value in values)
                for values in proposals_by_source
            )
        ):
            raise ModelError(
                "formation backend returned an invalid batch",
                reason="response_invalid",
                stage="form",
            )
        proposals_by_id: dict[str, tuple[FormationProposal, ...]] = {
            value.memory_id: proposals
            for value, proposals in zip(inputs, proposals_by_source, strict=True)
        }

        now = datetime.now(timezone.utc)
        pairs: builtins.list[tuple[PreparedMemory, str, float]] = []
        inputs_by_id = {value.memory_id: value for value in inputs}
        formed_sources = tuple(source for source in pending if source.id in inputs_by_id)
        refused = 0
        for source in formed_sources:
            proposals = proposals_by_id.get(source.id, ())
            # Application data and the symbolic place travel with the knowledge formed from them:
            # a host that filters recall by metadata expects the tag on the observation to hold
            # for what was learned from it, and `place_id` is a hard filter, so knowledge formed
            # from an observation in a room has to stand in that room too -- otherwise a
            # place-scoped question sees the raw observation and none of the entities, states, or
            # relations formed from it. One source agrees with itself; a record several sources
            # share keeps only what they all say, which `_commit_formation` settles.
            place_id, metadata = agreed_inheritance((source,))
            for proposal in proposals:
                evidence_ids = (
                    (source.id,) if proposal.evidence_ids is None else proposal.evidence_ids
                )
                if source.id not in evidence_ids or not set(evidence_ids) <= set(inputs_by_id):
                    raise ModelError(
                        "formation proposal cited an observation outside its batch",
                        reason="response_invalid",
                        stage="form",
                    )
                # One derived opinion the model grounded wrongly -- an affect cue naming a
                # modality the source never carried, a pose in another frame -- must not cost the
                # caller the observation it came from. `add` commits the source before formation
                # runs, so failing here fails a write that in fact succeeded, and every retry
                # re-runs formation and fails the same way. It is dropped and counted instead,
                # which is the policy the model adapter already applies to a malformed one, and
                # which consolidation already applies to this same rule. Damage to the batch
                # envelope still raises above: that is not one opinion.
                refusal = formation_refusal(proposal, inputs_by_id[source.id])
                if refusal is not None:
                    _LOGGER.warning(
                        "formation proposal refused for memory %s: %s",
                        source.id,
                        refusal,
                    )
                    refused += 1
                    continue
                prepared = prepare_memory(
                    self._materializer.prepare(proposal.content, operation),
                    occurred_at=source.occurred_at,
                    occurred_end=source.occurred_end,
                    metadata=metadata,
                    memory_type=formation_memory_type(proposal.kind),
                )
                context = formation_context(
                    source,
                    proposal,
                    model_id=self._backends.formation_model,
                    recipe=self._backends.formation_space,
                    recorded_at=now,
                    identity_id=self.bound_identity(proposal),
                )
                context = replace(context, evidence_ids=evidence_ids)
                pairs.append(
                    (
                        replace(
                            prepared,
                            memory_id=formation_memory_id(
                                source.id,
                                proposal,
                                recipe=self._backends.formation_space,
                                context=context,
                            ),
                            context=context,
                            place_id=place_id,
                        ),
                        source.id,
                        proposal.confidence,
                    )
                )
        grounded, conflicting = _grounded_formation_pairs(pairs)
        record_formation_refusals(refused + conflicting)
        self.commit(
            grounded,
            formed_sources,
            completed_at=now,
            joint_evidence_clauses=True,
        )

    def commit(
        self,
        pairs: Sequence[tuple[PreparedMemory, str | None, float]],
        sources: Sequence[MemoryRecord],
        *,
        completed_at: datetime,
        recipe: str | None = None,
        operation: StoredOperation | None = None,
        forget_ids: Sequence[str] = (),
        require_active: Sequence[str] = (),
        require_unretired: Sequence[str] = (),
        projection_identity_id: str | None = None,
        joint_evidence_clauses: bool = False,
        projection_factory: Callable[
            [], tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]]
        ]
        | None = None,
    ) -> StoredOperation | None:
        """Embed and commit derived records; return the log row when one was requested.

        A `None` source ID states that the record rests on no other memory, which is what an
        asserted claim such as a naming assertion is: it cites nothing and needs no evidence row.
        """
        recipe = self._backends.formation_space if recipe is None else recipe
        ordered_pairs = tuple(sorted(pairs, key=lambda value: (value[0].memory_id, value[1] or "")))
        by_id: dict[str, builtins.list[PreparedMemory]] = {}
        for prepared, _source, _score in ordered_pairs:
            by_id.setdefault(prepared.memory_id, []).append(prepared)
        with translate_storage_errors("check formed memories"):
            existing = self._store.records.read_memories(tuple(sorted(by_id)))
        existing_by_id = {value.memory_id: value for value in existing}
        unique = tuple(
            _agreed_columns(values, existing_by_id.get(memory_id))
            for memory_id, values in sorted(by_id.items())
        )
        # A record whose ID excludes its source -- an entity, a relation, an inferred trait -- is
        # written once and every later source only adds evidence to it, so what the first source
        # said would otherwise stand for evidence that disagrees. The columns it no longer agrees
        # on are cleared on the stored row inside the same transaction; nothing is re-embedded,
        # because neither column reaches the index.
        narrowed = tuple(
            (value.memory_id, value.place_id, value.metadata_json)
            for value in unique
            if (row := existing_by_id.get(value.memory_id)) is not None
            and (value.place_id, value.metadata_json) != (row.place_id, row.metadata_json)
        )
        missing = tuple(value for value in unique if value.memory_id not in existing_by_id)
        parts = tuple(
            (memory, object_part, model_input)
            for memory in missing
            for object_part, model_input in enumerate(self._embedding.inputs(memory.content))
        )
        vectors = self._embedding.embed(
            tuple(model_input for _memory, _part, model_input in parts),
            task=EmbedTask.DOCUMENT,
        )
        stored = tuple(
            StoredMemory(
                memory_id=memory.memory_id,
                content=memory.content.text,
                modality=memory.content.modality.value,
                memory_type=memory.memory_type.value,
                assets=(),
                metadata_json=memory.metadata_json,
                occurred_at=memory.occurred_at,
                occurred_end=memory.occurred_end,
                created_at=completed_at,
                updated_at=completed_at,
                place_id=memory.place_id,
                context=cast(MemoryContext, memory.context),
            )
            # Existing records travel too: what they skip is the embedding, not the write. The
            # store decides whether restating a claim it already holds is a no-op or a new
            # version, which is how re-asserting a retired one -- a name changed back, a
            # registration repeated after `rollback` -- reaches the lineage at all.
            for memory in unique
        )
        embeddings = tuple(
            StoredEmbedding(
                embedding_id=embedding_row_id(memory.memory_id, object_part),
                memory_id=memory.memory_id,
                values=vector,
                model_id=self._backends.embedding_model,
                space_id=self._backends.space_id,
                task=DOCUMENT_TASK,
                created_at=completed_at,
                object_part=object_part,
                normalized=True,
            )
            for (memory, object_part, _model_input), vector in zip(parts, vectors, strict=True)
        )
        if operation is not None:
            created = tuple(value.memory_id for value in missing)
            operation = replace(
                operation,
                created_ids=created,
                changed_ids=tuple(
                    memory_id
                    for memory_id in dict.fromkeys(
                        prepared.memory_id for prepared, _source, _score in ordered_pairs
                    )
                    if memory_id not in set(created)
                ),
            )
        projection_memories: tuple[StoredMemory, ...] = ()
        projection_embeddings: tuple[StoredEmbedding, ...] = ()
        if projection_factory is not None:
            # Derived assertion vectors are prepared first, then every speech document the new
            # projection affects. Neither reaches SQLite until both model calls have succeeded.
            projection_memories, projection_embeddings = projection_factory()
        with self._write_lock:
            with translate_storage_errors("commit automatic memory formation"):
                applied = self._store.semantics.apply_formation(
                    stored,
                    embeddings,
                    evidence_clauses=(
                        ()
                        if not joint_evidence_clauses
                        else tuple(
                            dict.fromkeys(
                                (
                                    prepared.memory_id,
                                    prepared.context.evidence_ids,
                                    confidence,
                                )
                                for prepared, _source_id, confidence in ordered_pairs
                                if isinstance(prepared.context, MemoryContext)
                                and prepared.context.evidence_ids
                            )
                        )
                    ),
                    evidence=tuple(
                        dict.fromkeys(
                            (prepared.memory_id, evidence_id, confidence)
                            for prepared, _source_id, confidence in ordered_pairs
                            for evidence_id in (
                                ()
                                if not isinstance(prepared.context, MemoryContext)
                                else prepared.context.evidence_ids
                            )
                        )
                    ),
                    source_memory_ids=tuple(source.id for source in sources),
                    narrowed=narrowed,
                    recipe=recipe,
                    completed_at=completed_at,
                    operation=operation,
                    forget_ids=forget_ids,
                    require_active=require_active,
                    require_unretired=require_unretired,
                    projection_identity_id=projection_identity_id,
                    projection_memories=projection_memories,
                    projection_embeddings=projection_embeddings,
                )
            self._projection.drain()
        if operation is None:
            return None
        if not applied:
            # A concurrent duplicate won the key inside the transaction. Its row is in the log
            # under the same key, but it is not this call's operation, so report the duplicate.
            return None
        with translate_storage_errors("read a memory operation"):
            logged = self._store.control.read_operations(operation_key=operation.operation_key)
        return logged[0] if logged else None

    def bound_identity(self, proposal: FormationProposal) -> str | None:
        """Resolve which recognized person a proposed claim is about, or None.

        Deterministic and never the model's decision: the proposal's subject has to match the
        canonical subject of one currently visible naming assertion, compared with the same
        NFKC casefold the lineage key uses. `_formation_lineage_id` then keys on the identity,
        so claims about one person converge however the model spelled the name that turn.

        An ENTITY proposal is deliberately never bound. A bound ENTITY row *is* a naming
        assertion, so binding one here would let a model's proposal rename a person; naming
        stays with the host and with the identify path.

        `consent` is a reserved predicate for the same reason and by the same route. A bound
        STATE row carrying it *is* a consent statement -- it is what `consent()` reads and what
        restrains enrolment and merging -- so binding one here would let a model manufacture or
        retract permission to process a person without ever reaching the control plane, which
        refuses a proposed `CONSENT` operation. The claim itself is kept, unbound: a model
        inferring that somebody withdrew consent is evidence of what the model thought, and it
        is recorded as an ordinary STATE about that subject with no identity attached.
        """
        if (
            proposal.kind is MemoryKind.ENTITY
            or proposal.subject is None
            or proposal.predicate == CONSENT_PREDICATE
        ):
            return None
        with translate_storage_errors("resolve a claim's subject"):
            return self._store.identities.identity_for_subject(proposal.subject)
