"""The memory control plane: proposing, applying, auditing, and reversing operations."""

from __future__ import annotations

import builtins
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from itertools import zip_longest

from opentelemetry.trace import Tracer

from mindbridge._telemetry import mark_model_requests
from mindbridge.control import dump_operation, load_operation, operation_key
from mindbridge.exceptions import MemoryNotFoundError, MindBridgeError, ModelError, ValidationError
from mindbridge.infrastructure.local.store import (
    StaleOperationError,
    StoredEmbedding,
    StoredMemory,
    StoredOperation,
)
from mindbridge.kernel.content import prepare_memory
from mindbridge.kernel.contracts import Backends
from mindbridge.kernel.formation import (
    Formation,
    RejectedOperation,
    agreed_inheritance,
    consolidation_primary,
    formation_context,
    formation_memory_id,
    formation_memory_type,
    is_bound_naming_assertion,
    is_consent_assertion,
    is_derived,
    naming_proposal,
    retiring_targets,
)
from mindbridge.kernel.hydration import Hydrator, operation_record
from mindbridge.kernel.identity import Identities
from mindbridge.kernel.lifecycle import Lifecycle, OperationAssets
from mindbridge.kernel.materialization import Materializer
from mindbridge.kernel.projection import Projection
from mindbridge.kernel.retrieval import Retrieval
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.settings import Settings
from mindbridge.kernel.tracing import Traced
from mindbridge.kernel.validation import validate_limit, validated_identifier
from mindbridge.types import (
    ConsolidationCandidate,
    ConsolidationReport,
    ContentInput,
    DeliberationReport,
    EvidenceBasis,
    IdentityChange,
    MemoryIntent,
    MemoryOperation,
    MemoryOperationRecord,
    MemoryOutcome,
    MemoryRecord,
    MemoryTrigger,
)

_LOGGER = logging.getLogger(__name__)


# One empty recall is a question nobody had asked before; two near-equal ones inside the
# configured window is a gap. Not configurable: below two there is no repetition to speak of, and
# a host that wants a stricter threshold narrows the window instead.
_REPEATED_FAILURES = 2


def operation_touches(
    record: MemoryOperationRecord,
    memory_ids: frozenset[str],
    identity_ids: frozenset[str],
) -> bool:
    """Report whether one logged operation moved any of these records or any of these people."""
    operation = record.operation
    named = {
        *operation.evidence_ids,
        *operation.target_ids,
        *record.created_ids,
        *record.changed_ids,
        *record.forgotten_ids,
        *(memory_id for memory_id, _version in record.superseded),
    }
    if named & memory_ids:
        return True
    people: set[str] = set()
    if operation.identity is not None:
        people.update((operation.identity.identity_id, *operation.identity.moved_ids))
    if operation.claim is not None:
        people.add(operation.claim.identity_id)
    if operation.consent is not None:
        people.add(operation.consent.identity_id)
    return bool(people & identity_ids)


class ControlPlane(Traced):
    """Propose, apply, audit, and reverse memory-control-plane operations."""

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
        projection: Projection,
        retrieval: Retrieval,
        formation: Formation,
        identities: Identities,
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
        self._projection = projection
        self._retrieval = retrieval
        self._formation = formation
        self._identities = identities

    # -- Agentic memory control plane ----------------------------------------------------------
    # One bounded memory-management loop (gate 3 of docs/context-os.md). The backend sees a
    # bounded evidence set and only proposes; every field is validated here, each accepted
    # operation commits with its own append-only log row, and `rollback()` reverses it. Physical
    # deletion is not an intent: it stays on `delete()` under host authority. None of these
    # methods is exposed on REST or MCP.

    def consolidation_candidates(
        self,
        *,
        limit: int = 32,
        idle: bool = False,
    ) -> tuple[ConsolidationCandidate, ...]:
        with (
            self._trace("mindbridge.consolidation_candidates", kind="operation"),
            self._lifecycle.operation() as assets,
        ):
            validate_limit(limit, maximum=100)
            if not isinstance(idle, bool):
                raise ValidationError("idle must be a boolean")
            with translate_storage_errors("list consolidation candidates"):
                rows = self._store.control.read_consolidation_candidates(
                    limit=limit,
                    idle=idle,
                    record_budget=self._settings.memory_budget_records,
                )
            derived = tuple(
                ConsolidationCandidate(
                    trigger=MemoryTrigger(row.trigger),
                    memory_ids=row.memory_ids,
                    evidence_count=row.evidence_count,
                )
                for row in rows
            )
            failures = self._query_failure_candidates(limit=limit, assets=assets)
            # Round robin rather than concatenation, so a store with many repeated failures
            # cannot push every evidence and contradiction row out of the window.
            return tuple(
                candidate
                for pair in zip_longest(derived, failures)
                for candidate in pair
                if candidate is not None
            )[:limit]

    def _query_failure_candidates(
        self,
        *,
        limit: int,
        assets: OperationAssets,
    ) -> tuple[ConsolidationCandidate, ...]:
        """Turn repeated empty recalls into candidates naming what the store does hold.

        A failed query has no evidence of its own -- that is what failing means -- so the
        candidate names the nearest active records to it. Those are what a backend can act on:
        the memory that should have matched and did not, or the gap it should record. Resolved
        through the ordinary retrieval kernel, and bounded by `limit`.
        """
        reference = datetime.now(timezone.utc)
        with translate_storage_errors("list repeated query failures"):
            failures = self._store.control.read_repeated_query_failures(
                limit=limit,
                since=reference - self._settings.query_failure_window,
                minimum=_REPEATED_FAILURES,
            )
        if not failures:
            return ()
        candidates: builtins.list[ConsolidationCandidate] = []
        for failure in failures:
            prepared = self._materializer.prepare(failure.query, assets)
            outcome = self._retrieval.search_prepared(
                prepared,
                limit=min(100, max(2, limit // 4)),
                operation=assets,
                memory_types=None,
                reference_at=reference,
                temporal_range=None,
                occurred_from=None,
                occurred_until=None,
                scope=None,
                require_unambiguous=False,
                capture_trace=False,
            )
            memory_ids = tuple(hit.id for hit in outcome.hits)
            if not memory_ids:
                # Nothing to weigh: the store holds no evidence anywhere near this query, so
                # there is no bounded evidence set a proposal could cite.
                continue
            with translate_storage_errors("read deliberation marks"):
                weighed = self._store.control.read_weighed_at(memory_ids)
            if weighed and failure.failed_at <= max(weighed.values()):
                continue
            candidates.append(
                ConsolidationCandidate(
                    trigger=MemoryTrigger.QUERY_FAILURE,
                    memory_ids=memory_ids,
                    evidence_count=failure.failures,
                )
            )
        return tuple(candidates)

    def deliberate(
        self,
        *,
        limit: int = 32,
        max_rounds: int = 4,
        idle: bool = False,
    ) -> DeliberationReport:
        with self._trace("mindbridge.deliberate", kind="operation"):
            validate_limit(limit, maximum=100)
            validate_limit(max_rounds, maximum=100)
            if not isinstance(idle, bool):
                raise ValidationError("idle must be a boolean")
            rounds = weighed = skipped = applied = rejected = calls = 0
            for _round in range(max_rounds):
                candidates = self.consolidation_candidates(limit=limit, idle=idle)
                if not candidates:
                    break
                rounds += 1
                for candidate in candidates:
                    report = self.consolidate(
                        evidence_ids=candidate.memory_ids,
                        limit=limit,
                        trigger=candidate.trigger,
                    )
                    if report.weighed:
                        weighed += 1
                        calls += 1
                    else:
                        skipped += 1
                    applied += len(report.operations)
                    rejected += len(report.rejected)
            return DeliberationReport(
                rounds=rounds,
                weighed=weighed,
                skipped=skipped,
                applied=applied,
                rejected=rejected,
                model_calls=calls,
            )

    def consolidate(
        self,
        *,
        evidence_ids: Sequence[str] | None = None,
        query: ContentInput | None = None,
        limit: int = 32,
        trigger: MemoryTrigger = MemoryTrigger.MANUAL,
    ) -> ConsolidationReport:
        with (
            self._trace("mindbridge.consolidate", kind="operation"),
            self._lifecycle.operation() as assets,
        ):
            validate_limit(limit, maximum=100)
            if self._backends.consolidator is None:
                raise ModelError(
                    "consolidation backend is not configured",
                    reason="backend_not_configured",
                    stage="consolidate",
                )
            if not isinstance(trigger, MemoryTrigger):
                raise ValidationError("trigger must be a MemoryTrigger")
            shown, window = self._consolidation_evidence(
                evidence_ids,
                query,
                limit=limit,
                operation=assets,
            )
            if not shown:
                return ConsolidationReport()
            applied: builtins.list[MemoryOperationRecord] = []
            rejected: builtins.list[tuple[MemoryOperation, str]] = []
            # One pass must not contradict itself: an operation may not name -- as evidence or as
            # a target -- a record an earlier accepted operation retired, nor retire evidence an
            # earlier one built on.
            # There is no way to submit a proposal built in some other pass, so this is the only
            # reachable form of the staleness the kernel is required to reject.
            consumed: set[str] = set()
            retired: set[str] = set()
            # Deliberately outside `_formation_lock`: the backend round trip is the slow part of
            # this call, and holding the formation lock across it makes a consolidation over
            # media evidence stall every concurrent `add()`. Scheduling between latency-sensitive
            # work and slow reasoning is the point of the plane, so the lock covers the apply
            # transactions only. Correctness does not rest on the lock: every proposal is
            # re-checked inside its own apply transaction (`require_active` and
            # `require_unretired`) and refused as stale if a target moved while the backend was
            # thinking. A model call that raised weighed nothing, so no marker is recorded and
            # the candidate stays due.
            proposals = self._propose_operations(tuple(shown.values()), trigger=trigger)
            with self._formation_lock:
                for operation in proposals:
                    targets = retiring_targets(operation)
                    named = set(operation.evidence_ids) | set(operation.target_ids)
                    if targets & consumed or named & retired:
                        _LOGGER.warning(
                            "consolidation refused a %s proposal: inconsistent_batch",
                            operation.intent.value,
                        )
                        rejected.append((operation, "inconsistent_batch"))
                        continue
                    try:
                        applied.append(
                            self._apply_memory_operation(
                                operation,
                                trigger=trigger,
                                model_id=self._backends.consolidation_model,
                                recipe=self._backends.consolidation_recipe,
                                shown=shown,
                                window=window,
                                assets=assets,
                            )
                        )
                    except RejectedOperation as rejection:
                        _LOGGER.warning(
                            "consolidation refused a %s proposal: %s",
                            operation.intent.value,
                            rejection.reason,
                        )
                        rejected.append((operation, rejection.reason))
                    else:
                        consumed.update(set(operation.evidence_ids) - targets)
                        retired.update(targets)
            # Recorded whatever the pass yielded, including nothing. Without this a candidate the
            # backend could not resolve -- or whose every proposal the kernel refused -- leaves no
            # trace and is derived again, and paid for again, every round.
            self._record_deliberation(
                trigger,
                tuple(shown),
                proposed=len(proposals),
                applied=len(applied),
                rejected=len(rejected),
            )
            return ConsolidationReport(
                operations=tuple(applied),
                rejected=tuple(rejected),
                weighed=len(shown),
            )

    def _record_deliberation(
        self,
        trigger: MemoryTrigger,
        memory_ids: Sequence[str],
        *,
        proposed: int,
        applied: int,
        rejected: int,
    ) -> None:
        """Mark one evidence set weighed so candidate derivation stops re-listing it."""
        with self._write_lock, translate_storage_errors("record a deliberation"):
            self._store.control.record_deliberation(
                trigger.value,
                memory_ids,
                weighed_at=datetime.now(timezone.utc),
                proposed=proposed,
                applied=applied,
                rejected=rejected,
            )

    def apply(self, operation: MemoryOperation) -> MemoryOperationRecord:
        with (
            self._trace("mindbridge.apply", kind="operation"),
            self._lifecycle.operation() as assets,
        ):
            if not isinstance(operation, MemoryOperation):
                raise ValidationError("operation must be a MemoryOperation")
            shown, _window = self._consolidation_evidence(
                operation.evidence_ids,
                None,
                limit=100,
                operation=assets,
            )
            try:
                with self._formation_lock:
                    return self._apply_memory_operation(
                        operation,
                        trigger=MemoryTrigger.MANUAL,
                        # The configured consolidation recipe and model, not the caller's word
                        # for them. A derived record's representation belongs to a recipe -- it
                        # is part of its identity and of the operation key -- so replaying a
                        # logged sequence reproduces the same derived IDs exactly when the
                        # store is configured with the recipe that produced them, and mints
                        # different ones when it is not. No backend is called either way.
                        model_id=self._backends.consolidation_model,
                        recipe=self._backends.consolidation_recipe,
                        shown=shown,
                        # The host names these IDs, the way it does for `forget()`, so the
                        # not-shown rule that bounds a backend to the window the kernel gathered
                        # does not apply. Every other rejection does.
                        window=None,
                        assets=assets,
                        # The host's own path, so a `RESPONSE_POLICY` proposal is authorized
                        # here and nowhere else.
                        host_authored=True,
                    )
            except RejectedOperation as rejection:
                raise ValidationError(
                    f"memory operation refused: {rejection.reason}",
                    reason=rejection.reason,
                ) from None

    def record_outcome(
        self,
        operation_id: int,
        outcome: MemoryOutcome,
        *,
        note: str | None = None,
    ) -> bool:
        with (
            self._trace("mindbridge.record_outcome", kind="operation"),
            self._lifecycle.operation(),
        ):
            if (
                isinstance(operation_id, bool)
                or not isinstance(operation_id, int)
                or operation_id <= 0
            ):
                raise ValidationError("operation_id must be a positive integer")
            if not isinstance(outcome, MemoryOutcome):
                raise ValidationError("outcome must be a MemoryOutcome")
            if note is not None and (not isinstance(note, str) or not note.strip()):
                raise ValidationError("note must be a non-empty string or None")
            with self._write_lock, translate_storage_errors("record an operation outcome"):
                return self._store.control.record_operation_outcome(
                    operation_id,
                    outcome=outcome.value,
                    note=None if note is None else note.strip(),
                )

    def forget(self, memory_ids: Sequence[str]) -> MemoryOperationRecord | None:
        with (
            self._trace("mindbridge.forget", kind="operation"),
            self._lifecycle.operation() as assets,
        ):
            if isinstance(memory_ids, (str, bytes)):
                raise ValidationError("memory_ids must be a sequence of memory IDs")
            try:
                targets = tuple(
                    validated_identifier(memory_id, "memory_id") for memory_id in memory_ids
                )
            except TypeError:
                raise ValidationError("memory_ids must be a sequence of memory IDs") from None
            if not targets:
                return None
            try:
                return self._apply_memory_operation(
                    MemoryOperation(intent=MemoryIntent.FORGET, target_ids=targets),
                    trigger=MemoryTrigger.MANUAL,
                    model_id=None,
                    recipe=None,
                    shown=None,
                    window=None,
                    assets=assets,
                )
            except RejectedOperation as rejection:
                if rejection.reason == "unknown_target":
                    missing = sorted(set(targets) - set(self._control_records(targets)))
                    raise MemoryNotFoundError(
                        f"memory does not exist: {', '.join(missing)}"
                    ) from None
                return None

    def rollback(self, operation_id: int) -> bool:
        with (
            self._trace("mindbridge.rollback", kind="operation"),
            self._lifecycle.operation(),
            self._write_lock,
        ):
            if (
                isinstance(operation_id, bool)
                or not isinstance(operation_id, int)
                or operation_id <= 0
            ):
                raise ValidationError("operation_id must be a positive integer")
            with translate_storage_errors("read a memory operation"):
                logged = self._store.control.read_operations(operation_id=operation_id)
            if not logged or logged[0].rolled_back_at is not None:
                return False
            row = logged[0]
            operation = load_operation(row.operation_json)
            if operation.identity is not None:
                return self._rollback_identity(row, operation, operation.identity)
            # A naming assertion is retracted, not deleted: the version it superseded has to
            # come back, and the record itself stays in the log so the audit trail shows both
            # names. That reversal rides the general `superseded` mechanism below. A consent
            # statement retracts the same way and for the same reason -- both are assertions a
            # host made about a person, and the history of what was asserted is the audit
            # trail -- but only naming feeds a projection that has to be repainted afterwards.
            naming = operation.intent is MemoryIntent.IDENTIFY
            assertion = naming or operation.intent is MemoryIntent.CONSENT
            with translate_storage_errors("roll back a memory operation"):
                # One transaction: the created records disappear in the same commit that marks
                # the operation rolled back, so a crash between the two cannot leave an active
                # operation whose recorded output is gone.
                reverted, orphaned = self._store.control.rollback_operation(
                    row.operation_id,
                    rolled_back_at=datetime.now(timezone.utc),
                    delete_memory_ids=(
                        row.created_ids if operation.intent is MemoryIntent.CONSOLIDATE else ()
                    ),
                    # `linked` is the evidence this operation actually inserted, so a link that
                    # predated it survives the reversal. Records deleted above take their own.
                    retire_evidence=(
                        () if row.clause_changes or row.linked_clauses else row.linked
                    ),
                    retire_clauses=row.linked_clauses,
                    reverse_clause_changes=row.clause_changes,
                    # A CORRECT retired the versions it named. A CONSOLIDATE may also have
                    # superseded records in the derived record's lineage that no backend ever
                    # saw; the log row names those exactly, so both halves reverse here.
                    restore_versions=(
                        row.changed_ids
                        if operation.intent is MemoryIntent.CORRECT
                        else row.superseded
                    ),
                    # An identity assertion is retracted rather than deleted: its own version
                    # is retired here and `restore_versions` brings back the one it displaced.
                    retire_versions=(
                        *row.activated_ids,
                        *(row.created_ids if assertion else ()),
                    ),
                    require_no_later_dependencies=((*row.created_ids, *row.activated_ids)),
                    require_in_force=(
                        (*row.created_ids, *row.changed_ids) if row.superseded else ()
                    ),
                    # Recorded per operation, so cognitive forgetting and the consolidation
                    # forgetting a CONSOLIDATE carried both reverse through one field. A FORGET
                    # row logged before that field existed carries the same IDs in `changed_ids`,
                    # and must still un-forget rather than silently do nothing.
                    clear_forgotten=row.forgotten_ids
                    or (row.changed_ids if operation.intent is MemoryIntent.FORGET else ()),
                )
            self._lifecycle.queue_asset_cleanup(orphaned)
            self._projection.drain()
            if reverted and naming:
                assert operation.claim is not None
                self._reproject_identity(operation.claim.identity_id)
            return reverted

    def _rollback_identity(
        self,
        row: StoredOperation,
        operation: MemoryOperation,
        change: IdentityChange,
    ) -> bool:
        """Reverse one identity-lifecycle operation: split a merge, or re-merge a split.

        The store refuses a reversal the identity graph no longer admits, which is what orders
        identity operations on one person newest first: a later split already removed the alias
        a merge would restore, and an erasure removed the person entirely.
        """
        if operation.intent is MemoryIntent.FORGET:
            # Physical forgetting. Nothing here can be restored, and reporting success would
            # claim a recovery that did not happen.
            return False
        absorbed = change.moved_ids[0]
        merging = operation.intent is MemoryIntent.CORRECT
        with translate_storage_errors("roll back an identity operation"):
            reverted, _orphaned = self._store.control.rollback_operation(
                row.operation_id,
                rolled_back_at=datetime.now(timezone.utc),
                split_identity=None if merging else absorbed,
                merge_identities=(change.identity_id, absorbed) if merging else None,
            )
        if reverted:
            # Repaint from the assertions that stand now, exactly as reversing a naming does.
            # A restored identity holds no assertion, so it projects as a nameless speaker.
            self._reproject_identity(change.identity_id)
            if not merging:
                self._reproject_identity(absorbed)
        return reverted

    def _reproject_identity(self, identity_id: str) -> None:
        """Repaint one identity's name and its indexed text from the assertion now current."""
        with self._lifecycle.operation() as operation:
            name, _relationship = self._identities.projected_identity(identity_id)
            memories, embeddings = self._identities.reindex_speech(
                identity_id,
                speaker_name=name,
                operation=operation,
            )
            with translate_storage_errors("refresh an identity projection"):
                self._store.identities.refresh_identity_projection(
                    identity_id,
                    memories=memories,
                    embeddings=embeddings,
                )
            self._projection.drain()

    def operations(self, *, limit: int = 100) -> tuple[MemoryOperationRecord, ...]:
        with self._trace("mindbridge.operations", kind="operation"), self._lifecycle.operation():
            validate_limit(limit, maximum=100)
            with translate_storage_errors("list memory operations"):
                logged = self._store.control.read_operations(limit=limit)
            return tuple(operation_record(row) for row in logged)

    def _consolidation_evidence(
        self,
        evidence_ids: Sequence[str] | None,
        query: ContentInput | None,
        *,
        limit: int,
        operation: OperationAssets,
    ) -> tuple[dict[str, MemoryRecord], frozenset[str]]:
        """Resolve what the backend may cite and, separately, what it may act on.

        The first value is the shown set: active, visible records the backend sees and is allowed
        to cite as evidence. The second is the window its `target_ids` are bounded by. They differ
        only for explicit `evidence_ids`, because a hidden derived record -- an inferred `TRAIT`
        below the visibility threshold, or a claim a `CORRECT` already retired -- is exactly what
        `REINFORCE` and `CORRECT` exist for and can never appear in the shown set. Naming it is
        the host's decision; a `query` or the default window never widens the window itself.
        """
        requested: tuple[str, ...] = ()
        if evidence_ids is not None:
            if isinstance(evidence_ids, (str, bytes)):
                raise ValidationError("evidence_ids must be a sequence of memory IDs")
            try:
                candidates = tuple(
                    dict.fromkeys(
                        validated_identifier(memory_id, "evidence_id") for memory_id in evidence_ids
                    )
                )
            except TypeError:
                raise ValidationError("evidence_ids must be a sequence of memory IDs") from None
            requested = candidates
        elif query is not None:
            prepared = self._materializer.prepare(query, operation)
            reference = datetime.now(timezone.utc)
            outcome = self._retrieval.search_prepared(
                prepared,
                limit=limit,
                operation=operation,
                memory_types=None,
                reference_at=reference,
                temporal_range=None,
                occurred_from=None,
                occurred_until=None,
                scope=None,
                require_unambiguous=False,
                capture_trace=False,
            )
            candidates = tuple(hit.id for hit in outcome.hits)
        else:
            # ponytail: over-fetch and filter rather than teach the keyset query about
            # visibility. Raise the factor only if a store of mostly forgotten records
            # measurably under-fills the window.
            with translate_storage_errors("list consolidation evidence"):
                newest = self._store.records.list_memories(limit=min(1_000, limit * 4))
            candidates = tuple(memory.memory_id for memory in newest)
        if not candidates:
            return {}, frozenset()
        with translate_storage_errors("read consolidation evidence"):
            memories = self._store.records.read_memories(candidates, active_only=True)
        memories = memories[:limit]
        self._lifecycle.lease_assets(
            tuple(asset for memory in memories for asset in memory.assets),
            operation.leased,
        )
        shown = {memory.memory_id: self._hydrator.memory_record(memory) for memory in memories}
        return shown, frozenset(shown) | frozenset(requested)

    def _propose_operations(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        assert self._backends.consolidator is not None
        try:
            with self._model_trace(
                "consolidation",
                "consolidate",
                model=self._backends.consolidation_model,
                batch_size=len(evidence),
                modalities=(record.modality for record in evidence),
            ):
                mark_model_requests(1)
                proposals = self._backends.consolidator.consolidate(evidence, trigger=trigger)
        except MindBridgeError:
            raise
        except Exception as error:
            raise ModelError(
                "memory consolidation failed",
                reason="model_failed",
                stage="consolidate",
            ) from error
        if not isinstance(proposals, tuple) or any(
            not isinstance(value, MemoryOperation) for value in proposals
        ):
            raise ModelError(
                "consolidation backend returned an invalid batch",
                reason="response_invalid",
                stage="consolidate",
            )
        return proposals

    def _apply_memory_operation(
        self,
        operation: MemoryOperation,
        *,
        trigger: MemoryTrigger,
        model_id: str | None,
        recipe: str | None,
        shown: Mapping[str, MemoryRecord] | None,
        window: frozenset[str] | None,
        assets: OperationAssets,
        # True only on `apply()`: the host authored this operation itself. A model-proposed
        # kind the kernel refuses on provenance -- a `RESPONSE_POLICY` -- turns on this flag.
        host_authored: bool = False,
    ) -> MemoryOperationRecord:
        if operation.intent is MemoryIntent.MERGE:
            # A cross-modal merge is committed by the kernel from corroboration evidence it
            # counted itself, never from a proposal: an agent-facing model must not be able to
            # fuse two people by asking. Refused here rather than in the value type, so a
            # backend that proposes one is reported instead of raising through the pass.
            raise RejectedOperation("unauthorized")
        if operation.intent is MemoryIntent.CONSENT:
            # The mirror of the rule above, for the same reason and by the same route: consent
            # is a statement its subject makes, so a model that could propose one could
            # manufacture permission to process a person. `record_consent()` is the only path.
            raise RejectedOperation("unauthorized")
        key = operation_key(operation, recipe=recipe)
        with translate_storage_errors("check a memory operation"):
            # Raise the rejection outside this block: `RejectedOperation` is not a
            # `MindBridgeError`, so the translator would turn it into a StorageError.
            duplicate = bool(self._store.control.read_operations(operation_key=key))
        if duplicate:
            raise RejectedOperation("duplicate")
        # A name is not an inference to be corrected, reinforced, or quietly forgotten: the
        # registry and the indexed speech text both project it, and only `rollback()` of the
        # operation that asserted it recomputes those. Retiring the assertion behind their back
        # leaves `identity()` and the stored transcripts answering to a name nothing asserts.
        # Checked for every intent here rather than per intent, because consolidation forgetting
        # names targets of its own and would otherwise walk straight past this.
        # A consent statement is somebody's own word about how they may be processed. Letting
        # the control plane retire, forget, or reinforce one would let a model change what a
        # person permitted; only `record_consent()` supersedes it and only `rollback()` retracts.
        # Checked here for the same reason as the rule above.
        targets = self._control_records(operation.target_ids)
        if any(is_bound_naming_assertion(targets, target) for target in operation.target_ids):
            raise RejectedOperation("naming_assertion")
        if any(is_consent_assertion(targets, target) for target in operation.target_ids):
            raise RejectedOperation("consent_assertion")
        pending = StoredOperation(
            operation_key=key,
            intent=operation.intent.value,
            trigger=trigger.value,
            model_id=model_id,
            recipe=recipe,
            operation_json=dump_operation(operation),
            applied_at=datetime.now(timezone.utc),
        )
        try:
            if operation.intent is MemoryIntent.CONSOLIDATE:
                return operation_record(
                    self._apply_consolidation(
                        operation,
                        pending,
                        shown=shown,
                        assets=assets,
                        host_authored=host_authored,
                    )
                )
            if operation.intent is MemoryIntent.IDENTIFY:
                return operation_record(
                    self._apply_identification(operation, pending, shown=shown, assets=assets)
                )
            return operation_record(
                self._apply_operation_effects(
                    operation,
                    pending,
                    targets=targets,
                    shown=shown,
                    window=window,
                )
            )
        except StaleOperationError:
            # The proposal was built on records the apply transaction found had moved. Nothing
            # was written; the in-transaction re-check is the whole guarantee, so there is no
            # expected-revision token to carry and no partial effect to undo.
            raise RejectedOperation("stale") from None

    def _apply_operation_effects(
        self,
        operation: MemoryOperation,
        pending: StoredOperation,
        *,
        targets: Mapping[str, StoredMemory],
        shown: Mapping[str, MemoryRecord] | None,
        window: frozenset[str] | None,
    ) -> StoredOperation:
        """Validate a REINFORCE, CORRECT, or FORGET proposal and commit its effect.

        Every named target must pass, or none of them is applied. A multi-target operation used
        to execute the eligible subset while its log row still listed the rest, so the log
        claimed effects that never happened; the whole proposal is now refused instead.
        """
        if set(targets) != set(operation.target_ids):
            raise RejectedOperation("unknown_target")
        # A backend may only act inside the window the kernel gathered for it. Consolidation
        # already gets this through `target_ids <= evidence_ids <= shown`; the three direct
        # intents need it stated, or a backend shown record A could retire or correct an
        # unrelated record B it never saw. `window` is None only for the host's own `forget()`,
        # where the host names the IDs and is the authority.
        if window is not None and not set(operation.target_ids) <= window:
            raise RejectedOperation("target_not_shown")
        reinforce: tuple[tuple[str, str], ...] = ()
        correct_ids: tuple[str, ...] = ()
        forget_ids: tuple[str, ...] = ()
        if operation.intent is MemoryIntent.REINFORCE:
            reinforce = self._reinforcement_pairs(operation, targets, shown=shown)
        elif operation.intent is MemoryIntent.CORRECT:
            if not all(is_derived(targets, memory_id) for memory_id in operation.target_ids):
                raise RejectedOperation("not_derived")
            correct_ids = operation.target_ids
        else:
            if any(targets[memory_id].forgotten_at is not None for memory_id in targets):
                raise RejectedOperation("already_forgotten")
            forget_ids = operation.target_ids
        with self._write_lock:
            with translate_storage_errors("apply a memory operation"):
                logged = self._store.control.apply_control_operation(
                    pending,
                    reinforce=reinforce,
                    correct_ids=correct_ids,
                    forget_ids=forget_ids,
                    # Re-checked inside the apply transaction, because everything above was read
                    # before it opened. A REINFORCE target must additionally still stand: a
                    # CORRECT between validation and here makes the proposal stale. `forget()`
                    # deliberately gets no such check -- forgetting a corrected record is legal.
                    require_active=(*operation.target_ids, *operation.evidence_ids),
                    require_unretired=(
                        operation.target_ids if operation.intent is MemoryIntent.REINFORCE else ()
                    ),
                )
            if logged is None:
                raise RejectedOperation("duplicate")
            self._projection.drain()
        return logged

    def _reinforcement_pairs(
        self,
        operation: MemoryOperation,
        targets: Mapping[str, StoredMemory],
        *,
        shown: Mapping[str, MemoryRecord] | None,
    ) -> tuple[tuple[str, str], ...]:
        target = operation.target_ids[0]
        record = targets[target]
        # A retired claim is not a standing derived claim: reinforcing one would attach fresh
        # independent support to a version a CORRECT already withdrew, in this pass or an
        # earlier one. `_control_records` reads the current version, so both cases land here.
        if not is_derived(targets, target) or (
            record.context is not None and record.context.retired_at is not None
        ):
            raise RejectedOperation("not_derived")
        # Self-citation first: a record that is its own evidence is malformed whatever set it
        # came from, and a hidden target is never in the shown set anyway.
        if target in operation.evidence_ids:
            raise RejectedOperation("target_is_evidence")
        if shown is not None and not set(operation.evidence_ids) <= set(shown):
            raise RejectedOperation("evidence_not_shown")
        if set(self._control_records(operation.evidence_ids)) != set(operation.evidence_ids):
            raise RejectedOperation("unknown_evidence")
        context = record.context
        linked = frozenset(() if context is None else context.evidence_ids)
        # Every cited source must be new. Reinforcement is a claim about independent support, so
        # a proposal that miscounts what already supports the record is refused, not trimmed.
        if any(source in linked for source in operation.evidence_ids):
            raise RejectedOperation("already_linked")
        return tuple((target, source) for source in operation.evidence_ids)

    def _apply_consolidation(
        self,
        operation: MemoryOperation,
        pending: StoredOperation,
        *,
        shown: Mapping[str, MemoryRecord] | None,
        assets: OperationAssets,
        host_authored: bool,
    ) -> StoredOperation:
        """Validate one consolidation proposal and commit it through the formation path."""
        if shown is None or not set(operation.evidence_ids) <= set(shown):
            raise RejectedOperation("evidence_not_shown")
        # Consolidation forgetting retires sources *this* derived record replaces, so a target
        # outside its own evidence has no lineage relationship to it and is refused. Retiring a
        # record no derived memory now covers is a FORGET, not a consolidation.
        if not set(operation.target_ids) <= set(operation.evidence_ids):
            raise RejectedOperation("target_not_evidence")
        proposal = operation.proposal
        assert proposal is not None
        sources = tuple(shown[memory_id] for memory_id in operation.evidence_ids)
        primary = consolidation_primary(proposal, sources, host_authored=host_authored)
        if primary is None:
            raise RejectedOperation("invalid_proposal")
        # One operation, one transaction time: the derived record, its evidence links, and any
        # consolidation forgetting all carry the timestamp the log row reports.
        now = pending.applied_at
        context = replace(
            formation_context(
                primary,
                proposal,
                model_id=self._backends.consolidation_model,
                recipe=self._backends.consolidation_recipe,
                recorded_at=now,
                identity_id=self._formation.bound_identity(proposal),
            ),
            evidence_ids=tuple(sorted(source.id for source in sources)),
        )
        place_id, metadata = agreed_inheritance(sources)
        prepared = replace(
            prepare_memory(
                self._materializer.prepare(proposal.content, assets),
                occurred_at=primary.occurred_at,
                occurred_end=primary.occurred_end,
                metadata=metadata,
                memory_type=formation_memory_type(proposal.kind),
            ),
            memory_id=formation_memory_id(
                "\n".join(sorted(source.id for source in sources)),
                proposal,
                recipe=self._backends.consolidation_recipe,
                context=context,
            ),
            context=context,
            place_id=place_id,
        )
        # A derived ID that ignores its episode source -- a RELATION, or a model-inferred TRAIT
        # -- is a function of the proposal alone, so re-proposing a claim that already stands
        # while citing that claim mints the cited record's own ID. Linking a record to itself is
        # malformed evidence whatever set it came from, exactly as it is for a REINFORCE, and it
        # is refused here rather than left to fail the storage constraint and take the whole
        # pass down.
        if prepared.memory_id in operation.evidence_ids:
            raise RejectedOperation("target_is_evidence")
        logged = self._formation.commit(
            tuple((prepared, source.id, proposal.confidence) for source in sources),
            (),
            completed_at=now,
            recipe=self._backends.consolidation_recipe,
            operation=pending,
            forget_ids=operation.target_ids,
            # `shown` was read before this transaction opened; re-check inside it that every
            # cited source still exists, is still active, and -- when it carries typed semantics
            # -- still stands, so a source corrected in between makes the proposal stale.
            require_active=operation.evidence_ids,
            require_unretired=operation.evidence_ids,
            # One consolidation operation presents one derived assertion over the complete cited
            # set. Treating those citations as singleton alternatives lets a summary that used A
            # and B survive with all of its prose after A is withdrawn. A later operation may add
            # a different complete set as another clause; REINFORCE remains the explicit way to
            # add independent singleton support.
            joint_evidence_clauses=True,
        )
        if logged is None:
            raise RejectedOperation("duplicate")
        return logged

    def _apply_identification(
        self,
        operation: MemoryOperation,
        pending: StoredOperation,
        *,
        shown: Mapping[str, MemoryRecord] | None,
        assets: OperationAssets,
    ) -> StoredOperation:
        """Validate one agent-proposed naming and commit it as a bound ENTITY assertion.

        The kernel builds the proposal from the claim, so a backend only names the identity and
        cites its evidence. `MODEL_INFERENCE` basis keeps the assertion, and with it the
        projected name, hidden until two independent evidence groups support it, exactly as an
        inferred trait is. Committing it recomputes the projection, so the registry and the
        indexed speech text never disagree with what is currently asserted.
        """
        claim = operation.claim
        assert claim is not None
        # The same window vocabulary the three direct intents use: a backend may only act on
        # what the kernel gathered for it, and citing outside that window is `target_not_shown`.
        if shown is None or not set(operation.evidence_ids) <= set(shown):
            raise RejectedOperation("target_not_shown")
        with translate_storage_errors("read identity memories"):
            identity_id = self._store.identities.resolve_identity_id(claim.identity_id)
            if (
                identity_id is not None
                and self._store.identities.identity_memory_ids(identity_id) is None
            ):
                identity_id = None
        if identity_id is None:
            raise RejectedOperation("unknown_identity")
        if set(self._control_records(operation.evidence_ids)) != set(operation.evidence_ids):
            raise RejectedOperation("unknown_evidence")
        with translate_storage_errors("read identity evidence"):
            involved = self._store.identities.identity_evidence_memory_ids(
                identity_id,
                operation.evidence_ids,
            )
        # A name may only be pinned on somebody the cited evidence actually contains, so an
        # agent cannot rename a person from a clip that person was never in. An empty citation
        # fails here too: nothing it cites involves them.
        if not involved:
            raise RejectedOperation("identity_not_in_evidence")
        proposal = naming_proposal(
            claim.name,
            claim.relationship,
            basis=EvidenceBasis.MODEL_INFERENCE,
        )
        sources = tuple(shown[memory_id] for memory_id in operation.evidence_ids)
        now = datetime.now(timezone.utc)
        context = replace(
            formation_context(
                None,
                proposal,
                model_id=self._backends.consolidation_model,
                recipe=self._backends.consolidation_recipe,
                recorded_at=now,
                identity_id=identity_id,
            ),
            evidence_ids=tuple(sorted(source.id for source in sources)),
        )
        prepared = replace(
            prepare_memory(
                self._materializer.prepare(proposal.content, assets),
                occurred_at=None,
                occurred_end=None,
                # A name is not observed anywhere: who somebody is stays true in the next room,
                # so the assertion inherits neither the place nor the tags of the clips they were
                # recognized in, exactly as `register_identity` asserts it from no evidence.
                metadata=None,
                memory_type=formation_memory_type(proposal.kind),
            ),
            memory_id=formation_memory_id(
                identity_id,
                proposal,
                recipe=self._backends.consolidation_recipe,
                context=context,
            ),
            context=context,
        )

        def projection() -> tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]]:
            with translate_storage_errors("preview an identity projection"):
                projected, _relationship = self._store.identities.naming_projection_after_assertion(
                    identity_id,
                    prepared.memory_id,
                    context,
                )
            return self._identities.reindex_speech(
                identity_id,
                speaker_name=projected,
                operation=assets,
            )

        # Hold the lock while `_commit_formation` prepares the assertion and projection vectors
        # and commits both, so no concurrent naming write can invalidate the preview.
        with self._write_lock:
            logged = self._formation.commit(
                tuple((prepared, source.id, proposal.confidence) for source in sources),
                (),
                completed_at=now,
                recipe=self._backends.consolidation_recipe,
                operation=pending,
                # Re-checked inside the apply transaction, because everything above was read
                # before it opened: cited evidence a CORRECT retired in between makes the naming
                # stale, and the dispatch turns that into a `stale` rejection.
                require_active=operation.evidence_ids,
                require_unretired=operation.evidence_ids,
                projection_identity_id=identity_id,
                projection_factory=projection,
            )
        if logged is None:
            raise RejectedOperation("duplicate")
        return logged

    def _control_records(self, memory_ids: Sequence[str]) -> dict[str, StoredMemory]:
        if not memory_ids:
            return {}
        with translate_storage_errors("read control-plane memories"):
            memories = self._store.records.read_memories(tuple(memory_ids))
        return {memory.memory_id: memory for memory in memories}
