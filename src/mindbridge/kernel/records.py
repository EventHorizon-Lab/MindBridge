"""Record-level reads, physical deletion, retention, and subject export."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from opentelemetry.trace import Tracer

from mindbridge.exceptions import IdentityNotFoundError, MemoryNotFoundError, ValidationError
from mindbridge.kernel.control_plane import operation_touches
from mindbridge.kernel.hydration import Hydrator, operation_record
from mindbridge.kernel.identity import Identities
from mindbridge.kernel.lifecycle import Lifecycle
from mindbridge.kernel.projection import Projection
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.settings import Settings
from mindbridge.kernel.tracing import Traced
from mindbridge.kernel.validation import (
    decode_cursor,
    encode_cursor,
    validate_limit,
    validated_identifier,
)
from mindbridge.types import (
    ExportBundle,
    IdentityProfile,
    MemoryOperationRecord,
    MemoryRecord,
    Page,
    RetentionReport,
)

# How much one retention pass may delete. Physical deletion is unrecoverable, so a pass is
# bounded and repeatable rather than unbounded: an operator runs it again until it reports
# nothing, and can stop after any pass.
_RETENTION_PAGE_SIZE = 1_000


class Records(Traced):
    """Record reads, physical deletion, retention, and data-subject export."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        settings: Settings,
        lifecycle: Lifecycle,
        hydrator: Hydrator,
        projection: Projection,
        identities: Identities,
    ) -> None:
        super().__init__(tracer)
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._settings = settings
        self._lifecycle = lifecycle
        self._hydrator = hydrator
        self._projection = projection
        self._identities = identities

    def get(self, memory_id: str) -> MemoryRecord:
        with (
            self._trace("mindbridge.get", kind="operation"),
            self._lifecycle.operation() as assets,
            self._write_lock,
        ):
            normalized_id = validated_identifier(memory_id, "memory_id")
            with translate_storage_errors("read memory"):
                memory = self._store.records.read_memory(normalized_id)
            if memory is None:
                raise MemoryNotFoundError(f"memory does not exist: {normalized_id}")
            self._lifecycle.lease_assets(memory.assets, assets.leased)
            return self._hydrator.memory_record(memory)

    def reinforce(self, memory_ids: Sequence[str]) -> int:
        if isinstance(memory_ids, (str, bytes)):
            raise ValidationError("memory_ids must be a sequence of memory IDs")
        try:
            normalized = tuple(
                dict.fromkeys(
                    validated_identifier(memory_id, "memory_id") for memory_id in memory_ids
                )
            )
        except TypeError:
            raise ValidationError("memory_ids must be a sequence of memory IDs") from None
        if not normalized:
            return 0
        with (
            self._trace("mindbridge.reinforce", kind="operation"),
            self._lifecycle.operation(),
            self._write_lock,
            translate_storage_errors("reinforce memories"),
        ):
            return self._store.records.reinforce_memories(
                normalized,
                accessed_at=datetime.now(timezone.utc),
            )

    def list(self, *, limit: int = 100, cursor: str | None = None) -> Page:
        with (
            self._trace("mindbridge.list", kind="operation"),
            self._lifecycle.operation() as assets,
            self._write_lock,
        ):
            validate_limit(limit, maximum=100)
            after = None if cursor is None else decode_cursor(cursor)
            with translate_storage_errors("list memories"):
                memories = self._store.records.list_memories(limit=limit + 1, after=after)
            has_next = len(memories) > limit
            visible = memories[:limit]
            self._lifecycle.lease_assets(
                tuple(asset for memory in visible for asset in memory.assets),
                assets.leased,
            )
            next_cursor = encode_cursor(visible[-1]) if has_next else None
            return Page(
                items=tuple(self._hydrator.memory_record(memory) for memory in visible),
                next_cursor=next_cursor,
            )

    def delete(self, memory_id: str) -> bool:
        with (
            self._trace("mindbridge.delete", kind="operation"),
            self._lifecycle.operation() as operation,
            self._write_lock,
        ):
            normalized_id = validated_identifier(memory_id, "memory_id")
            memories, embeddings = self._identities.deleted_naming_index(normalized_id, operation)
            with translate_storage_errors("delete memory"):
                deleted, orphaned = self._store.records.delete_memory_with_assets(
                    normalized_id,
                    memories=memories,
                    embeddings=embeddings,
                )
            self._lifecycle.queue_asset_cleanup(orphaned)
            self._projection.drain()
            return deleted

    def export(
        self,
        *,
        identity_id: str | None = None,
        memory_ids: Sequence[str] | None = None,
    ) -> ExportBundle:
        if (identity_id is None) == (memory_ids is None):
            raise ValidationError("export names exactly one of identity_id or memory_ids")
        with (
            self._trace("mindbridge.export", kind="operation"),
            self._lifecycle.operation(),
            self._write_lock,
        ):
            profiles: tuple[IdentityProfile, ...] = ()
            resolved_id: str | None = None
            if identity_id is not None:
                requested_id = validated_identifier(identity_id, "identity_id")
                with translate_storage_errors("read identity memories"):
                    resolved_id = self._store.identities.resolve_identity_id(requested_id)
                    occurrences = (
                        None
                        if resolved_id is None
                        else self._store.identities.identity_memory_ids(resolved_id)
                    )
                    if resolved_id is None or occurrences is None:
                        raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
                    asserted = self._store.identities.identity_assertion_memory_ids(resolved_id)
                    profile = self._store.identities.identity_profile(resolved_id)
                    aliases = self._store.identities.identity_equivalence_class(resolved_id) or (
                        resolved_id,
                    )
                selected = tuple(dict.fromkeys((*occurrences, *asserted)))
                profiles = () if profile is None else (profile,)
            else:
                assert memory_ids is not None
                if isinstance(memory_ids, (str, bytes)):
                    raise ValidationError("memory_ids must be a sequence of memory IDs")
                selected = tuple(
                    dict.fromkeys(validated_identifier(value, "memory_id") for value in memory_ids)
                )
                aliases = ()
            with translate_storage_errors("read exported memories"):
                stored = self._store.records.read_memories(selected)
            records = tuple(
                sorted(
                    (self._hydrator.memory_record(memory) for memory in stored),
                    key=lambda record: (record.created_at, record.id),
                )
            )
            return ExportBundle(
                exported_at=datetime.now(timezone.utc),
                identity_id=resolved_id,
                identities=profiles,
                records=records,
                operations=self._exported_operations(
                    frozenset(record.id for record in records),
                    frozenset(aliases),
                ),
            )

    def _exported_operations(
        self,
        memory_ids: frozenset[str],
        identity_ids: frozenset[str],
    ) -> tuple[MemoryOperationRecord, ...]:
        """Return every logged operation that moved one of these records or people, oldest first.

        The whole log is scanned in pages rather than read with a limit: an export that
        silently stopped at the hundredth row would answer the wrong question.
        """
        found: list[MemoryOperationRecord] = []
        before: int | None = None
        while True:
            with translate_storage_errors("read exported operations"):
                page = self._store.control.read_operations(limit=1_000, before_operation_id=before)
            if not page:
                break
            for row in page:
                record = operation_record(row)
                if operation_touches(record, memory_ids, identity_ids):
                    found.append(record)
            before = page[-1].operation_id
        return tuple(sorted(found, key=lambda record: record.operation_id))

    def apply_retention(self, *, dry_run: bool = False) -> RetentionReport:
        if not isinstance(dry_run, bool):
            raise ValidationError("dry_run must be a boolean")
        policy = self._settings.retention
        now = datetime.now(timezone.utc)
        with (
            self._trace("mindbridge.apply_retention", kind="operation"),
            self._lifecycle.operation(),
            self._write_lock,
        ):
            media_ids: tuple[str, ...] = ()
            aged_assets: tuple[str, ...] = ()
            if policy.media_days is not None:
                with translate_storage_errors("list retention candidates"):
                    candidates = self._store.media.asset_retention_candidates(
                        created_before=now - timedelta(days=policy.media_days),
                        limit=_RETENTION_PAGE_SIZE,
                    )
                    aged_assets = tuple(asset.asset_id for asset in candidates)
                    media_ids = self._store.media.asset_memory_ids(aged_assets)
            forgotten_ids: tuple[str, ...] = ()
            if policy.forgotten_days is not None:
                with translate_storage_errors("list forgotten memories"):
                    forgotten_ids = tuple(
                        memory_id
                        for memory_id in self._store.records.forgotten_memory_ids(
                            forgotten_before=now - timedelta(days=policy.forgotten_days),
                            limit=_RETENTION_PAGE_SIZE,
                        )
                        if memory_id not in set(media_ids)
                    )
            capture_ids: tuple[str, ...] = ()
            if policy.capture_failure_days is not None:
                cutoff = now - timedelta(days=policy.capture_failure_days)
                with translate_storage_errors("list pending captures"):
                    capture_ids = tuple(
                        capture.memory_id
                        for capture in self._store.captures.pending_captures(
                            limit=_RETENTION_PAGE_SIZE
                        )
                        if capture.attempts > 0 and capture.enqueued_at < cutoff
                    )
            direct_ids = (*media_ids, *forgotten_ids)
            with translate_storage_errors("predict retention cascade"):
                deleted_ids = self._store.records.deletion_cascade(direct_ids)
                direct_set = set(direct_ids)
                cascade_ids = tuple(
                    memory_id for memory_id in deleted_ids if memory_id not in direct_set
                )
                orphaned_assets = self._store.records.assets_orphaned_by_deletion(deleted_ids)
            reported_assets = tuple(dict.fromkeys((*aged_assets, *orphaned_assets)))
            if dry_run:
                return RetentionReport(
                    dry_run=True,
                    media_memory_ids=media_ids,
                    forgotten_memory_ids=forgotten_ids,
                    cascade_memory_ids=cascade_ids,
                    asset_ids=reported_assets,
                    capture_memory_ids=capture_ids,
                )
            # Each deletion forces an index flush when its drain runs, so the page is deleted
            # under the stream deferral and drained once: one flush, not one per record.
            with self._projection.deferred(defer=True):
                for memory_id in direct_ids:
                    self.delete(memory_id)
            self._projection.drain(force=True)
            removed: list[str] = []
            with translate_storage_errors("delete retained media"):
                # Whatever the deletions above orphaned is already gone; what is left here is
                # media no memory referenced in the first place, which nothing else collects
                # until the next open.
                self._lifecycle.cleanup_pending_assets()
                for asset_id in reported_assets:
                    if self._store.media.read_asset(asset_id) is None or (
                        self._store.media.delete_asset_if_unreferenced(asset_id)
                    ):
                        removed.append(asset_id)
                if capture_ids:
                    self._store.captures.complete_captures(capture_ids)
            return RetentionReport(
                dry_run=False,
                media_memory_ids=media_ids,
                forgotten_memory_ids=forgotten_ids,
                cascade_memory_ids=cascade_ids,
                asset_ids=tuple(removed),
                capture_memory_ids=capture_ids,
            )
