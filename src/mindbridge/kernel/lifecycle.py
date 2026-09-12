"""Operation leases, ownership, and media lease cleanup for one open memory."""

from __future__ import annotations

import builtins
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition

from mindbridge.exceptions import StorageError
from mindbridge.infrastructure.local.assets import AssetStoreError
from mindbridge.infrastructure.local.store import SpeechRollback, StoredAsset
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.models.base import SpeechAnalysis
from mindbridge.types import FaceObservation, SpeakerSegment


@dataclass(slots=True)
class OperationAssets:
    leased: builtins.list[StoredAsset]
    cleanup: builtins.list[StoredAsset]
    persisted: set[str]
    transcripts: dict[str, str]
    transcript_updates: dict[str, str]
    speech_updates: dict[str, SpeechAnalysis]
    speech_segments: dict[str, tuple[SpeakerSegment, ...]]
    speech_rollbacks: builtins.list[SpeechRollback]
    face_observations: dict[str, tuple[FaceObservation, ...]]
    descriptions: dict[str, str]
    # Identity ID -> the name a caption's facts stated for that diarised speaker. Staged rather
    # than applied on the spot: the naming assertion reindexes every memory that mentions the
    # person, and this write's own memory does not exist until it commits.
    speaker_names: dict[str, str]
    # A naming fact that named nobody: a label no recognizer in this clip produced, an identity
    # that no longer exists by binding time, or a name that failed validation. Counted rather
    # than only logged, alongside the refusals `_bind_speaker_names` already counts for a name
    # that conflicts with one a person already carries.
    speaker_names_refused: int = 0


class Lifecycle:
    """Operation leases and the media leases they hold, plus ownership and close coordination."""

    def __init__(
        self,
        *,
        storage: Storage,
    ) -> None:
        self._assets = storage.assets
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._owner_pid = os.getpid()
        self._lifecycle = Condition()
        self._active_operations = 0
        self._closing = False
        self._closed = True
        self._pending_asset_cleanup: dict[str, StoredAsset] = {}

    def mark_open(self) -> None:
        """Admit operations; the owner calls this once every resource is open."""
        with self._lifecycle:
            self._closed = False

    def mark_closed(self) -> None:
        with self._lifecycle:
            self._closed = True
            self._closing = False
            self._lifecycle.notify_all()

    def begin_close(self) -> bool:
        """Reject new work and wait for active operations; False when already closed."""
        self.require_owner_process()
        with self._lifecycle:
            while self._closing:
                self._lifecycle.wait()
            if self._closed:
                return False
            self._closing = True
            while self._active_operations:
                self._lifecycle.wait()
        return True

    @contextmanager
    def operation(self) -> Iterator[OperationAssets]:
        self.require_owner_process()
        with self._lifecycle:
            if self._closed or self._closing:
                raise StorageError("Memory is closed", reason="instance_unusable")
            self._active_operations += 1
        failed = False
        assets = OperationAssets(
            leased=[],
            cleanup=[],
            persisted=set(),
            transcripts={},
            transcript_updates={},
            speech_updates={},
            speech_segments={},
            speech_rollbacks=[],
            face_observations={},
            descriptions={},
            speaker_names={},
        )
        try:
            yield assets
        except BaseException:
            failed = True
            raise
        finally:
            cleanup_error: BaseException | None = None
            self.queue_asset_cleanup(assets.cleanup)
            try:
                self._assets.release(assets.leased)
                with self._lifecycle:
                    needs_cleanup = bool(self._pending_asset_cleanup)
                if needs_cleanup:
                    with self._write_lock:
                        self.cleanup_pending_assets()
            except BaseException as error:
                cleanup_error = error
            finally:
                with self._lifecycle:
                    self._active_operations -= 1
                    self._lifecycle.notify_all()
            if cleanup_error is not None and not failed:
                raise cleanup_error

    def require_owner_process(self) -> None:
        if os.getpid() != self._owner_pid:
            raise StorageError(
                "Memory cannot be used after fork; create a new instance with a different data_dir",
                reason="instance_unusable",
            )

    def require_open(self) -> None:
        self.require_owner_process()
        if self._closed or self._closing:
            raise StorageError("Memory is closed", reason="instance_unusable")

    def lease_assets(
        self,
        assets: Sequence[StoredAsset],
        leased: builtins.list[StoredAsset],
    ) -> None:
        unique = tuple({asset.asset_id: asset for asset in assets}.values())
        if not unique:
            return
        try:
            self._assets.acquire(unique)
        except AssetStoreError as error:
            raise StorageError("failed to lease local media", reason="io_failed") from error
        leased.extend(unique)

    def queue_asset_cleanup(self, assets: Sequence[StoredAsset]) -> None:
        if not assets:
            return
        with self._lifecycle:
            for asset in assets:
                self._pending_asset_cleanup.setdefault(asset.asset_id, asset)

    def cleanup_pending_assets(self) -> None:
        with self._lifecycle:
            assets = tuple(self._pending_asset_cleanup.values())
            for asset in assets:
                self._pending_asset_cleanup.pop(asset.asset_id, None)
        if not assets:
            return
        remaining = 0
        try:
            asset_ids = tuple(asset.asset_id for asset in assets)
            with translate_storage_errors("check temporary media ownership"):
                persisted_assets = self._store.media.read_assets(asset_ids)
                unreferenced_assets = self._store.media.read_unreferenced_assets(asset_ids)
            persisted = {asset.asset_id for asset in persisted_assets}
            unreferenced = {asset.asset_id: asset for asset in unreferenced_assets}
            for index, asset in enumerate(assets):
                remaining = index
                asset_id = asset.asset_id
                if asset_id in persisted and asset_id not in unreferenced:
                    remaining = index + 1
                    continue
                deleted = self._assets.delete_if_unleased(unreferenced.get(asset_id, asset))
                if not deleted:
                    self.queue_asset_cleanup((asset,))
                    remaining = index + 1
                    continue
                if asset_id in unreferenced:
                    with translate_storage_errors("delete orphaned media metadata"):
                        if not self._store.media.delete_asset_if_unreferenced(asset_id):
                            raise StorageError(
                                "orphaned media became referenced during cleanup",
                                reason="io_failed",
                            )
                remaining = index + 1
        except AssetStoreError as error:
            self.queue_asset_cleanup(assets[remaining:])
            raise StorageError("failed to clean up orphaned media", reason="io_failed") from error
        except BaseException:
            self.queue_asset_cleanup(assets[remaining:])
            raise

    def collect_orphan_assets(self, *, scan_physical: bool) -> None:
        while True:
            with translate_storage_errors("list orphaned media"):
                orphaned = self._store.media.list_unreferenced_assets(limit=256)
            if not orphaned:
                break
            self._delete_orphan_assets(orphaned)
        if not scan_physical:
            return
        try:
            physical_ids = self._assets.list_ids()
        except AssetStoreError as error:
            raise StorageError("failed to scan local media", reason="io_failed") from error
        with translate_storage_errors("reconcile local media"):
            tracked_ids = {asset.asset_id for asset in self._store.media.read_assets(physical_ids)}
        for asset_id in physical_ids:
            if asset_id in tracked_ids:
                continue
            try:
                self._assets.delete_id(asset_id)
            except AssetStoreError as error:
                raise StorageError(
                    "failed to delete untracked local media", reason="io_failed"
                ) from error

    def _delete_orphan_assets(self, orphaned: Sequence[StoredAsset]) -> None:
        for asset in orphaned:
            try:
                self._assets.delete(asset)
            except AssetStoreError as error:
                raise StorageError("failed to delete orphaned media", reason="io_failed") from error
            with translate_storage_errors("delete orphaned media metadata"):
                self._store.media.delete_asset_if_unreferenced(asset.asset_id)
