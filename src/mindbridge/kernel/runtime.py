"""Shared runtime references and boundary error translation for the kernel planes."""

from __future__ import annotations

import errno
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Protocol

from mindbridge.exceptions import IndexUnavailableError, MindBridgeError, StorageError
from mindbridge.infrastructure.local._lock import DataDirectoryInUseError
from mindbridge.infrastructure.local.assets import AssetStore
from mindbridge.infrastructure.local.store import (
    IndexDocument,
    LocalStore,
    StaleOperationError,
    UnsupportedSchemaError,
)
from mindbridge.infrastructure.local.zvec_index import IndexHit


class Index(Protocol):
    def upsert(self, documents: Sequence[IndexDocument]) -> None: ...

    def delete(self, ids: Sequence[str]) -> None: ...

    def search(
        self,
        values: Sequence[float],
        *,
        limit: int = 10,
        space_id: str | None = None,
        task: str | None = None,
        memory_type: str | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
        ef: int | None = None,
        exact: bool = False,
    ) -> tuple[IndexHit, ...]: ...

    def lexical_search(
        self,
        text: str,
        *,
        limit: int = 10,
        space_id: str | None = None,
        task: str | None = None,
        memory_type: str | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
    ) -> tuple[IndexHit, ...]: ...

    def flush(self) -> None: ...

    def optimize(self, *, concurrency: int = 0) -> None: ...

    def optimize_if_needed(self, *, minimum_unindexed: int = 100_000) -> bool: ...

    def rebuild(
        self,
        documents: Iterable[IndexDocument],
        *,
        batch_size: int = 1_024,
        optimize_concurrency: int = 0,
    ) -> int: ...

    def close(self) -> None: ...


def open_store(data_dir: Path) -> LocalStore:
    try:
        return LocalStore(data_dir)
    except DataDirectoryInUseError as error:
        # The path is the caller's own configuration, but it is server state to every transport,
        # so it travels in `subject` instead of the message every surface forwards.
        raise StorageError(
            "the data directory is already in use by another live MindBridge instance",
            reason="data_dir_in_use",
            stage="open",
            subject=str(data_dir),
        ) from error
    except UnsupportedSchemaError as error:
        raise StorageError(str(error), reason="schema_unsupported", stage="open") from error
    except Exception as error:
        raise StorageError(
            "failed to open the local memory store", reason="io_failed", stage="open"
        ) from error


@contextmanager
def translate_storage_errors(action: str) -> Iterator[None]:
    try:
        yield
    # A stale control-plane precondition is kernel policy, not an IO failure: the two apply call
    # sites turn it into a `"stale"` rejection, and no other store method raises it.
    except (MindBridgeError, StaleOperationError):
        raise
    except Exception as error:
        # `io_failed` is this wrapper's pre-existing contract and a test pins it. It is coarse by
        # design -- the wrapper catches whatever any storage action raised -- and deliberately not
        # in `RETRYABLE_REASONS`, so a caller does not retry a failure it cannot know is transient.
        raise StorageError(f"failed to {action}", reason="io_failed") from error


@contextmanager
def translate_index_errors(action: str) -> Iterator[None]:
    try:
        yield
    except MindBridgeError:
        raise
    except Exception as error:
        detail = ""
        if isinstance(error, OSError) and error.errno == errno.EMFILE:
            detail = f": {error.strerror or 'too many open files'}"
        raise IndexUnavailableError(f"failed to {action}{detail}") from error


@dataclass(frozen=True, slots=True)
class Storage:
    """The authoritative store, the media store, and the process-local locks over them.

    SQLite serializes its own writers; these locks order what SQLite cannot see. `write_lock`
    protects outbox application, destructive operations, index replacement, final record and
    asset hydration, and add-time speaker identity updates, whose identity changes must roll
    back with a failed add. `formation_lock` serializes the formation backend's proposals per
    source. `settle_lock` keeps two settlers from running the model stages over one queued row.
    """

    store: LocalStore
    assets: AssetStore
    write_lock: RLock
    formation_lock: RLock
    settle_lock: RLock
