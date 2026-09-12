"""Shared runtime references and boundary error translation for the kernel planes."""

from __future__ import annotations

import errno
import shutil
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Protocol, cast

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
from mindbridge.kernel.contracts import STORE_METADATA_KEYS, Backends, known_metadata_upgrade
from mindbridge.kernel.settings import Settings


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


def ensure_store_metadata(
    store: LocalStore,
    backends: Backends,
    settings: Settings,
    index_path: Path,
) -> tuple[bool, bool]:
    """Reconcile the store's markers with the wired backends and settings.

    Returns `(rebuild_index, rebuild_embeddings)`: whether the Zvec collection must be rebuilt and
    whether every memory must be re-embedded first. A marker the store has never recorded is
    written; a known upgrade is accepted; anything else is a mismatch the caller must not paper
    over.
    """
    expected = {
        STORE_METADATA_KEYS["model"]: backends.embedding_model,
        STORE_METADATA_KEYS["space"]: backends.space_id,
        STORE_METADATA_KEYS["transcription"]: backends.transcription_space,
        STORE_METADATA_KEYS["dimension"]: str(backends.embedding_dimension),
        STORE_METADATA_KEYS["index"]: settings.index_recipe,
    }
    if backends.face_analyzer is not None:
        expected[STORE_METADATA_KEYS["face"]] = backends.face_space
        expected[STORE_METADATA_KEYS["face_analysis"]] = backends.face_analysis_space
    rebuild_index = False
    rebuild_embeddings = False
    legacy_embedding_spaces = cast(
        frozenset[str],
        getattr(backends.embedder, "_legacy_embedding_spaces", frozenset()),
    )
    with translate_storage_errors("validate local store metadata"):
        for key, value in expected.items():
            stored = store.get_metadata(key)
            if stored is None:
                if key == STORE_METADATA_KEYS["index"] and index_path.exists():
                    rebuild_index = True
                    rebuild_embeddings = True
                else:
                    store.set_metadata(key, value)
            elif stored == value:
                continue
            elif (
                requires_reembedding := known_metadata_upgrade(
                    key,
                    stored,
                    legacy_embedding_spaces,
                )
            ) is not None:
                rebuild_index = True
                rebuild_embeddings = rebuild_embeddings or requires_reembedding
            else:
                raise StorageError(
                    f"local store metadata mismatch for {key}: expected {value!r}, found {stored!r}"
                )
        if rebuild_index:
            if index_path.exists():
                shutil.rmtree(index_path)
            if not rebuild_embeddings:
                store.set_metadata(STORE_METADATA_KEYS["index"], settings.index_recipe)
    return rebuild_index, rebuild_embeddings
