"""Startup validation of store markers and recovery of the disposable index."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from mindbridge.kernel.contracts import Backends
from mindbridge.kernel.metadata import (
    EVIDENCE_RECIPE_KEY,
    STORE_METADATA_KEYS,
    plan_metadata,
)
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.settings import Settings

_LOGGER = logging.getLogger(__name__)


class Recovery:
    """Reconcile one exclusively owned directory before opening its search index."""

    def __init__(self, *, storage: Storage, backends: Backends, settings: Settings) -> None:
        self._store = storage.store
        self._backends = backends
        self._settings = settings

    def prepare(self, index_path: Path) -> None:
        expected = {
            STORE_METADATA_KEYS["model"]: self._backends.embedding_model,
            STORE_METADATA_KEYS["space"]: self._backends.space_id,
            STORE_METADATA_KEYS["transcription"]: self._backends.transcription_space,
            STORE_METADATA_KEYS["dimension"]: str(self._backends.embedding_dimension),
            STORE_METADATA_KEYS["index"]: self._settings.index_recipe,
            EVIDENCE_RECIPE_KEY: (
                "independent-v1" if self._settings.independent_evidence else "complete-v1"
            ),
        }
        if self._backends.face_analyzer is not None:
            expected[STORE_METADATA_KEYS["face"]] = self._backends.face_space
            expected[STORE_METADATA_KEYS["face_analysis"]] = self._backends.face_analysis_space
        missing = not index_path.exists()
        with translate_storage_errors("validate local store metadata"):
            stored = {key: self._store.get_metadata(key) for key in expected}
            face_keys = frozenset(
                {STORE_METADATA_KEYS["face"], STORE_METADATA_KEYS["face_analysis"]}
            )
            initializing_faces = (
                self._backends.face_analyzer is not None
                and all(stored[key] is None for key in face_keys)
                and not self._store.media.has_face_analyses()
            )
            plan = plan_metadata(
                expected,
                stored,
                has_records=bool(self._store.records.list_memories(limit=1)),
                initializable=face_keys if initializing_faces else frozenset(),
            )
        with translate_storage_errors("reconcile local store metadata"):
            if plan.rebuild_index and index_path.exists():
                shutil.rmtree(index_path)
            if plan.updates:
                self._store.update_metadata(plan.updates)
        if missing or plan.rebuild_index:
            # Checkpoint before Zvec creates a directory. An interrupted collection open must
            # leave enough durable work to finish recovery even though that directory exists.
            with translate_storage_errors("checkpoint a missing search index"):
                queued = self._store.index.queue_all_embeddings()
            if queued:
                _LOGGER.info(
                    "rebuilding the search index for %d embeddings (missing=%s, recipe changed=%s)",
                    queued,
                    missing,
                    plan.rebuild_index,
                )
