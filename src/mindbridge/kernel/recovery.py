"""Startup recovery of store markers, canonical embeddings, and the disposable index."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from mindbridge.kernel.contracts import Backends
from mindbridge.kernel.embedding import Embedding
from mindbridge.kernel.metadata import (
    EVIDENCE_RECIPE_KEY,
    MIGRATION_TARGET_KEY,
    STORE_METADATA_KEYS,
    plan_metadata,
)
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.settings import Settings

_LOGGER = logging.getLogger(__name__)


class Recovery:
    """Reconcile one exclusively owned directory before opening its search index."""

    def __init__(
        self, *, storage: Storage, backends: Backends, settings: Settings, embedding: Embedding
    ) -> None:
        self._store = storage.store
        self._backends = backends
        self._settings = settings
        self._embedding = embedding

    def prepare(self, index_path: Path) -> None:
        expected = {
            STORE_METADATA_KEYS["model"]: self._backends.embedding_model,
            STORE_METADATA_KEYS["space"]: self._backends.space_id,
            STORE_METADATA_KEYS["transcription"]: self._backends.transcription_space,
            STORE_METADATA_KEYS["dimension"]: str(self._backends.embedding_dimension),
            STORE_METADATA_KEYS["index"]: self._settings.index_recipe,
            EVIDENCE_RECIPE_KEY: (
                "independent-v1" if self._settings.independent_evidence else "legacy-v1"
            ),
        }
        if self._backends.face_analyzer is not None:
            expected[STORE_METADATA_KEYS["face"]] = self._backends.face_space
            expected[STORE_METADATA_KEYS["face_analysis"]] = self._backends.face_analysis_space
        missing = not index_path.exists()
        with translate_storage_errors("validate local store metadata"):
            stored = {
                key: self._store.get_metadata(key) for key in (*expected, MIGRATION_TARGET_KEY)
            }
            plan = plan_metadata(
                expected,
                stored,
                has_records=(
                    stored[EVIDENCE_RECIPE_KEY] is None
                    and bool(self._store.records.list_memories(limit=1))
                ),
                index_exists=not missing,
                legacy_embedding_spaces=self._backends.legacy_embedding_spaces,
            )
            if plan.rebuild_embeddings:
                # Each model batch commits independently. Persist its target first so a failed
                # batch can never make mixed vectors appear to belong to the previous space.
                self._store.update_metadata(
                    {**plan.updates, MIGRATION_TARGET_KEY: plan.migration_target}
                )
        if plan.rebuild_embeddings:
            _LOGGER.warning("re-embedding stored memories for space %s", self._backends.space_id)
            self._embedding.reembed_memories()
        with translate_storage_errors("reconcile local store metadata"):
            if plan.rebuild_index and index_path.exists():
                shutil.rmtree(index_path)
            updates: dict[str, str | None] = dict(plan.updates)
            if plan.rebuild_index:
                updates[STORE_METADATA_KEYS["index"]] = self._settings.index_recipe
            if plan.rebuild_embeddings:
                updates[STORE_METADATA_KEYS["space"]] = self._backends.space_id
                updates[MIGRATION_TARGET_KEY] = None
            if updates:
                self._store.update_metadata(updates)
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
