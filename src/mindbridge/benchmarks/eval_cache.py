"""SQLite response cache compatible with lmms-eval's directory and file path shapes."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CachedAnswer:
    """The stable answer fields evaluation artifacts retain."""

    prediction: str
    confidence: float
    memory_ids: tuple[str, ...]


class ResponseCache:
    """Read a shared cache and write through a run-local SQLite shard."""

    def __init__(self, path: Path, run_id: str, namespace: str) -> None:
        requested = path.expanduser().resolve()
        layered = requested.name == "cache.db" or requested.suffix != ".db"
        directory = requested.parent if requested.name == "cache.db" else requested
        self.root_path = directory / "cache.db" if layered else requested
        self.write_path = directory / "runs" / run_id / "cache.db" if layered else self.root_path
        self.namespace = namespace
        self._layered = layered
        self._closed = False
        self._root = _open(self.root_path)
        self._write = self._root if not layered else _open(self.write_path)

    def get(self, task: str, unit_id: str, question_id: str) -> CachedAnswer | None:
        key = self._key(task, unit_id, question_id)
        row = self._write.execute(
            "SELECT payload FROM responses WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None and self._write is not self._root:
            row = self._root.execute(
                "SELECT payload FROM responses WHERE cache_key = ?", (key,)
            ).fetchone()
        return None if row is None else _answer(str(row[0]))

    def put(self, task: str, unit_id: str, question_id: str, answer: CachedAnswer) -> None:
        payload = json.dumps(
            {
                "prediction": answer.prediction,
                "confidence": answer.confidence,
                "memory_ids": answer.memory_ids,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._write.execute(
            "INSERT OR REPLACE INTO responses(cache_key, payload) VALUES (?, ?)",
            (self._key(task, unit_id, question_id), payload),
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._write is not self._root:
            self._write.close()
            self._root.execute("ATTACH DATABASE ? AS run_cache", (str(self.write_path),))
            self._root.execute("BEGIN IMMEDIATE")
            try:
                self._root.execute(
                    "INSERT OR IGNORE INTO responses(cache_key, payload) "
                    "SELECT cache_key, payload FROM run_cache.responses"
                )
                self._root.execute("COMMIT")
            except BaseException:
                self._root.execute("ROLLBACK")
                raise
            finally:
                self._root.execute("DETACH DATABASE run_cache")
        self._root.close()
        self._closed = True

    def _key(self, task: str, unit_id: str, question_id: str) -> str:
        encoded = json.dumps(
            (self.namespace, task, unit_id, question_id),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _open(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60, isolation_level=None)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS responses ("
        "cache_key TEXT PRIMARY KEY NOT NULL, payload TEXT NOT NULL) WITHOUT ROWID"
    )
    if os.name == "posix":
        os.chmod(path, 0o600)
    return connection


def _answer(payload: str) -> CachedAnswer:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("response cache payload must be an object")
    prediction, confidence, memory_ids = (
        value.get("prediction"),
        value.get("confidence"),
        value.get("memory_ids"),
    )
    if (
        not isinstance(prediction, str)
        or isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not 0 <= float(confidence) <= 1
        or not isinstance(memory_ids, list)
        or any(not isinstance(item, str) for item in memory_ids)
    ):
        raise ValueError("response cache payload is invalid")
    return CachedAnswer(prediction, float(confidence), tuple(memory_ids))
