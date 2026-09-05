"""SQLite response cache compatible with lmms-eval's directory and file path shapes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvidenceInterval:
    """One retrieved benchmark memory and its original source interval."""

    memory_id: str
    source_id: str | None
    start_seconds: float | None
    end_seconds: float | None

    def __post_init__(self) -> None:
        if not self.memory_id:
            raise ValueError("evidence memory_id must not be empty")
        if self.source_id is not None and not self.source_id:
            raise ValueError("evidence source_id must not be empty")
        for value in (self.start_seconds, self.end_seconds):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value < 0
            ):
                raise ValueError("evidence interval values must be finite and non-negative")
        if self.end_seconds is not None and (
            self.start_seconds is None or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("evidence end_seconds must be later than start_seconds")

    def json(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "source_id": self.source_id,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }


@dataclass(frozen=True, slots=True)
class CachedAnswer:
    """The stable answer fields evaluation artifacts retain."""

    prediction: str
    confidence: float
    memory_ids: tuple[str, ...]
    evidence: tuple[EvidenceInterval, ...] = ()
    abstained: bool = False
    abstention_reason: str | None = None
    # None means an older cache entry did not retain the diagnostic. An empty tuple is a
    # successful diagnostic that returned no candidates; those two cases score differently.
    ranked_source_ids: tuple[str, ...] | None = None


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
        self._root = _open(self.root_path, _RESPONSES_DDL)
        self._write = self._root if not layered else _open(self.write_path, _RESPONSES_DDL)

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
        key = self._key(task, unit_id, question_id)
        payload = json.dumps(
            {
                "prediction": answer.prediction,
                "confidence": answer.confidence,
                "memory_ids": answer.memory_ids,
                "evidence": tuple(item.json() for item in answer.evidence),
                "abstained": answer.abstained,
                "abstention_reason": answer.abstention_reason,
                "ranked_source_ids": answer.ranked_source_ids,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._write.execute(
            "INSERT OR REPLACE INTO responses(cache_key, payload) VALUES (?, ?)",
            (key, payload),
        )
        if self._write is not self._root:
            self._root.execute(
                "INSERT OR IGNORE INTO responses(cache_key, payload) VALUES (?, ?)",
                (key, payload),
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


_RESPONSES_DDL = (
    "CREATE TABLE IF NOT EXISTS responses ("
    "cache_key TEXT PRIMARY KEY NOT NULL, payload TEXT NOT NULL) WITHOUT ROWID"
)
_DESCRIPTIONS_DDL = (
    "CREATE TABLE IF NOT EXISTS descriptions ("
    "cache_key TEXT PRIMARY KEY NOT NULL, description TEXT NOT NULL) WITHOUT ROWID"
)


def _open(path: Path, ddl: str, *, check_same_thread: bool = True) -> sqlite3.Connection:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    connection = sqlite3.connect(
        path, timeout=60, isolation_level=None, check_same_thread=check_same_thread
    )
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(ddl)
    if os.name == "posix":
        os.chmod(path, 0o600)
    return connection


def _answer(payload: str) -> CachedAnswer:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("response cache payload must be an object")
    (
        prediction,
        confidence,
        memory_ids,
        evidence,
        abstained,
        abstention_reason,
        ranked_source_ids,
    ) = (
        value.get("prediction"),
        value.get("confidence"),
        value.get("memory_ids"),
        value.get("evidence", []),
        value.get("abstained", False),
        value.get("abstention_reason"),
        value.get("ranked_source_ids"),
    )
    if (
        not isinstance(prediction, str)
        or isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not 0 <= float(confidence) <= 1
        or not isinstance(memory_ids, list)
        or any(not isinstance(item, str) for item in memory_ids)
        or not isinstance(evidence, list)
        or not isinstance(abstained, bool)
        or (abstention_reason is not None and not isinstance(abstention_reason, str))
        or abstained != (abstention_reason is not None)
        or (
            ranked_source_ids is not None
            and (
                not isinstance(ranked_source_ids, list)
                or any(not isinstance(item, str) for item in ranked_source_ids)
            )
        )
    ):
        raise ValueError("response cache payload is invalid")
    try:
        intervals = tuple(_evidence(item) for item in evidence)
    except (TypeError, ValueError):
        raise ValueError("response cache payload is invalid") from None
    return CachedAnswer(
        prediction,
        float(confidence),
        tuple(memory_ids),
        intervals,
        abstained,
        abstention_reason,
        None if ranked_source_ids is None else tuple(ranked_source_ids),
    )


def _evidence(value: object) -> EvidenceInterval:
    if not isinstance(value, dict) or set(value) != {
        "memory_id",
        "source_id",
        "start_seconds",
        "end_seconds",
    }:
        raise ValueError("invalid evidence interval")
    memory_id = value["memory_id"]
    source_id = value["source_id"]
    start = value["start_seconds"]
    end = value["end_seconds"]
    if (
        not isinstance(memory_id, str)
        or (source_id is not None and not isinstance(source_id, str))
        or not _optional_number(start)
        or not _optional_number(end)
    ):
        raise ValueError("invalid evidence interval")
    return EvidenceInterval(
        memory_id,
        source_id,
        None if start is None else float(start),
        None if end is None else float(end),
    )


def _optional_number(value: object) -> bool:
    return value is None or (not isinstance(value, bool) and isinstance(value, int | float))


class DescriptionCache:
    """Persist one visual description per asset content hash, shared by every run.

    A caption becomes indexed text, and the measured generation endpoint returns a different
    completion for the same image on every call even at temperature 0 with a fixed seed. Two
    ingests of one corpus would therefore build different libraries, which makes a paired arm
    incomparable with itself. Keying on the asset's own SHA-256 plus the describer's vision space
    makes the second ingest reproduce the first exactly and spend nothing. The space, not the
    model, is the key the store itself uses: a cache keyed on the model alone would serve captions
    written under an older prompt forever, so editing the prompt would measure a no-op.
    """

    def __init__(self, path: Path, space: str) -> None:
        if not space:
            raise ValueError("description cache space must not be empty")
        self.path = path.expanduser().resolve()
        self.space = space
        self.hits = 0
        self.misses = 0
        self._closed = False
        # Unlike `ResponseCache`, which every unit opens for itself, one description cache serves
        # the whole run from the thread that built the backend pool while units ingest on worker
        # threads. SQLite's per-thread binding is lifted and the calls are serialised here instead.
        self._lock = threading.Lock()
        self._connection = _open(self.path, _DESCRIPTIONS_DDL, check_same_thread=False)

    def get(self, digest: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT description FROM descriptions WHERE cache_key = ?",
                (self._key(digest),),
            ).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
        return str(row[0])

    def put(self, digest: str, description: str) -> None:
        self.put_many(((digest, description),))

    def put_many(self, items: Sequence[tuple[str, str]]) -> None:
        """Write one batch of captions in one transaction.

        `synchronous=FULL` costs an fsync per commit, and a described batch is written as a
        batch, so committing each caption on its own made the cache slower than the endpoint.
        """
        rows = []
        for digest, description in items:
            if not description.strip():
                raise ValueError("description must not be blank")
            rows.append((self._key(digest), description))
        if not rows:
            return
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.executemany(
                    "INSERT OR REPLACE INTO descriptions(cache_key, description) VALUES (?, ?)",
                    rows,
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def _key(self, digest: str) -> str:
        if not digest:
            raise ValueError("description cache digest must not be empty")
        encoded = json.dumps((self.space, digest), ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()
