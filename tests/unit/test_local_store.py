from __future__ import annotations

import math
import multiprocessing
import os
import random
import sqlite3
import stat
import struct
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from mindbridge import (
    EvidenceBasis,
    IdentityClaim,
    IdentityProfile,
    MemoryContext,
    MemoryIntent,
    MemoryKind,
    MemoryOperation,
    SpatialAnchor,
    SpatialContext,
)
from mindbridge.control import dump_operation, operation_key
from mindbridge.infrastructure.local import (
    DataDirectoryInUseError,
    LocalStore,
    LocalStoreClosedError,
    StoredAsset,
    StoredEmbedding,
    StoredMemory,
    StoredOperation,
    UnsupportedSchemaError,
)
from mindbridge.infrastructure.local import store as store_module
from mindbridge.infrastructure.local.store import (
    _PRE_VISUAL_DESCRIPTION_VERSION,
    _SCHEMA_VERSION,
)
from mindbridge.models.base import (
    FaceAnalysis,
    FaceEmbedding,
    SpeakerEmbedding,
    SpeechAnalysis,
    SpeechTurn,
)


def _attempt_store_open(data_dir: str, sender: Connection) -> None:
    try:
        with LocalStore(data_dir):
            sender.send("opened")
    except DataDirectoryInUseError:
        sender.send("locked")
    finally:
        sender.close()


def _open_result(data_dir: Path) -> str:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_attempt_store_open, args=(str(data_dir), sender))
    process.start()
    sender.close()
    try:
        if not receiver.poll(10):
            process.terminate()
            pytest.fail("child process did not report its lock result")
        result = receiver.recv()
    finally:
        receiver.close()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
    assert process.exitcode == 0
    assert isinstance(result, str)
    return result


def _memory(
    memory_id: str = "memory-1",
    content: str = "A red tool is in drawer two",
    *,
    created_at: datetime | None = None,
) -> StoredMemory:
    now = created_at or datetime(2026, 8, 27, 1, 2, 3, tzinfo=timezone.utc)
    return StoredMemory(
        memory_id=memory_id,
        content=content,
        metadata_json='{"room": "workshop", "priority": 2}',
        created_at=now,
        updated_at=now,
        occurred_at=now - timedelta(hours=1),
        occurred_end=now - timedelta(minutes=30),
    )


def _embedding(
    embedding_id: str = "embedding-1",
    memory_id: str = "memory-1",
    *,
    object_part: int = 0,
) -> StoredEmbedding:
    return StoredEmbedding(
        embedding_id=embedding_id,
        memory_id=memory_id,
        values=(0.6, 0.8),
        model_id="test-embedder",
        space_id="test-space",
        task="retrieval.document",
        created_at=datetime(2026, 8, 27, 1, 2, 3, tzinfo=timezone.utc),
        object_part=object_part,
        normalized=True,
    )


def _asset(
    content: bytes = b"image bytes",
    *,
    modality: str = "image",
    mime_type: str = "image/png",
    name: str | None = "image.png",
    transcript: str | None = None,
) -> StoredAsset:
    digest = sha256(content).hexdigest()
    return StoredAsset(
        asset_id=digest,
        modality=modality,
        mime_type=mime_type,
        size_bytes=len(content),
        sha256=digest,
        relative_path=f"assets/{digest[:2]}/{digest}",
        name=name,
        transcript=transcript,
        created_at=datetime(2026, 8, 27, 1, 2, 3, tzinfo=timezone.utc),
    )


def _state_memory(
    memory_id: str,
    source_memory_id: str,
    value: str,
    *,
    valid_from: datetime,
    recorded_at: datetime,
    valid_until: datetime | None = None,
) -> StoredMemory:
    return replace(
        _memory(memory_id, f"preferred drink is {value}", created_at=recorded_at),
        context=MemoryContext(
            kind=MemoryKind.STATE,
            basis=EvidenceBasis.MODEL_INFERENCE,
            confidence=0.9,
            valid_from=valid_from,
            valid_until=valid_until,
            recorded_at=recorded_at,
            lineage_id="user:preferred_drink",
            subject="user",
            predicate="preferred_drink",
            value=value,
            evidence_ids=(source_memory_id,),
        ),
    )


def _install_legacy_identity_schema(
    connection: sqlite3.Connection,
    *,
    with_name: bool,
) -> None:
    connection.executescript(
        """
        DROP TABLE face_observations;
        DROP TABLE face_analyses;
        DROP TABLE speech_segments;
        DROP TABLE speech_analyses;
        DROP TABLE identity_exemplars;
        DROP TABLE identity_aliases;
        DROP TABLE identities;
        CREATE TABLE speech_analyses (
            asset_id TEXT PRIMARY KEY REFERENCES media_assets (asset_id) ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            space_id TEXT NOT NULL,
            transcript TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    name_column = "name TEXT," if with_name else ""
    connection.executescript(
        f"""
        CREATE TABLE speaker_identities (
            speaker_id TEXT PRIMARY KEY,
            {name_column}
            model_id TEXT NOT NULL,
            space_id TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            centroid BLOB NOT NULL,
            observations INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE speech_segments (
            asset_id TEXT NOT NULL REFERENCES speech_analyses (asset_id) ON DELETE CASCADE,
            position INTEGER NOT NULL,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            transcript TEXT NOT NULL,
            speaker_id TEXT REFERENCES speaker_identities (speaker_id) ON DELETE SET NULL,
            identity_score REAL,
            PRIMARY KEY (asset_id, position)
        );
        """
    )


def _video_asset(store: LocalStore, label: str) -> StoredAsset:
    """Write one memory that owns one fresh video asset."""
    asset = _asset(
        f"{label} frames".encode(),
        modality="video",
        mime_type="video/mp4",
        name=f"{label}.mp4",
    )
    store.write_memory(replace(_memory(label), content="", modality="video", assets=(asset,)))
    return asset


def _face_identity(store: LocalStore, asset: StoredAsset, vector: tuple[float, float]) -> str:
    return store.write_faces(
        asset.asset_id,
        FaceAnalysis((FaceEmbedding("face-0", vector, (0.1, 0.1, 0.4, 0.5)),)),
        model_id="sface",
        space_id="sface:test",
        minimum_similarity=0.99,
    )[0].identity_id


def _voice_identity(store: LocalStore, asset: StoredAsset, vector: tuple[float, float]) -> str:
    speaker_id = store.write_speech(
        asset.asset_id,
        SpeechAnalysis(
            turns=(SpeechTurn(0, 500, "hello", "0"),),
            speakers=(SpeakerEmbedding("0", vector),),
        ),
        model_id="cam++",
        space_id="cam++:test",
        minimum_similarity=0.99,
    )[0].speaker_id
    assert speaker_id is not None
    return speaker_id


def _full_identity(store: LocalStore, asset: StoredAsset, vector: tuple[float, float]) -> str:
    """Merge one face-only and one voice-only identity into a face-and-voice identity."""
    face_id = _face_identity(store, asset, vector)
    voice_id = _voice_identity(store, asset, vector)
    linked = store.link_identities(face_id, voice_id)
    assert linked is not None
    return linked


def test_data_directory_has_one_process_owner(tmp_path: Path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"

    with LocalStore(first_path):
        with pytest.raises(DataDirectoryInUseError, match="already in use"):
            LocalStore(first_path)
        assert _open_result(first_path) == "locked"
        assert _open_result(second_path) == "opened"

    with LocalStore(first_path):
        pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not available")
def test_store_secures_existing_directory_and_created_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "shared"
    data_dir.mkdir(mode=0o755)
    data_dir.chmod(0o755)

    with LocalStore(data_dir) as store:
        assert stat.S_IMODE(data_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(store.database_path.stat().st_mode) == 0o600

    created_dir = tmp_path / "created"
    with LocalStore(created_dir):
        assert stat.S_IMODE(created_dir.stat().st_mode) == 0o700


def test_schema_is_local_and_enforces_foreign_keys(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        with closing(sqlite3.connect(store.database_path)) as connection:
            schema = "\n".join(
                str(row[0])
                for row in connection.execute(
                    "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name"
                )
            )
            assert connection.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert "tenant" not in schema.casefold()

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            store.write_embedding(_embedding(memory_id="missing-memory"))


def test_schema_v1_is_migrated_atomically_with_text_modality(tmp_path: Path) -> None:
    data_dir = tmp_path / "v1"
    data_dir.mkdir()
    database = data_dir / "state.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        connection.executescript(
            """
            CREATE TABLE memory_records (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL CHECK (length(trim(content)) > 0),
                metadata_json TEXT NOT NULL,
                occurred_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE embeddings (
                embedding_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES memory_records (memory_id) ON DELETE CASCADE,
                object_part INTEGER NOT NULL DEFAULT 0,
                model_id TEXT NOT NULL,
                space_id TEXT NOT NULL,
                task TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                normalized INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (memory_id, object_part, model_id, task)
            );
            CREATE TABLE store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE search_index_queue (
                operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_id TEXT NOT NULL,
                action TEXT NOT NULL,
                enqueued_at TEXT NOT NULL
            );
            INSERT INTO memory_records (
                memory_id, content, metadata_json, occurred_at, created_at, updated_at
            ) VALUES (
                'legacy', 'legacy text', '{}', NULL,
                '2026-08-27T01:02:03.000000Z', '2026-08-27T01:02:03.000000Z'
            );
            PRAGMA user_version = 1;
            """
        )

    with LocalStore(data_dir) as store:
        legacy = store.read_memory("legacy")
        assert legacy is not None
        assert legacy.content == "legacy text"
        assert legacy.modality == "text"
        assert legacy.memory_type == "semantic"
        assert legacy.access_count == 0
        assert legacy.assets == ()
        with closing(sqlite3.connect(store.database_path)) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_schema_v2_is_migrated_without_losing_memories(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        store.write_memory(_memory("preserved"))
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            """
            DROP TABLE speech_segments;
            DROP TABLE speech_analyses;
            DROP TABLE face_observations;
            DROP TABLE face_analyses;
            DROP TABLE identity_exemplars;
            DROP TABLE identity_aliases;
            DROP TABLE identities;
            PRAGMA user_version = 2;
            """
        )

    with LocalStore(tmp_path) as store:
        assert store.read_memory("preserved") is not None
        with closing(sqlite3.connect(store.database_path)) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.skipif(sqlite3.sqlite_version_info < (3, 35), reason="DROP COLUMN needs SQLite 3.35")
def test_schema_v3_adds_optional_speaker_names(tmp_path: Path) -> None:
    with LocalStore(tmp_path):
        pass
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        _install_legacy_identity_schema(connection, with_name=False)
        connection.executescript(
            """
            PRAGMA user_version = 3;
            """
        )

    with LocalStore(tmp_path) as store:
        with closing(sqlite3.connect(store.database_path)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(identities)")}
            assert connection.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
        assert "name" in columns


@pytest.mark.skipif(sqlite3.sqlite_version_info < (3, 35), reason="DROP COLUMN needs SQLite 3.35")
def test_schema_v5_adds_optional_event_end(tmp_path: Path) -> None:
    with LocalStore(tmp_path):
        pass
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        _install_legacy_identity_schema(connection, with_name=True)
        connection.executescript(
            """
            ALTER TABLE memory_records DROP COLUMN occurred_end;
            PRAGMA user_version = 5;
            """
        )

    with LocalStore(tmp_path) as store:
        with closing(sqlite3.connect(store.database_path)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(memory_records)")}
            assert connection.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
        assert "occurred_end" in columns


def test_schema_v6_migrates_the_centroid_into_a_voice_exemplar(tmp_path: Path) -> None:
    audio = _asset(b"legacy voice", modality="audio", mime_type="audio/wav", name="voice.wav")
    memory = replace(_memory(), content="", modality="audio", assets=(audio,))
    with LocalStore(tmp_path) as store:
        store.write_memory(memory)
    timestamp = "2026-08-27T01:02:03.000000Z"
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        _install_legacy_identity_schema(connection, with_name=True)
        connection.execute(
            """
            INSERT INTO speaker_identities (
                speaker_id, name, model_id, space_id, dimension, centroid,
                observations, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "speaker_legacy",
                "Alice",
                "cam++",
                "cam++:legacy",
                2,
                struct.pack("<2f", 0.6, 0.8),
                9,
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO speech_analyses (asset_id, model_id, space_id, transcript, created_at)
            VALUES (?, 'cam++', 'cam++:legacy', 'hello', ?)
            """,
            (audio.asset_id, timestamp),
        )
        connection.execute(
            """
            INSERT INTO speech_segments (
                asset_id, position, start_ms, end_ms, transcript, speaker_id, identity_score
            ) VALUES (?, 0, 0, 500, 'hello', 'speaker_legacy', 0.9)
            """,
            (audio.asset_id,),
        )
        connection.execute("PRAGMA user_version = 6")
        connection.commit()

    with LocalStore(tmp_path) as store:
        segments = store.read_speech(audio.asset_id, space_id="cam++:legacy")
        with closing(sqlite3.connect(store.database_path)) as connection:
            exemplar = connection.execute(
                """
                SELECT identity_id, modality, position, vector
                FROM identity_exemplars
                """
            ).fetchone()
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }

    assert segments is not None and segments[0].speaker_name == "Alice"
    assert exemplar[:3] == ("speaker_legacy", "voice", 0)
    assert struct.unpack("<2f", exemplar[3]) == pytest.approx((0.6, 0.8))
    assert "speaker_identities" not in tables


def test_schema_v7_adds_transactional_evidence_without_losing_records(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        store.write_memory(_memory("preserved"))
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            """
            DROP TABLE formation_runs;
            DROP TABLE memory_evidence;
            DROP TABLE memory_versions;
            DROP TABLE memory_semantics;
            PRAGMA user_version = 7;
            """
        )

    with LocalStore(tmp_path) as store:
        preserved = store.read_memory("preserved")
        with closing(sqlite3.connect(store.database_path)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(memory_evidence)")}
            version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert preserved is not None and preserved.content == "A red tool is in drawer two"
    assert {"source_group_id", "recorded_at", "retired_at"} <= columns
    assert version == _SCHEMA_VERSION


@pytest.mark.skipif(sqlite3.sqlite_version_info < (3, 35), reason="DROP COLUMN needs SQLite 3.35")
def test_schema_v8_adds_identity_profiles_and_link_evidence(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        face_id = _face_identity(store, clip, (1.0, 0.0))
        voice_id = _voice_identity(store, clip, (1.0, 0.0))
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            """
            DROP TABLE identity_link_evidence;
            ALTER TABLE identities DROP COLUMN relationship;
            ALTER TABLE identity_aliases DROP COLUMN contributed_modality;
            PRAGMA user_version = 8;
            """
        )

    with LocalStore(tmp_path) as store:
        faces = store.read_faces(clip.asset_id, space_id="sface:test")
        profile = store.identity_profile(face_id)
        recorded = store.record_identity_link_evidence(voice_id, face_id, clip.asset_id)
        assert store.register_identity(face_id, "Alice", relationship="neighbour") is True
        named = store.identity_profile(face_id)
        with closing(sqlite3.connect(store.database_path)) as connection:
            alias_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(identity_aliases)")
            }
            version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert faces is not None and faces[0].identity_id == face_id
    assert profile == IdentityProfile(identity_id=face_id)
    assert recorded == 1
    assert named == IdentityProfile(identity_id=face_id, name="Alice", relationship="neighbour")
    assert "contributed_modality" in alias_columns
    assert version == _SCHEMA_VERSION


def test_memory_embedding_and_outbox_round_trip(tmp_path: Path) -> None:
    memory = _memory()
    embedding = _embedding()

    with LocalStore(tmp_path) as store:
        assert store.write_memory(memory, (embedding,)) is True
        assert store.write_memory(memory, ()) is False

        stored_memory = store.read_memory(memory.memory_id)
        assert stored_memory is not None
        assert stored_memory.metadata_json == '{"priority":2,"room":"workshop"}'
        assert stored_memory.occurred_at == memory.occurred_at
        assert stored_memory.occurred_end == memory.occurred_end
        assert stored_memory.memory_type == "semantic"
        assert stored_memory.access_count == 0
        stored_embedding = store.read_embedding(embedding.embedding_id)
        assert stored_embedding is not None
        assert stored_embedding.values == pytest.approx(embedding.values)

        operations = store.pending_index_operations()
        assert [operation.action for operation in operations] == ["upsert"]
        assert store.acknowledge_index_operations(operations) == 1

        changed = replace(
            memory,
            content="A red tool is now on the workbench",
            updated_at=memory.updated_at + timedelta(seconds=1),
        )
        assert store.write_memory(changed) is False
        changed_operation = store.pending_index_operations()
        assert len(changed_operation) == 1
        assert changed_operation[0].action == "upsert"

        document = store.read_index_document(embedding.embedding_id)
        assert document is not None
        assert document.content == changed.content
        assert document.memory_type == "semantic"
        assert document.occurred_at == changed.occurred_at
        assert document.occurred_end == changed.occurred_end
        assert document.embedding.embedding_id == embedding.embedding_id

        assert store.acknowledge_index_operations(changed_operation) == 1
        assert store.delete_memory(memory.memory_id) is True
        assert store.read_memory(memory.memory_id) is None
        assert store.read_embedding(embedding.embedding_id) is None
        assert store.read_index_document(embedding.embedding_id) is None
        deletion = store.pending_index_operations()
        assert len(deletion) == 1
        assert deletion[0].embedding_id == embedding.embedding_id
        assert deletion[0].action == "delete"

        store.set_metadata("embedding.dimension", "2")
        assert store.get_metadata("embedding.dimension") == "2"
        assert store.delete_metadata("embedding.dimension") is True
        assert store.get_metadata("embedding.dimension") is None

    with pytest.raises(LocalStoreClosedError, match="closed"):
        store.read_memory(memory.memory_id)


def test_read_memory_index_documents_returns_every_part_in_order(tmp_path: Path) -> None:
    memory = _memory()
    aggregate = _embedding(memory_id=memory.memory_id)
    child = _embedding("embedding-2", memory.memory_id, object_part=1)

    with LocalStore(tmp_path) as store:
        store.write_memory(memory, (child, aggregate))

        documents = store.read_memory_index_documents((memory.memory_id, "missing"))

    assert [document.embedding.embedding_id for document in documents] == [
        aggregate.embedding_id,
        child.embedding_id,
    ]


def test_retrieval_reinforcement_is_bounded_and_monotonic(tmp_path: Path) -> None:
    first_access = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    later_access = first_access + timedelta(hours=1)
    with LocalStore(tmp_path) as store:
        store.write_memory(_memory())
        for _attempt in range(25):
            assert store.reinforce_memories(("memory-1",), accessed_at=first_access) == 1
        assert store.reinforce_memories(("memory-1",), accessed_at=later_access) == 1
        assert store.reinforce_memories(("memory-1",), accessed_at=first_access) == 1

        reinforced = store.read_memory("memory-1")
        assert reinforced is not None
        assert reinforced.access_count == 20
        assert reinforced.last_accessed_at == later_access


def test_memory_and_embeddings_commit_atomically(tmp_path: Path) -> None:
    memory = _memory()
    first = _embedding("embedding-1")
    conflicting = _embedding("embedding-2")

    with LocalStore(tmp_path) as store:
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            store.write_memory(memory, (first, conflicting))

        assert store.read_memory(memory.memory_id) is None
        assert store.read_embedding(first.embedding_id) is None
        assert store.pending_index_operations() == ()


def test_batch_write_ranked_hydration_and_keyset_listing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 27, 1, 2, 3, tzinfo=timezone.utc)
    oldest = _memory("memory-a", created_at=base)
    newer_a = _memory("memory-b", created_at=base + timedelta(seconds=1))
    newer_b = _memory("memory-c", created_at=base + timedelta(seconds=1))
    monkeypatch.setattr("mindbridge.infrastructure.local.store._SQLITE_PARAMETER_BATCH", 2)

    with LocalStore(tmp_path) as store:
        assert store.write_memories((oldest, newer_a, newer_b)) == (True, True, True)
        hydrated = store.read_memories(
            (newer_a.memory_id, "missing", oldest.memory_id, newer_a.memory_id)
        )
        assert [memory.memory_id for memory in hydrated] == [
            newer_a.memory_id,
            oldest.memory_id,
            newer_a.memory_id,
        ]

        first_page = store.list_memories(limit=2)
        assert [memory.memory_id for memory in first_page] == ["memory-c", "memory-b"]
        cursor = (first_page[-1].created_at, first_page[-1].memory_id)
        second_page = store.list_memories(limit=2, after=cursor)
        assert [memory.memory_id for memory in second_page] == ["memory-a"]


def test_embedding_ids_in_range_uses_half_open_intervals_and_type(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    until = start + timedelta(days=1)
    records = (
        replace(_memory("point-from"), occurred_at=start, occurred_end=None),
        replace(_memory("point-until"), occurred_at=until, occurred_end=None),
        replace(
            _memory("span-ending-at-from"),
            occurred_at=start - timedelta(hours=2),
            occurred_end=start,
        ),
        replace(
            _memory("span-crossing-from"),
            occurred_at=start - timedelta(hours=1),
            occurred_end=start + timedelta(hours=1),
        ),
        replace(
            _memory("episodic-point"),
            occurred_at=start + timedelta(hours=2),
            occurred_end=None,
            memory_type="episodic",
        ),
    )
    with LocalStore(tmp_path) as store:
        for record in records:
            embeddings: tuple[StoredEmbedding, ...] = (
                _embedding(f"embedding-{record.memory_id}", record.memory_id),
            )
            if record.memory_id == "point-from":
                embeddings += (
                    _embedding(
                        "embedding-point-from-part-1",
                        record.memory_id,
                        object_part=1,
                    ),
                )
            store.write_memory(record, embeddings)

        embedding_ids, total = store.embedding_ids_in_range(
            start.astimezone(timezone(timedelta(hours=8))),
            until.astimezone(timezone(timedelta(hours=8))),
            space_id="test-space",
            task="retrieval.document",
        )
        episodic_ids, episodic_total = store.embedding_ids_in_range(
            start,
            until,
            space_id="test-space",
            task="retrieval.document",
            memory_type="episodic",
        )

    assert embedding_ids == frozenset(
        {
            "embedding-point-from",
            "embedding-span-crossing-from",
            "embedding-episodic-point",
        }
    )
    assert total == 5
    assert episodic_ids == frozenset({"embedding-episodic-point"})
    assert episodic_total == 1


def test_media_assets_round_trip_in_order_and_cache_transcript(tmp_path: Path) -> None:
    first = _asset()
    second = _asset(b"second image", name="second.png")
    memory = replace(
        _memory(),
        content="",
        modality="image",
        assets=(second, first),
    )

    with LocalStore(tmp_path) as store:
        embedding = _embedding(memory_id=memory.memory_id)
        assert store.write_memory(memory, (embedding,)) is True
        assert store.read_memory(memory.memory_id) == memory
        assert store.read_memories((memory.memory_id,)) == (memory,)
        assert store.list_memories() == (memory,)
        document = store.read_index_document(embedding.embedding_id)
        assert document is not None
        assert document.content == ""
        assert store.read_assets((first.asset_id, "0" * 64, first.asset_id)) == (
            first,
            first,
        )
        assert store.set_asset_transcript(first.asset_id, "a red workbench") is True
        assert store.read_asset(first.asset_id) == replace(
            first,
            transcript="a red workbench",
        )
        assert store.set_asset_transcript(first.asset_id, "") is True
        assert store.read_asset(first.asset_id) == replace(first, transcript="")
        assert store.set_asset_transcript("f" * 64, "missing") is False


def test_visual_descriptions_are_keyed_by_asset_and_space(tmp_path: Path) -> None:
    """One caption per (asset content, vision space), so a new recipe cannot serve an old caption.

    `media_assets` enforces `asset_id = sha256`, so this key is the picture's own digest: a second
    memory over the same bytes reads the row the first one wrote.
    """
    first = _asset()
    second = _asset(b"second image", name="second.png")
    memory = replace(_memory(), content="", modality="image", assets=(first, second))

    with LocalStore(tmp_path) as store:
        store.write_memory(memory, (_embedding(memory_id=memory.memory_id),))
        assert store.read_visual_descriptions((), space_id="caption-v1") == {}
        assert store.read_visual_descriptions((first.asset_id,), space_id="caption-v1") == {}

        written = store.write_visual_descriptions(
            {first.asset_id: "a red bicycle", second.asset_id: "a blue door"},
            model_id="describer",
            space_id="caption-v1",
        )
        assert written == 2
        assert store.read_visual_descriptions(
            (second.asset_id, first.asset_id, "0" * 64),
            space_id="caption-v1",
        ) == {first.asset_id: "a red bicycle", second.asset_id: "a blue door"}

        # A different recipe shares no captions with the first, and does not evict it.
        assert store.read_visual_descriptions((first.asset_id,), space_id="caption-v2") == {}
        store.write_visual_descriptions(
            {first.asset_id: "a bicycle against a fence"},
            model_id="describer",
            space_id="caption-v2",
        )
        assert store.read_visual_descriptions((first.asset_id,), space_id="caption-v1") == {
            first.asset_id: "a red bicycle"
        }
        assert store.read_visual_descriptions((first.asset_id,), space_id="caption-v2") == {
            first.asset_id: "a bicycle against a fence"
        }

        # Re-describing an asset keeps the stored caption: identical documents is the whole point.
        assert (
            store.write_visual_descriptions(
                {first.asset_id: "something else entirely"},
                model_id="describer",
                space_id="caption-v1",
            )
            == 0
        )
        assert store.read_visual_descriptions((first.asset_id,), space_id="caption-v1") == {
            first.asset_id: "a red bicycle"
        }

        for blank in ("", "   "):
            with pytest.raises(ValueError, match="must not be blank"):
                store.write_visual_descriptions(
                    {first.asset_id: blank},
                    model_id="describer",
                    space_id="caption-v1",
                )
        with pytest.raises(ValueError):
            store.write_visual_descriptions(
                {first.asset_id: "orphan"},
                model_id="describer",
                space_id="   ",
            )
        # The foreign key is the asset: a caption for media this store never stored is refused.
        with pytest.raises(sqlite3.IntegrityError):
            store.write_visual_descriptions(
                {"0" * 64: "never stored"},
                model_id="describer",
                space_id="caption-v1",
            )


def test_visual_descriptions_outlive_the_memory_and_die_with_the_asset(tmp_path: Path) -> None:
    """Deleting a memory must not make re-adding the same picture pay for a caption again."""
    picture = _asset()
    memory = replace(_memory(), content="", modality="image", assets=(picture,))

    with LocalStore(tmp_path) as store:
        store.write_memory(memory, (_embedding(memory_id=memory.memory_id),))
        store.write_visual_descriptions(
            {picture.asset_id: "a red bicycle"},
            model_id="describer",
            space_id="caption-v1",
        )
        deleted, orphaned = store.delete_memory_with_assets(memory.memory_id)
        assert deleted is True
        assert tuple(asset.asset_id for asset in orphaned) == (picture.asset_id,)
        assert store.read_visual_descriptions((picture.asset_id,), space_id="caption-v1") == {
            picture.asset_id: "a red bicycle"
        }

        # It is the asset row that owns the caption, so erasing the asset cascades.
        with closing(sqlite3.connect(store.database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("DELETE FROM media_assets WHERE asset_id = ?", (picture.asset_id,))
            connection.commit()
        assert store.read_visual_descriptions((picture.asset_id,), space_id="caption-v1") == {}


def test_a_store_without_visual_descriptions_gains_the_table(tmp_path: Path) -> None:
    """An existing library one step behind opens, keeps every row, and gains the caption cache.

    Parametric on the version pair rather than on literals: this migration step is renumbered
    when it merges alongside parallel schema work, and then only the two constants move.
    """
    previous_version = _PRE_VISUAL_DESCRIPTION_VERSION
    picture = _asset()
    memory = replace(_memory(), content="", modality="image", assets=(picture,))
    with LocalStore(tmp_path) as store:
        store.write_memory(memory, (_embedding(memory_id=memory.memory_id),))
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            f"""
            DROP TABLE visual_descriptions;
            PRAGMA user_version = {previous_version};
            """
        )
        assert connection.execute("PRAGMA user_version").fetchone()[0] == previous_version

    with LocalStore(tmp_path) as store:
        preserved = store.read_memory(memory.memory_id)
        # The migrated table is usable, not merely present.
        store.write_visual_descriptions(
            {picture.asset_id: "a red bicycle"},
            model_id="describer",
            space_id="caption-v1",
        )
        captions = store.read_visual_descriptions((picture.asset_id,), space_id="caption-v1")
        with closing(sqlite3.connect(store.database_path)) as connection:
            tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master")}
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert connection.execute("PRAGMA foreign_key_check").fetchone() is None

    assert preserved is not None and preserved.assets == (picture,)
    assert "visual_descriptions" in tables
    assert version == _SCHEMA_VERSION
    assert captions == {picture.asset_id: "a red bicycle"}


def test_a_store_missing_the_caption_table_at_the_current_version_is_refused(
    tmp_path: Path,
) -> None:
    """`_REQUIRED_TABLES` is what stops a half-migrated library from opening and writing."""
    with LocalStore(tmp_path):
        pass
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript("DROP TABLE visual_descriptions;")

    with pytest.raises(UnsupportedSchemaError, match="visual_descriptions"):
        LocalStore(tmp_path).close()


def test_speech_identity_is_stable_across_assets(tmp_path: Path) -> None:
    first = _asset(b"first voice", modality="audio", mime_type="audio/wav", name="first.wav")
    second = _asset(
        b"second voice",
        modality="audio",
        mime_type="audio/wav",
        name="second.wav",
    )
    first_memory = replace(_memory("first"), content="", modality="audio", assets=(first,))
    second_memory = replace(_memory("second"), content="", modality="audio", assets=(second,))

    with LocalStore(tmp_path) as store:
        store.write_memories((first_memory, second_memory))
        first_segments = store.write_speech(
            first.asset_id,
            SpeechAnalysis(
                turns=(
                    SpeechTurn(0, 800, "hello", "0"),
                    SpeechTurn(800, 1_600, "second speaker", "1"),
                ),
                speakers=(
                    SpeakerEmbedding("0", (1.0, 0.0)),
                    SpeakerEmbedding("1", (0.0, 1.0)),
                ),
            ),
            model_id="cam++",
            space_id="cam++:test",
        )
        second_segments = store.write_speech(
            second.asset_id,
            SpeechAnalysis(
                turns=(SpeechTurn(20, 900, "hello again", "0"),),
                speakers=(SpeakerEmbedding("0", (0.99, 0.1)),),
            ),
            model_id="cam++",
            space_id="cam++:test",
        )

        speaker_id = first_segments[0].speaker_id
        assert speaker_id is not None
        assert first_segments[1].speaker_id != first_segments[0].speaker_id
        assert second_segments[0].speaker_id == speaker_id
        assert first_segments[0].identity_score is None
        assert second_segments[0].identity_score is not None
        assert second_segments[0].identity_score > 0.99
        assert store.register_speaker(speaker_id, "Alice") is True
        renamed = store.read_speech(first.asset_id, space_id="cam++:test")
        assert renamed is not None and renamed[0].speaker_name == "Alice"
        assert store.register_speaker(speaker_id, "Alicia") is True
        renamed = store.read_speech(second.asset_id, space_id="cam++:test")
        assert renamed is not None and renamed[0].speaker_name == "Alicia"
        assert store.register_speaker("speaker_missing", "Nobody") is False
        assert store.read_asset(second.asset_id) == replace(second, transcript="hello again")


def test_speech_identity_uses_max_over_bounded_exemplars(tmp_path: Path) -> None:
    assets = tuple(
        _asset(
            f"voice-{index}".encode(),
            modality="audio",
            mime_type="audio/wav",
            name=f"voice-{index}.wav",
        )
        for index in range(4)
    )
    memories = tuple(
        replace(_memory(f"memory-{index}"), content="", modality="audio", assets=(asset,))
        for index, asset in enumerate(assets)
    )
    vectors = (
        (1.0, 0.0),
        (math.sqrt(3) / 2, 0.5),
        (0.5, math.sqrt(3) / 2),
        (0.5, math.sqrt(3) / 2),
    )

    with LocalStore(tmp_path) as store:
        store.write_memories(memories)
        identities = []
        scores = []
        for asset, vector in zip(assets, vectors, strict=True):
            segment = store.write_speech(
                asset.asset_id,
                SpeechAnalysis(
                    turns=(SpeechTurn(0, 500, "voice", "0"),),
                    speakers=(SpeakerEmbedding("0", vector),),
                ),
                model_id="cam++",
                space_id="cam++:test",
                minimum_similarity=0.8,
            )[0]
            identities.append(segment.speaker_id)
            scores.append(segment.identity_score)
        with closing(sqlite3.connect(store.database_path)) as connection:
            exemplars = connection.execute(
                "SELECT COUNT(*) FROM identity_exemplars WHERE modality = 'voice'"
            ).fetchone()[0]

    assert len(set(identities)) == 1
    assert scores[2] == pytest.approx(math.sqrt(3) / 2)
    assert exemplars == 3


def test_voice_exemplar_collection_keeps_a_hard_twenty_vector_bound(tmp_path: Path) -> None:
    assets = tuple(
        _asset(
            f"bounded-voice-{index}".encode(),
            modality="audio",
            mime_type="audio/wav",
            name=f"voice-{index}.wav",
        )
        for index in range(21)
    )
    memories = tuple(
        replace(_memory(f"bounded-{index}"), content="", modality="audio", assets=(asset,))
        for index, asset in enumerate(assets)
    )

    with LocalStore(tmp_path) as store:
        store.write_memories(memories)
        identity_ids = []
        for index, asset in enumerate(assets):
            angle = math.radians(index * 10)
            segment = store.write_speech(
                asset.asset_id,
                SpeechAnalysis(
                    turns=(SpeechTurn(0, 500, "voice", "0"),),
                    speakers=(SpeakerEmbedding("0", (math.cos(angle), math.sin(angle))),),
                ),
                model_id="cam++",
                space_id="cam++:test",
                minimum_similarity=0.98,
            )[0]
            identity_ids.append(segment.speaker_id)
        with closing(sqlite3.connect(store.database_path)) as connection:
            exemplar_count = connection.execute(
                "SELECT COUNT(*) FROM identity_exemplars WHERE modality = 'voice'"
            ).fetchone()[0]

    assert len(set(identity_ids)) == 1
    assert exemplar_count == 20


def test_face_and_voice_share_one_identity_only_with_explicit_singleton_link(
    tmp_path: Path,
) -> None:
    video = _asset(
        b"one person video",
        modality="video",
        mime_type="video/mp4",
        name="person.mp4",
    )
    memory = replace(_memory(), content="", modality="video", assets=(video,))

    with LocalStore(tmp_path) as store:
        store.write_memory(memory)
        speaker = store.write_speech(
            video.asset_id,
            SpeechAnalysis(
                turns=(SpeechTurn(0, 500, "hello", "0"),),
                speakers=(SpeakerEmbedding("0", (1.0, 0.0)),),
            ),
            model_id="cam++",
            space_id="cam++:test",
        )[0]
        assert speaker.speaker_id is not None
        faces = store.write_faces(
            video.asset_id,
            FaceAnalysis((FaceEmbedding("face-0", (0.0, 1.0), (0.1, 0.1, 0.4, 0.5), 100),)),
            model_id="sface",
            space_id="sface:test",
            preferred_identity=speaker.speaker_id,
        )

        assert faces[0].identity_id == speaker.speaker_id
        assert store.register_identity(speaker.speaker_id, "Alice") is True
        named_faces = store.read_faces(video.asset_id, space_id="sface:test")
        named_speech = store.read_speech(video.asset_id, space_id="cam++:test")

    assert named_faces is not None and named_faces[0].identity_name == "Alice"
    assert named_speech is not None and named_speech[0].speaker_name == "Alice"


def test_two_faces_in_one_frame_are_never_collapsed_into_the_single_speaker(
    tmp_path: Path,
) -> None:
    video = _asset(
        b"two person video",
        modality="video",
        mime_type="video/mp4",
        name="people.mp4",
    )
    memory = replace(_memory(), content="", modality="video", assets=(video,))

    with LocalStore(tmp_path) as store:
        store.write_memory(memory)
        speaker = store.write_speech(
            video.asset_id,
            SpeechAnalysis(
                turns=(SpeechTurn(0, 500, "hello", "0"),),
                speakers=(SpeakerEmbedding("0", (1.0, 0.0)),),
            ),
            model_id="cam++",
            space_id="cam++:test",
        )[0]
        assert speaker.speaker_id is not None
        faces = store.write_faces(
            video.asset_id,
            FaceAnalysis(
                (
                    FaceEmbedding("face-0", (0.0, 1.0), (0.05, 0.1, 0.3, 0.5), 100),
                    FaceEmbedding("face-1", (0.0, 1.0), (0.55, 0.1, 0.3, 0.5), 100),
                )
            ),
            model_id="sface",
            space_id="sface:test",
            preferred_identity=speaker.speaker_id,
        )

    face_ids = {face.identity_id for face in faces}
    assert len(face_ids) == 2
    assert speaker.speaker_id not in face_ids


def test_preferred_identity_only_enrolls_a_missing_modality(tmp_path: Path) -> None:
    first = _asset(b"first video", modality="video", mime_type="video/mp4", name="first.mp4")
    second = _asset(
        b"second video",
        modality="video",
        mime_type="video/mp4",
        name="second.mp4",
    )
    memories = (
        replace(_memory("first"), content="", modality="video", assets=(first,)),
        replace(_memory("second"), content="", modality="video", assets=(second,)),
    )

    with LocalStore(tmp_path) as store:
        store.write_memories(memories)
        speaker = store.write_speech(
            first.asset_id,
            SpeechAnalysis(
                turns=(SpeechTurn(0, 500, "hello", "0"),),
                speakers=(SpeakerEmbedding("0", (1.0, 0.0)),),
            ),
            model_id="cam++",
            space_id="cam++:test",
        )[0]
        assert speaker.speaker_id is not None
        enrolled = store.write_faces(
            first.asset_id,
            FaceAnalysis((FaceEmbedding("face-0", (1.0, 0.0), (0.1, 0.1, 0.4, 0.5), 100),)),
            model_id="sface",
            space_id="sface:test",
            preferred_identity=speaker.speaker_id,
        )[0]
        unrelated = store.write_faces(
            second.asset_id,
            FaceAnalysis((FaceEmbedding("face-0", (0.0, 1.0), (0.1, 0.1, 0.4, 0.5), 100),)),
            model_id="sface",
            space_id="sface:test",
            minimum_similarity=0.9,
            preferred_identity=speaker.speaker_id,
        )[0]

    assert enrolled.identity_id == speaker.speaker_id
    assert unrelated.identity_id != speaker.speaker_id


def test_face_trajectory_matching_preserves_observation_order(tmp_path: Path) -> None:
    video = _asset(
        b"ordered faces",
        modality="video",
        mime_type="video/mp4",
        name="ordered.mp4",
    )
    memory = replace(_memory(), content="", modality="video", assets=(video,))

    with LocalStore(tmp_path) as store:
        store.write_memory(memory)
        observations = store.write_faces(
            video.asset_id,
            FaceAnalysis(
                (
                    FaceEmbedding("face_0", (1.0, 0.0), (0.1, 0.1, 0.2, 0.2), 0),
                    FaceEmbedding(
                        "face_2",
                        (math.sqrt(3) / 2, 0.5),
                        (0.1, 0.1, 0.2, 0.2),
                        1_000,
                    ),
                    FaceEmbedding(
                        "face_10",
                        (0.5, math.sqrt(3) / 2),
                        (0.1, 0.1, 0.2, 0.2),
                        2_000,
                    ),
                )
            ),
            model_id="sface",
            space_id="sface:test",
            minimum_similarity=0.8,
            minimum_margin=0.05,
        )

    assert len({observation.identity_id for observation in observations}) == 1


def test_linked_identity_keeps_the_source_id_as_an_alias(tmp_path: Path) -> None:
    image = _asset(b"known face")
    audio = _asset(b"known voice", modality="audio", mime_type="audio/wav", name="voice.wav")
    memories = (
        replace(_memory("face"), content="", modality="image", assets=(image,)),
        replace(_memory("voice"), content="", modality="audio", assets=(audio,)),
    )

    with LocalStore(tmp_path) as store:
        store.write_memories(memories)
        face_id = store.write_faces(
            image.asset_id,
            FaceAnalysis((FaceEmbedding("face-0", (1.0, 0.0), (0.1, 0.1, 0.4, 0.5)),)),
            model_id="sface",
            space_id="sface:test",
        )[0].identity_id
        assert store.register_identity(face_id, "Alice") is True
        voice_id = store.write_speech(
            audio.asset_id,
            SpeechAnalysis(
                turns=(SpeechTurn(0, 500, "hello", "0"),),
                speakers=(SpeakerEmbedding("0", (1.0, 0.0)),),
            ),
            model_id="cam++",
            space_id="cam++:test",
        )[0].speaker_id
        assert voice_id is not None and voice_id != face_id

        assert store.link_identities(face_id, voice_id) == face_id
        assert store.resolve_identity_id(voice_id) == face_id
        assert store.register_speaker(voice_id, "Alicia") is True
        speech = store.read_speech(audio.asset_id, space_id="cam++:test")

    assert speech is not None
    assert speech[0].speaker_id == face_id
    assert speech[0].speaker_name == "Alicia"


def test_identity_link_evidence_counts_one_row_per_distinct_asset(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        other = _video_asset(store, "other")
        face_id = _face_identity(store, clip, (1.0, 0.0))
        voice_id = _voice_identity(store, clip, (0.0, 1.0))

        assert store.record_identity_link_evidence(voice_id, face_id, clip.asset_id) == 1
        assert store.record_identity_link_evidence(voice_id, face_id, clip.asset_id) == 1
        assert store.record_identity_link_evidence(voice_id, face_id, other.asset_id) == 2
        assert store.record_identity_link_evidence(voice_id, "identity_missing", clip.asset_id) == 0
        with pytest.raises(ValueError, match="stored media asset"):
            store.record_identity_link_evidence(
                voice_id,
                face_id,
                sha256(b"never stored").hexdigest(),
            )


def test_merging_identities_repoints_link_evidence_onto_the_target(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        left = _video_asset(store, "left")
        right = _video_asset(store, "right")
        face_id = _face_identity(store, clip, (1.0, 0.0))
        first_voice_id = _voice_identity(store, clip, (1.0, 0.0))
        second_voice_id = _voice_identity(store, right, (0.0, 1.0))
        assert store.record_identity_link_evidence(first_voice_id, face_id, clip.asset_id) == 1
        assert store.record_identity_link_evidence(first_voice_id, face_id, left.asset_id) == 2
        assert store.record_identity_link_evidence(second_voice_id, face_id, clip.asset_id) == 1
        assert store.record_identity_link_evidence(second_voice_id, face_id, right.asset_id) == 2

        plan = store.identity_link_plan(
            first_voice_id,
            second_voice_id,
            allow_shared_modality=True,
        )
        assert plan is not None
        target = store.link_identities(
            first_voice_id,
            second_voice_id,
            expected=plan,
            allow_shared_modality=True,
        )
        assert target == plan.target_id
        with closing(sqlite3.connect(store.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT voice_id, face_id, asset_id
                FROM identity_link_evidence
                ORDER BY voice_id, face_id, asset_id
                """
            ).fetchall()
        # The collided clip row survives once and both plain rows follow the target.
        assert rows == sorted(
            [
                (target, face_id, clip.asset_id),
                (target, face_id, left.asset_id),
                (target, face_id, right.asset_id),
            ]
        )
        assert store.record_identity_link_evidence(plan.source_id, face_id, clip.asset_id) == 3


def test_identity_link_plan_refuses_a_shared_modality_by_default(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        fragment = _video_asset(store, "fragment")
        full_id = _full_identity(store, clip, (1.0, 0.0))
        fragment_id = _face_identity(store, fragment, (0.0, 1.0))

        assert store.identity_link_plan(full_id, fragment_id) is None
        assert store.link_identities(full_id, fragment_id) is None
        assert store.resolve_identity_id(fragment_id) == fragment_id


def test_shared_modality_link_absorbs_a_face_fragment_into_a_full_identity(
    tmp_path: Path,
) -> None:
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        fragment = _video_asset(store, "fragment")
        full_id = _full_identity(store, clip, (1.0, 0.0))
        assert store.register_identity(full_id, "Alice") is True
        fragment_id = _face_identity(store, fragment, (0.0, 1.0))

        plan = store.identity_link_plan(full_id, fragment_id, allow_shared_modality=True)
        assert plan is not None and plan.target_id == full_id and plan.source_id == fragment_id
        linked = store.link_identities(
            full_id,
            fragment_id,
            expected=plan,
            allow_shared_modality=True,
        )
        faces = store.read_faces(fragment.asset_id, space_id="sface:test")
        profile = store.identity_profile(fragment_id)

    assert linked == full_id
    assert faces is not None and faces[0].identity_id == full_id
    assert profile == IdentityProfile(identity_id=full_id, name="Alice")


def test_shared_modality_link_still_refuses_two_full_identities(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        first = _video_asset(store, "first")
        second = _video_asset(store, "second")
        first_id = _full_identity(store, first, (1.0, 0.0))
        second_id = _full_identity(store, second, (0.0, 1.0))

        assert store.identity_link_plan(first_id, second_id, allow_shared_modality=True) is None
        assert store.link_identities(first_id, second_id, allow_shared_modality=True) is None
        assert store.resolve_identity_id(second_id) == second_id


def test_shared_modality_link_still_refuses_a_conflicting_name(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        fragment = _video_asset(store, "fragment")
        full_id = _full_identity(store, clip, (1.0, 0.0))
        assert store.register_identity(full_id, "Alice") is True
        fragment_id = _face_identity(store, fragment, (0.0, 1.0))
        assert store.register_identity(fragment_id, "Bruno") is True

        assert store.identity_link_plan(full_id, fragment_id, allow_shared_modality=True) is None
        assert store.link_identities(full_id, fragment_id, allow_shared_modality=True) is None
        assert store.resolve_identity_id(fragment_id) == fragment_id


def test_link_evidence_refuses_a_pair_that_is_already_one_identity(tmp_path: Path) -> None:
    """An identity co-occurring with itself is not evidence, so it is never recorded."""
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        face_id = _face_identity(store, clip, (1.0, 0.0))
        voice_id = _voice_identity(store, clip, (1.0, 0.0))

        assert store.record_identity_link_evidence(voice_id, face_id, clip.asset_id) == 1
        assert store.record_identity_link_evidence(face_id, face_id, clip.asset_id) == 0
        assert store.link_identities(face_id, voice_id) == face_id
        # Both arguments now resolve through the alias to one identity.
        assert store.record_identity_link_evidence(voice_id, face_id, clip.asset_id) == 0

        with closing(sqlite3.connect(store.database_path)) as connection:
            self_pairs = connection.execute(
                "SELECT COUNT(*) FROM identity_link_evidence WHERE voice_id = face_id"
            ).fetchone()[0]

    assert self_pairs == 0


def test_unlink_identity_returns_only_the_contributed_modality(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        face_id = _face_identity(store, clip, (1.0, 0.0))
        assert store.register_identity(face_id, "Alice", relationship="sister") is True
        voice_id = _voice_identity(store, clip, (1.0, 0.0))
        assert store.record_identity_link_evidence(voice_id, face_id, clip.asset_id) == 1
        assert store.link_identities(face_id, voice_id) == face_id
        # Merging folds this pair's evidence into one identity paired with itself, which is
        # meaningless and is dropped, so a merged pair records nothing further.
        assert store.record_identity_link_evidence(voice_id, face_id, clip.asset_id) == 0

        assert store.unlink_identity(voice_id) == voice_id
        speech = store.read_speech(clip.asset_id, space_id="cam++:test")
        faces = store.read_faces(clip.asset_id, space_id="sface:test")
        with closing(sqlite3.connect(store.database_path)) as connection:
            exemplars = connection.execute(
                "SELECT identity_id, modality FROM identity_exemplars"
            ).fetchall()
            aliases = connection.execute("SELECT COUNT(*) FROM identity_aliases").fetchone()[0]
            evidence = connection.execute("SELECT COUNT(*) FROM identity_link_evidence").fetchone()[
                0
            ]
        profile = store.identity_profile(voice_id)

    assert set(exemplars) == {(face_id, "face"), (voice_id, "voice")}
    assert speech is not None and speech[0].speaker_id == voice_id
    assert speech[0].speaker_name is None
    assert faces is not None and faces[0].identity_id == face_id
    assert faces[0].identity_name == "Alice"
    assert profile == IdentityProfile(identity_id=voice_id)
    assert aliases == 0
    assert evidence == 0


def test_unlink_identity_refuses_an_alias_without_a_recorded_modality(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        face_id = _face_identity(store, clip, (1.0, 0.0))
        voice_id = _voice_identity(store, clip, (1.0, 0.0))
        assert store.link_identities(face_id, voice_id) == face_id
        assert store.unlink_identity("identity_missing") is None
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.execute("UPDATE identity_aliases SET contributed_modality = NULL")
        connection.commit()

    with LocalStore(tmp_path) as store:
        assert store.unlink_identity(voice_id) is None
        assert store.resolve_identity_id(voice_id) == face_id
        with closing(sqlite3.connect(store.database_path)) as connection:
            exemplars = connection.execute(
                "SELECT identity_id, modality FROM identity_exemplars"
            ).fetchall()

    assert set(exemplars) == {(face_id, "face"), (face_id, "voice")}


def _drop_asset(store: LocalStore, label: str, asset: StoredAsset) -> None:
    """Delete the memory that owns one asset and then collect the orphaned asset."""
    deleted, unreferenced = store.delete_memory_with_assets(label)
    assert deleted is True
    assert [row.asset_id for row in unreferenced] == [asset.asset_id]
    assert store.delete_asset_if_unreferenced(asset.asset_id) is True


def _identity_rows(store: LocalStore) -> tuple[set[str], set[tuple[str, str]]]:
    with closing(sqlite3.connect(store.database_path)) as connection:
        identities = {
            str(row[0]) for row in connection.execute("SELECT identity_id FROM identities")
        }
        exemplars = {
            (str(row[0]), str(row[1]))
            for row in connection.execute("SELECT identity_id, modality FROM identity_exemplars")
        }
    return identities, exemplars


def test_deleting_an_asset_collects_the_anonymous_identities_it_alone_observed(
    tmp_path: Path,
) -> None:
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        anonymous = _face_identity(store, clip, (1.0, 0.0))
        named = _voice_identity(store, clip, (0.0, 1.0))
        assert store.register_identity(named, "Alice") is True

        _drop_asset(store, "clip", clip)
        identities, exemplars = _identity_rows(store)

    # The anonymous identity was a by-product of the deleted recording, so its exemplar goes
    # with it. The named one is a person the caller asserted; `forget_identity()` erases that.
    assert anonymous not in identities
    assert identities == {named}
    assert exemplars == {(named, "voice")}


def test_an_identity_another_asset_still_observes_survives_the_collection(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        first = _video_asset(store, "first")
        second = _video_asset(store, "second")
        face_id = _face_identity(store, first, (1.0, 0.0))
        voice_id = _voice_identity(store, first, (0.0, 1.0))
        # The same person recorded twice resolves to the same identity both times.
        assert _face_identity(store, second, (1.0, 0.0)) == face_id
        assert _voice_identity(store, second, (0.0, 1.0)) == voice_id

        # `face_observations.identity_id` RESTRICTs, so a regression raises here, but
        # `speech_segments.speaker_id` only SET NULLs, so the speaker has to be read back.
        _drop_asset(store, "first", first)
        identities, _ = _identity_rows(store)
        faces = store.read_faces(second.asset_id, space_id="sface:test")
        speech = store.read_speech(second.asset_id, space_id="cam++:test")

    assert identities == {face_id, voice_id}
    assert faces is not None and faces[0].identity_id == face_id
    assert speech is not None and speech[0].speaker_id == voice_id


def test_collecting_one_asset_leaves_an_unlinked_identity_waiting_for_corroboration(
    tmp_path: Path,
) -> None:
    """An identity `unlink_identity()` restored is untouched by an unrelated asset's collection."""
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        face_id = _face_identity(store, clip, (1.0, 0.0))
        voice_id = _voice_identity(store, clip, (1.0, 0.0))
        assert store.link_identities(face_id, voice_id) == face_id

        # Reversing the merge after the recording is gone leaves two anonymous identities that
        # hold exemplars and observe nothing, which is what continued ingestion re-corroborates.
        _drop_asset(store, "clip", clip)
        assert store.unlink_identity(voice_id) == voice_id

        other = _video_asset(store, "other")
        stranger = _face_identity(store, other, (0.0, 1.0))
        _drop_asset(store, "other", other)
        identities, exemplars = _identity_rows(store)

    assert stranger not in identities
    assert identities == {face_id, voice_id}
    assert exemplars == {(face_id, "face"), (voice_id, "voice")}


def test_unlink_identity_refuses_to_strip_the_target_of_every_exemplar(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        first = _video_asset(store, "first")
        second = _video_asset(store, "second")
        first_id = _face_identity(store, first, (1.0, 0.0))
        second_id = _face_identity(store, second, (0.0, 1.0))
        target = store.link_identities(first_id, second_id, allow_shared_modality=True)
        assert target is not None
        source = second_id if target == first_id else first_id

        assert store.unlink_identity(source) is None
        assert store.resolve_identity_id(source) == target
        with closing(sqlite3.connect(store.database_path)) as connection:
            exemplars = connection.execute(
                "SELECT DISTINCT identity_id FROM identity_exemplars"
            ).fetchall()

    assert exemplars == [(target,)]


def test_identity_relationship_is_stored_and_validated(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        face_id = _face_identity(store, clip, (1.0, 0.0))

        assert store.identity_profile(face_id) == IdentityProfile(identity_id=face_id)
        assert store.identity_profile("identity_missing") is None
        assert store.register_identity(face_id, "Alice", relationship="sister") is True
        assert store.identity_profile(face_id) == IdentityProfile(
            identity_id=face_id, name="Alice", relationship="sister"
        )
        assert store.register_identity(face_id, "Alicia") is True
        assert store.identity_profile(face_id) == IdentityProfile(
            identity_id=face_id, name="Alicia", relationship="sister"
        )
        for invalid in ("x" * 256, "side\nkick"):
            with pytest.raises(ValueError, match="identity relationship must be at most 255"):
                store.register_identity(face_id, "Alice", relationship=invalid)
        with pytest.raises(ValueError, match="identity relationship must be non-empty"):
            store.register_identity(face_id, "Alice", relationship="   ")
        assert store.identity_profile(face_id) == IdentityProfile(
            identity_id=face_id, name="Alicia", relationship="sister"
        )


def test_memory_hydration_uses_one_wal_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    asset = _asset()
    expected = replace(_memory(), content="", modality="image", assets=(asset,))

    with LocalStore(tmp_path) as store:
        store.write_memory(expected)
        original = LocalStore._read_memory_assets
        deleted = False

        def delete_between_selects(
            connection: sqlite3.Connection,
            memory_ids: tuple[str, ...],
        ) -> dict[str, tuple[StoredAsset, ...]]:
            nonlocal deleted
            if not deleted:
                deleted = True
                with closing(sqlite3.connect(store.database_path, isolation_level=None)) as writer:
                    writer.execute("PRAGMA foreign_keys = ON")
                    writer.execute(
                        "DELETE FROM memory_records WHERE memory_id = ?",
                        (expected.memory_id,),
                    )
            return original(connection, memory_ids)

        monkeypatch.setattr(LocalStore, "_read_memory_assets", staticmethod(delete_between_selects))
        assert store.read_memory(expected.memory_id) == expected
        assert store.read_memory(expected.memory_id) is None


@pytest.mark.parametrize(
    ("modality", "assets"),
    [
        ("text", (_asset(),)),
        ("image", ()),
        ("image", (_asset(modality="audio", mime_type="audio/wav"),)),
        ("omni", (_asset(),)),
    ],
)
def test_stored_memory_rejects_modality_asset_mismatches(
    modality: str,
    assets: tuple[StoredAsset, ...],
) -> None:
    with pytest.raises(ValueError, match=r"memories|modality|assets"):
        replace(_memory(), modality=modality, assets=assets)


def test_shared_assets_are_collected_only_after_the_last_reference(tmp_path: Path) -> None:
    asset = _asset()
    first = replace(_memory("memory-1"), modality="image", assets=(asset,))
    second = replace(_memory("memory-2"), modality="image", assets=(asset,))

    with LocalStore(tmp_path) as store:
        assert store.write_memories((first, second)) == (True, True)
        assert store.delete_asset_if_unreferenced(asset.asset_id) is False
        assert store.delete_memory_with_assets(first.memory_id) == (True, ())
        deleted, unreferenced = store.delete_memory_with_assets(second.memory_id)
        assert deleted is True
        assert unreferenced == (asset,)
        assert store.list_unreferenced_assets() == (asset,)
        assert store.delete_asset_if_unreferenced(asset.asset_id) is True
        assert store.delete_asset_if_unreferenced(asset.asset_id) is False


def test_conflicting_asset_metadata_rolls_back_the_memory_write(tmp_path: Path) -> None:
    image = _asset()
    audio = replace(image, modality="audio", mime_type="audio/wav", name="audio.wav")
    first = replace(_memory("memory-1"), modality="image", assets=(image,))
    conflicting = replace(_memory("memory-2"), modality="audio", assets=(audio,))

    with LocalStore(tmp_path) as store:
        store.write_memory(first)
        with pytest.raises(ValueError, match="conflicts with stored metadata"):
            store.write_memory(conflicting)
        assert store.read_memory(conflicting.memory_id) is None
        assert store.read_memory(first.memory_id) == first


def test_state_order_uses_storage_batch_not_wall_clock_equality(tmp_path: Path) -> None:
    recorded_at = datetime(2026, 8, 27, 1, 2, 3, tzinfo=timezone.utc)
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    february = datetime(2026, 2, 1, tzinfo=timezone.utc)
    march = datetime(2026, 3, 1, tzinfo=timezone.utc)
    known_at = recorded_at + timedelta(microseconds=2)
    tea_source = _memory("tea-source", "tea evidence", created_at=recorded_at)
    coffee_source = _memory("coffee-source", "coffee evidence", created_at=recorded_at)
    tea = _state_memory(
        "tea-state",
        tea_source.memory_id,
        "tea",
        valid_from=january,
        recorded_at=recorded_at,
    )
    coffee = _state_memory(
        "coffee-state",
        coffee_source.memory_id,
        "coffee",
        valid_from=february,
        recorded_at=recorded_at,
    )

    with LocalStore(tmp_path / "sequential") as store:
        store.write_memories((tea_source, tea))
        store.write_memories((coffee_source, coffee))
        current = store.read_memories(
            (tea.memory_id, coffee.memory_id),
            valid_at=march,
            known_at=known_at,
            active_only=True,
        )

    with LocalStore(tmp_path / "same-batch") as store:
        store.write_memories((tea_source, coffee_source, tea, coffee))
        conflicting = store.read_memories(
            (tea.memory_id, coffee.memory_id),
            valid_at=march,
            known_at=known_at,
            active_only=True,
        )

    assert [memory.context.value for memory in current if memory.context] == ["coffee"]
    assert {memory.context.value for memory in conflicting if memory.context} == {
        "coffee",
        "tea",
    }


def test_separate_lineage_writes_advance_a_frozen_transaction_clock(tmp_path: Path) -> None:
    recorded_at = datetime(2026, 8, 27, 1, 2, 3, tzinfo=timezone.utc)
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    february = datetime(2026, 2, 1, tzinfo=timezone.utc)
    march = datetime(2026, 3, 1, tzinfo=timezone.utc)
    april = datetime(2026, 4, 1, tzinfo=timezone.utc)
    first_source = _memory("first-source", "first evidence", created_at=recorded_at)
    second_source = _memory("second-source", "second evidence", created_at=recorded_at)
    first = _state_memory(
        "first-state",
        first_source.memory_id,
        "tea",
        valid_from=january,
        valid_until=february,
        recorded_at=recorded_at,
    )
    second = _state_memory(
        "second-state",
        second_source.memory_id,
        "coffee",
        valid_from=march,
        valid_until=april,
        recorded_at=recorded_at,
    )
    next_transaction = recorded_at + timedelta(microseconds=1)

    with LocalStore(tmp_path) as store:
        store.write_memories((first_source, first))
        store.write_memories((second_source, second))
        before = store.read_memories(
            (second.memory_id,),
            valid_at=march,
            known_at=recorded_at,
            active_only=True,
        )
        after = store.read_memories(
            (second.memory_id,),
            valid_at=march,
            known_at=next_transaction,
            active_only=True,
        )
        with closing(sqlite3.connect(store.database_path)) as connection:
            evidence_time = connection.execute(
                "SELECT recorded_at FROM memory_evidence WHERE memory_id = ?",
                (second.memory_id,),
            ).fetchone()

    assert before == ()
    assert len(after) == 1
    assert after[0].context is not None
    assert after[0].context.recorded_at == next_transaction
    assert evidence_time == (
        next_transaction.isoformat(timespec="microseconds").replace("+00:00", "Z"),
    )


def test_evidence_and_projection_share_one_transaction_time(tmp_path: Path) -> None:
    recorded_at = datetime(2026, 8, 27, 1, 2, 3, tzinfo=timezone.utc)
    first = _memory("source-1", "first witness", created_at=recorded_at)
    second = _memory("source-2", "second witness", created_at=recorded_at)
    entity = replace(
        _memory("entity", "the user", created_at=recorded_at),
        context=MemoryContext(
            kind=MemoryKind.ENTITY,
            basis=EvidenceBasis.MODEL_INFERENCE,
            confidence=0.8,
            valid_from=None,
            valid_until=None,
            recorded_at=recorded_at,
            lineage_id="user",
            subject="user",
            evidence_ids=(first.memory_id,),
        ),
    )

    with LocalStore(tmp_path) as store:
        store.write_memories((first, second, entity))
        assert store.add_memory_evidence(
            entity.memory_id,
            second.memory_id,
            confidence=0.8,
            recorded_at=recorded_at,
        )
        prior = store.read_memories(
            (entity.memory_id,),
            known_at=recorded_at,
            active_only=True,
        )[0].context
        current = store.read_memory(entity.memory_id)
        assert current is not None and current.context is not None

        store.delete_memory(second.memory_id)
        retracted = store.read_memory(entity.memory_id)
        assert retracted is not None and retracted.context is not None
        before_retraction = store.read_memories(
            (entity.memory_id,),
            known_at=retracted.context.recorded_at - timedelta(microseconds=1),
            active_only=True,
        )[0].context

    assert prior is not None
    assert prior.evidence_ids == (first.memory_id,)
    assert prior.confidence == pytest.approx(0.8)
    assert current.context.evidence_ids == (first.memory_id, second.memory_id)
    assert current.context.confidence == pytest.approx(0.96)
    assert before_retraction is not None
    assert before_retraction.evidence_ids == (first.memory_id, second.memory_id)
    assert before_retraction.confidence == pytest.approx(0.96)
    assert retracted.context.evidence_ids == (first.memory_id,)
    assert retracted.context.confidence == pytest.approx(0.8)


def test_schema_v10_adds_forgetting_and_control_plane_state(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        store.write_memories((_memory(),), (_embedding(),))
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            """
            DROP TABLE capture_queue;
            DROP TABLE memory_operations;
            ALTER TABLE memory_records DROP COLUMN forgotten_at;
            PRAGMA user_version = 10;
            """
        )

    with LocalStore(tmp_path) as store:
        memory = store.read_memory("memory-1")
        with closing(sqlite3.connect(store.database_path)) as connection:
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert memory is not None and memory.forgotten_at is None
    assert {"capture_queue", "memory_operations"} <= tables
    assert version == _SCHEMA_VERSION


def test_schema_v11_adds_the_identity_binding_on_typed_claims(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        store.write_memories((_memory(),), (_embedding(),))
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            """
            DROP INDEX memory_semantics_identity_idx;
            ALTER TABLE memory_semantics DROP COLUMN identity_id;
            PRAGMA user_version = 11;
            """
        )

    with LocalStore(tmp_path) as store:
        memory = store.read_memory("memory-1")
        with closing(sqlite3.connect(store.database_path)) as connection:
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(memory_semantics)")
            }
            indexes = {
                str(row[1]) for row in connection.execute("PRAGMA index_list(memory_semantics)")
            }
            version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert memory is not None
    assert "identity_id" in columns
    assert "memory_semantics_identity_idx" in indexes
    assert version == _SCHEMA_VERSION


def test_forgotten_memories_leave_active_reads_but_stay_auditable(tmp_path: Path) -> None:
    forgotten_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        store.write_memories(
            (_memory("memory-1"), _memory("memory-2")),
            (_embedding("e-1", "memory-1"), _embedding("e-2", "memory-2")),
        )

        assert store.set_forgotten(("memory-1", "memory-1"), forgotten_at=forgotten_at) == (
            "memory-1",
        )
        assert store.set_forgotten(("memory-1",), forgotten_at=forgotten_at) == ()

        active = store.read_memories(("memory-1", "memory-2"), active_only=True)
        audit = store.read_memories(("memory-1", "memory-2"))
        listed = store.list_memories(limit=10)

        assert [memory.memory_id for memory in active] == ["memory-2"]
        assert [memory.forgotten_at for memory in audit] == [forgotten_at, None]
        assert {memory.memory_id for memory in listed} == {"memory-1", "memory-2"}

        assert store.set_forgotten(("memory-1", "memory-2"), forgotten_at=None) == ("memory-1",)
        restored = store.read_memories(("memory-1", "memory-2"), active_only=True)
        assert [memory.memory_id for memory in restored] == ["memory-1", "memory-2"]


def test_index_candidates_project_the_columns_ranking_reads(tmp_path: Path) -> None:
    with LocalStore(tmp_path / "state.sqlite3") as store:
        store.write_memories(
            (_memory("first", content="winter"), _memory("second", content="summer")),
            (
                _embedding("embedding-first", "first"),
                _embedding("embedding-second", "second"),
            ),
        )
        # Input order is the ranking order, so the projection has to preserve it rather than
        # return whatever order SQLite chose, and an unknown id has to drop out silently the way
        # the full-document read does.
        candidates = store.read_index_candidates(
            ("embedding-second", "embedding-missing", "embedding-first")
        )
        documents = store.read_index_documents(("embedding-first",))

    assert [candidate.memory_id for candidate in candidates] == ["second", "first"]
    # The projection has to agree with the read it replaces on every column ranking consults.
    assert candidates[1].embedding_id == documents[0].embedding.embedding_id
    assert candidates[1].occurred_at == documents[0].occurred_at
    assert candidates[1].occurred_end == documents[0].occurred_end


def _unit(values: list[float]) -> list[float]:
    scale = math.sqrt(math.fsum(value * value for value in values))
    return [value / scale for value in values]


def _speaker_population(
    people: int,
    per_person: int,
    *,
    shared: float = 0.42,
    personal: float = 0.32,
    seed: int = 5,
) -> dict[tuple[int, int], tuple[float, ...]]:
    """Speaker vectors laid out over three orthogonal blocks, at a controlled separability.

    ``v = sqrt(shared)*g + sqrt(personal)*q_k + sqrt(rest)*n_i``. Two utterances of one speaker
    have cosine ``shared + personal`` in expectation and two speakers have ``shared``; the
    spread around both means comes from the random blocks, whose widths are their dimensions.

    The four constants are fitted to a real corpus rather than chosen: an m3-bench-robot unit
    (68 memories, 158 CAM++ voice exemplars, 192 dimensions) measures within-identity cosine
    0.7385 +/- 0.0717 and 0.4244 +/- 0.2043 between identities, and this generator reproduces
    0.7432 +/- 0.0716 and 0.4325 +/- 0.2043. Real speaker embeddings are that badly separated:
    the two distributions are single broad humps that overlap, not two modes with a valley.
    """
    person_dimension, utterance_dimension = 3, 14
    rest = 1.0 - shared - personal
    generator = random.Random(seed)
    directions = [
        _unit([generator.gauss(0.0, 1.0) for _ in range(person_dimension)]) for _ in range(people)
    ]
    return {
        (person, index): tuple(
            [math.sqrt(shared)]
            + [math.sqrt(personal) * value for value in directions[person]]
            + [
                math.sqrt(rest) * value
                for value in _unit([generator.gauss(0.0, 1.0) for _ in range(utterance_dimension)])
            ]
        )
        for person in range(people)
        for index in range(per_person)
    }


def _cluster_speakers(
    store: LocalStore,
    population: dict[tuple[int, int], tuple[float, ...]],
    *,
    minimum_similarity: float,
    minimum_margin: float,
    order_seed: int = 1,
) -> tuple[int, float, int, int]:
    """Match a whole population one utterance at a time; return count, purity and pair recall.

    Purity is reported beside the identity count on purpose. Alone it is worthless: a store
    that files every utterance under its own identity scores a perfect 1.000.
    """
    keys = list(population)
    random.Random(order_seed).shuffle(keys)
    assigned: dict[str, list[int]] = {}
    for position, key in enumerate(keys):
        asset = _asset(f"utterance {position}".encode(), modality="audio", mime_type="audio/wav")
        store.write_memory(
            replace(
                _memory(f"memory-{position}"),
                content="",
                modality="audio",
                assets=(asset,),
            )
        )
        speaker = store.write_speech(
            asset.asset_id,
            SpeechAnalysis(
                turns=(SpeechTurn(0, 500, "spoken", "0"),),
                speakers=(SpeakerEmbedding("0", population[key]),),
            ),
            model_id="cam++",
            space_id="cam++:test",
            minimum_similarity=minimum_similarity,
            minimum_margin=minimum_margin,
        )[0].speaker_id
        assert speaker is not None
        assigned.setdefault(speaker, []).append(key[0])
    purity = math.fsum(
        max(members.count(person) for person in set(members)) for members in assigned.values()
    ) / len(keys)
    linked = sum(
        members.count(person) * (members.count(person) - 1) // 2
        for members in assigned.values()
        for person in set(members)
    )
    per_person = len(keys) // len({key[0] for key in keys})
    gold = len({key[0] for key in keys}) * per_person * (per_person - 1) // 2
    return len(assigned), purity, linked, gold


def test_no_similarity_floor_recovers_the_speakers_at_a_realistic_separability(
    tmp_path: Path,
) -> None:
    """Speaker matching trades purity against recall; no floor and margin buys both.

    Six speakers say fourteen things each, at the separability measured on a real CAM++
    population. A low floor merges strangers, a high floor files everyone separately, and the
    shipped default sits at the second extreme: it keeps purity by declining to match. There
    is no setting in between, because the within-speaker and between-speaker cosines overlap.

    If this test ever fails because some grid point reaches both, the speaker representation
    has genuinely improved and this characterisation is obsolete. Read the numbers in the
    message before changing the assertion.
    """
    population = _speaker_population(6, 14)
    outcomes = {}
    for floor, margin in ((0.60, 0.05), (0.70, 0.0), (0.78, 0.0), (0.78, 0.05), (0.90, 0.05)):
        with LocalStore(tmp_path / f"floor{floor}margin{margin}") as store:
            outcomes[(floor, margin)] = _cluster_speakers(
                store, population, minimum_similarity=floor, minimum_margin=margin
            )

    solved = {
        setting: outcome
        for setting, outcome in outcomes.items()
        if outcome[1] >= 0.99 and outcome[2] >= outcome[3] // 2
    }
    assert not solved, f"a threshold recovered the speakers: {outcomes}"
    # Both failure modes are present, so the grid really does span the trade-off.
    assert min(outcome[1] for outcome in outcomes.values()) < 0.8
    assert max(outcome[0] for outcome in outcomes.values()) > 6 * 6
    # Purity on its own cannot see the second one: the strictest floor scores a perfect 1.000
    # while linking almost nothing.
    strictest = outcomes[(0.90, 0.05)]
    assert strictest[1] == pytest.approx(1.0)
    assert strictest[2] < strictest[3] // 20
    # At one floor, the margin is the whole trade: it declines the matches whose runner-up is
    # too close, which is most of them once the store holds rivals.
    without_margin, with_margin = outcomes[(0.78, 0.0)], outcomes[(0.78, 0.05)]
    assert with_margin[1] > without_margin[1]
    assert with_margin[2] < without_margin[2]


def test_valid_at_keeps_records_without_a_declared_validity_interval(tmp_path: Path) -> None:
    recorded_at = datetime(2026, 8, 27, 1, 2, 3, tzinfo=timezone.utc)
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    march = datetime(2026, 3, 1, tzinfo=timezone.utc)
    last_year = datetime(2025, 6, 1, tzinfo=timezone.utc)
    plain = _memory("plain", "the user drank tea", created_at=recorded_at)
    source = _memory("state-source", "tea evidence", created_at=recorded_at)
    typed = _state_memory(
        "typed",
        source.memory_id,
        "tea",
        valid_from=january,
        recorded_at=recorded_at,
    )
    elsewhere = SpatialContext(
        frame_id="workshop",
        anchor=SpatialAnchor.OBSERVER,
        x=0.0,
        y=0.0,
        z=0.0,
    )

    with LocalStore(tmp_path) as store:
        assert store.write_memories((plain, source, typed)) == (True, True, True)
        selected = (plain.memory_id, typed.memory_id)
        unscoped = store.read_memories(selected, active_only=True)
        during = store.read_memories(selected, valid_at=march, active_only=True)
        before = store.read_memories(selected, valid_at=last_year, active_only=True)
        unknown = store.read_memories(
            selected,
            valid_at=march,
            known_at=recorded_at - timedelta(microseconds=1),
            active_only=True,
        )
        nowhere = store.read_memories(
            selected,
            near=elsewhere,
            radius_m=1.0,
            active_only=True,
        )

    # A record with no validity interval answers every `valid_at`, and only the typed interval
    # can refuse one. A record with no pose is at no location, so `near` still drops both.
    assert [memory.memory_id for memory in unscoped] == ["plain", "typed"]
    assert [memory.memory_id for memory in during] == ["plain", "typed"]
    assert [memory.memory_id for memory in before] == ["plain"]
    assert unknown == ()
    assert nowhere == ()


_COUNT_RECORDED_AT = datetime(2026, 8, 27, 1, 2, 3, tzinfo=timezone.utc)
_COUNT_HERE = SpatialContext(
    frame_id="workshop",
    anchor=SpatialAnchor.OBSERVER,
    x=1.0,
    y=1.0,
    z=0.0,
)


def _count_corpus() -> tuple[StoredMemory, ...]:
    """One record per scope axis: none, validity interval, pose, symbolic place."""
    january = datetime(2026, 1, 1, tzinfo=timezone.utc)
    february = datetime(2026, 2, 1, tzinfo=timezone.utc)
    source = _memory("evidence", "tea evidence", created_at=_COUNT_RECORDED_AT)
    current = _state_memory(
        "typed-current",
        source.memory_id,
        "tea",
        valid_from=january,
        recorded_at=_COUNT_RECORDED_AT,
    )
    return (
        _memory("plain", "the user drank tea", created_at=_COUNT_RECORDED_AT),
        source,
        current,
        _state_memory(
            "typed-expired",
            source.memory_id,
            "coffee",
            valid_from=january,
            valid_until=february,
            recorded_at=_COUNT_RECORDED_AT,
        ),
        replace(
            current,
            memory_id="typed-here",
            context=replace(_require_context(current.context), spatial=_COUNT_HERE),
        ),
        replace(_memory("placed", "in the kitchen", created_at=_COUNT_RECORDED_AT), place_id="k1"),
    )


def _require_context(context: MemoryContext | None) -> MemoryContext:
    assert context is not None
    return context


# Every scope axis `read_memories` accepts, plus the requested-ID quirks the count has to
# reproduce: a missing ID contributes nothing and a repeated ID is counted twice.
_COUNT_SCOPES: tuple[tuple[str, dict[str, object], int], ...] = (
    ("unscoped", {}, 7),
    ("active_only", {"active_only": True}, 6),
    (
        "valid_at inside the expired interval",
        {"active_only": True, "valid_at": datetime(2026, 1, 15, tzinfo=timezone.utc)},
        7,
    ),
    (
        "valid_at after it",
        {"active_only": True, "valid_at": datetime(2026, 3, 1, tzinfo=timezone.utc)},
        6,
    ),
    ("valid_at without active_only", {"valid_at": datetime(2026, 3, 1, tzinfo=timezone.utc)}, 7),
    (
        "known_at before anything was recorded",
        {"active_only": True, "known_at": _COUNT_RECORDED_AT - timedelta(microseconds=1)},
        0,
    ),
    ("known_at after", {"active_only": True, "known_at": _COUNT_RECORDED_AT}, 6),
    ("near the pose", {"active_only": True, "near": _COUNT_HERE, "radius_m": 0.5}, 1),
    ("near without active_only", {"near": _COUNT_HERE, "radius_m": 0.5}, 7),
    (
        "out of radius",
        {"active_only": True, "near": replace(_COUNT_HERE, x=100.0), "radius_m": 0.5},
        0,
    ),
    ("place_id", {"place_id": "k1"}, 1),
    ("place_id with active_only", {"place_id": "k1", "active_only": True}, 1),
    ("unknown place_id", {"place_id": "nowhere", "active_only": True}, 0),
)


@pytest.mark.parametrize(
    ("scope", "expected"),
    [pytest.param(scope, expected, id=label) for label, scope, expected in _COUNT_SCOPES],
)
def test_count_memories_matches_the_scoped_hydration_it_replaces(
    tmp_path: Path,
    scope: dict[str, object],
    expected: int,
) -> None:
    """The count drives candidate widening, so it has to equal the hydration exactly.

    `expected` is asserted as well as the equality: two implementations that both return
    nothing agree on every axis.
    """
    corpus = _count_corpus()
    requested = (*(memory.memory_id for memory in corpus), "missing", "plain")

    with LocalStore(tmp_path) as store:
        assert store.write_memories(corpus) == (True,) * len(corpus)
        hydrated = store.read_memories(requested, **scope)  # type: ignore[arg-type]
        counted = store.count_memories(requested, **scope)  # type: ignore[arg-type]
        assert store.count_memories((), **scope) == 0  # type: ignore[arg-type]

    assert counted == len(hydrated) == expected


def _naming_assertion(
    memory_id: str,
    identity_id: str,
    name: str,
    *,
    recorded_at: datetime,
    relationship: str | None = None,
    basis: EvidenceBasis = EvidenceBasis.USER_STATEMENT,
    evidence_ids: tuple[str, ...] = (),
) -> StoredMemory:
    """One naming assertion, shaped the way `Memory.register_identity` writes it."""
    return replace(
        _memory(memory_id, f"{name} is a recognized person", created_at=recorded_at),
        context=MemoryContext(
            kind=MemoryKind.ENTITY,
            basis=basis,
            confidence=1.0,
            valid_from=None,
            valid_until=None,
            recorded_at=recorded_at,
            lineage_id=f"naming:{identity_id}",
            subject=name,
            predicate="identity",
            value=relationship,
            identity_id=identity_id,
            evidence_ids=evidence_ids,
        ),
    )


def _bound_claim(
    memory_id: str,
    identity_id: str,
    *,
    evidence_ids: tuple[str, ...],
    recorded_at: datetime,
) -> StoredMemory:
    return replace(
        _memory(memory_id, "the neighbour prefers tea", created_at=recorded_at),
        context=MemoryContext(
            kind=MemoryKind.STATE,
            basis=EvidenceBasis.MODEL_INFERENCE,
            confidence=0.9,
            valid_from=recorded_at,
            valid_until=None,
            recorded_at=recorded_at,
            lineage_id=f"claim:{memory_id}",
            subject="the neighbour",
            predicate="preferred_drink",
            value="tea",
            identity_id=identity_id,
            evidence_ids=evidence_ids,
        ),
    )


def _semantic_bindings(store: LocalStore) -> dict[str, str | None]:
    with closing(sqlite3.connect(store.database_path)) as connection:
        return {
            str(row[0]): None if row[1] is None else str(row[1])
            for row in connection.execute("SELECT memory_id, identity_id FROM memory_semantics")
        }


def test_the_projection_follows_every_visibility_change_of_a_naming_assertion(
    tmp_path: Path,
) -> None:
    """`identities.name` is the name of the currently visible assertion, or nothing.

    The registry is a projection, so no path that retires, hides or deletes the assertion may
    leave a name standing that nothing asserts -- including the paths that know nothing about
    identities, which is every one of them but naming.
    """
    recorded_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        voice_id = _voice_identity(store, clip, (1.0, 0.0))
        store.write_memories(
            (
                _naming_assertion("naming-1", voice_id, "Li", recorded_at=recorded_at),
                _naming_assertion(
                    "naming-2",
                    voice_id,
                    "Li Hua",
                    recorded_at=recorded_at + timedelta(minutes=1),
                ),
            ),
            (_embedding("e-naming-1", "naming-1"), _embedding("e-naming-2", "naming-2")),
        )
        # The store projects on write: the newer assertion supersedes the older one.
        assert store.identity_profile(voice_id) == IdentityProfile(
            identity_id=voice_id,
            name="Li Hua",
            confirmed=True,
        )

        # Cognitive forgetting hides the record, so it stops naming anybody.
        store.set_forgotten(("naming-2",), forgotten_at=recorded_at + timedelta(minutes=2))
        assert store.projected_identity_name(voice_id) == ("Li", None)

        store.set_forgotten(("naming-2",), forgotten_at=None)
        assert store.projected_identity_name(voice_id) == ("Li Hua", None)

        # A correction through the control plane retires the version.
        applied = store.apply_control_operation(
            StoredOperation(
                operation_key="correct-naming-2",
                intent="correct",
                trigger="manual",
                operation_json="{}",
                applied_at=recorded_at + timedelta(minutes=3),
            ),
            correct_ids=("naming-2",),
        )
        assert applied is not None
        assert store.projected_identity_name(voice_id) == ("Li", None)

        # And deleting the last one leaves nobody named, in the ordinary delete path.
        assert store.delete_memory("naming-1") is True
        speech = store.read_speech(clip.asset_id, space_id="cam++:test")
        profile = store.identity_profile(voice_id)

    assert profile == IdentityProfile(identity_id=voice_id)
    assert speech is not None and speech[0].speaker_name is None


def test_a_backdated_forget_does_not_break_the_identity_projection(tmp_path: Path) -> None:
    """`identities.updated_at` is transaction time, not the caller's semantic time.

    A host may forget or retract a record with a timestamp in the past -- backfilling something it
    learned late. The name projection is rewritten in that same transaction, so stamping the
    caller's time would put the identity row's `updated_at` before its own `created_at` and the
    CHECK constraint would surface as an `IntegrityError` out of a public store method.
    """
    long_ago = datetime(2020, 1, 1, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        voice_id = _voice_identity(store, clip, (1.0, 0.0))
        store.write_memories(
            (_naming_assertion("naming-1", voice_id, "Li", recorded_at=long_ago),),
            (_embedding("e-naming-1", "naming-1"),),
        )
        assert store.projected_identity_name(voice_id) == ("Li", None)

        assert store.set_forgotten(("naming-1",), forgotten_at=long_ago) == ("naming-1",)

        assert store.projected_identity_name(voice_id) == (None, None)


def test_merging_identities_repoints_the_naming_assertion_onto_the_survivor(
    tmp_path: Path,
) -> None:
    """A merge must not orphan the source's assertion, or the name becomes unsupported."""
    recorded_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        face_id = _face_identity(store, clip, (1.0, 0.0))
        voice_id = _voice_identity(store, clip, (1.0, 0.0))
        store.write_memories(
            (
                _naming_assertion("naming-face", face_id, "Alice", recorded_at=recorded_at),
                _naming_assertion(
                    "naming-voice",
                    voice_id,
                    "Alice",
                    recorded_at=recorded_at + timedelta(minutes=1),
                ),
            ),
            (_embedding("e-face", "naming-face"), _embedding("e-voice", "naming-voice")),
        )
        plan = store.identity_link_plan(face_id, voice_id)
        assert plan is not None and plan.name == "Alice"
        target = store.link_identities(face_id, voice_id, expected=plan)

        assert target == plan.target_id
        assert _semantic_bindings(store) == {"naming-face": target, "naming-voice": target}
        assert store.identity_profile(target) == IdentityProfile(
            identity_id=target,
            name="Alice",
            confirmed=True,
        )


def test_unlinking_re_attributes_or_unbinds_the_claims_made_while_two_were_one(
    tmp_path: Path,
) -> None:
    """Scenario step 6: a wrong merge must not leave claims attached to the wrong person."""
    recorded_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        seen = _video_asset(store, "seen")
        heard = _video_asset(store, "heard")
        face_id = _face_identity(store, seen, (1.0, 0.0))
        assert store.register_identity(face_id, "Li") is True
        voice_id = _voice_identity(store, heard, (0.0, 1.0))
        assert store.link_identities(face_id, voice_id) == face_id
        store.write_memories(
            (
                _bound_claim(
                    "heard-only",
                    face_id,
                    evidence_ids=("heard",),
                    recorded_at=recorded_at,
                ),
                _bound_claim(
                    "both",
                    face_id,
                    evidence_ids=("seen", "heard"),
                    recorded_at=recorded_at,
                ),
                _bound_claim(
                    "seen-only",
                    face_id,
                    evidence_ids=("seen",),
                    recorded_at=recorded_at,
                ),
            ),
            (
                _embedding("e-heard-only", "heard-only"),
                _embedding("e-both", "both"),
                _embedding("e-seen-only", "seen-only"),
            ),
        )

        assert store.unlink_identity(voice_id) == voice_id
        bindings = _semantic_bindings(store)

    # The claim resting only on the clip that moved back follows the restored person; the one
    # resting on both people's media is attributed to nobody; the one that never involved the
    # restored person is untouched.
    assert bindings["heard-only"] == voice_id
    assert bindings["both"] is None
    assert bindings["seen-only"] == face_id


def test_provisional_identities_name_the_people_no_assertion_names(tmp_path: Path) -> None:
    recorded_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        seen = _video_asset(store, "seen")
        heard = _video_asset(store, "heard")
        face_id = _face_identity(store, seen, (1.0, 0.0))
        voice_id = _voice_identity(store, heard, (0.0, 1.0))
        store.write_memories(
            (_naming_assertion("naming-face", face_id, "Li", recorded_at=recorded_at),),
            (_embedding("e-naming-face", "naming-face"),),
        )

        assert store.identity_for_subject("  LI  ") == face_id
        assert store.identity_for_subject("nobody") is None
        assert store.provisional_identities(("seen", "heard", "memory-missing")) == {
            "heard": (voice_id,)
        }


def test_rolling_back_an_operation_deletes_its_records_in_the_same_transaction(
    tmp_path: Path,
) -> None:
    applied_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        store.write_memories(
            (_memory("kept"), _memory("created")),
            (_embedding("e-kept", "kept"), _embedding("e-created", "created")),
        )
        logged = store.apply_control_operation(
            StoredOperation(
                operation_key="op-1",
                intent="forget",
                trigger="manual",
                operation_json="{}",
                applied_at=applied_at,
            ),
            forget_ids=("kept",),
        )
        assert logged is not None

        reverted, orphaned = store.rollback_operation(
            logged.operation_id,
            rolled_back_at=applied_at + timedelta(minutes=1),
            clear_forgotten=("kept",),
            delete_memory_ids=("created",),
        )

        assert reverted is True and orphaned == ()
        assert store.read_memory("created") is None
        kept = store.read_memory("kept")
        assert kept is not None and kept.forgotten_at is None
        assert store.read_operations(operation_id=logged.operation_id)[0].rolled_back_at is not None
        assert store.rollback_operation(
            logged.operation_id,
            rolled_back_at=applied_at + timedelta(minutes=2),
            delete_memory_ids=("kept",),
        ) == (False, ())
        assert store.read_memory("kept") is not None


def test_a_failed_control_schema_step_leaves_the_v10_store_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_migrate_v10` runs inside one transaction, so a failure rolls the whole rung back.

    `executescript` used to be how the control tables were created, and it issues an implicit
    COMMIT: the `forgotten_at` column landed before the tables did, and a failure there left a
    store at `user_version = 10` with half the rung applied and no way to open it again.
    """
    with LocalStore(tmp_path) as store:
        store.write_memories((_memory(),), (_embedding(),))
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            """
            DROP TABLE capture_queue;
            DROP TABLE memory_operations;
            ALTER TABLE memory_records DROP COLUMN forgotten_at;
            PRAGMA user_version = 10;
            """
        )
    monkeypatch.setattr(
        store_module,
        "_CONTROL_SCHEMA",
        (store_module._CONTROL_SCHEMA[0], "CREATE TABLE memory_records (broken TEXT)"),
    )

    with pytest.raises(sqlite3.OperationalError):
        LocalStore(tmp_path)

    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(memory_records)")}
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert version == 10
    assert "forgotten_at" not in columns
    assert "capture_queue" not in tables


def test_pending_captures_are_globally_oldest_first_across_id_chunks(tmp_path: Path) -> None:
    """More than 900 ids are queried in chunks; the oldest rows may all be in the last one."""
    enqueued_at = datetime(2026, 9, 4, 9, tzinfo=timezone.utc)
    recent = tuple(f"recent-{index:04d}" for index in range(900))
    oldest = tuple(f"oldest-{index}" for index in range(10))
    with LocalStore(tmp_path) as store:
        store.write_captures(
            (_memory(memory_id) for memory_id in oldest),
            enqueued_at=enqueued_at,
        )
        store.write_captures(
            (_memory(memory_id) for memory_id in recent),
            enqueued_at=enqueued_at + timedelta(hours=1),
        )
        # The recent ids fill the first chunk, so the oldest are only visible in the second.
        pending = store.pending_captures(limit=5, memory_ids=(*recent, *oldest))

    assert [row.memory_id for row in pending] == list(oldest[:5])


def test_the_current_schema_indexes_recalled_records(tmp_path: Path) -> None:
    """`_feedback_candidates` polls `last_accessed_at DESC, memory_id` on every candidate sweep."""
    with LocalStore(tmp_path) as store:
        database_path = store.database_path
    with closing(sqlite3.connect(database_path)) as connection:
        plan = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT memory_id, last_accessed_at, access_count
            FROM memory_records
            WHERE last_accessed_at IS NOT NULL AND forgotten_at IS NULL
            ORDER BY last_accessed_at DESC, memory_id
            LIMIT 4
            """
        ).fetchall()

    assert any("memory_records_accessed_idx" in str(row[3]) for row in plan)


def test_the_last_rung_indexes_recalled_records_and_rekeys_the_operation_log(
    tmp_path: Path,
) -> None:
    """A store one step behind gains the index and has its stale operation keys recomputed.

    `operation_key` gained a `claim` field without re-keying what was already logged, so every
    stored key stopped matching what the kernel recomputes and deduplication silently stopped.
    """
    applied_at = datetime(2026, 9, 4, 10, tzinfo=timezone.utc)
    operation = MemoryOperation(
        intent=MemoryIntent.IDENTIFY,
        claim=IdentityClaim(identity_id="identity-1", name="Li"),
        rationale="the host said so",
    )
    logged = StoredOperation(
        operation_key="a-key-the-old-algorithm-produced",
        intent="identify",
        trigger="manual",
        recipe="test-recipe",
        operation_json=dump_operation(operation),
        applied_at=applied_at,
    )
    with LocalStore(tmp_path) as store:
        assert store.apply_control_operation(logged) is not None
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            """
            DROP INDEX memory_records_accessed_idx;
            PRAGMA user_version = 13;
            """
        )

    expected_key = operation_key(operation, recipe="test-recipe")
    with LocalStore(tmp_path) as store:
        (rekeyed,) = store.read_operations(limit=10)
        # Deduplication works again: the same proposal is refused rather than applied twice.
        replayed = store.apply_control_operation(replace(logged, operation_key=expected_key))
        with closing(sqlite3.connect(store.database_path)) as connection:
            indexes = {
                str(row[1]) for row in connection.execute("PRAGMA index_list(memory_records)")
            }
            version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert rekeyed.operation_key == expected_key
    assert replayed is None
    assert "memory_records_accessed_idx" in indexes
    assert version == _SCHEMA_VERSION


def test_deleting_the_current_naming_assertion_restores_the_name_it_displaced(
    tmp_path: Path,
) -> None:
    """A rename is a supersession, so undoing it by deleting the record leaves the old name.

    Deleting the assertion that renamed somebody used to leave the predecessor retired for good,
    so the person ended up nameless -- the registry reporting nobody where the audit trail plainly
    still held one standing assertion.
    """
    recorded_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        voice_id = _voice_identity(store, clip, (1.0, 0.0))
        # Two writes, not one batch: assertions made in one storage batch stay conflicting
        # rather than superseding, and superseding is what this is about.
        store.write_memories(
            (_naming_assertion("naming-1", voice_id, "Li", recorded_at=recorded_at),),
            (_embedding("e-naming-1", "naming-1"),),
        )
        store.write_memories(
            (
                _naming_assertion(
                    "naming-2",
                    voice_id,
                    "Li Hua",
                    recorded_at=recorded_at + timedelta(minutes=1),
                ),
            ),
            (_embedding("e-naming-2", "naming-2"),),
        )
        assert store.projected_identity_name(voice_id) == ("Li Hua", None)

        assert store.delete_memory("naming-2") is True

        assert store.projected_identity_name(voice_id) == ("Li", None)
        # Deleting the one that was already superseded restores nothing: it displaced nothing.
        assert store.delete_memory("naming-1") is True
        assert store.projected_identity_name(voice_id) == (None, None)


def test_consolidation_forgetting_a_naming_assertion_unprojects_it_in_the_same_commit(
    tmp_path: Path,
) -> None:
    """The projection is computed after the forget, not against the row about to be forgotten."""
    recorded_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        clip = _video_asset(store, "clip")
        voice_id = _voice_identity(store, clip, (1.0, 0.0))
        store.write_memories(
            (_naming_assertion("naming-1", voice_id, "Li", recorded_at=recorded_at),),
            (_embedding("e-naming-1", "naming-1"),),
        )
        assert store.projected_identity_name(voice_id) == ("Li", None)

        assert (
            store.apply_formation(
                (_memory("derived", "the neighbour is called Li"),),
                (_embedding("e-derived", "derived"),),
                evidence=(),
                source_memory_ids=(),
                recipe="test-consolidator",
                completed_at=recorded_at + timedelta(minutes=1),
                operation=StoredOperation(
                    operation_key="forget-naming-1",
                    intent="forget",
                    trigger="manual",
                    operation_json="{}",
                    applied_at=recorded_at + timedelta(minutes=1),
                ),
                forget_ids=("naming-1",),
            )
            is True
        )

        assert store.projected_identity_name(voice_id) == (None, None)
        assert store.identity_profile(voice_id) == IdentityProfile(identity_id=voice_id)


def test_provisional_identities_only_asks_about_the_people_the_memories_observed(
    tmp_path: Path,
) -> None:
    """Named strangers elsewhere in the store never enter the answer, or the scan."""
    recorded_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        seen = _video_asset(store, "seen")
        elsewhere = _video_asset(store, "elsewhere")
        unnamed_id = _voice_identity(store, seen, (1.0, 0.0))
        named_id = _voice_identity(store, elsewhere, (0.0, 1.0))
        store.write_memories(
            (
                replace(_memory("observed"), modality="video", assets=(seen,)),
                _naming_assertion("naming-1", named_id, "Li", recorded_at=recorded_at),
            ),
            (_embedding("e-naming-1", "naming-1"),),
        )

        assert store.provisional_identities(("observed",)) == {"observed": (unnamed_id,)}
        # Naming the observed person is what empties the answer, not naming anybody at all.
        store.write_memories(
            (
                _naming_assertion(
                    "naming-2",
                    unnamed_id,
                    "Ana",
                    recorded_at=recorded_at + timedelta(minutes=1),
                ),
            ),
            (_embedding("e-naming-2", "naming-2"),),
        )
        assert store.provisional_identities(("observed",)) == {}


def test_schema_v14_backfills_a_naming_assertion_for_a_name_registered_before_it(
    tmp_path: Path,
) -> None:
    """A name written before naming became a claim has to gain the claim its reads now need.

    Every projection derives `identities.name` from a visible ENTITY assertion, so a legacy row
    with nothing behind it reported unconfirmed and never answered `identity_for_subject`.
    """
    with LocalStore(tmp_path) as store:
        store.write_memories((_memory(),), (_embedding(),))
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            """
            INSERT INTO identities (identity_id, name, relationship, created_at, updated_at)
            VALUES ('identity-legacy', 'Alice', 'sister', '2026-01-01T00:00:00+00:00',
                    '2026-01-01T00:00:00+00:00');
            PRAGMA user_version = 13;
            """
        )

    with LocalStore(tmp_path) as store:
        profile = store.identity_profile("identity-legacy")
        assert profile == IdentityProfile(
            identity_id="identity-legacy", name="Alice", relationship="sister", confirmed=True
        )
        assert store.identity_for_subject("Alice") == "identity-legacy"

        # Re-running the rung must not mint a second assertion beside the first.
        with closing(sqlite3.connect(store.database_path)) as connection:
            store_module._v14_backfill_naming_assertions(connection)
            connection.commit()
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM memory_semantics WHERE identity_id = 'identity-legacy'"
                ).fetchone()[0]
                == 1
            )


def test_the_backfilled_naming_assertion_carries_the_id_the_kernel_would_mint(
    tmp_path: Path,
) -> None:
    """The migration restates two hashes `memory.py` owns; this is the pin that keeps them equal.

    If they drift, a legacy store re-registered under the same name grows a second assertion
    instead of treating the repeat as the no-op it is documented to be.
    """
    import mindbridge.memory as memory_module

    identity_id = "identity-legacy"
    proposal = memory_module._naming_proposal("Alice", "sister", basis=EvidenceBasis.USER_STATEMENT)
    context = memory_module._formation_context(
        None,
        proposal,
        model_id=None,
        recipe=memory_module._NAMING_RECIPE,
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        identity_id=identity_id,
    )
    kernel_memory_id = memory_module._formation_memory_id(
        identity_id,
        proposal,
        recipe=memory_module._NAMING_RECIPE,
        context=context,
    )

    assert store_module._naming_assertion_ids(identity_id, "Alice", "sister") == (
        kernel_memory_id,
        context.lineage_id,
    )
    assert store_module._NAMING_RECIPE == memory_module._NAMING_RECIPE
    assert store_module._NAMING_PREDICATE == memory_module._NAMING_PREDICATE
