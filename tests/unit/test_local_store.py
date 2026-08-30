from __future__ import annotations

import math
import multiprocessing
import os
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

from mindbridge.infrastructure.local import (
    DataDirectoryInUseError,
    LocalStore,
    LocalStoreClosedError,
    StoredAsset,
    StoredEmbedding,
    StoredMemory,
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
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 7
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
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 7
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
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 7
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
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 7
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
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 7
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
