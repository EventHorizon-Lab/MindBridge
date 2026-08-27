from __future__ import annotations

import multiprocessing
import os
import sqlite3
import stat
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

import mindbridge.infrastructure.local.store as store_module
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
    FaceDetection,
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
    )


def _embedding(
    embedding_id: str = "embedding-1",
    memory_id: str = "memory-1",
) -> StoredEmbedding:
    return StoredEmbedding(
        embedding_id=embedding_id,
        memory_id=memory_id,
        values=(0.6, 0.8),
        model_id="test-embedder",
        space_id="test-space",
        task="retrieval.document",
        created_at=datetime(2026, 8, 27, 1, 2, 3, tzinfo=timezone.utc),
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
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
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
        assert legacy.assets == ()
        with closing(sqlite3.connect(store.database_path)) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_schema_v2_is_migrated_without_losing_memories(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        store.write_memory(_memory("preserved"))
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            """
            DROP TABLE face_observations;
            DROP TABLE face_analyses;
            DROP TABLE speech_segments;
            DROP TABLE speech_analyses;
            DROP TABLE identity_profiles;
            DROP TABLE identities;
            PRAGMA user_version = 2;
            """
        )

    with LocalStore(tmp_path) as store:
        assert store.read_memory("preserved") is not None
        with closing(sqlite3.connect(store.database_path)) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.skipif(sqlite3.sqlite_version_info < (3, 35), reason="DROP COLUMN needs SQLite 3.35")
def test_schema_v3_adds_optional_speaker_names(tmp_path: Path) -> None:
    with LocalStore(tmp_path):
        pass
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            """
            DROP TABLE face_observations;
            DROP TABLE face_analyses;
            DROP TABLE speech_segments;
            DROP TABLE speech_analyses;
            DROP TABLE identity_profiles;
            DROP TABLE identities;
            """
        )
        connection.executescript(store_module._SPEECH_SCHEMA_V4)
        connection.executescript(
            """
            ALTER TABLE speaker_identities DROP COLUMN name;
            INSERT INTO speaker_identities (
                speaker_id, model_id, space_id, dimension, centroid,
                observations, created_at, updated_at
            ) VALUES (
                'speaker_legacy', 'cam++', 'cam++:legacy', 2,
                X'0000803F00000000', 1,
                '2026-08-27T01:02:03.000000Z', '2026-08-27T01:02:03.000000Z'
            );
            PRAGMA user_version = 3;
            """
        )

    with LocalStore(tmp_path) as store:
        with closing(sqlite3.connect(store.database_path)) as connection:
            identity = connection.execute(
                "SELECT identity_id, name FROM identities WHERE identity_id = 'speaker_legacy'"
            ).fetchone()
            profile = connection.execute(
                """
                SELECT kind, space_id FROM identity_profiles
                WHERE identity_id = 'speaker_legacy'
                """
            ).fetchone()
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        assert identity == ("speaker_legacy", None)
        assert profile == ("voice", "cam++:legacy")


def test_schema_v4_preserves_named_speaker_turns_as_identities(tmp_path: Path) -> None:
    audio = _asset(b"legacy voice", modality="audio", mime_type="audio/wav", name="voice.wav")
    memory = replace(_memory("legacy"), content="", modality="audio", assets=(audio,))
    with LocalStore(tmp_path) as store:
        store.write_memory(memory)
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            """
            DROP TABLE face_observations;
            DROP TABLE face_analyses;
            DROP TABLE speech_segments;
            DROP TABLE speech_analyses;
            DROP TABLE identity_profiles;
            DROP TABLE identities;
            """
        )
        connection.executescript(store_module._SPEECH_SCHEMA_V4)
        connection.execute(
            """
            INSERT INTO speaker_identities (
                speaker_id, name, model_id, space_id, dimension, centroid,
                observations, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 2, ?, 1, ?, ?)
            """,
            (
                "speaker_legacy",
                "Ada",
                "cam++",
                "cam++:legacy",
                bytes.fromhex("0000803F00000000"),
                "2026-08-27T01:02:03.000000Z",
                "2026-08-27T01:02:03.000000Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO speech_analyses (asset_id, model_id, space_id, transcript, created_at)
            VALUES (?, 'cam++', 'cam++:legacy', 'hello', '2026-08-27T01:02:03.000000Z')
            """,
            (audio.asset_id,),
        )
        connection.execute(
            """
            INSERT INTO speech_segments (
                asset_id, position, start_ms, end_ms, transcript, speaker_id, identity_score
            ) VALUES (?, 0, 0, 1000, 'hello', 'speaker_legacy', 0.9)
            """,
            (audio.asset_id,),
        )
        connection.execute("PRAGMA user_version = 4")
        connection.commit()

    with LocalStore(tmp_path) as store:
        turns = store.read_speech(audio.asset_id, space_id="cam++:legacy")
        assert turns is not None
        assert turns[0].identity_id == "speaker_legacy"
        assert turns[0].identity_name == "Ada"
        assert turns[0].identity_score == pytest.approx(0.9)


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


def test_face_and_voice_profiles_share_mergeable_identities(tmp_path: Path) -> None:
    group = _asset(b"group photo")
    portrait = _asset(b"portrait")
    voice = _asset(b"voice", modality="audio", mime_type="audio/wav", name="voice.wav")
    memories = (
        replace(_memory("group"), content="", modality="image", assets=(group,)),
        replace(_memory("portrait"), content="", modality="image", assets=(portrait,)),
        replace(_memory("voice"), content="", modality="audio", assets=(voice,)),
    )

    with LocalStore(tmp_path) as store:
        store.write_memories(memories)
        group_faces = store.write_faces(
            group.asset_id,
            FaceAnalysis(
                detections=(
                    FaceDetection("alice", (0.1, 0.1, 0.4, 0.8), 0.98),
                    FaceDetection("bob", (0.6, 0.1, 0.9, 0.8), 0.97),
                ),
                faces=(
                    FaceEmbedding("alice", (1.0, 0.0)),
                    FaceEmbedding("bob", (0.0, 1.0)),
                ),
            ),
            model_id="insightface/buffalo_l",
            space_id="insightface:buffalo_l:test",
        )
        portrait_faces = store.write_faces(
            portrait.asset_id,
            FaceAnalysis(
                detections=(FaceDetection("alice", (0.2, 0.1, 0.8, 0.9), 0.96),),
                faces=(FaceEmbedding("alice", (0.99, 0.1)),),
            ),
            model_id="insightface/buffalo_l",
            space_id="insightface:buffalo_l:test",
        )
        voice_segments = store.write_speech(
            voice.asset_id,
            SpeechAnalysis(
                turns=(SpeechTurn(0, 1_000, "hello", "0"),),
                speakers=(SpeakerEmbedding("0", (1.0, 0.0)),),
            ),
            model_id="cam++",
            space_id="cam++:test",
        )

        alice_id = group_faces[0].identity_id
        voice_id = voice_segments[0].identity_id
        assert group_faces[1].identity_id != alice_id
        assert portrait_faces[0].identity_id == alice_id
        assert voice_id is not None and voice_id != alice_id
        assert store.register_identity(alice_id, "Alice") is True
        assert store.merge_identities(alice_id, voice_id) is True

        cached_voice = store.read_speech(voice.asset_id, space_id="cam++:test")
        cached_face = store.read_faces(portrait.asset_id, space_id="insightface:buffalo_l:test")
        assert cached_voice is not None and cached_voice[0].identity_id == alice_id
        assert cached_voice[0].identity_name == "Alice"
        assert cached_face is not None and cached_face[0].identity_name == "Alice"


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
