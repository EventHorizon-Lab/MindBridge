"""Erasing one person's biometric template, and the retention queries around it.

`delete_memory` reaches memories, their embeddings, their outbox rows and their assets, but it
never touched `identities` or `identity_exemplars`. A household robot asked to forget a person
therefore kept their fp32 face and voice templates forever. These tests pin what erasure removes,
what it deliberately keeps, and -- because a privacy deletion a hexdump can undo is not one --
that the vector bytes are actually gone from the database files.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from contextlib import closing
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from mindbridge.infrastructure.local import (
    LocalStore,
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

_NOW = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
_DIMENSION = 48
_FACE_SPACE = "sface:test"
_VOICE_SPACE = "cam++:test"


def _one_hot(index: int) -> tuple[float, ...]:
    return tuple(1.0 if position == index else 0.0 for position in range(_DIMENSION))


def _distinctive(seed: int) -> tuple[float, ...]:
    """A unit vector orthogonal to every other seed's, with a distinctive fp32 byte pattern.

    Two non-zero coordinates, not one: a normalised one-hot vector is stored as the very
    common four bytes for 1.0, which a byte-level search cannot tell from any other vector.
    Seeds are laid out on disjoint coordinate pairs so no two identities ever match.
    """
    slot = ((seed % 100) - 1) + 7 * (seed // 100)
    first, second = 2 * slot, 2 * slot + 1
    assert 0 <= first < second < _DIMENSION, f"seed {seed} has no coordinate pair"
    raw = [0.0] * _DIMENSION
    raw[first] = 0.6234567 + 0.000131 * seed
    raw[second] = 0.7134891 - 0.000117 * seed
    norm = math.sqrt(raw[first] ** 2 + raw[second] ** 2)
    return tuple(value / norm for value in raw)


def _asset(label: str, *, modality: str = "video", created_at: datetime = _NOW) -> StoredAsset:
    content = f"{label} bytes".encode()
    digest = sha256(content).hexdigest()
    suffix = {"video": ("video/mp4", "mp4"), "audio": ("audio/wav", "wav")}[modality]
    return StoredAsset(
        asset_id=digest,
        modality=modality,
        mime_type=suffix[0],
        size_bytes=len(content),
        sha256=digest,
        relative_path=f"assets/{digest[:2]}/{digest}",
        name=f"{label}.{suffix[1]}",
        created_at=created_at,
    )


def _write_clip(
    store: LocalStore,
    label: str,
    *,
    created_at: datetime = _NOW,
    modality: str = "video",
) -> StoredAsset:
    asset = _asset(label, modality=modality, created_at=created_at)
    store.write_memory(
        StoredMemory(
            memory_id=label,
            content="",
            metadata_json="{}",
            created_at=created_at,
            updated_at=created_at,
            modality=modality,
            assets=(asset,),
        )
    )
    return asset


def _face(store: LocalStore, asset: StoredAsset, vector: tuple[float, ...]) -> str:
    return store.write_faces(
        asset.asset_id,
        FaceAnalysis((FaceEmbedding("face-0", vector, (0.1, 0.1, 0.4, 0.5)),)),
        model_id="sface",
        space_id=_FACE_SPACE,
        minimum_similarity=0.99,
    )[0].identity_id


def _voice(store: LocalStore, asset: StoredAsset, vector: tuple[float, ...]) -> str:
    speaker_id = store.write_speech(
        asset.asset_id,
        SpeechAnalysis(
            turns=(SpeechTurn(0, 500, "the kettle is on", "0"),),
            speakers=(SpeakerEmbedding("0", vector),),
        ),
        model_id="cam++",
        space_id=_VOICE_SPACE,
        minimum_similarity=0.99,
    )[0].speaker_id
    assert speaker_id is not None
    return speaker_id


def _person(store: LocalStore, label: str, seed: int) -> tuple[str, tuple[str, ...]]:
    """Build one identity the way the product write path does: fragments, then merges.

    Returns the canonical identity and the two clip memory IDs that mention it.
    """
    first = _write_clip(store, f"{label}-first")
    second = _write_clip(store, f"{label}-second")
    face_id = _face(store, first, _distinctive(seed))
    voice_id = _voice(store, first, _distinctive(seed + 100))
    merged = store.link_identities(face_id, voice_id)
    assert merged is not None
    # A second voice fragment re-absorbed, exactly as `Memory._link_asset_identity` does it.
    extra_voice = _voice(store, second, _distinctive(seed + 200))
    merged = store.link_identities(merged, extra_voice, allow_shared_modality=True)
    assert merged is not None
    return merged, (first.asset_id, second.asset_id)


def _counts(store: LocalStore) -> dict[str, int]:
    with closing(sqlite3.connect(store.database_path)) as connection:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "identities",
                "identity_aliases",
                "identity_exemplars",
                "identity_link_evidence",
                "face_observations",
                "speech_segments",
                "memory_records",
                "media_assets",
            )
        }


def test_forgetting_an_identity_removes_the_whole_cluster_and_keeps_the_memories(
    tmp_path: Path,
) -> None:
    with LocalStore(tmp_path) as store:
        forgotten, forgotten_assets = _person(store, "alice", 1)
        assert store.register_identity(forgotten, "Alice", relationship="neighbour") is True
        kept, _kept_assets = _person(store, "bob", 2)
        aliases = store.identity_equivalence_class(forgotten)
        assert aliases is not None and len(aliases) == 3 and aliases[0] == forgotten
        before = _counts(store)

        erasure = store.forget_identity(forgotten)

        assert erasure is not None
        assert erasure.identity_id == forgotten
        assert erasure.alias_ids == aliases[1:]
        assert erasure.face_exemplars == 1
        assert erasure.voice_exemplars == 2
        assert erasure.face_observations == 1
        assert erasure.speech_segments == 2
        # Every entrance to the forgotten person is closed, including each merged alias.
        for member in aliases:
            assert store.resolve_identity_id(member) is None
            assert store.identity_profile(member) is None
            assert store.identity_memory_ids(member) is None
            assert store.identity_equivalence_class(member) is None
        # The other person in the house is untouched.
        assert store.identity_profile(kept) is not None
        after = _counts(store)
        assert after["identities"] == before["identities"] - 1
        assert after["identity_aliases"] == before["identity_aliases"] - 2
        assert after["identity_exemplars"] == before["identity_exemplars"] - 3
        assert after["face_observations"] == before["face_observations"] - 1
        # The family's memories of the events, and their media, survive intact.
        assert after["memory_records"] == before["memory_records"]
        assert after["media_assets"] == before["media_assets"]
        assert after["speech_segments"] == before["speech_segments"]
        # Only the first clip was face-analysed; the second carried voice alone.
        assert store.read_faces(forgotten_assets[0], space_id=_FACE_SPACE) == ()
        assert store.read_faces(forgotten_assets[1], space_id=_FACE_SPACE) is None
        for asset_id in forgotten_assets:
            speech = store.read_speech(asset_id, space_id=_VOICE_SPACE)
            assert speech is not None
            # The transcript is evidence and stays; only the attribution is scrubbed.
            assert [segment.text for segment in speech] == ["the kettle is on"]
            assert [segment.speaker_id for segment in speech] == [None]


def test_forgotten_exemplar_vectors_are_not_recoverable_from_the_database_files(
    tmp_path: Path,
) -> None:
    """A privacy deletion a `.db` hexdump can undo is not a privacy deletion.

    The reader held open across the erasure is the point: SQLite only checkpoints the WAL on
    the last connection close, so without an explicit truncating checkpoint the pre-deletion
    page image -- vector bytes and all -- stays in the main database file.
    """
    face_vector = _distinctive(7)
    voice_vector = _distinctive(107)
    face_bytes = struct.pack(f"<{_DIMENSION}f", *face_vector)
    voice_bytes = struct.pack(f"<{_DIMENSION}f", *voice_vector)
    files = tuple(tmp_path / name for name in ("state.sqlite3", "state.sqlite3-wal"))
    with LocalStore(tmp_path) as store:
        assert store.database_path == files[0]
        clip = _write_clip(store, "clip")
        face_id = _face(store, clip, face_vector)
        voice_id = _voice(store, clip, voice_vector)
        merged = store.link_identities(face_id, voice_id)
        assert merged is not None
        with closing(sqlite3.connect(store.database_path)) as reader:
            reader.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            present = {
                path.name for path in files if path.exists() and face_bytes in path.read_bytes()
            }
            assert present, "the face exemplar was never durable, so this proves nothing"

            store.forget_identity(merged)

            for probe, label in ((face_bytes, "face"), (voice_bytes, "voice")):
                leaked = {
                    path.name for path in files if path.exists() and probe in path.read_bytes()
                }
                assert not leaked, f"the {label} exemplar is still readable in {sorted(leaked)}"


def test_forgetting_an_unknown_identity_reports_nothing_and_changes_nothing(
    tmp_path: Path,
) -> None:
    with LocalStore(tmp_path) as store:
        kept, _assets = _person(store, "alice", 3)
        before = _counts(store)

        assert store.forget_identity("identity_never_existed") is None

        assert _counts(store) == before
        assert store.identity_profile(kept) is not None
        with pytest.raises(ValueError, match="identity_id"):
            store.forget_identity("   ")


def test_forgetting_replaces_the_indexed_memories_in_the_same_commit(tmp_path: Path) -> None:
    """The indexed document carries the person's name, so erasure must re-index atomically."""
    with LocalStore(tmp_path) as store:
        forgotten, _assets = _person(store, "alice", 4)
        assert store.register_identity(forgotten, "Alice") is True
        named = StoredMemory(
            memory_id="alice-first",
            content="Alice: the kettle is on",
            metadata_json="{}",
            created_at=_NOW,
            updated_at=_NOW,
            modality="video",
            assets=(_asset("alice-first"),),
        )
        embedding = StoredEmbedding(
            embedding_id="alice-first:0",
            memory_id="alice-first",
            values=_one_hot(0),
            model_id="jina",
            space_id="jina:test",
            task="retrieval.passage",
            created_at=_NOW,
        )
        store.write_memories((named,), (embedding,))
        assert store.acknowledge_index_operations(store.pending_index_operations()) > 0
        scrubbed = StoredMemory(
            memory_id="alice-first",
            content="Speaker 1: the kettle is on",
            metadata_json="{}",
            created_at=_NOW,
            updated_at=_NOW,
            modality="video",
            assets=(_asset("alice-first"),),
        )

        erasure = store.forget_identity(
            forgotten,
            memories=(scrubbed,),
            embeddings=(embedding,),
        )

        assert erasure is not None
        stored = store.read_memory("alice-first")
        assert stored is not None and "Alice" not in stored.content
        document = store.read_index_document("alice-first:0")
        assert document is not None and "Alice" not in document.content
        # The projection is only told after SQLite committed, through the durable outbox.
        assert [operation.embedding_id for operation in store.pending_index_operations()] == [
            "alice-first:0",
            "alice-first:0",
        ]


def test_re_analysing_a_stored_asset_does_not_re_mint_a_forgotten_identity(
    tmp_path: Path,
) -> None:
    """The cached analysis row survives erasure, so the same media cannot resurrect the person."""
    with LocalStore(tmp_path) as store:
        clip = _write_clip(store, "clip")
        vector = _distinctive(5)
        face_id = _face(store, clip, vector)
        assert store.forget_identity(face_id) is not None

        assert _face_analysis_is_cached(store, clip.asset_id)
        assert (
            store.write_faces(
                clip.asset_id,
                FaceAnalysis((FaceEmbedding("face-0", vector, (0.1, 0.1, 0.4, 0.5)),)),
                model_id="sface",
                space_id=_FACE_SPACE,
                minimum_similarity=0.99,
            )
            == ()
        )
        assert _counts(store)["identities"] == 0


def _face_analysis_is_cached(store: LocalStore, asset_id: str) -> bool:
    with closing(sqlite3.connect(store.database_path)) as connection:
        row = connection.execute(
            "SELECT 1 FROM face_analyses WHERE asset_id = ?", (asset_id,)
        ).fetchone()
    return row is not None


def test_equivalence_class_resolves_from_any_member_and_is_none_when_unknown(
    tmp_path: Path,
) -> None:
    with LocalStore(tmp_path) as store:
        clip = _write_clip(store, "clip")
        face_id = _face(store, clip, _one_hot(0))
        assert store.identity_equivalence_class(face_id) == (face_id,)
        voice_id = _voice(store, clip, _one_hot(1))
        merged = store.link_identities(face_id, voice_id)
        assert merged is not None
        expected = (merged, *sorted({face_id, voice_id} - {merged}))

        assert store.identity_equivalence_class(merged) == expected
        assert store.identity_equivalence_class(face_id) == expected
        assert store.identity_equivalence_class(voice_id) == expected
        assert store.identity_equivalence_class("identity_never_existed") is None


def test_asset_retention_candidates_are_oldest_first_and_filter_by_age(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        old = _write_clip(store, "old", created_at=_NOW - timedelta(days=400))
        middle = _write_clip(store, "middle", created_at=_NOW - timedelta(days=30))
        recent = _write_clip(store, "recent", created_at=_NOW)
        # The descriptor's own created_at is what a retention window must read, not the
        # memory's: an asset outlives the memory that first referenced it.
        assert [asset.asset_id for asset in store.asset_retention_candidates()] == [
            old.asset_id,
            middle.asset_id,
            recent.asset_id,
        ]
        assert [
            asset.asset_id
            for asset in store.asset_retention_candidates(created_before=_NOW - timedelta(days=90))
        ] == [old.asset_id]
        assert [asset.asset_id for asset in store.asset_retention_candidates(limit=1)] == [
            old.asset_id
        ]
        # Stored timestamps are normalised to UTC text, so the boundary must be too. An
        # unnormalised `+09:00` boundary compares its local wall clock against `Z` strings and
        # sweeps in nine extra hours -- here, the asset stored at exactly the boundary.
        elsewhere = _NOW.astimezone(timezone(timedelta(hours=9)))
        assert [
            asset.asset_id for asset in store.asset_retention_candidates(created_before=elsewhere)
        ] == [old.asset_id, middle.asset_id]
        assert store.asset_storage_bytes() == sum(
            asset.size_bytes for asset in (old, middle, recent)
        )
        with pytest.raises(ValueError, match="limit"):
            store.asset_retention_candidates(limit=0)
        with pytest.raises(ValueError, match="created_before"):
            store.asset_retention_candidates(created_before=_NOW.replace(tzinfo=None))
