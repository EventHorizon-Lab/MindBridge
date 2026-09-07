"""The identity axis: "only memories about this person".

The graph edge has always existed -- `memory_semantics.identity_id` for a bound assertion, and
`face_observations` / `speech_segments` for everyone a clip shows or records -- and SQLite even
indexes it. What was missing was any way for a caller to ask for it. These tests pin the
membership rule, the fact that a merge follows the person rather than the ID a caller happens to
hold, and that both network transports accept the axis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from mindbridge import (
    AssetRef,
    Blob,
    EmbedTask,
    FaceAnalysis,
    FaceEmbedding,
    Memory,
    Modality,
    ModelInput,
    RetrievalScope,
    SpeakerEmbedding,
    SpeechAnalysis,
    SpeechTurn,
    ValidationError,
)
from mindbridge.infrastructure.local.store import (
    _IDENTITY_MEMORIES_SQL,
    _IDENTITY_MEMORY_SQL,
    _IDENTITY_SCOPE_CLAUSE,
)


class _Embedder:
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})
    embedding_model = "identity-scope-test"
    embedding_space = "identity-scope-test:2"
    embedding_dimension = 2

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        del task
        return tuple((1.0, 0.0) for _value in inputs)

    def close(self) -> None:
        pass


class _PerAssetFace:
    """One face per asset, its vector taken from the asset name, so people stay distinct."""

    face_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    face_model = "identity-scope-face"
    face_space = "identity-scope-face:2"
    face_analysis_space = "identity-scope-detector:1"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        return tuple(
            FaceAnalysis((FaceEmbedding("face-0", _unit(asset.name or ""), (0.1, 0.1, 0.4, 0.5)),))
            for asset in assets
        )

    def close(self) -> None:
        pass


class _Speech:
    transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})
    transcription_model = "identity-scope-speech"
    transcription_space = "identity-scope-speech:1"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        return tuple(
            SpeechAnalysis(
                turns=(SpeechTurn(0, 900, "the kettle is on", "0"),),
                speakers=(SpeakerEmbedding("0", (1.0, 0.0)),),
            )
            for _asset in assets
        )

    def close(self) -> None:
        pass


def _unit(name: str) -> tuple[float, ...]:
    """A distinct unit vector per person, far enough apart to never be matched together."""
    angle = sum(name.encode()) % 360 * math.pi / 180.0
    return (math.cos(angle), math.sin(angle))


def test_identity_scope_selects_only_the_memories_that_person_is_in(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        face_analyzer=_PerAssetFace(),
        minimum_relevance=0,
    ) as memory:
        first = memory.add(Blob(b"one", "image/png", "alice.png"))
        second = memory.add(Blob(b"two", "image/png", "alice.png"))
        other = memory.add(Blob(b"three", "image/png", "bob.png"))
        text_only = memory.add("a kettle on a worktop")

        alice = memory.faces(first.id)[0].identity_id
        bob = memory.faces(other.id)[0].identity_id
        assert alice != bob
        assert memory.faces(second.id)[0].identity_id == alice

        scoped = memory.search("kettle", limit=10, scope=RetrievalScope(identity_id=alice))

        assert {hit.id for hit in scoped} == {first.id, second.id}
        assert text_only.id not in {hit.id for hit in scoped}
        assert {hit.id for hit in memory.search("kettle", limit=10)} >= {
            first.id,
            second.id,
            other.id,
            text_only.id,
        }


def test_identity_scope_includes_the_person_a_claim_is_about(tmp_path: Path) -> None:
    """Membership is the semantic subject too, not only who a clip shows."""
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        face_analyzer=_PerAssetFace(),
        minimum_relevance=0,
    ) as memory:
        record = memory.add(Blob(b"one", "image/png", "alice.png"))
        alice = memory.faces(record.id)[0].identity_id
        memory.register_identity(alice, "Alice")

        scoped = memory.search("Alice", limit=10, scope=RetrievalScope(identity_id=alice))

        # The clip plus the naming assertion the registration wrote about her.
        assert len(scoped) == 2
        assert record.id in {hit.id for hit in scoped}


def test_identity_scope_follows_a_merge_to_the_surviving_person(tmp_path: Path) -> None:
    """The index projection is refreshed through the outbox, so recall survives the merge."""
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        face_analyzer=_PerAssetFace(),
        transcriber=_Speech(),
        minimum_relevance=0,
    ) as memory:
        first = memory.add(Blob(b"one", "video/mp4", "alice.mp4"))
        face_id = memory.faces(first.id)[0].identity_id
        voice_id = memory.speech(first.id)[0].speaker_id
        assert voice_id is not None and face_id != voice_id

        second = memory.add(Blob(b"two", "video/mp4", "alice.mp4"))
        memory.faces(second.id)
        survivor = memory.faces(first.id)[0].identity_id
        retired = face_id if survivor == voice_id else voice_id
        assert survivor != retired

        scoped = memory.search("kettle", limit=10, scope=RetrievalScope(identity_id=survivor))
        assert {hit.id for hit in scoped} == {first.id, second.id}

        # A merged-away ID resolves to the person it became, exactly as `identity()` does: a
        # caller holding a pre-merge observation must still reach them.
        assert {
            hit.id
            for hit in memory.search("kettle", limit=10, scope=RetrievalScope(identity_id=retired))
        } == {first.id, second.id}


def test_a_merge_reprojects_a_memory_that_carried_only_the_retired_identity(
    tmp_path: Path,
) -> None:
    """A memory holding both merged IDs matches either one, so this pins the case that does not.

    A memory that only ever saw the face carries no trace of the voice it turns out to be, and a
    merge rewrites the speech it re-narrates but never touches that one. Once the face retires,
    nothing in its indexed projection matches the surviving ID, and without the reprojection the
    pushed-down filter drops it from every scoped read.
    """
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        face_analyzer=_PerAssetFace(),
        transcriber=_Speech(),
        minimum_relevance=0,
    ) as memory:
        first = memory.add(Blob(b"one", "video/mp4", "alice.mp4"))
        voice_id = memory.speech(first.id)[0].speaker_id
        assert voice_id is not None
        # A link plan keeps the named identity, so naming the voice retires the face.
        memory.register_speaker(voice_id, "Alice")
        # Face analysis is lazy, and the link needs the pair corroborated on two assets.
        assert memory.faces(first.id)[0].identity_id != voice_id

        # Seen and never heard: its projection is the face alone.
        seen = memory.add(Blob(b"two", "image/png", "alice.mp4"))
        face_id = memory.faces(seen.id)[0].identity_id
        assert face_id != voice_id

        second = memory.add(Blob(b"three", "video/mp4", "alice.mp4"))
        memory.faces(second.id)
        assert memory.faces(seen.id)[0].identity_id == voice_id

        scoped = memory.search("kettle", limit=10, scope=RetrievalScope(identity_id=voice_id))

    assert seen.id in {hit.id for hit in scoped}


def test_an_unknown_identity_scopes_to_nothing(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_Embedder(), minimum_relevance=0) as memory:
        memory.add("a kettle on a worktop")

        assert memory.search("kettle", scope=RetrievalScope(identity_id="nobody")) == ()


def test_identity_membership_sql_agrees_in_both_directions(tmp_path: Path) -> None:
    """One membership rule, written twice: per document for the index, per identity for a scope.

    The scope used to reuse the per-document statement correlated against the row being
    filtered, so SQLite re-ran its three-branch UNION once per candidate record (and once per
    embedding in the store on a merge). Resolving the identity's memory set once is the same
    membership -- these two directions must keep agreeing, and the plan must stay uncorrelated.
    """
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        face_analyzer=_PerAssetFace(),
        transcriber=_Speech(),
        minimum_relevance=0,
    ) as memory:
        clip = memory.add(Blob(b"one", "video/mp4", "alice.mp4"))
        memory.add(Blob(b"two", "image/png", "bob.png"))
        memory.add("a kettle nobody is in")
        alice = memory.faces(clip.id)[0].identity_id
        memory.register_identity(alice, "Alice")
        memory.speech(clip.id)

        with memory._store._read_transaction() as connection:
            per_document = {
                (row["memory_id"], row["identity_id"])
                for row in connection.execute(_IDENTITY_MEMORY_SQL.format(predicate="1"))
            }
            per_identity = {
                (row["memory_id"], identity)
                for identity in {identity for _memory_id, identity in per_document}
                for row in connection.execute(
                    _IDENTITY_MEMORIES_SQL,
                    (identity, identity, identity),
                )
            }
            plan = " ".join(
                str(row["detail"])
                for row in connection.execute(
                    "EXPLAIN QUERY PLAN "
                    f"SELECT memory_id FROM memory_records WHERE 1 {_IDENTITY_SCOPE_CLAUSE}",
                    (alice, alice, alice),
                )
            )

    assert per_document
    assert per_identity == per_document
    assert "CORRELATED" not in plan


def test_a_scoped_identity_must_be_real_text() -> None:
    for value in ("", " ", "alice "):
        with pytest.raises(ValidationError, match="identity_id"):
            RetrievalScope(identity_id=value)
    assert RetrievalScope().identity_id is None
