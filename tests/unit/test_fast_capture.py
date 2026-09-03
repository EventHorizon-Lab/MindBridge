"""Fast capture: durable acknowledgement before any model work, settled explicitly later."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder

from mindbridge import (
    AssetRef,
    Blob,
    EmbedTask,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryKind,
    Modality,
    ModelError,
    ModelInput,
)
from mindbridge.memory import AsyncMemory


class CountingEmbedder:
    """`TinyEmbedder` that counts its calls and can refuse one poisoned input."""

    embedding_model = TinyEmbedder.embedding_model
    embedding_space = TinyEmbedder.embedding_space
    embedding_dimension = TinyEmbedder.embedding_dimension

    def __init__(self, *, capabilities: frozenset[Modality] = ATOMIC_MODALITIES) -> None:
        self.embedding_capabilities = capabilities
        self.calls = 0
        self.poison: str | None = None
        self._inner = TinyEmbedder()

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        if self.poison is not None and any(self.poison in value.text for value in inputs):
            raise RuntimeError("simulated embedding failure")
        return self._inner.embed(inputs, task)

    def close(self) -> None:
        self._inner.close()


class CountingTranscriber:
    transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})
    transcription_model = "capture-test-asr"
    transcription_space = "capture-test-asr:v1"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
        self.calls += 1
        return tuple("the spare key is in the blue toolbox" for _asset in assets)

    def close(self) -> None:
        pass


class CountingFormer:
    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "capture-test-former"
    formation_space = "capture-test-former:v1"

    def __init__(self) -> None:
        self.calls = 0

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        self.calls += 1
        return tuple(
            (
                FormationProposal(
                    kind=MemoryKind.ENTITY,
                    content="The user is an entity",
                    subject="user",
                    confidence=0.9,
                ),
            )
            for _value in inputs
        )

    def close(self) -> None:
        pass


def _queue_rows(data_dir: Path) -> list[tuple[str, int, str | None]]:
    with closing(sqlite3.connect(data_dir / "state.sqlite3")) as connection:
        return [
            (str(row[0]), int(row[1]), row[2])
            for row in connection.execute(
                "SELECT memory_id, attempts, last_error FROM capture_queue ORDER BY enqueued_at"
            ).fetchall()
        ]


def test_a_capture_is_durable_and_listable_but_only_searchable_once_settled(
    tmp_path: Path,
) -> None:
    embedder = CountingEmbedder()
    former = CountingFormer()
    with Memory(
        tmp_path,
        embedder=embedder,
        former=former,
        minimum_relevance=0,
    ) as memory:
        record = memory.capture("the spare key is in the blue toolbox")

        assert embedder.calls == 0
        assert former.calls == 0
        assert memory.pending_captures() == 1
        assert memory.get(record.id) == record
        assert [item.id for item in memory.list().items] == [record.id]
        assert memory.search("spare key") == ()

        assert memory.settle() == 1
        assert memory.pending_captures() == 0
        assert record.id in {hit.id for hit in memory.search("spare key")}
        assert former.calls == 1

        # A second pass has nothing left to do and never re-forms a settled source.
        assert memory.settle() == 0
        assert former.calls == 1


def test_repeating_a_capture_is_idempotent_and_still_calls_no_model(tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0) as memory:
        first = memory.capture("red screwdriver in drawer two")
        second = memory.capture("red screwdriver in drawer two")

        assert second == first
        assert embedder.calls == 0
        assert memory.pending_captures() == 1

        assert memory.settle() == 1
        # Capturing settled content neither re-enqueues work nor rewrites the derived record.
        assert memory.capture("red screwdriver in drawer two") == memory.get(first.id)
        assert memory.pending_captures() == 0
        assert [hit.id for hit in memory.search("red screwdriver")] == [first.id]


def test_adding_captured_content_settles_it_under_the_same_id(tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    former = CountingFormer()
    with Memory(
        tmp_path,
        embedder=embedder,
        former=former,
        minimum_relevance=0,
    ) as memory:
        captured = memory.capture("the blue toolbox sits on the workbench")
        added = memory.add("the blue toolbox sits on the workbench")

        assert added.id == captured.id
        assert memory.pending_captures() == 0
        assert former.calls == 1
        assert captured.id in {hit.id for hit in memory.search("blue toolbox")}

        # `add_many` shares the same enrichment path.
        queued = memory.capture("the yellow hammer hangs by the door")
        assert memory.add_many(("the yellow hammer hangs by the door",))[0].id == queued.id
        assert memory.pending_captures() == 0
        assert queued.id in {hit.id for hit in memory.search("yellow hammer")}


def test_settling_stores_the_content_a_blocking_add_would_have_stored(tmp_path: Path) -> None:
    audio = Blob(b"captured speech bytes", "audio/wav", "speech.wav")
    text_only = frozenset({Modality.TEXT})

    deferred_transcriber = CountingTranscriber()
    with Memory(
        tmp_path / "deferred",
        embedder=CountingEmbedder(capabilities=text_only),
        transcriber=deferred_transcriber,
        minimum_relevance=0,
    ) as memory:
        captured = memory.capture(audio)
        assert deferred_transcriber.calls == 0
        assert "[transcript:" not in memory.get(captured.id).content

        assert memory.settle() == 1
        assert deferred_transcriber.calls == 1
        settled = memory.get(captured.id)

    with Memory(
        tmp_path / "blocking",
        embedder=CountingEmbedder(capabilities=text_only),
        transcriber=CountingTranscriber(),
        minimum_relevance=0,
    ) as memory:
        added = memory.add(audio)

    assert settled.id == added.id
    assert settled.content == added.content
    assert "[transcript:" in settled.content


def test_a_model_failure_keeps_its_record_queued_and_names_it(tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    embedder.poison = "unreachable"
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0) as memory:
        healthy = memory.capture("a healthy observation")
        failing = memory.capture("an unreachable observation")

        with pytest.raises(ModelError) as failure:
            memory.settle()

        # The queue is processed in enqueue order, so the healthy record settled before the
        # failing one aborted the batch.
        assert failure.value.subject == failing.id
        assert memory.pending_captures() == 1
        assert {hit.id for hit in memory.search("observation")} == {healthy.id}

        queued = _queue_rows(tmp_path)
        assert [row[0] for row in queued] == [failing.id]
        assert queued[0][1] == 1
        assert queued[0][2]

        with pytest.raises(ModelError):
            memory.settle()
        assert _queue_rows(tmp_path)[0][1] == 2

        embedder.poison = None
        assert memory.settle() == 1
        assert _queue_rows(tmp_path) == []


def test_a_capture_that_survived_a_crash_settles_under_a_new_owner(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=CountingEmbedder(), minimum_relevance=0) as memory:
        record = memory.capture("the fuse box is behind the coats")

    assert [row[0] for row in _queue_rows(tmp_path)] == [record.id]

    with Memory(tmp_path, embedder=CountingEmbedder(), minimum_relevance=0) as memory:
        assert memory.pending_captures() == 1
        assert memory.settle() == 1
        assert [hit.id for hit in memory.search("fuse box")] == [record.id]
    assert _queue_rows(tmp_path) == []


def test_async_capture_settle_and_pending_mirror_the_synchronous_surface(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with AsyncMemory(
            tmp_path,
            embedder=CountingEmbedder(),
            minimum_relevance=0,
        ) as memory:
            record = await memory.capture("the ladder is in the garage")

            assert await memory.pending_captures() == 1
            assert await memory.search("ladder") == ()
            assert await memory.settle() == 1
            assert await memory.pending_captures() == 0
            assert [hit.id for hit in await memory.search("ladder")] == [record.id]

    asyncio.run(scenario())


def test_the_capture_operations_reach_the_command_line() -> None:
    from mindbridge import cli

    assert {"capture", "settle", "pending_captures"} <= set(cli.OPERATIONS)
    assert {"capture", "settle", "pending-captures"} <= set(cli.COMMANDS)
    assert {"capture", "settle", "pending-captures"} <= set(cli._LOCAL)


class FlakyFormer(CountingFormer):
    """A former that fails on its first call and behaves on the next."""

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        if self.calls == 0:
            self.calls += 1
            raise RuntimeError("simulated formation outage")
        return super().form(inputs)


def test_a_formation_failure_keeps_the_capture_queued_until_it_forms(tmp_path: Path) -> None:
    former = FlakyFormer()
    with Memory(
        tmp_path, embedder=CountingEmbedder(), former=former, minimum_relevance=0
    ) as memory:
        record = memory.capture("the spare key is in the blue toolbox")

        with pytest.raises(ModelError) as failure:
            memory.settle()

        # The record is embedded and searchable, but formation is still owed, so the queue row
        # survives with the failure recorded instead of the source staying silently unformed.
        assert failure.value.subject == record.id
        assert {hit.id for hit in memory.search("spare key")} == {record.id}
        assert memory.pending_captures() == 1
        assert _queue_rows(tmp_path)[0][1] == 1

        assert memory.settle() == 1
        assert memory.pending_captures() == 0
        assert former.calls == 2
        derived = [
            item
            for item in memory.list().items
            if item.context is not None and item.context.kind is MemoryKind.ENTITY
        ]
        assert len(derived) == 1 and derived[0].context is not None
        assert derived[0].context.evidence_ids == (record.id,)


def test_settling_describes_an_image_a_text_only_embedder_cannot_take(tmp_path: Path) -> None:
    from test_streaming_observations import RecordingVisionDescriber

    describer = RecordingVisionDescriber()
    with Memory(
        tmp_path,
        embedder=CountingEmbedder(capabilities=frozenset({Modality.TEXT})),
        vision_describer=describer,
        minimum_relevance=0,
    ) as memory:
        record = memory.capture(("the red toolbox", Blob(b"frame", "image/png")))
        assert describer.inputs == []

        assert memory.settle() == 1

        settled = memory.get(record.id)
        assert len(describer.inputs) == 1
        assert "[visual description:" in settled.content
        assert "automatically described red toolbox" in settled.content
        assert {hit.id for hit in memory.search("red toolbox")} == {record.id}
