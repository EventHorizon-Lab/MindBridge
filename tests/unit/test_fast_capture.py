"""Fast capture: durable acknowledgement before any model work, settled explicitly later."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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
from mindbridge._telemetry import CAPTURE_FAILED, CAPTURE_SETTLED, CAPTURE_TIME_TO_SEARCHABLE
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
        assert len(memory.pending_captures()) == 1
        assert memory.get(record.id) == record
        assert [item.id for item in memory.list().items] == [record.id]
        assert memory.search("spare key") == ()

        assert memory.settle() == 1
        assert memory.pending_captures() == ()
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
        assert len(memory.pending_captures()) == 1

        assert memory.settle() == 1
        # Capturing settled content neither re-enqueues work nor rewrites the derived record.
        assert memory.capture("red screwdriver in drawer two") == memory.get(first.id)
        assert memory.pending_captures() == ()
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
        assert memory.pending_captures() == ()
        assert former.calls == 1
        assert captured.id in {hit.id for hit in memory.search("blue toolbox")}

        # `add_many` shares the same enrichment path.
        queued = memory.capture("the yellow hammer hangs by the door")
        assert memory.add_many(("the yellow hammer hangs by the door",))[0].id == queued.id
        assert memory.pending_captures() == ()
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

        # Every readable record is attempted; the failure is raised once the batch is done.
        assert failure.value.subject == failing.id
        assert len(memory.pending_captures()) == 1
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


def test_a_poisoned_capture_neither_blocks_nor_hides_the_records_behind_it(
    tmp_path: Path,
) -> None:
    embedder = CountingEmbedder()
    embedder.poison = "unreachable"
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0) as memory:
        poisoned = memory.capture("an unreachable observation")

        for _attempt in range(3):
            with pytest.raises(ModelError):
                memory.settle()

        # Captured after the poisoned record, so enqueue order would starve it forever.
        later = memory.capture("the ladder is in the garage")

        # The ceiling retires the poisoned row from the work set. It stays queued and visible
        # with its reason, and the record behind it settles.
        assert memory.settle() == 1
        assert {hit.id for hit in memory.search("ladder")} == {later.id}
        (pending,) = memory.pending_captures()
        assert pending.memory_id == poisoned.id
        assert pending.attempts == 3
        assert pending.last_error

        # Asking about specific records answers "is this one searchable yet".
        assert memory.pending_captures(memory_ids=(later.id,)) == ()
        assert [row.memory_id for row in memory.pending_captures(memory_ids=(poisoned.id,))] == [
            poisoned.id
        ]

        # Raising the ceiling is what retries it.
        with pytest.raises(ModelError):
            memory.settle(max_attempts=4)
        assert memory.pending_captures()[0].attempts == 4


def test_capture_refuses_media_no_configured_model_will_ever_embed(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=CountingEmbedder(capabilities=frozenset({Modality.TEXT})),
        minimum_relevance=0,
    ) as memory:
        image = Blob(b"frame", "image/png", "frame.png")
        with pytest.raises(ModelError) as refusal:
            memory.add(image)
        with pytest.raises(ModelError) as captured:
            memory.capture(image)

        # Committing it would have made an unsettleable row durable, so capture rejects exactly
        # what add rejects, before the write.
        assert captured.value.reason == refusal.value.reason == "unsupported_modality"
        assert _queue_rows(tmp_path) == []
        assert memory.list().items == ()

    # An embedder that takes audio reaches `add()`'s second rejection instead of its first, so
    # the capture guard has to be the same unconditional check rather than an audio-only mirror.
    with Memory(
        tmp_path,
        embedder=CountingEmbedder(capabilities=frozenset({Modality.TEXT, Modality.AUDIO})),
        minimum_relevance=0,
    ) as memory:
        image = Blob(b"frame", "image/png", "frame.png")
        with pytest.raises(ModelError) as refusal:
            memory.add(image)
        with pytest.raises(ModelError) as captured:
            memory.capture(image)

        assert captured.value.reason == refusal.value.reason == "unsupported_modality"
        assert _queue_rows(tmp_path) == []
        assert memory.list().items == ()


def test_an_add_that_crashed_before_forming_is_completed_by_the_next_settle(
    tmp_path: Path,
) -> None:
    def crash(*_arguments: object, **_keywords: object) -> None:
        raise RuntimeError("simulated crash after the add commit")

    former = CountingFormer()
    embedder = CountingEmbedder()
    with Memory(tmp_path, embedder=embedder, former=former, minimum_relevance=0) as memory:
        memory._form_sources = crash  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            memory.add("the fuse box is behind the coats")

    # The record is committed and searchable but unformed, and the queue row is the only thing
    # that says so.
    (queued,) = _queue_rows(tmp_path)
    embedded = embedder.calls
    with Memory(tmp_path, embedder=embedder, former=former, minimum_relevance=0) as memory:
        assert [row.memory_id for row in memory.pending_captures()] == [queued[0]]
        # Settling an already-embedded row owes formation only; re-running the model stages would
        # buy the same vectors twice.
        memory._enrich_row = crash  # type: ignore[method-assign]
        assert memory.settle() == 1

        # Formation ran exactly once. The single extra embed is the record it proposed.
        assert former.calls == 1
        assert embedder.calls == embedded + 1
        assert memory.pending_captures() == ()
        derived = [
            item
            for item in memory.list().items
            if item.context is not None and item.context.kind is MemoryKind.ENTITY
        ]
        assert len(derived) == 1

        # A second pass has nothing left to do.
        assert memory.settle() == 0
        assert former.calls == 1


def test_the_settle_span_reports_time_to_searchable_separately(tmp_path: Path) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    embedder = CountingEmbedder()
    embedder.poison = "unreachable"
    with Memory(
        tmp_path,
        embedder=embedder,
        minimum_relevance=0,
        tracer=provider.get_tracer("test"),
    ) as memory:
        memory.capture("a healthy observation")
        memory.capture("an unreachable observation")
        with pytest.raises(ModelError):
            memory.settle()
    provider.shutdown()

    (settle,) = [span for span in exporter.get_finished_spans() if span.name == "mindbridge.settle"]
    assert settle.attributes is not None
    assert settle.attributes[CAPTURE_SETTLED] == 1
    assert settle.attributes[CAPTURE_FAILED] == 1
    # The capture-to-searchable interval, which neither the capture span nor the settle span's own
    # duration reports.
    assert cast(float, settle.attributes[CAPTURE_TIME_TO_SEARCHABLE]) >= 0.0


def test_a_capture_that_survived_a_crash_settles_under_a_new_owner(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=CountingEmbedder(), minimum_relevance=0) as memory:
        record = memory.capture("the fuse box is behind the coats")

    assert [row[0] for row in _queue_rows(tmp_path)] == [record.id]

    with Memory(tmp_path, embedder=CountingEmbedder(), minimum_relevance=0) as memory:
        assert len(memory.pending_captures()) == 1
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

            assert len(await memory.pending_captures()) == 1
            assert await memory.search("ladder") == ()
            assert await memory.settle() == 1
            assert await memory.pending_captures() == ()
            assert [hit.id for hit in await memory.search("ladder")] == [record.id]

    asyncio.run(scenario())


def test_the_capture_operations_reach_the_command_line(tmp_path: Path) -> None:
    from mindbridge import cli

    assert {"capture", "settle", "pending_captures"} <= set(cli.OPERATIONS)
    assert {"capture", "settle", "pending-captures"} <= set(cli.COMMANDS)
    assert {"capture", "settle", "pending-captures"} <= set(cli._LOCAL)

    parser = cli._parser()
    with Memory(tmp_path, embedder=CountingEmbedder(), minimum_relevance=0) as memory:
        record = memory.capture("the ladder is in the garage")
        reported = cli._LOCAL["pending-captures"](memory, parser.parse_args(["pending-captures"]))
        settled = cli._LOCAL["settle"](memory, parser.parse_args(["settle", "--max-attempts", "2"]))

    assert reported == {
        "pending": [
            {
                "memory_id": record.id,
                "enqueued_at": reported["pending"][0]["enqueued_at"],  # type: ignore[index]
                "attempts": 0,
                "last_error": None,
                "awaiting": "enrichment",
            }
        ]
    }
    assert settled == {"settled": 1}


def test_settle_names_records_on_the_command_line(tmp_path: Path) -> None:
    from mindbridge import cli

    parser = cli._parser()
    embedder = CountingEmbedder()
    embedder.poison = "unreachable"
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0) as memory:
        poisoned = memory.capture("an unreachable observation")
        for _attempt in range(3):
            with pytest.raises(ModelError):
                memory.settle()
        later = memory.capture("the ladder is in the garage")

        # Naming the healthy record settles it alone; the poisoned one keeps its ceiling.
        assert cli._LOCAL["settle"](memory, parser.parse_args(["settle", later.id])) == {
            "settled": 1
        }
        assert [row.memory_id for row in memory.pending_captures()] == [poisoned.id]


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
        assert len(memory.pending_captures()) == 1
        assert _queue_rows(tmp_path)[0][1] == 1

        assert memory.settle() == 1
        assert memory.pending_captures() == ()
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


def test_settling_appends_derived_text_and_leaves_the_captured_evidence_intact(
    tmp_path: Path,
) -> None:
    note = "the tenant said this on the phone"
    audio = Blob(b"captured speech bytes", "audio/wav", "speech.wav")
    with Memory(
        tmp_path,
        embedder=CountingEmbedder(capabilities=frozenset({Modality.TEXT})),
        transcriber=CountingTranscriber(),
        minimum_relevance=0,
    ) as memory:
        captured = memory.capture((note, audio))
        assert captured.content == note
        assets = captured.assets

        assert memory.settle() == 1
        settled = memory.get(captured.id)

        # Settlement appends; it never rewrites the observation. The caller's text is still the
        # byte-identical front of `content`, the transcript sits behind its own asset-keyed
        # marker, and the raw media is the same content-addressed object.
        marker = f"[transcript:{assets[0].id}]"
        assert settled.content == f"{note}\n\n{marker}\nthe spare key is in the blue toolbox"
        assert settled.content.encode()[: len(note.encode())] == note.encode()
        assert [(asset.id, asset.sha256) for asset in settled.assets] == [
            (asset.id, asset.sha256) for asset in assets
        ]


def test_pending_captures_name_the_stage_a_record_is_stopped_at(tmp_path: Path) -> None:
    former = FlakyFormer()
    with Memory(
        tmp_path,
        embedder=CountingEmbedder(),
        former=former,
        minimum_relevance=0,
    ) as memory:
        record = memory.capture("the spare key is in the blue toolbox")
        (pending,) = memory.pending_captures()
        assert pending.awaiting == "enrichment"
        assert memory.search("spare key") == ()

        with pytest.raises(ModelError):
            memory.settle()

        # Embedded, indexed, and searchable already; only formation is still owed, which the
        # attempt count and the error text alone could not tell an operator.
        (pending,) = memory.pending_captures()
        assert pending.awaiting == "formation"
        assert {hit.id for hit in memory.search("spare key")} == {record.id}

        assert memory.settle() == 1
        assert memory.pending_captures() == ()


def test_settle_can_name_the_records_it_runs_and_ignores_the_ceiling_for_them(
    tmp_path: Path,
) -> None:
    embedder = CountingEmbedder()
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0) as memory:
        first = memory.capture("the ladder is in the garage")
        second = memory.capture("the fuse box is behind the coats")

        # Only the named record settles, whatever the enqueue order says.
        assert memory.settle(memory_ids=(second.id,)) == 1
        assert [row.memory_id for row in memory.pending_captures()] == [first.id]
        assert {hit.id for hit in memory.search("fuse box")} == {second.id}

        # A settled or unknown ID is skipped rather than raised.
        assert memory.settle(memory_ids=(second.id, "0" * 64)) == 0

        embedder.poison = "unreachable"
        poisoned = memory.capture("an unreachable observation")
        for _attempt in range(3):
            with pytest.raises(ModelError):
                memory.settle(memory_ids=(poisoned.id,))

        # The ordinary queue now skips it and settles the record behind it instead.
        assert memory.settle() == 1
        assert [row.memory_id for row in memory.pending_captures()] == [poisoned.id]

        # Naming it is the host asking by hand, so the ceiling does not apply.
        with pytest.raises(ModelError):
            memory.settle(memory_ids=(poisoned.id,))
        assert memory.pending_captures()[0].attempts == 4
        embedder.poison = None
        assert memory.settle(memory_ids=(poisoned.id,)) == 1
        assert memory.pending_captures() == ()


class SlowEmbedder(CountingEmbedder):
    """`CountingEmbedder` slow enough that a second thread reaches the queue mid-embedding."""

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        time.sleep(0.2)
        return super().embed(inputs, task)


def test_two_concurrent_settles_never_run_the_models_over_one_record_twice(
    tmp_path: Path,
) -> None:
    embedder = SlowEmbedder()
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0) as memory:
        record = memory.capture("the spare key is in the blue toolbox")

        started = threading.Barrier(2)
        settled: list[int] = []

        def worker() -> None:
            started.wait()
            settled.append(memory.settle())

        threads = [threading.Thread(target=worker) for _index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # One settlement at a time: the loser waits and then finds the queue already drained,
        # rather than paying for the same embedding and discovering the race at the commit.
        assert sorted(settled) == [0, 1]
        assert embedder.calls == 1
        assert memory.pending_captures() == ()
        assert {hit.id for hit in memory.search("spare key")} == {record.id}
