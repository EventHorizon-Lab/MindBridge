from __future__ import annotations

import asyncio
import threading
import wave
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest
from _feature_support import TinyEmbedder

from mindbridge import (
    AcousticBoundary,
    ASRPartial,
    AsyncAudioStream,
    AsyncCaptureStream,
    AsyncMemory,
    AsyncOmniPrefetch,
    AsyncVisionStream,
    AudioBoundary,
    AudioStreamPacket,
    Blob,
    EmbedTask,
    EvidenceBasis,
    MemoryRecord,
    MemoryType,
    Modality,
    ModelError,
    ModelInput,
    ObservationContext,
    PCMChunk,
    SceneBoundary,
    SpatialAnchor,
    SpatialContext,
    StorageError,
    StreamEvent,
    StreamInput,
    StreamPhase,
    VADPacket,
    ValidationError,
    VisionBoundary,
    VisionFrame,
    VisionPartial,
    VisionStreamPacket,
)


async def _events(*values: StreamEvent) -> AsyncIterator[StreamEvent]:
    for value in values:
        yield value


async def _audio_packets(*values: AudioStreamPacket) -> AsyncIterator[AudioStreamPacket]:
    for value in values:
        yield value


async def _vision_packets(*values: VisionStreamPacket) -> AsyncIterator[VisionStreamPacket]:
    for value in values:
        yield value


class RecordingEmbedder:
    embedding_model = TinyEmbedder.embedding_model
    embedding_space = TinyEmbedder.embedding_space
    embedding_dimension = TinyEmbedder.embedding_dimension

    def __init__(self, capabilities: frozenset[Modality]) -> None:
        self.embedding_capabilities = capabilities
        self.inputs: list[ModelInput] = []
        self._embedder = TinyEmbedder()

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        self.inputs.extend(inputs)
        return self._embedder.embed(inputs, task)

    def close(self) -> None:
        self._embedder.close()


class RecordingVisionDescriber:
    vision_capabilities = frozenset({Modality.IMAGE})
    vision_model = "test-vision"

    def __init__(self) -> None:
        self.inputs: list[ModelInput] = []
        self.closed = False

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        self.inputs.extend(inputs)
        return tuple("automatically described red toolbox" for _value in inputs)

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_partial_snapshots_are_speculative_and_only_final_is_durable(
    tmp_path: Path,
) -> None:
    memory = AsyncMemory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0)
    try:
        stream = AsyncCaptureStream(memory)
        commits = [
            value
            async for value in stream.consume(
                _events(
                    StreamEvent(StreamPhase.UPDATE, "find"),
                    StreamEvent(StreamPhase.UPDATE, "find the red toolbox"),
                    StreamEvent(
                        StreamPhase.FINAL,
                        StreamInput("find the red toolbox"),
                    ),
                )
            )
        ]

        assert len(commits) == 1
        assert commits[0].record.content == "find the red toolbox"
        assert commits[0].prefetch is not None
        assert commits[0].prefetch.revision == 2
        assert [item.content for item in (await memory.list()).items] == ["find the red toolbox"]
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_cancel_and_eof_never_promote_a_partial_snapshot(tmp_path: Path) -> None:
    memory = AsyncMemory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0)
    try:
        stream = AsyncCaptureStream(memory)
        assert [
            item
            async for item in stream.consume(
                _events(
                    StreamEvent(StreamPhase.UPDATE, "unfinished"),
                    StreamEvent(StreamPhase.CANCEL),
                )
            )
        ] == []
        assert [
            item
            async for item in stream.consume(
                _events(StreamEvent(StreamPhase.UPDATE, "also unfinished"))
            )
        ] == []
        assert (await memory.list()).items == ()
    finally:
        await memory.close()


def test_stream_event_boundary_is_modality_agnostic_and_strict() -> None:
    audio = Blob(b"audio", "audio/wav")
    assert StreamEvent(StreamPhase.UPDATE, audio).item == audio
    assert StreamEvent(StreamPhase.FINAL, StreamInput(audio)).item is not None
    assert StreamEvent(StreamPhase.UPDATE, audio, "microphone-1").stream_id == "microphone-1"
    with pytest.raises(ValidationError):
        StreamEvent(StreamPhase.UPDATE)
    with pytest.raises(ValidationError):
        StreamEvent(StreamPhase.CANCEL, "must be empty")
    for invalid in (b"bytes", "", ()):
        with pytest.raises(ValidationError):
            StreamEvent(StreamPhase.FINAL, cast(Any, invalid))


@pytest.mark.asyncio
async def test_interleaved_stream_events_keep_prefetch_and_commits_associated(
    tmp_path: Path,
) -> None:
    memory = AsyncMemory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0)
    try:
        commits = [
            value
            async for value in AsyncCaptureStream(memory).consume(
                _events(
                    StreamEvent(StreamPhase.UPDATE, "alpha", "left"),
                    StreamEvent(StreamPhase.UPDATE, "beta", "right"),
                    StreamEvent(StreamPhase.FINAL, "alpha", "left"),
                    StreamEvent(StreamPhase.FINAL, "beta", "right"),
                )
            )
        ]

        assert [commit.stream_id for commit in commits] == ["left", "right"]
        assert [commit.prefetch.revision for commit in commits if commit.prefetch] == [1, 1]
        assert {item.content for item in (await memory.list()).items} == {"alpha", "beta"}
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_pcm_uses_native_audio_embedding_and_acoustic_boundary(
    tmp_path: Path,
) -> None:
    embedder = RecordingEmbedder(TinyEmbedder.embedding_capabilities)
    memory = AsyncMemory(tmp_path, embedder=embedder, minimum_relevance=0)
    pcm = b"\x00\x00\x01\x00" * 80
    try:
        commits = [
            value
            async for value in AsyncAudioStream(memory).consume(
                _audio_packets(
                    PCMChunk(pcm, stream_id="room"),
                    AcousticBoundary(AudioBoundary.END, stream_id="room"),
                )
            )
        ]

        assert len(commits) == 1
        assert commits[0].stream_id == "room"
        assert commits[0].record.modality is Modality.AUDIO
        assert any(value.modalities == {Modality.AUDIO} for value in embedder.inputs)
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_asr_partial_routes_pcm_to_text_embedding_and_preserves_audio(
    tmp_path: Path,
) -> None:
    embedder = RecordingEmbedder(frozenset({Modality.TEXT}))
    memory = AsyncMemory(tmp_path, embedder=embedder, minimum_relevance=0)
    started_at = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    ended_at = started_at + timedelta(milliseconds=20)
    pcm = b"\x00\x00\x01\x00" * 80
    try:
        commits = [
            value
            async for value in AsyncAudioStream(memory).consume(
                _audio_packets(
                    VADPacket(True, stream_id="headset", occurred_at=started_at),
                    PCMChunk(pcm, stream_id="headset"),
                    ASRPartial("red toolbox", stream_id="headset"),
                    VADPacket(False, stream_id="headset", occurred_at=ended_at),
                )
            )
        ]

        assert len(commits) == 1
        commit = commits[0]
        assert commit.stream_id == "headset"
        assert commit.prefetch is not None and commit.prefetch.revision == 1
        assert commit.record.modality is Modality.AUDIO
        assert "red toolbox" in commit.record.content
        assert commit.record.occurred_at == started_at
        assert commit.record.occurred_end == ended_at
        assert all(value.modalities == {Modality.TEXT} for value in embedder.inputs)
        audio_path = commit.record.assets[0].path
        assert audio_path is not None
        with wave.open(str(audio_path), "rb") as recording:
            assert recording.getframerate() == 16_000
            assert recording.getnchannels() == 1
            assert recording.getsampwidth() == 2
            assert recording.readframes(recording.getnframes()) == pcm
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_vision_frames_use_native_image_embedding_and_keep_latest_keyframe(
    tmp_path: Path,
) -> None:
    embedder = RecordingEmbedder(frozenset({Modality.IMAGE}))
    memory = AsyncMemory(tmp_path, embedder=embedder, minimum_relevance=0)
    started_at = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    ended_at = started_at + timedelta(seconds=1)
    try:
        commits = [
            value
            async for value in AsyncVisionStream(memory).consume(
                _vision_packets(
                    VisionFrame(
                        Blob(b"first frame", "image/jpeg", "first.jpg"),
                        stream_id="camera",
                        occurred_at=started_at,
                    ),
                    VisionFrame(
                        Blob(b"latest frame", "image/jpeg", "latest.jpg"),
                        stream_id="camera",
                    ),
                    SceneBoundary(
                        VisionBoundary.END,
                        stream_id="camera",
                        occurred_at=ended_at,
                    ),
                )
            )
        ]

        assert len(commits) == 1
        commit = commits[0]
        assert commit.stream_id == "camera"
        assert commit.record.modality is Modality.IMAGE
        assert commit.record.occurred_at == started_at
        assert commit.record.occurred_end == ended_at
        assert commit.record.assets[0].path is not None
        assert commit.record.assets[0].path.read_bytes() == b"latest frame"
        assert all(value.modalities == {Modality.IMAGE} for value in embedder.inputs)
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_visual_partials_route_interleaved_frames_to_text_embedding(
    tmp_path: Path,
) -> None:
    embedder = RecordingEmbedder(frozenset({Modality.TEXT}))
    memory = AsyncMemory(tmp_path, embedder=embedder, minimum_relevance=0)
    try:
        commits = [
            value
            async for value in AsyncVisionStream(memory).consume(
                _vision_packets(
                    VisionFrame(Blob(b"left", "image/png"), stream_id="left"),
                    VisionFrame(Blob(b"right", "image/png"), stream_id="right"),
                    VisionPartial("red toolbox", stream_id="left"),
                    VisionPartial("blue door", stream_id="right"),
                    SceneBoundary(VisionBoundary.END, stream_id="right"),
                    SceneBoundary(VisionBoundary.END, stream_id="left"),
                )
            )
        ]

        assert [commit.stream_id for commit in commits] == ["right", "left"]
        paths = [commit.record.assets[0].path for commit in commits]
        assert all(path is not None for path in paths)
        assert [path.read_bytes() for path in paths if path is not None] == [
            b"right",
            b"left",
        ]
        assert [commit.record.content.splitlines()[-1] for commit in commits] == [
            "blue door",
            "red toolbox",
        ]
        assert all(value.modalities == {Modality.TEXT} for value in embedder.inputs)
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_visual_backend_describes_the_final_frame_for_text_embedding(
    tmp_path: Path,
) -> None:
    embedder = RecordingEmbedder(frozenset({Modality.TEXT}))
    describer = RecordingVisionDescriber()
    memory = AsyncMemory(
        tmp_path,
        embedder=embedder,
        vision_describer=describer,
        minimum_relevance=0,
    )
    try:
        commits = [
            value
            async for value in AsyncVisionStream(memory).consume(
                _vision_packets(
                    VisionFrame(Blob(b"frame", "image/png")),
                    SceneBoundary(VisionBoundary.END),
                )
            )
        ]

        assert len(commits) == 1
        assert commits[0].record.content.splitlines()[-1] == ("automatically described red toolbox")
        assert [value.modalities for value in describer.inputs] == [{Modality.IMAGE}]
        assert all(value.modalities == {Modality.TEXT} for value in embedder.inputs)
    finally:
        await memory.close()
    assert describer.closed is True


@pytest.mark.asyncio
async def test_visual_fallback_never_drops_an_undescribed_frame(tmp_path: Path) -> None:
    memory = AsyncMemory(
        tmp_path,
        embedder=RecordingEmbedder(frozenset({Modality.TEXT})),
        minimum_relevance=0,
    )
    try:
        with pytest.raises(ModelError, match="image"):
            await anext(
                AsyncVisionStream(memory).consume(
                    _vision_packets(
                        VisionFrame(Blob(b"frame", "image/png")),
                        SceneBoundary(VisionBoundary.END),
                    )
                )
            )
        assert (await memory.list()).items == ()
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_retrieval_failure_is_visible_without_losing_the_final_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = AsyncMemory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0)

    async def fail_search(*_args: object, **_kwargs: object) -> object:
        raise StorageError("index unavailable")

    monkeypatch.setattr(memory, "search", fail_search)
    try:
        commits = [
            value
            async for value in AsyncCaptureStream(memory).consume(
                _events(StreamEvent(StreamPhase.FINAL, "door closed"))
            )
        ]
        assert len(commits) == 1
        assert commits[0].record.content == "door closed"
        assert commits[0].prefetch is None
        assert commits[0].retrieval_error == StorageError.code
        assert [item.content for item in (await memory.list()).items] == ["door closed"]
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_cancellation_during_final_retrieval_never_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = AsyncMemory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def block_search(*_args: object, **_kwargs: object) -> tuple[()]:
        started.set()
        await release.wait()
        return ()

    monkeypatch.setattr(memory, "search", block_search)

    async def collect() -> list[object]:
        return [
            value
            async for value in AsyncCaptureStream(memory).consume(
                _events(StreamEvent(StreamPhase.FINAL, "do not persist"))
            )
        ]

    try:
        task = asyncio.create_task(collect())
        await started.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await memory.list()).items == ()
    finally:
        release.set()
        await memory.close()


@pytest.mark.asyncio
async def test_cancellation_after_final_commit_starts_waits_for_the_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = AsyncMemory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0)
    started = threading.Event()
    release = threading.Event()
    add = cast(Any, memory._memory.add)

    def block_add(*args: object, **kwargs: object) -> object:
        started.set()
        release.wait()
        return add(*args, **kwargs)

    monkeypatch.setattr(memory._memory, "add", block_add)

    async def collect() -> list[object]:
        return [
            value
            async for value in AsyncCaptureStream(memory).consume(
                _events(StreamEvent(StreamPhase.FINAL, "persist exactly once"))
            )
        ]

    try:
        task = asyncio.create_task(collect())
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [item.content for item in (await memory.list()).items] == ["persist exactly once"]
    finally:
        release.set()
        await memory.close()


@pytest.mark.asyncio
async def test_cancellation_wins_after_a_started_final_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = AsyncMemory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0)
    started = threading.Event()
    release = threading.Event()

    def fail_add(*_args: object, **_kwargs: object) -> object:
        started.set()
        release.wait()
        raise StorageError("write failed")

    monkeypatch.setattr(memory._memory, "add", fail_add)

    async def collect() -> list[object]:
        return [
            value
            async for value in AsyncCaptureStream(memory).consume(
                _events(StreamEvent(StreamPhase.FINAL, "failed final"))
            )
        ]

    try:
        task = asyncio.create_task(collect())
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await memory.list()).items == ()
    finally:
        release.set()
        await memory.close()


@pytest.mark.asyncio
async def test_a_rejected_final_query_still_drains_its_speculative_search(
    tmp_path: Path,
) -> None:
    memory = AsyncMemory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0)
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    prefetches: list[AsyncOmniPrefetch] = []
    build = AsyncCaptureStream._prefetch_for

    def record(
        self: AsyncCaptureStream,
        active: dict[str, AsyncOmniPrefetch],
        stream_id: str,
    ) -> AsyncOmniPrefetch:
        prefetch = build(self, active, stream_id)
        if prefetch not in prefetches:
            prefetches.append(prefetch)
        return prefetch

    # A Path is legal in StreamInput but rejected as a speculative snapshot, so finalize()
    # raises before it closes and drains the worker it started for the earlier UPDATE.
    cast(Any, AsyncCaptureStream)._prefetch_for = record
    try:
        await memory.add("the key is in the blue toolbox")
        commits = [
            value
            async for value in AsyncCaptureStream(memory).consume(
                _events(
                    StreamEvent(StreamPhase.UPDATE, "where is the key", stream_id="s1"),
                    StreamEvent(
                        StreamPhase.FINAL,
                        StreamInput(("the key is here", frame)),
                        stream_id="s1",
                    ),
                )
            )
        ]
        assert len(commits) == 1
        assert commits[0].retrieval_error == ValidationError.code
        assert [prefetch._closed for prefetch in prefetches] == [True]
        assert all(prefetch._worker is None or prefetch._worker.done() for prefetch in prefetches)
    finally:
        cast(Any, AsyncCaptureStream)._prefetch_for = build
        await memory.close()


@pytest.mark.asyncio
async def test_streamed_observations_carry_the_pose_sampled_at_each_boundary(
    tmp_path: Path,
) -> None:
    """A moving robot's pose belongs to the observation, not to the stream that outlives it.

    Both adapters previously built `StreamInput` from timing and text alone, so every memory a
    microphone or camera wrote had a null pose and the default `MemoryType` regardless of what
    the caller asked for. The context is read once per closed observation, so the second scene
    is stamped with where the robot was when that scene ended, not where it started.
    """
    memory = AsyncMemory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0)
    poses = iter(
        (
            ObservationContext(
                basis=EvidenceBasis.OBSERVATION,
                spatial=SpatialContext(frame_id="map", anchor=SpatialAnchor.OBSERVER, x=1.0, y=2.0),
            ),
            ObservationContext(
                basis=EvidenceBasis.OBSERVATION,
                spatial=SpatialContext(frame_id="map", anchor=SpatialAnchor.OBSERVER, x=9.0, y=8.0),
            ),
        )
    )
    try:
        commits = [
            value
            async for value in AsyncVisionStream(
                memory,
                memory_type=MemoryType.EPISODIC,
                context=lambda: next(poses),
            ).consume(
                _vision_packets(
                    VisionPartial("a red toolbox on the bench", stream_id="head"),
                    SceneBoundary(VisionBoundary.END, stream_id="head"),
                    VisionPartial("a blue mug by the sink", stream_id="head"),
                    SceneBoundary(VisionBoundary.END, stream_id="head"),
                )
            )
        ]

        assert len(commits) == 2
        records = [await memory.get(commit.record.id) for commit in commits]
        assert [record.memory_type for record in records] == [MemoryType.EPISODIC] * 2
        spatial = [record.context.spatial for record in records if record.context is not None]
        assert [(value.x, value.y) for value in spatial if value is not None] == [
            (1.0, 2.0),
            (9.0, 8.0),
        ]

        heard = [
            value
            async for value in AsyncAudioStream(
                memory,
                memory_type=MemoryType.EPISODIC,
                context=ObservationContext(
                    spatial=SpatialContext(
                        frame_id="map", anchor=SpatialAnchor.OBSERVER, x=4.0, y=5.0
                    )
                ),
            ).consume(
                _audio_packets(
                    ASRPartial("put the wrench away", stream_id="mic"),
                    AcousticBoundary(AudioBoundary.END, stream_id="mic"),
                )
            )
        ]

        assert len(heard) == 1
        spoken = await memory.get(heard[0].record.id)
        assert spoken.memory_type is MemoryType.EPISODIC
        assert spoken.context is not None and spoken.context.spatial is not None
        assert (spoken.context.spatial.x, spoken.context.spatial.y) == (4.0, 5.0)
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_a_captured_final_acknowledges_before_the_models_and_settles_later(
    tmp_path: Path,
) -> None:
    memory = AsyncMemory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0)
    try:
        commits = [
            value
            async for value in AsyncCaptureStream(memory, capture=True).consume(
                _events(
                    StreamEvent(StreamPhase.UPDATE, "find"),
                    StreamEvent(StreamPhase.FINAL, StreamInput("find the red toolbox")),
                )
            )
        ]

        (commit,) = commits
        assert commit.pending_settlement is True
        assert [row.memory_id for row in await memory.pending_captures()] == [commit.record.id]
        assert await memory.search("red toolbox") == ()

        assert await memory.settle() == 1
        assert await memory.pending_captures() == ()
        assert [hit.id for hit in await memory.search("red toolbox")] == [commit.record.id]
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_a_captured_audio_final_settles_into_the_record_add_would_have_written(
    tmp_path: Path,
) -> None:
    pcm = b"\x00\x00\x01\x00" * 80
    started_at = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    ended_at = started_at + timedelta(milliseconds=20)

    async def commit_one(directory: Path, *, capture: bool) -> MemoryRecord:
        memory = AsyncMemory(
            directory,
            embedder=RecordingEmbedder(frozenset({Modality.TEXT})),
            minimum_relevance=0,
        )
        try:
            commits = [
                value
                async for value in AsyncAudioStream(memory, capture=capture).consume(
                    _audio_packets(
                        VADPacket(True, stream_id="headset", occurred_at=started_at),
                        PCMChunk(pcm, stream_id="headset"),
                        ASRPartial("red toolbox", stream_id="headset"),
                        VADPacket(False, stream_id="headset", occurred_at=ended_at),
                    )
                )
            ]
            (commit,) = commits
            assert commit.pending_settlement is capture
            if capture:
                assert await memory.settle() == 1
            return await memory.get(commit.record.id)
        finally:
            await memory.close()

    captured = await commit_one(tmp_path / "captured", capture=True)
    added = await commit_one(tmp_path / "added", capture=False)

    # The ASR hypothesis the reducer already holds is folded in at capture time, so the deferred
    # path lands on the same content-addressed record as the strong one.
    assert captured.id == added.id
    assert captured.content == added.content
    assert "red toolbox" in captured.content


@pytest.mark.asyncio
async def test_a_captured_vision_final_reports_its_pending_settlement(tmp_path: Path) -> None:
    memory = AsyncMemory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0)
    try:
        commits = [
            value
            async for value in AsyncVisionStream(memory, capture=True).consume(
                _vision_packets(
                    VisionFrame(Blob(b"frame", "image/png"), stream_id="camera"),
                    VisionPartial("a red toolbox beside the door", stream_id="camera"),
                    SceneBoundary(VisionBoundary.END, stream_id="camera"),
                )
            )
        ]

        (commit,) = commits
        assert commit.pending_settlement is True
        assert [row.memory_id for row in await memory.pending_captures()] == [commit.record.id]
        assert await memory.settle() == 1
        assert [hit.id for hit in await memory.search("red toolbox")] == [commit.record.id]
    finally:
        await memory.close()


def test_add_stream_can_capture_instead_of_adding(tmp_path: Path) -> None:
    from mindbridge import Memory

    with Memory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0) as memory:
        records = list(
            memory.add_stream(
                (StreamInput("the ladder is in the garage"), "the fuse box is behind the coats"),
                capture=True,
            )
        )

        assert [row.memory_id for row in memory.pending_captures()] == [
            record.id for record in records
        ]
        assert memory.search("ladder") == ()
        assert memory.settle() == 2
        assert records[0].id in {hit.id for hit in memory.search("ladder")}
