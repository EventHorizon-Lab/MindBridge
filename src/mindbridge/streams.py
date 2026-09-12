"""Async observation streams over `AsyncMemory`: omni prefetch, capture, audio, and vision."""

from __future__ import annotations

import asyncio
import builtins
import io
import wave
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from mindbridge.exceptions import MindBridgeError, ValidationError
from mindbridge.kernel.content import content_atoms, declared_atom_modality, snapshot_content
from mindbridge.kernel.contracts import positive_dimension
from mindbridge.kernel.temporal import resolved_reference_at, search_occurrence_range
from mindbridge.kernel.validation import optional_memory_type, validate_limit
from mindbridge.memory import AsyncMemory
from mindbridge.types import (
    AcousticBoundary,
    ASRPartial,
    AudioBoundary,
    AudioStreamPacket,
    Blob,
    ContentAtom,
    ContentInput,
    MemoryRecord,
    MemoryType,
    Modality,
    ObservationContext,
    PCMChunk,
    PrefetchResult,
    SceneBoundary,
    StreamCommit,
    StreamEvent,
    StreamInput,
    StreamPhase,
    VADPacket,
    VisionBoundary,
    VisionFrame,
    VisionPartial,
    VisionStreamPacket,
)


@dataclass(slots=True)
class _AudioStreamState:
    pcm: bytearray
    sample_rate_hz: int | None = None
    channels: int | None = None
    sample_width_bytes: int | None = None
    transcript: str = ""
    occurred_at: datetime | None = None


@dataclass(slots=True)
class _VisionStreamState:
    image: Blob | None = None
    description: str = ""
    occurred_at: datetime | None = None
    last_occurred_at: datetime | None = None


class AsyncOmniPrefetch:
    """Coalesce evolving omni query snapshots into one in-flight search per turn."""

    def __init__(
        self,
        memory: AsyncMemory,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
    ) -> None:
        validate_limit(limit, maximum=100)
        occurred_from, occurred_until = search_occurrence_range(
            occurred_from,
            occurred_until,
        )
        self._memory = memory
        self.validate_limit = limit
        self.validated_memory_type = optional_memory_type(memory_type)
        self.resolved_reference_at = resolved_reference_at(reference_at)
        self._occurred_from = occurred_from
        self._occurred_until = occurred_until
        self._revision = 0
        self._submitted: tuple[int, ContentInput] | None = None
        self._pending: tuple[int, ContentInput] | None = None
        self._latest: PrefetchResult | None = None
        self._failure: tuple[int, Exception] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def latest(self) -> PrefetchResult | None:
        """Return the newest completed result without waiting."""
        return self._latest

    def submit(self, query: ContentInput) -> int:
        """Queue a complete current snapshot, replacing any not-yet-started snapshot."""
        if self._closed:
            raise ValidationError("prefetch is closed")
        loop = asyncio.get_running_loop()
        snapshot = snapshot_content(query)
        if self._submitted is not None and self._submitted[1] == snapshot:
            failed = self._failure is not None and self._failure[0] == self._submitted[0]
            if not failed:
                return self._submitted[0]
        self._revision += 1
        revision = self._revision
        self._submitted = (revision, snapshot)
        self._pending = self._submitted
        self._failure = None
        if self._worker is None:
            self._worker = loop.create_task(self._run())
        return revision

    async def finalize(self, query: ContentInput | None = None) -> PrefetchResult:
        """Finish the turn and return a result for the exact final snapshot."""
        if self._closed:
            raise ValidationError("prefetch is closed")
        if query is None:
            if self._submitted is None:
                raise ValidationError("prefetch has no submitted query")
            target = self._submitted[0]
        else:
            target = self.submit(query)
        self._closed = True
        worker = self._worker
        if worker is not None:
            await asyncio.shield(worker)
        if self._latest is not None and self._latest.revision == target:
            return self._latest
        if self._failure is not None and self._failure[0] == target:
            raise self._failure[1]
        raise RuntimeError("prefetch completed without its final revision")

    async def close(self) -> None:
        """Discard queued snapshots and drain the one search that may already be running."""
        self._closed = True
        self._pending = None
        worker = self._worker
        if worker is not None:
            await asyncio.shield(worker)

    async def _run(self) -> None:
        try:
            while self._pending is not None:
                revision, query = self._pending
                self._pending = None
                try:
                    hits = await self._memory.search(
                        query,
                        limit=self.validate_limit,
                        memory_type=self.validated_memory_type,
                        reference_at=self.resolved_reference_at,
                        occurred_from=self._occurred_from,
                        occurred_until=self._occurred_until,
                    )
                except Exception as error:
                    self._failure = (revision, error)
                    continue
                self._latest = PrefetchResult(revision=revision, hits=hits)
        finally:
            self._worker = None


class AsyncCaptureStream:
    """Reduce associated capture streams into retrieval and durable memories.

    `capture=True` commits each `FINAL` through `Memory.capture()` instead of `Memory.add()`, so
    the acknowledgement leaves the model path and every `StreamCommit` reports
    `pending_settlement`. The host then owes `settle()`; the default stays the strong `add()`.
    """

    def __init__(
        self,
        memory: AsyncMemory,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        max_streams: int = 32,
        capture: bool = False,
    ) -> None:
        if not isinstance(memory, AsyncMemory):
            raise ValidationError("memory must be an AsyncMemory")
        validate_limit(limit, maximum=100)
        if not isinstance(capture, bool):
            raise ValidationError("capture must be a boolean")
        self._memory = memory
        self.validate_limit = limit
        self.validated_memory_type = optional_memory_type(memory_type)
        self.resolved_reference_at = resolved_reference_at(reference_at)
        self._max_streams = positive_dimension(max_streams, "max_streams")
        self._capture = capture

    async def consume(  # noqa: C901 - the three-state reducer is intentionally inline
        self,
        events: AsyncIterable[StreamEvent],
    ) -> AsyncIterator[StreamCommit]:
        """Yield only final observations; partials, cancellation, and EOF never write."""
        if not isinstance(events, AsyncIterable):
            raise ValidationError("events must be an async iterable of StreamEvent values")
        prefetches: dict[str, AsyncOmniPrefetch] = {}
        try:
            async for event in events:
                if not isinstance(event, StreamEvent):
                    raise ValidationError("events must contain StreamEvent values")
                stream_id = event.stream_id
                if event.phase is StreamPhase.UPDATE:
                    prefetch = self._prefetch_for(prefetches, stream_id)
                    assert event.item is not None and not isinstance(event.item, StreamInput)
                    prefetch.submit(event.item)
                    continue
                if event.phase is StreamPhase.CANCEL:
                    cancelled_prefetch = (
                        prefetches.pop(stream_id) if stream_id in prefetches else None
                    )
                    if cancelled_prefetch is not None:
                        await cancelled_prefetch.close()
                    continue
                assert event.item is not None
                prefetch = self._prefetch_for(prefetches, stream_id)
                item = event.item
                final_content = self._final_query(item)
                retrieval_error: Exception | None = None
                retrieval: PrefetchResult | None = None
                try:
                    retrieval = await prefetch.finalize(final_content)
                except Exception as error:
                    retrieval_error = error
                    # finalize() drains its own worker only once it has started closing, so an
                    # early rejection would otherwise abandon a search already in flight.
                    await prefetch.close()
                prefetches.pop(stream_id, None)
                # ponytail: FINAL is the commit point; a cancellable storage transaction would
                # be needed to revoke it safely once the worker thread has started.
                commit = asyncio.create_task(self._add_final(item))
                cancelled = False
                while not commit.done():
                    try:
                        await asyncio.shield(commit)
                    except asyncio.CancelledError:
                        cancelled = True
                    except Exception:
                        if cancelled:
                            raise asyncio.CancelledError from None
                        raise
                try:
                    record = commit.result()
                except Exception:
                    if cancelled:
                        raise asyncio.CancelledError from None
                    raise
                if cancelled:
                    raise asyncio.CancelledError
                if retrieval_error is not None:
                    yield StreamCommit(
                        record=record,
                        prefetch=None,
                        retrieval_error=(
                            retrieval_error.code
                            if isinstance(retrieval_error, MindBridgeError)
                            else "retrieval_failed"
                        ),
                        stream_id=stream_id,
                        pending_settlement=self._capture,
                    )
                else:
                    assert retrieval is not None
                    yield StreamCommit(
                        record=record,
                        prefetch=retrieval,
                        stream_id=stream_id,
                        pending_settlement=self._capture,
                    )
        finally:
            for prefetch in tuple(prefetches.values()):
                await prefetch.close()

    def _prefetch_for(
        self,
        prefetches: dict[str, AsyncOmniPrefetch],
        stream_id: str,
    ) -> AsyncOmniPrefetch:
        prefetch = prefetches.get(stream_id)
        if prefetch is not None:
            return prefetch
        if len(prefetches) >= self._max_streams:
            raise ValidationError("capture stream exceeds max_streams")
        prefetch = self._new_prefetch()
        prefetches[stream_id] = prefetch
        return prefetch

    def _final_query(self, item: ContentInput | StreamInput) -> ContentInput:
        if not isinstance(item, StreamInput) or (
            item.transcript is None and item.description is None
        ):
            return item.content if isinstance(item, StreamInput) else item
        capabilities = self._memory.capabilities.embedding
        if Modality.TEXT not in capabilities:
            return item.content
        routed: builtins.list[ContentAtom] = [
            value for value in (item.transcript, item.description) if value is not None
        ]
        for atom in content_atoms(item.content):
            modality = declared_atom_modality(atom)
            if (
                modality is Modality.AUDIO
                and item.transcript is not None
                and modality not in capabilities
            ):
                continue
            if (
                modality in {Modality.IMAGE, Modality.VIDEO}
                and item.description is not None
                and modality not in capabilities
            ):
                continue
            routed.append(atom)
        return routed[0] if len(routed) == 1 else tuple(routed)

    def _new_prefetch(self) -> AsyncOmniPrefetch:
        return AsyncOmniPrefetch(
            self._memory,
            limit=self.validate_limit,
            memory_type=self.validated_memory_type,
            reference_at=self.resolved_reference_at,
        )

    async def _add_final(self, item: ContentInput | StreamInput) -> MemoryRecord:
        if isinstance(item, StreamInput):
            if self._capture:
                return await self._memory._capture_stream_input(item)
            return await self._memory._add_stream_input(item)
        if self._capture:
            return await self._memory.capture(item)
        return await self._memory.add(item)


StreamContext = ObservationContext | Callable[[], ObservationContext | None] | None


"""Provenance for streamed observations: one fixed value, or a sampler read at each boundary."""


def _stream_context(context: StreamContext) -> Callable[[], ObservationContext | None]:
    """Normalize a fixed context or a per-observation sampler into one sampler.

    A capture stream outlives the observations it commits, so a robot's pose is not a property
    of the stream. A callable is read once per closed observation, which is the only moment at
    which the adapter knows which interval the pose belongs to. A fixed `ObservationContext`
    stays accepted because a static camera really does have one.
    """
    if context is None:
        return lambda: None
    if isinstance(context, ObservationContext):
        return lambda: context
    if not callable(context):
        raise ValidationError("context must be an ObservationContext or a callable returning one")
    return context


class AsyncAudioStream:
    """Normalize PCM, VAD, ASR, and acoustic boundaries into associated capture events."""

    def __init__(
        self,
        memory: AsyncMemory,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        context: StreamContext = None,
        reference_at: datetime | None = None,
        max_streams: int = 32,
        capture: bool = False,
    ) -> None:
        if not isinstance(memory, AsyncMemory):
            raise ValidationError("memory must be an AsyncMemory")
        self._memory = memory
        self._max_streams = positive_dimension(max_streams, "max_streams")
        self._written_type = optional_memory_type(memory_type) or MemoryType.SEMANTIC
        self._context = _stream_context(context)
        capabilities = memory._memory._backends.embedding_capabilities
        self._native_audio = Modality.AUDIO in capabilities
        self._native_text = Modality.TEXT in capabilities
        self._max_pcm_bytes = memory._memory._storage.assets.max_bytes - 44
        self._capture = AsyncCaptureStream(
            memory,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            max_streams=max_streams,
            capture=capture,
        )

    async def consume(
        self,
        packets: AsyncIterable[AudioStreamPacket],
    ) -> AsyncIterator[StreamCommit]:
        """Yield one associated commit for each VAD or acoustic end boundary."""
        if not isinstance(packets, AsyncIterable):
            raise ValidationError("packets must be an async iterable of audio stream values")
        async for commit in self._capture.consume(self._events(packets)):
            yield commit

    async def _events(  # noqa: C901 - packet kinds form one explicit stream state machine
        self,
        packets: AsyncIterable[AudioStreamPacket],
    ) -> AsyncIterator[StreamEvent]:
        states: dict[str, _AudioStreamState] = {}
        async for packet in packets:
            if not isinstance(packet, (PCMChunk, VADPacket, ASRPartial, AcousticBoundary)):
                raise ValidationError("packets contain an invalid audio stream value")
            stream_id = packet.stream_id
            if isinstance(packet, AcousticBoundary):
                if packet.boundary is AudioBoundary.CANCEL:
                    states.pop(stream_id, None)
                    yield StreamEvent(StreamPhase.CANCEL, stream_id=stream_id)
                    continue
                if packet.boundary is AudioBoundary.END:
                    final = self._finish(states, stream_id, packet.occurred_at)
                    if final is not None:
                        yield final
                    continue
                current = states.get(stream_id)
                if current is not None and (current.pcm or current.transcript):
                    raise ValidationError("audio stream started before its prior boundary")
                if current is None:
                    self._state_for(states, stream_id, packet.occurred_at)
                else:
                    current.occurred_at = packet.occurred_at
                continue
            if isinstance(packet, VADPacket):
                if not packet.active:
                    final = self._finish(states, stream_id, packet.occurred_at)
                    if final is not None:
                        yield final
                else:
                    self._state_for(states, stream_id, packet.occurred_at)
                continue
            state = self._state_for(states, stream_id, packet.occurred_at)
            if isinstance(packet, ASRPartial):
                state.transcript = packet.text
            else:
                self._append_pcm(state, packet)
            query = self._query(state)
            if query is not None:
                yield StreamEvent(StreamPhase.UPDATE, query, stream_id)

    def _state_for(
        self,
        states: dict[str, _AudioStreamState],
        stream_id: str,
        occurred_at: datetime | None,
    ) -> _AudioStreamState:
        state = states.get(stream_id)
        if state is not None:
            if state.occurred_at is None:
                state.occurred_at = occurred_at
            return state
        if len(states) >= self._max_streams:
            raise ValidationError("audio stream exceeds max_streams")
        state = _AudioStreamState(bytearray(), occurred_at=occurred_at)
        states[stream_id] = state
        return state

    def _append_pcm(self, state: _AudioStreamState, packet: PCMChunk) -> None:
        sample_format = (
            packet.sample_rate_hz,
            packet.channels,
            packet.sample_width_bytes,
        )
        current_format = (
            state.sample_rate_hz,
            state.channels,
            state.sample_width_bytes,
        )
        if state.sample_rate_hz is None:
            state.sample_rate_hz, state.channels, state.sample_width_bytes = sample_format
        elif current_format != sample_format:
            raise ValidationError("PCM format changed before an audio boundary")
        if len(state.pcm) + len(packet.data) > self._max_pcm_bytes:
            raise ValidationError("PCM stream exceeds the local asset size limit")
        state.pcm.extend(packet.data)

    def _query(self, state: _AudioStreamState) -> ContentInput | None:
        if self._native_audio and state.pcm:
            audio = _pcm_blob(state)
            if self._native_text and state.transcript:
                return (state.transcript, audio)
            return audio
        return state.transcript or None

    def _finish(
        self,
        states: dict[str, _AudioStreamState],
        stream_id: str,
        occurred_end: datetime | None,
    ) -> StreamEvent | None:
        state = states.pop(stream_id, None)
        if state is None:
            return None
        occurred_at, occurred_end = _audio_interval(state, occurred_end)
        # Sampled here, at the boundary, so the pose belongs to the interval being committed.
        context = self._context()
        if state.pcm:
            item = StreamInput(
                _pcm_blob(state),
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                memory_type=self._written_type,
                context=context,
                transcript=state.transcript or None,
            )
        elif state.transcript:
            item = StreamInput(
                state.transcript,
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                memory_type=self._written_type,
                context=context,
            )
        else:
            return StreamEvent(StreamPhase.CANCEL, stream_id=stream_id)
        return StreamEvent(StreamPhase.FINAL, item, stream_id)


class AsyncVisionStream:
    """Normalize image frames, visual descriptions, and scene boundaries."""

    def __init__(
        self,
        memory: AsyncMemory,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        context: StreamContext = None,
        reference_at: datetime | None = None,
        max_streams: int = 32,
        capture: bool = False,
    ) -> None:
        if not isinstance(memory, AsyncMemory):
            raise ValidationError("memory must be an AsyncMemory")
        self._memory = memory
        self._max_streams = positive_dimension(max_streams, "max_streams")
        self._written_type = optional_memory_type(memory_type) or MemoryType.SEMANTIC
        self._context = _stream_context(context)
        capabilities = memory._memory._backends.embedding_capabilities
        self._native_image = Modality.IMAGE in capabilities
        self._native_text = Modality.TEXT in capabilities
        self._capture = AsyncCaptureStream(
            memory,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            max_streams=max_streams,
            capture=capture,
        )

    async def consume(
        self,
        packets: AsyncIterable[VisionStreamPacket],
    ) -> AsyncIterator[StreamCommit]:
        """Yield one associated commit for each completed visual scene."""
        if not isinstance(packets, AsyncIterable):
            raise ValidationError("packets must be an async iterable of vision stream values")
        async for commit in self._capture.consume(self._events(packets)):
            yield commit

    async def _events(  # noqa: C901 - packet kinds form one explicit stream state machine
        self,
        packets: AsyncIterable[VisionStreamPacket],
    ) -> AsyncIterator[StreamEvent]:
        states: dict[str, _VisionStreamState] = {}
        async for packet in packets:
            if not isinstance(packet, (VisionFrame, VisionPartial, SceneBoundary)):
                raise ValidationError("packets contain an invalid vision stream value")
            stream_id = packet.stream_id
            if isinstance(packet, SceneBoundary):
                if packet.boundary is VisionBoundary.CANCEL:
                    states.pop(stream_id, None)
                    yield StreamEvent(StreamPhase.CANCEL, stream_id=stream_id)
                    continue
                if packet.boundary is VisionBoundary.END:
                    final = await self._finish(states, stream_id, packet.occurred_at)
                    if final is not None:
                        yield final
                    continue
                current = states.get(stream_id)
                if current is not None and (current.image is not None or current.description):
                    raise ValidationError("vision stream started before its prior boundary")
                if current is None:
                    self._state_for(states, stream_id, packet.occurred_at)
                else:
                    current.occurred_at = packet.occurred_at
                    current.last_occurred_at = packet.occurred_at
                continue
            state = self._state_for(states, stream_id, packet.occurred_at)
            if isinstance(packet, VisionPartial):
                state.description = packet.text
            else:
                # ponytail: retain one keyframe per scene; add a bounded sampler only when
                # multi-frame retrieval quality demonstrates that the latest frame is insufficient.
                state.image = packet.image
            query = self._query(state)
            if query is not None:
                yield StreamEvent(StreamPhase.UPDATE, query, stream_id)

    def _state_for(
        self,
        states: dict[str, _VisionStreamState],
        stream_id: str,
        occurred_at: datetime | None,
    ) -> _VisionStreamState:
        state = states.get(stream_id)
        if state is None:
            if len(states) >= self._max_streams:
                raise ValidationError("vision stream exceeds max_streams")
            state = _VisionStreamState(occurred_at=occurred_at)
            states[stream_id] = state
        elif state.occurred_at is None:
            state.occurred_at = occurred_at
        if occurred_at is not None:
            state.last_occurred_at = occurred_at
        return state

    def _query(self, state: _VisionStreamState) -> ContentInput | None:
        if self._native_image and state.image is not None:
            if self._native_text and state.description:
                return (state.description, state.image)
            return state.image
        return state.description or None

    async def _finish(
        self,
        states: dict[str, _VisionStreamState],
        stream_id: str,
        occurred_end: datetime | None,
    ) -> StreamEvent | None:
        state = states.pop(stream_id, None)
        if state is None:
            return None
        occurred_end = occurred_end or state.last_occurred_at
        if (
            state.image is not None
            and not state.description
            and not self._native_image
            and self._native_text
            and self._memory._memory._backends.vision_describer is not None
        ):
            state.description = (
                await asyncio.to_thread(self._memory._memory._vision.describe, (state.image,))
            )[0]
        # Sampled here, at the boundary, so the pose belongs to the scene being committed.
        context = self._context()
        if state.image is not None:
            item = StreamInput(
                state.image,
                occurred_at=state.occurred_at,
                occurred_end=occurred_end,
                memory_type=self._written_type,
                context=context,
                description=state.description or None,
            )
        elif state.description:
            item = StreamInput(
                state.description,
                occurred_at=state.occurred_at,
                occurred_end=occurred_end,
                memory_type=self._written_type,
                context=context,
            )
        else:
            return StreamEvent(StreamPhase.CANCEL, stream_id=stream_id)
        return StreamEvent(StreamPhase.FINAL, item, stream_id)


def _pcm_blob(state: _AudioStreamState) -> Blob:
    assert (
        state.sample_rate_hz is not None
        and state.channels is not None
        and state.sample_width_bytes is not None
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(state.channels)
        audio.setsampwidth(state.sample_width_bytes)
        audio.setframerate(state.sample_rate_hz)
        audio.writeframes(state.pcm)
    return Blob(output.getvalue(), "audio/wav", "capture.wav")


def _audio_interval(
    state: _AudioStreamState,
    occurred_end: datetime | None,
) -> tuple[datetime | None, datetime | None]:
    occurred_at = state.occurred_at
    if not state.pcm:
        return occurred_at, occurred_end
    assert (
        state.sample_rate_hz is not None
        and state.channels is not None
        and state.sample_width_bytes is not None
    )
    frames = len(state.pcm) // (state.channels * state.sample_width_bytes)
    duration = timedelta(seconds=frames / state.sample_rate_hz)
    if occurred_at is None and occurred_end is not None:
        occurred_at = occurred_end - duration
    elif occurred_at is not None and occurred_end is None:
        occurred_end = occurred_at + duration
    return occurred_at, occurred_end
