# Omni streams, speculative recall, and interaction records

MindBridge distinguishes three states in a continuous application:

1. A changing query snapshot used only for speculative search.
2. A completed observation that can be written durably.
3. An application-derived record that cites the observation it came from.

MindBridge does not capture sensors, reconnect streams, select frames, parse partial containers, or
detect turn boundaries. The application decides when an observation is complete and normalizes it
to `ContentInput`.

The snippets assume an open memory plus already-read immutable media bytes. They show lifecycle and
provenance boundaries; media capture and decoding remain application work.

## Add completed observations

`Memory.add_stream` consumes a lazy iterable of `ContentInput` or `StreamInput` values. Each item
uses the ordinary `add` path and is durable and searchable before the next item is pulled. The
stream is not one transaction: a later item or source failure leaves the committed prefix intact.

Use `StreamInput` for per-observation event time, metadata, or memory type:

```python
from datetime import datetime, timedelta, timezone

from mindbridge import Blob, MemoryType, StreamInput

started = datetime(2026, 8, 31, 9, tzinfo=timezone.utc)
observation = StreamInput(
    (
        "The user is looking for the red toolbox.",
        Blob(frame_bytes, "image/png", "frame.png"),
        Blob(audio_bytes, "audio/wav", "speech.wav"),
    ),
    occurred_at=started,
    occurred_end=started + timedelta(seconds=2),
    metadata={"source": "companion_robot", "sequence": 17},
    memory_type=MemoryType.EPISODIC,
)

for source_record in memory.add_stream((observation,)):
    print(source_record.id)
```

The async form accepts an `AsyncIterable` and yields through `async for`. Both forms preserve
source backpressure and name a failing item's zero-based `contents[index]` in the public error.
Mutable ASR hypotheses, incomplete media, and repeatedly overwritten file paths are not completed
observations; keep them out of durable ingestion.

## Prefetch a changing query

`AsyncOmniPrefetch` wraps an open `AsyncMemory` (named `async_memory` below) and overlaps retrieval
with an unfinished turn without storing its partial input. Every submission is a complete current
snapshot, not a delta:

```python
from mindbridge import AsyncOmniPrefetch

prefetch = AsyncOmniPrefetch(async_memory, limit=8)
prefetch.submit((partial_text, latest_frame, current_audio_window))
prefetch.submit((newer_text, latest_frame, current_audio_window))

result = await prefetch.finalize((final_text, final_frame, final_audio_window))
hits = result.hits
```

Only one search runs at a time. While it runs, a newer submission replaces the snapshot that has
not started yet. `latest` returns the newest completed `PrefetchResult` and its revision without
waiting.

`finalize` returns only a result for the exact final snapshot. It reuses an already completed
revision when the values match and that revision succeeded; otherwise it waits for a search of the
final snapshot. It then closes that per-turn prefetcher.

Snapshots accept immutable text, `Blob`, and `AssetRef` values. Raw `Path` values are rejected
because the file may change after submission. If a turn is abandoned, `await prefetch.close()`
drops queued work and waits for any search already dispatched through `asyncio.to_thread`.
Cancelling that task would not stop its synchronous model or index call, so the helper drains it.

Speculative search does not reinforce returned memories. Call `reinforce` only after the
application has observed positive use or feedback.

## Interaction-derived records

Interaction memory is an application convention over existing records, not another store or a
fourth `MemoryType`:

| Derived information | Type |
| --- | --- |
| Explicit preference or supported stable fact | `SEMANTIC` |
| One situated interaction with event time | `EPISODIC` |
| Response guidance supported by feedback | `PROCEDURAL` |

Store the completed source observation first. Continuing with the `source_record` returned above, a
later analysis can write an ordinary record whose metadata cites the source ID:

```python
from mindbridge import MemoryType

guidance = memory.add(
    "When deadline pressure is present, acknowledge it before proposing steps.",
    metadata={
        "basis": "response_feedback",
        "evidence_ids": [source_record.id],
        "confidence": 0.8,
    },
    memory_type=MemoryType.PROCEDURAL,
)
```

MindBridge stores that metadata but does not interpret its provenance or confidence. Emotion,
trait, and policy inference remain application work; the memory kernel does not run them
automatically. Keep the source observation when a derived interpretation changes. Add corrected
evidence or delete the incorrect derived record instead of presenting a rewrite as the original
observation.

Applications can query roles separately when one role must not crowd out another:

```python
semantic = memory.search(query, memory_type=MemoryType.SEMANTIC, limit=2)
episodic = memory.search(query, memory_type=MemoryType.EPISODIC, limit=2)
procedural = memory.search(query, memory_type=MemoryType.PROCEDURAL, limit=1)
```

Procedural records are evidence for the application; MindBridge never executes their text.
Metadata and evidence IDs do not create an authorization or isolation boundary.

## Observability

Every completed stream item emits an ordinary `mindbridge.add` operation span. Every speculative
revision that starts emits an ordinary `mindbridge.search` span. This keeps model request, token,
latency, and failure accounting comparable with non-streaming calls. See
[operations](operations.md#telemetry) for the attribute contract.
