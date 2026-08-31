# Omni streaming, speculative recall, and interaction memory

MindBridge separates three lifecycles that a continuous embodied agent must not confuse:

1. A changing **working snapshot** used only for speculative search.
2. A completed **observation** that is safe to store durably.
3. Derived **interaction memory** whose evidence and cognitive role remain explicit.

This separation applies equally to text, image, video, audio, and combined omni input. MindBridge
does not own cameras, microphones, stream containers, ASR partials, frame selection, or turn
detection. Applications normalize those sources into the existing `ContentInput` contract.

## Streaming design decisions

Speculative recall searches a changing observation before the turn boundary, then consumes the
prefetched result when the turn ends. The query snapshot uses `ContentInput`, so text, images,
video, audio, and their ordered combinations follow one path. One search remains in flight while
queued snapshots coalesce, and the result returned by `finalize` must match the exact final
snapshot. Cancelling an `asyncio.to_thread` task cannot cancel its synchronous work, and silently
reusing a result that omits the final input can retrieve the wrong memory.

Interaction memory separately records affective episodes, supported traits, and learned response
experiences. MindBridge maps those roles onto the existing episodic, semantic, and procedural
memory types instead of adding another store. Emotion and trait inference do not belong in the
kernel: perception quality, privacy, model cost, and supported modalities belong to an explicitly
selected application or analysis plugin, while MindBridge keeps the resulting claim, confidence,
time, and source evidence durable and searchable.

## Completed omni observations

`Memory.add_stream` consumes completed observations lazily. Each item follows the ordinary `add`
path and is durable and searchable before the next item is requested. The stream is therefore not
one transaction: a later failure leaves the committed prefix intact.

Use `StreamInput` when an item needs its own event time, metadata, or memory type. Its `content` is
the same ordered multimodal union accepted by `add`:

```python
from datetime import datetime, timedelta, timezone

from mindbridge import Blob, MemoryType, StreamInput

observed_at = datetime(2026, 8, 31, 9, tzinfo=timezone.utc)
observation = StreamInput(
    (
        "The user is looking for the red toolbox.",
        Blob(frame_bytes, "image/png", "frame.png"),
        Blob(audio_bytes, "audio/wav", "speech.wav"),
    ),
    occurred_at=observed_at,
    occurred_end=observed_at + timedelta(seconds=2),
    metadata={"source": "companion_robot", "sequence": 17},
    memory_type=MemoryType.EPISODIC,
)

for record in memory.add_stream((observation,)):
    print(record.id)
```

An async owner consumes an `AsyncIterable` with `async for`. Both forms preserve backpressure and
use the same SQLite, outbox, and Zvec consistency path as `add`. Mutable ASR hypotheses, incomplete
container fragments, and repeatedly overwritten frame paths are not completed observations and
must not be passed to `add_stream`.

The CLI `add-stream` command provides the same per-line commit behavior for finite JSONL imports.
It still emits one JSON document per invocation, so it collects result records until EOF. Use the
Python iterator for an unbounded source.

## Speculative omni recall

`AsyncOmniPrefetch` hides retrieval latency inside an ongoing turn without persisting partial
input. Every submission is the complete current snapshot, not a delta. A snapshot may combine any
modalities the configured embedder supports. Use immutable text, `Blob`, or stored `AssetRef`
values; raw `Path` values are rejected because their contents can change after submission:

```python
from mindbridge import AsyncOmniPrefetch

prefetch = AsyncOmniPrefetch(memory, limit=8)

prefetch.submit((partial_text, latest_frame, current_audio_window))
prefetch.submit((newer_text, latest_frame, current_audio_window))

result = await prefetch.finalize((final_text, final_frame, final_audio_window))
hits = result.hits
```

Only one actual search runs at a time. While it runs, newer submissions replace the queued
snapshot, so intermediate revisions do not create a thread storm. `latest` exposes the newest
completed `PrefetchResult` and its revision without waiting. `finalize` reuses the result only when
the final snapshot is exactly the most recently submitted value and that revision did not fail.
Otherwise it searches the final snapshot before returning. It then closes that per-turn prefetcher.

If a turn is abandoned, `await prefetch.close()` drops the queued snapshot and drains the search
already in progress. It deliberately does not cancel `asyncio.to_thread` work, because cancelling
the task cannot stop its underlying synchronous model or index call.

Search does not reinforce results. Call `reinforce` only after the application observes that the
agent actually used a memory and received positive feedback. Applications with a hard response
deadline may read `latest` and choose their own fallback, but a stale revision must remain visible
rather than being presented as the final query's result.

Useful end-to-end measurements are:

- the fraction of turns with a completed prefetch at the final boundary;
- final-boundary-to-context latency at p50, p95, and p99;
- the fraction of final snapshots that require another search;
- searches started per turn and maximum concurrent searches per prefetcher;
- retrieval quality for the final snapshot, not only for its earlier prefixes.

## Interaction memory with existing types

Personalized interaction behavior is a storage and retrieval policy over the existing memory
roles, not a new store or `MemoryType`:

| Evidence | Memory type | Trust and use |
| --- | --- | --- |
| Explicit preference or sufficiently supported stable trait | `SEMANTIC` | A user fact or a clearly labelled hypothesis |
| A situated affective interaction with event time and cause | `EPISODIC` | What happened on one occasion |
| A response approach that feedback showed was helpful or harmful | `PROCEDURAL` | Internal interaction guidance, never executable code |

The application or an explicitly configured analysis plugin produces these values after the
observation is final. The memory kernel does not infer emotion or personality. Store the source
observation first, then preserve its stable ID in application metadata on any derived record:

```python
from mindbridge import MemoryType, StreamInput

interaction_items = (
    StreamInput(
        "The user explicitly prefers short, calm explanations.",
        metadata={
            "basis": "user_statement",
            "evidence_ids": [source_record.id],
            "confidence": 1.0,
        },
        memory_type=MemoryType.SEMANTIC,
    ),
    StreamInput(
        "The user sounded tense while discussing the deadline.",
        occurred_at=source_record.occurred_at,
        occurred_end=source_record.occurred_end,
        metadata={
            "basis": "affect_observation",
            "evidence_ids": [source_record.id],
            "confidence": 0.74,
            "target": "deadline",
        },
        memory_type=MemoryType.EPISODIC,
    ),
    StreamInput(
        "When deadline pressure is present, acknowledge it before proposing steps.",
        metadata={
            "basis": "response_feedback",
            "evidence_ids": [source_record.id],
            "confidence": 0.8,
        },
        memory_type=MemoryType.PROCEDURAL,
    ),
)

tuple(memory.add_stream(interaction_items))
```

Metadata is provenance-bearing application data, not an isolation or authorization boundary.
Keep the original observation even when a derived interpretation later changes. A correction adds
new temporal evidence or deletes the incorrect derived record; it does not rewrite history.

For retrieval, request the roles separately so one category cannot crowd out the others:

```python
from mindbridge import MemoryType

semantic = memory.search(query, memory_type=MemoryType.SEMANTIC, limit=2)
episodic = memory.search(query, memory_type=MemoryType.EPISODIC, limit=2)
procedural = memory.search(query, memory_type=MemoryType.PROCEDURAL, limit=1)
```

Treat the sections differently when constructing an agent prompt:

- explicit semantic statements may ground an answer;
- inferred semantic traits are hypotheses and should influence tone softly;
- episodes retain target, cause, time, confidence, and source evidence;
- procedures are private response guidance and must not be quoted as user facts.

Do not promote one emotional episode into a stable semantic trait. An inferred trait needs repeated,
independent evidence or an explicit user statement. User correction outranks model inference.

## Emotion and other analysis plugins

This design does not add an emotion plugin without an implementation that proves its contract. A
future plugin must declare supported atomic modalities, typed output, confidence and provenance,
model identity, lifecycle and concurrency behavior, privacy/export boundary, and stable failure
mapping. Audio prosody, text, face, and posture observations should retain their separate evidence
and be fused late; a single hard label must not erase disagreement between modalities.

Such a plugin may perform inference only. It cannot write around `Memory`, own another store, or
turn metadata into routing or isolation policy. Omitting it must leave stream ingestion, prefetch,
and ordinary memory behavior unchanged.

## Evidence gate for a graph projection

No graph is added by this feature. First establish that the current hybrid retrieval loses useful
top-k positions because a large distractor pool dilutes otherwise retrievable evidence.

Use the public SDK path and fixed dataset/model/runtime revisions. ATM-Bench already reports
retrieval Recall@1/5/10 and Joint@K, while Mem-Gallery reports retrieval Precision/Recall/Hit@10.
Run the same task at small and larger recall limits and retain sample diagnostics:

```bash
mindbridge-bench eval \
  --tasks atm-bench,mem-gallery \
  --recall-limit 5 \
  --log-samples \
  --output_path .benchmarks/results/left-gate-k5

mindbridge-bench eval \
  --tasks atm-bench,mem-gallery \
  --recall-limit 20 \
  --log-samples \
  --output_path .benchmarks/results/left-gate-k20
```

A graph candidate is justified only when all of these are true:

1. Gold evidence is commonly retrieved at a larger K but displaced from the product K; missing
   evidence caused by unsupported modality, failed transcription, or weak embeddings is not a
   graph problem.
2. The failure appears across independent memory units, not a handful of questions sharing one
   store.
3. A product-path prototype improves retrieval and joint answer metrics with a paired confidence
   interval above zero, without a statistically supported regression elsewhere.
4. Its p95/p99 latency, sustained ingestion cost, index size, and rebuild behavior remain within
   the target deployment's explicit budget.

If those gates pass, the smallest acceptable implementation is an additive entity/topic projection
derived from authoritative SQLite records and rebuilt through the existing outbox/index lifecycle.
It must not introduce a graph database, become a second source of truth, create logical account
scopes, or learn from abandoned speculative queries. Accepted final queries may inform a later
measured projection only when that durable, privacy-sensitive behavior is explicit and replayable.
