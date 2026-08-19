# Python SDK

`mindbridge.MindBridge` is a small asynchronous client over the [REST contract](rest.md). It
ships in the base package — `uv add mindbridge` with no extras — because an application that
only *calls* MindBridge should not pull in FastAPI, psycopg, or torch.

Every request and response type is importable from the top-level package, so the first call
costs one import rather than two.

## Connecting

```python
from mindbridge import MindBridge

async with MindBridge.connect(
    base_url="https://memory.example.com",
    api_key="at-least-32-characters-long",
    timeout_seconds=120.0,
) as memory:
    ...
```

`connect()` validates eagerly: a blank `base_url`, a blank `api_key`, or a non-positive timeout
raises `ValueError` at construction rather than at first use. `api_key` may be `None` only if
the deployment sits behind a gateway that adds the header.

The default timeout is 120 seconds because a recall in `answer` mode does real model work — a
30-second default would turn normal latency into a client-side error.

Outside a context manager, close it yourself:

```python
memory = MindBridge.connect(base_url=..., api_key=...)
try:
    ...
finally:
    await memory.close()
```

## Methods

| Method | REST equivalent |
| --- | --- |
| `observe(request)` | `POST /v1/observations` |
| `remember(request)` | `POST /v1/memories` |
| `recall(request)` | `POST /v1/recall` |
| `get_memory(tenant_id, memory_id)` | `GET /v1/memories/{memory_id}` |
| `record_feedback(request)` | `POST /v1/feedback` |
| `forget(request)` | `POST /v1/forget` |
| `get_forget_status(tenant_id, tombstone_id)` | `GET /v1/deletions/{tombstone_id}` |
| `list_deletions(request)` | `GET /v1/deletions` |
| `get_observation_job(tenant_id, job_id)` | `GET /v1/jobs/{job_id}` |
| `stream_observation_job(tenant_id, job_id, *, last_event_id=None)` | `GET /v1/jobs/{job_id}/events` |
| `close()` | — |

## Writing and recalling

```python
from datetime import datetime, timezone

from mindbridge import MemoryType, RecallQuery, RecallRequest, RememberRequest

written = await memory.remember(
    RememberRequest(
        tenant_id="tenant_01",
        summary="The red screwdriver went into the blue toolbox on the workbench.",
        memory_type=MemoryType.EPISODIC,
        occurred_at=datetime.now(timezone.utc),
    )
)
assert written.status.value in {"created", "duplicate"}

result = await memory.recall(
    RecallRequest(
        tenant_id="tenant_01",
        query=RecallQuery(text="Where did the red screwdriver end up?"),
    )
)
```

`result.answer` is `None` when nothing supports an answer. That is a correct outcome, not an
error — check it rather than assuming a string.

Verifying an answer needs no second call to storage: `result.evidence` already carries signed
`media_url`s pointing at the exact `start_ms`–`end_ms` slice of the original recording.

### Grounded follow-up

Pass IDs from a previous result to scope the next question to exactly those memories:

```python
follow_up = await memory.recall(
    RecallRequest(
        tenant_id="tenant_01",
        query=RecallQuery(text="Was anyone else in the room?"),
        memory_ids=tuple(m.memory_id for m in result.memories[:5]),
    )
)
```

`memory_ids` is a strict scope, not a ranking hint. Tenant, lifecycle, deletion, and evidence
checks still apply, but nothing outside that set is searched.

### Searching without answering

```python
from mindbridge import RecallMode

hits = await memory.recall(
    RecallRequest(
        tenant_id="tenant_01",
        query=RecallQuery(text="workbench"),
        mode=RecallMode.SEARCH,
        limit=50,
    )
)
```

`search` skips the generator entirely. Use it when you want ranked memories and intend to do
your own reasoning — it is substantially cheaper and lower-latency than `answer`.

### Recalling by media

```python
result = await memory.recall(
    RecallRequest(
        tenant_id="tenant_01",
        query=RecallQuery(media_object_ids=("media_a1b2",)),
    )
)
```

Text and media are not privileged relative to each other. Either alone is a valid query, and
supplying both encodes them as one query vector.

## Observing, and following the job

`observe()` returns as soon as the observation is durable. The memory does not exist yet.

```python
receipt = await memory.observe(request)
print(receipt.status, receipt.processing_job_id)
```

Poll:

```python
job = await memory.get_observation_job("tenant_01", receipt.processing_job_id)
if job.state.value == "succeeded":
    for memory_id in job.memory_ids:
        ...
```

Or stream, which is usually what you want — deriving memory from raw media takes far longer than
the request that submitted it:

```python
async for event in memory.stream_observation_job("tenant_01", receipt.processing_job_id):
    print(event.job.state, event.job.attempt)
    last_seen = event.event_id
```

Every event carries the complete job view, so resuming after a dropped connection needs only the
last ID:

```python
async for event in memory.stream_observation_job("tenant_01", job_id, last_event_id=last_seen):
    ...
```

Two things to internalise about the stream:

- **`failed` settles the attempt, not the job.** The stale-job sweep can retry it later. Do not
  treat `failed` as terminal unless you have decided it is.
- **Intermediate states are coalesced.** You always observe the newer state, but not necessarily
  every `attempt`. Reconnecting is your decision; the server does not do it for you.

## Feedback

```python
from mindbridge import FeedbackRequest, FeedbackType

await memory.record_feedback(
    FeedbackRequest(
        tenant_id="tenant_01",
        feedback_type=FeedbackType.CORRECTION,
        memory_id=result.memories[0].memory_id,
        correction_summary="It was the green screwdriver, not the red one.",
    )
)
```

A correction writes a new version that supersedes the original; the receipt returns it as
`corrected_memory_id`. Nothing is overwritten.

When recall found nothing usable, report it against the recall rather than a memory:

```python
await memory.record_feedback(
    FeedbackRequest(
        tenant_id="tenant_01",
        feedback_type=FeedbackType.MISSING,
        recall_trace_id=result.trace_id,
    )
)
```

This is why `RecallResult.trace_id` is worth keeping: it is the handle that ties a retrieval
failure back to what was asked.

## Forgetting

```python
from mindbridge import ForgetRequest, ForgetTargetType

receipt = await memory.forget(
    ForgetRequest(
        tenant_id="tenant_01",
        target_type=ForgetTargetType.OBSERVATION,
        target_id="obs_...",
    )
)
```

The receipt reports that erasure was recorded and started. Poll for completion:

```python
status = await memory.get_forget_status("tenant_01", receipt.tombstone_id)
done = status.propagation_state.value == "complete"
```

Only `complete` means every copy is gone, including on devices that were offline when the
deletion was issued.

## Error handling

Every failure raises `MindBridgeError`:

```python
from mindbridge import MindBridgeError

try:
    result = await memory.recall(request)
except MindBridgeError as error:
    error.code  # stable machine-readable code
    error.status_code  # HTTP status, or None for a transport failure
    error.trace_id  # correlates with your telemetry backend
```

Branch on `code`, never on the message. Beyond the [server codes](rest.md#error-codes), the
client itself produces four:

| Code | Meaning |
| --- | --- |
| `transport_error` | The request never got an answer — wrong address, nothing listening, or timed out. `status_code` is `None`. |
| `http_error` | Non-success response whose body was not a parseable error envelope. |
| `invalid_response` | A success response that failed contract validation. Usually a version skew between client and server. |
| `stream_error` | The job stream reported a failure whose payload could not be parsed. |

A minimal retry policy that matches how the server actually fails:

```python
import asyncio

RETRYABLE = {
    "transport_error",
    "database_unavailable",
    "model_unavailable",
    "object_storage_unavailable",
    "task_broker_unavailable",
}


async def with_retry(call, attempts: int = 3):
    for attempt in range(attempts):
        try:
            return await call()
        except MindBridgeError as error:
            if error.code not in RETRYABLE or attempt == attempts - 1:
                raise
            await asyncio.sleep(2**attempt)
```

Retrying a write is safe: idempotency keys are derived from content when omitted, so a duplicate
submission returns the original record rather than creating a second one.

## Types

Everything importable from `mindbridge`:

**Requests** — `ObserveRequest`, `RememberRequest`, `RecallRequest`, `RecallQuery`,
`RecallFilters`, `FeedbackRequest`, `ForgetRequest`, `DeletionListRequest`, `GetMemoryRequest`,
`GetObservationJobRequest`, `MediaObjectInput`, `IdentityObservationInput`.

**Responses** — `ObservationReceipt`, `ObservationProcessingJobView`, `RememberResult`,
`MemoryResult`, `MemoryView`, `RecallResult`, `EvidenceView`, `FeedbackReceipt`, `ForgetReceipt`,
`DeletionPage`, `DeletionTombstoneView`, `HealthResponse`, `ErrorResponse`, `ValidationIssue`.

**Enums** — `MemoryType`, `MemoryState`, `MemoryWriteStatus`, `ObservationStatus`, `JobState`,
`RecallMode`, `FeedbackType`, `ForgetTargetType`, `DeletionPropagationState`,
`VerificationStatus`, `MediaKind`, `SensorKind`, `IdentityKind`, `IdentityScope`.

**Client** — `MindBridge`, `MindBridgeError`, `ObservationJobEvent`.

All contracts are frozen Pydantic models with `extra="forbid"`. A misspelled field is a
construction error, not a value silently dropped on the wire.

`RememberResult` and `MemoryResult` are **siblings**, not parent and child. A write result
carries `status`, which a read has no value for; making it a subclass would let it pass wherever
a read result is accepted and silently drop that field on serialization. Annotate each position
with the one it actually receives.
