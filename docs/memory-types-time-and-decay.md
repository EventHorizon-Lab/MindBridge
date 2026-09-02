# Memory types, time, and decay

This page owns memory-role, event-time, typed-assertion, retrieval-scope, and ranking semantics.
These controls all use the same SQLite record and Zvec projection path; none creates another store
or isolation scope.

| Control | Behavior |
| --- | --- |
| `memory_type` | Hard exact-role filter when supplied |
| `occurred_from` / `occurred_until` | Hard event-overlap filter |
| `RetrievalScope` | Hard valid-time, known-time, and same-frame spatial filters |
| Temporal phrases in query text | Soft event-time ranking signal |
| Reinforcement and decay | Soft ranking signals; content is never rewritten or deleted |

## Memory types

| Type | Store | Kernel behavior |
| --- | --- | --- |
| `MemoryType.SEMANTIC` | Facts and stable application knowledge | Default type |
| `MemoryType.EPISODIC` | Events and observations | Usually paired with event time |
| `MemoryType.PROCEDURAL` | Instructions and reusable routines | Returned as evidence, never executed |

**Contract:** The caller classifies ordinary content. MindBridge does not promote one type into
another unless an explicit `FormationBackend` creates a separate derived record. Memory type is
part of stable identity, and `search()` and `ask()` accept an optional exact type filter.

The snippets below assume an open `memory`:

```python
from datetime import datetime, timezone

from mindbridge import MemoryType

memory.add(
    "The deployment failed because the token expired.",
    memory_type=MemoryType.EPISODIC,
    occurred_at=datetime(2026, 8, 20, 9, tzinfo=timezone.utc),
)
memory.add(
    "Refresh the token, retry once, then escalate.",
    memory_type=MemoryType.PROCEDURAL,
)

episodes = memory.search(
    "What failed last week?",
    memory_type=MemoryType.EPISODIC,
    reference_at=datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
)
```

**Guidance:** Use types to keep cognitive roles distinct, not to enforce access control. Metadata
and types are application data; separate security domains require separate data directories.

## Event time and strict filters

`occurred_at` is an event start. `occurred_end` is an optional exclusive end; it requires a start
and must be later. Both values must be timezone-aware. `created_at` is storage time and is not used
as a substitute when event time is absent.

`search(..., occurred_from=..., occurred_until=...)` applies a hard half-open overlap filter. Either
bound may be omitted. Supplying any bound excludes records without event time. An interval ending
at `occurred_from`, or starting at `occurred_until`, does not match; an instant matches when its
timestamp falls inside the requested range.

**Contract:** SQLite rechecks the event range after Zvec candidate selection. Use these explicit
bounds whenever records outside the interval must be excluded.

## Temporal phrases

Temporal phrases in query text influence ranking; they do not replace explicit event filters. The
bounded parser recognizes:

- ISO dates and date ranges;
- English month-years, Chinese year-months, and calendar years from 1900 through 2199;
- today, yesterday, tomorrow, and the adjacent two-day phrases in English and Chinese;
- last, this, next, and rolling weeks; last, this, and next months or years;
- `N days ago` and rolling past or recent `N` days in English and Chinese.

`reference_at` supplies the timezone-aware clock for relative phrases. Without it, the current UTC
time is used. When no explicit reference is passed, `Today is <date>` may anchor the query. `ask()`
includes the resolved reference in generation input when relative time is involved.

For a detected range, retrieval considers both in-range and global candidates. In-range events are
boosted, nearby events decay smoothly with distance, and records without event time are downranked.

**Guidance:** Use temporal language for natural recall where event boundaries may be noisy. Use
`occurred_from` and `occurred_until` for strict exclusion.

## Typed context and formation

`ObservationContext` is caller input. It attaches provenance, confidence, optional world validity,
and optional spatial pose to a source observation:

```python
from mindbridge import EvidenceBasis, ObservationContext

source = memory.add(
    "The mug is on the kitchen table.",
    context=ObservationContext(
        basis=EvidenceBasis.OBSERVATION,
        source_id="camera-1:frame-42",
        confidence=0.94,
    ),
)
```

The returned record exposes the persisted `MemoryContext`. A source context has
`MemoryKind.OBSERVATION`. An optional former may propose these derived kinds:

| Derived kind | Memory type | Meaning |
| --- | --- | --- |
| `ENTITY` | Semantic | A referenced person, object, or concept |
| `EVENT` | Episodic | A formed event assertion |
| `STATE` | Semantic | A value that can change over valid time |
| `RELATION` | Semantic | A typed relation between subjects |
| `AFFECT` | Episodic | A situated affect cue |
| `TRAIT` | Semantic | A longer-horizon characteristic |
| `RESPONSE_POLICY` | Procedural | Feedback-grounded response guidance |

**Contract:** The source commits before formation. The former only proposes; the kernel validates
source modality and spatial binding, assigns IDs, links evidence, versions conflicting state, and
commits derived records. A formation failure leaves the source durable, and repeating the same add
can complete the missing formation recipe.

Formation never rewrites the caller's source record. `AFFECT` must name a cue modality present in
that source. A model-inferred `TRAIT` remains hidden from active retrieval until two independent
sources support the same normalized claim; a trusted `USER_STATEMENT` can be visible immediately.
Deleting evidence recomputes derived confidence and visibility, and removes a derived record when
no support remains. Source observations are deleted only by an explicit caller action.

**Guidance:** Keep raw observations even when an interpretation changes. Correct typed state with
new evidence or remove the incorrect derived record; do not present a derived rewrite as the
original observation.

## Valid time and transaction time

Occurrence time and typed assertion time answer different questions:

| Field | Question answered |
| --- | --- |
| `occurred_at` / `occurred_end` | When did the captured episode happen? |
| `MemoryContext.valid_from` / `valid_until` | When was this assertion true in the represented world? |
| `MemoryContext.recorded_at` / `retired_at` | When did MindBridge know this assertion version? |

`RetrievalScope(valid_at=..., known_at=...)` combines the last two axes. `valid_at` selects a typed
assertion whose half-open world interval contains that instant. `known_at` selects the transaction
version active then and excludes raw records created later. Evidence links use the same transaction
bounds, so historical retrieval does not expose support added later.

### Spatial scope

Spatial scope uses the same value:

```python
from mindbridge import RetrievalScope, SpatialAnchor, SpatialContext

scope = RetrievalScope(
    valid_at=event_time,
    known_at=audit_time,
    near=SpatialContext(
        frame_id="home/map",
        anchor=SpatialAnchor.SUBJECT,
        x=2.0,
        y=1.0,
    ),
    radius_m=0.75,
)
hits = memory.search("Where was the red toolbox?", scope=scope)
```

`near` and `radius_m` must be supplied together. Spatial search compares only matching coordinate
frames and anchors; position uncertainty expands the conservative intersection test. MindBridge
does not infer coordinate transforms.

**Contract:** Supplying `valid_at` excludes records without typed validity. `get()` and `list()` can
still expose the latest retired or hidden context for audit, while ordinary search returns only
active visible versions.

## Decay and reinforcement

Decay is disabled by default. Enable it with a positive `decay_half_life_days` setting. Decay
changes query-time ranking only; it does not delete or rewrite records, assets, embeddings, or
outbox rows.

Retention is anchored to the newest eligible explicit reinforcement, otherwise event end, event
start, or last update time. Repeated confirmations slow decay. Confirmations also provide a small
ranking boost when decay is disabled, and public scores remain in `[0, 1]`.

Search never reinforces a hit. Record positive application feedback explicitly:

```python
used = memory.search("How should I recover the deployment?", limit=1)
if used and user_confirmed_helpful:
    memory.reinforce((used[0].id,))
```

`reinforce()` de-duplicates IDs, ignores missing records, and caps each record's confirmation count
at 20. A confirmation later than the query's ranking reference is ignored for that query, which
prevents future feedback from leaking into historical evaluation.

**Guidance:** Reinforce only observed positive use or feedback. Use `search_with_trace()` when one
query needs its temporal, reinforcement, retention, or rejection factors explained.
