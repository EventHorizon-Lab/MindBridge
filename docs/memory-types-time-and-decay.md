# Memory types, time, and decay

This page owns memory-role, event-time, typed-assertion, retrieval-scope, and ranking semantics.
These controls all use the same SQLite record and Zvec projection path; none creates another store
or isolation scope.

| Control | Behavior |
| --- | --- |
| `memory_type` | Hard exact-role filter when supplied |
| `occurred_from` / `occurred_until` | Hard event-overlap filter |
| `RetrievalScope` | Hard valid-time, known-time, symbolic-place, and same-frame metric filters |
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
and optional metric pose or symbolic `place_id` to a source observation:

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

The returned record exposes the persisted `MemoryContext` and keeps a symbolic place separately as
`MemoryRecord.place_id`. A source context has `MemoryKind.OBSERVATION`. An optional former may
propose these derived kinds:

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
| `occurred_at` / `occurred_end` | When the captured episode happened |
| `MemoryContext.valid_from` / `valid_until` | When the assertion is true in the represented world |
| `MemoryContext.recorded_at` / `retired_at` | When MindBridge knew that assertion version |

Both validity columns and both transaction columns are stored on every typed version, so this is
full bitemporal invalidation rather than a single timestamp. A formation backend is not required to
reach them: `add(..., context=ObservationContext(valid_from=..., valid_until=...))` writes the same
validity axis directly, and correcting an assertion later sets `retired_at` on the version it
replaces instead of rewriting it.

`RetrievalScope(valid_at=..., known_at=...)` combines the last two axes. `valid_at` selects an
assertion whose half-open world interval contains that instant; `known_at` selects the transaction
version active then. Supplying either excludes records without the corresponding typed semantic
version, and excludes raw records created after `known_at`. Evidence links carry the same
recorded/retired bounds, so a historical result never exposes support added later. Each evidence
change and its semantic projection share one monotonically allocated transaction instant even when
the device wall clock repeats.

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

`ObservationContext(place_id="kitchen")` stores a trimmed symbolic label when metric localization is
not available. `RetrievalScope(place_id="kitchen")` applies indexed equality in SQLite and excludes
unlabelled records. Symbolic and metric scopes are independent and both must match when combined.

`get` and `list` expose the latest typed context even when it is retired or hidden, while default
search uses only active visible versions. Forgetting is evidence-aware: deleting evidence
recalculates derived confidence and visibility, and removing the last evidence removes the
unsupported derived record. Removing a superseding source, or deleting a derived assertion, rebuilds
current validity segments from the remaining supported assertions. Source records are deleted only
by an explicit caller action.

### Expiring a memory without deleting it

Because default retrieval considers only active validity, an explicit `valid_until` is a soft
forget. Give the observation a validity interval that ends, and after that instant the record stops
appearing in default `search` and `ask` while `get` and `list` still return it and its context:

```python
from datetime import datetime, timedelta, timezone

from mindbridge import ObservationContext

now = datetime.now(timezone.utc)
expiring = memory.add(
    "The guest wifi password is swordfish.",
    context=ObservationContext(valid_from=now, valid_until=now + timedelta(days=1)),
)
```

`valid_until` requires `valid_from`; supplying an end alone raises `ValidationError`, because a
world interval with no beginning cannot be placed on the validity axis. The end is exclusive and
must be later than the start.

This is the right tool when a fact was true and stopped being true — a temporary access code, a
guest who has left, a plan that has been superseded. It is not a privacy control: the content,
assets, embedding, and typed context all remain on disk and remain readable, and a
`RetrievalScope(valid_at=...)` inside the expired window still retrieves the record, which is the
point of keeping it. Use `delete` when the bytes must go.

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
