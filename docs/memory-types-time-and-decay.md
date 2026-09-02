# Memory types, time, and decay

This page owns memory-role, event-time, typed-assertion, retrieval-scope, and ranking semantics.
MindBridge stores semantic, episodic, and procedural roles on the same durable record, and an
optional typed context refines that record into observations, entities, events, state, relations,
affect, traits, and response policies carrying source evidence, confidence, and their own validity
and transaction time. Every control below uses the ordinary SQLite record and Zvec projection path;
none creates another store or isolation scope.

| Control | Behavior |
| --- | --- |
| `memory_type` | Hard exact-role filter when supplied |
| `occurred_from` / `occurred_until` | Hard event-overlap filter |
| `RetrievalScope` | Hard valid-time, known-time, and same-frame spatial filters |
| Temporal phrases in query text | Soft event-time ranking signal relative to `reference_at` |
| Reinforcement and decay | Soft ranking signals; content is never rewritten or deleted |

## Memory types

| Type | Intended content | Kernel behavior |
| --- | --- | --- |
| `MemoryType.SEMANTIC` | Facts and stable application knowledge | Default type |
| `MemoryType.EPISODIC` | Events and observations | Usually paired with event time |
| `MemoryType.PROCEDURAL` | Instructions and reusable routines | Returned as evidence, never executed |

The caller classifies ordinary content. MindBridge does not extract facts, segment episodes, or
promote one type into another unless an explicit `FormationBackend` is configured, and even then
formation proposes a separate typed record rather than rewriting the caller's role. Memory type is
part of stable identity, so otherwise identical semantic, episodic, and procedural records have
different IDs. `search()` and `ask()` accept an optional exact type filter.

Every snippet on this page assumes an open `memory`:

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

Use types to keep cognitive roles distinct, not to enforce access control. Metadata and types are
application data; separate security domains require separate data directories.

## Typed context and formation

`ObservationContext` is caller input. It attaches provenance, confidence, optional world validity,
and optional spatial pose to a source observation without leaving the common path:

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
`MemoryKind.OBSERVATION`. An optional former may propose these derived kinds, each with the memory
type the kernel assigns it:

| Derived kind | Memory type | Meaning |
| --- | --- | --- |
| `ENTITY` | Semantic | A referenced person, object, or concept |
| `EVENT` | Episodic | A formed event assertion |
| `STATE` | Semantic | A value that can change over valid time |
| `RELATION` | Semantic | A typed relation between subjects |
| `AFFECT` | Episodic | A situated affect cue |
| `TRAIT` | Semantic | A longer-horizon characteristic |
| `RESPONSE_POLICY` | Procedural | Feedback-grounded response guidance |

The source commits before any typed proposal is formed, so a formation model error leaves the
observation durable and the formation retryable. The former only proposes; the kernel validates
source modality and spatial binding, assigns IDs, links evidence, versions conflicting state, and
commits derived records. Two kinds carry extra visibility rules:

- `MemoryKind.AFFECT` records a situated cue together with the source modality it came from, so a
  model cannot claim an audio cue for a source that has no audio.
- `MemoryKind.TRAIT` stays hidden from active retrieval until two independent sources support the
  same typed claim, combining their independent confidence with a noisy-OR projection. A trusted
  `EvidenceBasis.USER_STATEMENT` trait is visible immediately.

Keep raw observations even when an interpretation changes. Correct typed state with new evidence,
or remove the incorrect derived record; do not present a derived rewrite as the original
observation.

See [omni streaming and interaction memory](omni-streaming-and-interaction-memory.md) for the
formation, affect, and spatial examples that arrive from capture streams.

## Event time and strict filters

`occurred_at` is an event start; `occurred_end` is an optional exclusive end and must be later than
the start. Both must include a timezone. `created_at` is storage time and is not substituted when a
record has no event time.

`search(..., occurred_from=..., occurred_until=...)` applies a hard half-open overlap filter. Either
bound may be omitted. Any bound excludes records without event time. MindBridge pushes the filter
into Zvec for candidate selection and rechecks it after SQLite hydration because SQLite is
authoritative.

An instant event is treated internally as a one-microsecond interval. A stored interval matches
when it overlaps the requested range; an event ending exactly at `occurred_from`, or starting
exactly at `occurred_until`, does not match.

Use these explicit bounds whenever records outside the interval must be excluded; temporal
language in the query is a ranking signal, not a filter.

## Valid time and transaction time

Raw occurrence and typed assertion time answer different questions:

| Field | Question answered |
| --- | --- |
| `occurred_at` / `occurred_end` | When did the captured episode happen? |
| `MemoryContext.valid_from` / `valid_until` | When was this assertion true in the represented world? |
| `MemoryContext.recorded_at` / `retired_at` | When did MindBridge know this assertion version? |

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

```python
from mindbridge import RetrievalScope

what_we_believed_then = memory.search(
    "What drink did Alex prefer?",
    scope=RetrievalScope(valid_at=event_time, known_at=audit_time),
)
```

Overlapping state assertions share a deterministic lineage keyed by kind, normalized subject,
predicate, and spatial frame/anchor. A later assertion retires the previous transaction version and
splits any unaffected before/after validity segments into carry-forward versions, which is what
supports bounded backfill and A to B to A evolution. Assertions conflict only within one SQLite
write batch; equal wall-clock timestamps in separate transactions are still ordered.

`get` and `list` expose the latest typed context even when it is retired or hidden, while default
search uses only active visible versions. Forgetting is evidence-aware: deleting evidence
recalculates derived confidence and visibility, and removing the last evidence removes the
unsupported derived record. Removing a superseding source, or deleting a derived assertion, rebuilds
current validity segments from the remaining supported assertions. Source records are deleted only
by an explicit caller action.

### Spatial scope

`RetrievalScope` carries the spatial filter on the same value:

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

`near` and `radius_m` must be supplied together, and `radius_m` must be non-negative. Spatial
search compares only matching coordinate frames and anchors, and position uncertainty expands the
conservative intersection test. MindBridge infers no coordinate transform, so a scope in one frame
never matches a memory stored in another, and it excludes every memory stored without a spatial
context.

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

## Temporal phrases

Temporal phrases in query text add a soft event-time ranking range. They do not replace explicit
`occurred_from` and `occurred_until` filters.

The bounded parser recognizes:

- ISO dates and ranges;
- named English month-years, Chinese year-months, and calendar years from 1900 through 2199;
- today, yesterday, tomorrow, the day before yesterday, and the day after tomorrow in English and
  Chinese;
- last, this, next, and rolling weeks; last, this, and next months or years;
- `N days ago` and rolling past or recent `N` days in English and Chinese.

`reference_at` resolves relative phrases in its timezone. Without it, MindBridge uses the current
UTC time. A `Today is <date>` declaration can supply the reference date when no explicit reference
was passed; the declaration is removed before the remaining temporal phrase is parsed. `ask`
includes the resolved reference time in the generation input when relative time is involved.

For a detected range, retrieval considers in-range and global candidates. In-range events receive a
boost; nearby events decay smoothly with distance, and records without event time are downranked.
This is deliberately soft because event boundaries may be noisy. Use explicit bounds when outside
events must be excluded.

## Decay and reinforcement

Decay is disabled by default. Enable search-time decay with
`Memory(..., decay_half_life_days=<positive number>)` or the equivalent `MemoryConfig` setting.
It changes ranking only: no record, asset, embedding, or outbox row is deleted or rewritten.

When enabled, retention uses the most recent eligible explicit reinforcement; otherwise it uses
event end, event start, or last update time. The configured half-life applies exponential decay
with a nonzero floor, and repeated confirmations slow that decay. Public scores remain bounded to
`[0, 1]`. Eligible confirmations also provide a small ranking boost when decay is disabled.

`search()` never reinforces a result. `ask()` does reinforce the evidence it cited, because
answering is itself a usage signal; set `reinforce_on_answer=False` to turn that off, which
measurement needs, since reinforcing mid-run makes one question's retrieval depend on which
earlier questions answered. Record other positive application feedback explicitly:

```python
used = memory.search("How should I recover the deployment?", limit=1)
if used and user_confirmed_helpful:
    memory.reinforce((used[0].id,))
```

`reinforce()` de-duplicates IDs, skips missing records, and caps each record's confirmation count
at 20. Confirmations after the query's ranking reference are ignored so evaluation cannot leak
future feedback into the past.

Reinforce only observed positive use or feedback. Use `search_with_trace()` when one query needs
its temporal, reinforcement, retention, or rejection factors explained; normal telemetry omits
per-memory candidate identifiers.
