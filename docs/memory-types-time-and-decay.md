# Memory types, time, and decay

MindBridge stores semantic, episodic, and procedural roles on the same durable record. Event time,
type filtering, optional temporal parsing, and optional decay all use the ordinary SQLite and Zvec
path; they do not create separate stores or scopes.

| Control | Effect |
| --- | --- |
| `memory_type` | Hard exact-role filter when supplied. |
| `occurred_from` / `occurred_until` | Hard event-overlap filter. |
| Temporal phrases in query text | Soft ranking signal relative to `reference_at`. |
| Decay and reinforcement | Soft ranking signals; they never delete or rewrite content. |

## Memory types

| Type | Intended content | Kernel behavior |
| --- | --- | --- |
| `MemoryType.SEMANTIC` | Facts and stable application knowledge | Default type. |
| `MemoryType.EPISODIC` | An event or observation | Usually paired with event time. |
| `MemoryType.PROCEDURAL` | Instructions or reusable routines | Retrieved as evidence; never executed by MindBridge. |

The caller classifies content. MindBridge does not extract facts, segment episodes, or promote one
type into another. Type is part of stable identity, so otherwise identical semantic, episodic, and
procedural records have different IDs. `search` and `ask` accept an optional exact type filter.

```python
from datetime import datetime, timezone

from mindbridge import JinaOmniEmbedder, Memory, MemoryType

with Memory("./data/agent", embedder=JinaOmniEmbedder()) as memory:
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

Metadata remains application data. A type or metadata value is not an authorization or isolation
boundary.

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

Search never reinforces a result. Record positive application feedback explicitly:

```python
used = memory.search("How should I recover the deployment?", limit=1)
if used and user_confirmed_helpful:
    memory.reinforce((used[0].id,))
```

`reinforce` de-duplicates IDs, updates existing records, and caps confirmation count at 20.
Confirmations after the query's ranking reference are ignored so evaluation cannot leak future
feedback into the past.

Use `search_with_trace()` when one query needs its temporal, reinforcement, and retention factors
explained. Normal telemetry omits per-memory candidate identifiers.
