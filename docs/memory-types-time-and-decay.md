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
| `RetrievalScope` | Hard valid-time, known-time, symbolic-place, and same-frame metric filters |
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

## Event time and strict filters

`occurred_at` is an event start. `occurred_end` is an optional exclusive end; it requires a start
and must be later. Both values must be timezone-aware. `created_at` is storage time and is not used
as a substitute when event time is absent.

`search(..., occurred_from=..., occurred_until=...)` applies a hard half-open overlap filter. Either
bound may be omitted. Supplying any bound excludes records without event time. An interval ending
at `occurred_from`, or starting at `occurred_until`, does not match; an instant matches when its
timestamp falls inside the requested range, and is treated internally as a one-microsecond
interval.

**Contract:** MindBridge pushes the filter into Zvec for candidate selection and rechecks it after
SQLite hydration, because SQLite is authoritative. Use these explicit bounds whenever records
outside the interval must be excluded.

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
appends the resolved reference to the generation input of every question the answerer reads as
text, whether or not the parser recognized a phrase in it: "how long ago did grandpa visit" narrows
no retrieval window yet still needs the answering clock. It is appended after routing, so a spoken
question carries it once transcription has given it text; a question routed with no text at all is
handed over unchanged.

For a detected range, retrieval considers both in-range and global candidates. In-range events are
boosted; events outside the range and records without event time keep the score their relevance
earned and are neither downranked nor gated out. The signal is deliberately additive, because event
boundaries may be noisy and a relevant memory that happened at the wrong time is still the answer
more often than a well-timed irrelevant one.

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

The source commits before any typed proposal is formed, so a formation model error leaves the
observation durable and the formation retryable. The former only proposes; the kernel validates
source modality and spatial binding, assigns IDs, links evidence, versions conflicting state, and
commits derived records. Two kinds carry extra visibility rules:

- `MemoryKind.AFFECT` records a situated cue together with the source modality it came from, so a
  model cannot claim an audio cue for a source that has no audio.
- `MemoryKind.TRAIT` stays hidden from active retrieval until two independent sources support the
  same typed claim, combining their independent confidence with a noisy-OR projection. A trusted
  `EvidenceBasis.USER_STATEMENT` trait is visible immediately.

A proposal that fails one of those per-proposal rules — an `AFFECT` cue naming a modality the
source never carried, a pose in another coordinate frame — is dropped, counted on the
`mindbridge.formation.refused_proposals` span attribute, and costs only itself: the observation's
remaining proposals commit and the write succeeds. One badly grounded opinion must not fail a write
whose source is already durable, because every retry would re-run the model and fail the same way.
Two proposals from one source that contradict each other — one record proposed twice with different
content, or two overlapping states in one lineage — are refused the same way, and the first of the
pair stands. Only damage to the batch envelope — a backend returning the wrong number of results,
or a shape that is not a batch of proposals — fails the write. A refusal is final for that
formation recipe: the source is marked formed, so nothing retries the proposal and re-adding the
same content forms nothing new.

Formation never rewrites the caller's source record. A derived record inherits its source's event
time, valid interval, and metric pose, and inherits the symbolic `place_id` and the `metadata` that
every cited source agrees on, so a place-scoped or metadata-filtered recall reaches the knowledge
formed in a room and not only the raw observation it came from. Agreement is the rule for formation
too, because an `ENTITY`, `RELATION`, inferred `TRAIT`, or `RESPONSE_POLICY` is written once and
later sources only add evidence to it: when a second source disagrees, the shared record keeps
neither place nor metadata rather than the first source's, since a hard retrieval filter must not
be guessed. Both columns follow the live evidence rather than the moment the record was written:
deleting a source or rolling an operation back recomputes them over the sources that remain, so a
record the survivors agree on is scoped again. A derived record carries no media assets of its own:
it is text, and its evidence link points at the observation that holds the media.

### Naming a person is a typed assertion

`register_speaker` and `register_identity` write an `ENTITY` assertion whose
`MemoryContext.identity_id` binds it to the recognized person, whose `subject` is the name, and
whose `value` is the recorded relationship. Its basis is `USER_STATEMENT`, so it is visible at
once, and it needs no former: naming rests on the host's authority, not on a model. It inherits
neither place nor metadata from the evidence an agent cites for it, whoever asserts it: who
somebody is does not stop being true in another room.

Every naming assertion about one identity shares a lineage keyed on that identity rather than on
the spelling of the name, so renaming the person supersedes the previous assertion, exactly as a
new `STATE` supersedes the one it overlaps. The retracted name stops reaching active retrieval;
the retired version stays in the log.

`identities.name` and `identities.relationship` are a projection of the current visible naming
assertion, recomputed the way evidence recomputes confidence and visibility. Every recompute
rewrites the indexed transcript text in the same commit, so what search matches and what
`identity()` reports are always the same assertion.

Naming is logged like any other control-plane operation with `MemoryTrigger.MANUAL`, so
`operations()` shows it and `rollback(operation_id)` retracts the assertion, restores the one it
superseded, and repaints both the projection and the index. Registering a name that already
stands changes nothing and logs nothing.

The consequence is deliberate and visible: naming a person creates a searchable memory record, and
it appears in `list()` and `search()` results. That is what makes a name retrievable knowledge
rather than a label on a registry row.

**Guidance:** Keep raw observations even when an interpretation changes. Correct typed state with
new evidence or remove the incorrect derived record; do not present a derived rewrite as the
original observation.

## Valid time and transaction time

Raw occurrence and typed assertion time answer different questions:

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
unlabelled records. Derived records inherit the label their evidence agrees on, so the scope
reaches formed entities, states, and relations as well as observations. Symbolic and metric scopes
are independent and both must match when combined.

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

## Memory management loop

Formation reads one committed observation. The memory management loop reads a bounded set of
records that already exist and proposes what should change about them. It is one loop with a
stable vocabulary, not a swarm of specialized agents, and it needs a `consolidator`:

```python
report = memory.consolidate(query="what do we know about Ana?", trigger=MemoryTrigger.EVIDENCE)
for record in report.operations:
    print(record.operation_id, record.operation.intent, record.changed_ids)
for operation, reason in report.rejected:
    print(operation.intent, "refused:", reason)
```

The evidence set comes from explicit `evidence_ids`, else the active search result for `query`,
else the newest `limit` active records. Forgotten and hidden records never reach the backend, and
the backend may cite only IDs it was shown. Its `target_ids` are bounded too: they must fall
inside the window the kernel gathered, which is the shown set plus whatever the host named in
`evidence_ids`. That extra allowance is how a hidden derived record — the reason `REINFORCE` and
`CORRECT` exist — stays reachable, without letting a backend act on records nobody put in front
of it.

Named targets and cited evidence are what the window bounds. Lineage supersession is not a
backend choice at all: a new `STATE` or user-stated `TRAIT` retires the current version of every
other record in its lineage whose validity it overlaps, by the kernel's own deterministic rule,
including records the backend never saw. Those versions are recorded on the log row as
`MemoryOperationRecord.superseded` and `rollback()` restores exactly them. Because a later
consolidation can supersede an earlier one's record, operations on one lineage reverse newest
first: `rollback()` returns `False` for an operation a standing later one has built on.

| Intent | Kernel semantics |
| --- | --- |
| `REINFORCE` | Link an independent source to an existing derived record. Confidence recombines by noisy-OR over independent sources, and a hidden inferred `TRAIT` can become visible. |
| `CONSOLIDATE` | Derive one new record citing several sources. The sources stay as evidence, and stay in recall unless the same proposal names them in `target_ids`. |
| `CORRECT` | Retire the current version of a bad derived inference. History is preserved, not overwritten. |
| `FORGET` | Set `forgotten_at`. Recall skips the record; audit keeps it. |
| `IDENTIFY` | Name a recognized person. The kernel turns the `IdentityClaim` into an `ENTITY` assertion bound to that identity, and `identities.name` is a projection of the assertion currently visible. |

`IDENTIFY` is checked hardest, because a name is the one claim that changes what an agent will
say out loud about somebody: the identity must exist and at least one cited memory must actually
contain that person through a speech or face observation, so a name cannot be pinned on someone
from evidence that never contained them. A proposed name also carries basis `MODEL_INFERENCE`,
so it stays hidden until two independent evidence groups support it, and `identities.name` keeps
projecting whatever is visible instead. `register_identity` names somebody on the host's
authority and is visible at once. `REINFORCE`, `CORRECT`, and `FORGET` refuse a naming assertion
outright; the way to retract a name is `rollback()` of the `IDENTIFY` that asserted it, which
recomputes both the projected name and the indexed speech text.

The backend proposes and never writes. Each proposal is validated against the shown evidence set
and its intent's rules, then committed in its own transaction together with an append-only log
row. A pass is therefore not atomic, by design: a proposal refused after an earlier one committed
does not undo it, and `ConsolidationReport` names exactly which applied and which were rejected.
A refused proposal is reported with a reason instead of raising, so one bad proposal does not
discard the pass. Every operation is identified by `sha256(canonical operation JSON + recipe)`, so
re-proposing the same operation is rejected as `"duplicate"` rather than applied twice.
`rollback(operation_id)` reverses one operation and `operations()` lists the log newest first.
Deletion is not an intent.

A proposal is all or nothing within itself. Every target it names must be eligible or the whole
proposal is refused, so an applied row's `target_ids` are the IDs it actually acted on. The apply
transaction re-checks what validation read: a target or cited source that moved in between makes
the proposal `"stale"` with nothing written.

Two things share the name "reinforce" and are not the same mechanism. `reinforce()` is the
*ranking-utility* signal: it bumps a bounded confirmation count, changes retrieval order only,
writes no log row, and cannot be rolled back. `MemoryIntent.REINFORCE` is the *evidence-linkage*
signal: it attaches an independent source to a derived record, is logged, and is reversible. Only
`FORGET` has a host entry point for its intent, because `forget()` is a policy decision a host
makes without a model; the other three intents exist only as backend proposals.

"Update" is not a separate intent. It is what a `CONSOLIDATE` into an existing lineage does: the
new version supersedes the prior one at transaction time and history is kept. That reconciliation
runs for `STATE` and for `USER_STATEMENT` `TRAIT`. Any other kind gains a second record in the
same lineage rather than superseding, which is why a model-inferred `TRAIT` can end up
contradicting itself — and why the loop is given a way to see that. Proposing a replacement value
for a wrong inference is a `CORRECT` and a `CONSOLIDATE` in the same batch, which the
consumed-evidence rule allows: correcting a derived record retires that record, not the sources
the replacement is built from.

### What needs deliberation

`consolidation_candidates()` derives due work from committed state rather than from a clock, so
the loop has a durable trigger instead of a timer, and `deliberate()` is that loop:

```python
report = memory.deliberate(idle=is_charging_and_idle)
```

One round asks what is due, consolidates each row under the row's own trigger, and repeats
because applying operations can make further work due. It ends when nothing is due, which the
report distinguishes from hitting `max_rounds`. The two primitives stay available for a host that
wants to schedule the halves itself.

`EVIDENCE` rows are derived records that gained independent evidence nothing has weighed — what
the formation path leaves behind. `CONTRADICTION` rows are lineages whose current visible claims
disagree. `FEEDBACK` rows are records confirmed through `reinforce()`, or cited by an `ask()`
answer under the default `reinforce_on_answer`. `QUERY_FAILURE` rows are near-equal recalls that
came back empty at least twice inside the configured window, named against the nearest records
the store does hold. `PRESSURE` rows appear only over a declared `memory_budget_records`, because
growth alone is not evidence that forgetting is useful. `IDLE` rows appear only when the operator
passes `idle=True`, because whether the device is idle or charging is the host's knowledge.

A periodic timer alone is not evidence that the work is useful — and neither is a candidate that
always comes back, so each pass records that it weighed its evidence set whatever it yielded, and
a candidate is not derived again until its own signal moves. See the
[Python SDK reference](api/python-sdk.md#memory-management-operations) for signatures, effects,
outcome recording, and rollback behavior.

## Decay and reinforcement

MindBridge separates five forms of forgetting and never conflates them:

| Form | Call | Effect |
| --- | --- | --- |
| Expiring validity | `valid_until` | Leaves default retrieval when the interval ends; a `valid_at` scope inside the window still retrieves it. |
| Ranking decay | `decay_half_life_days` | Downranks stale records at query time. Nothing is removed or rewritten. |
| Cognitive forgetting | `forget()` | Excludes a record from recall while `get()`, `list()`, and `MemoryRecord.forgotten_at` retain it. Reversible through `rollback()`. |
| Consolidation forgetting | `CONSOLIDATE` naming `target_ids` | Retires the detail a new derived record replaces, in that record's own transaction and under its lineage. Logged on the `CONSOLIDATE` row as `forgotten_ids`, and reversed with it. The lineage versions the same write superseded are logged beside them as `superseded`. |
| Physical deletion | `delete()` | Removes the record and any media no other record references. Not recoverable, and never something a model proposes. |

`forget()` is cognitive only. It is the host entry point for the `FORGET` intent, so it takes the
same log row and the same rollback path as a proposed operation — and, like one, applies all of
its IDs or none. An unknown ID raises `MemoryNotFoundError` the way `get()` and `delete()` do; an
empty sequence, or a set containing an already-forgotten record, returns `None` having changed
nothing. The host names the IDs and is the authority, so no evidence window bounds it. Unlike an
expiring validity interval, it is a policy state a host sets rather than a property of the world.

Consolidation forgetting sets the same `forgotten_at` column, reached a different way: the memory
loop proposes it as part of the consolidation that justifies it, so the kernel can keep the
evidence links and reverse both halves together. Telling the three apart in the log needs no
convention — `delete()` leaves no row, cognitive forgetting is a `FORGET` row, and consolidation
forgetting is a `CONSOLIDATE` row with `forgotten_ids`.

Decay is disabled by default. Enable it with a positive `decay_half_life_days` setting. Decay
changes query-time ranking only; it does not delete or rewrite records, assets, embeddings, or
outbox rows.

When enabled, retention uses the most recent eligible explicit reinforcement; otherwise it uses
event end, event start, or last update time. The configured half-life applies exponential decay
with a nonzero floor, and repeated confirmations slow that decay. Public scores remain bounded to
`[0, 1]`. Eligible confirmations also provide a small ranking boost when decay is disabled.

Search never reinforces a hit. `ask()` does reinforce the hits its answerer actually cited,
because a citation is observed utility rather than retrieval: something read the evidence and
used it. That is a default, not a rule — set `reinforce_on_answer=False` when one question must
not change later rankings, which is what an evaluation needs. Record positive application
feedback explicitly:

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
