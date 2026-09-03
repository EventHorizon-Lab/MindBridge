# Context OS round 1: fast capture, memory control plane, context compiler

Date: 2026-09-03. Status: approved for implementation by the team lead under the direction in
[docs/context-os.md](../../context-os.md). This spec turns evolution gates 2, 3, and 4 of that page
into concrete Python contracts. It does not change `add()`, `search()`, or `ask()` semantics.

Amended 2026-09-03: the `ContextBudget.max_chars` default below stayed at 6000 in this record, but
shipped as 16000 -- one video part is priced at 12 000 characters, so no default compilation could
afford media evidence. [docs/context-compilation.md](../../context-compilation.md) is the contract.

## Starting point (verified against source on 2026-09-03)

- `add()` blocks on every stage: asset materialization, speech recognition, transcription,
  embedding, the SQLite commit, the Zvec outbox flush, and formation. Nothing is deferred.
- The only durable work queue is `search_index_queue`, fed by triggers on `embeddings`. A record
  without embeddings is durable but invisible to retrieval and enqueues nothing.
- `formation_runs` is a completion marker keyed by `(source_memory_id, recipe)`; formation runs
  over one committed observation at a time and every proposal cites exactly one source.
- `memory_evidence`, `memory_versions`, and the noisy-OR confidence projection already support
  several independent sources per derived record; only the write path never uses that.
- Forgetting does not exist. `visible = 0` is computed from evidence (hidden inferred traits) and
  `retired_at` is bitemporal supersession; neither is a policy state a host can set.
- `ask()` selects grounding hits by modality round robin plus an optional character budget and
  returns a flat hit list. There is no structured, budgeted context view.

## Shared foundation (team lead, lands first)

Schema version 10. One migration adds:

| Change | Purpose |
| --- | --- |
| `memory_records.forgotten_at TEXT` | Cognitive forgetting state for any record, raw or derived. `NULL` means active. |
| `capture_queue (memory_id PK → memory_records ON DELETE CASCADE, enqueued_at, attempts, last_error)` | Durable deferred-enrichment work for fast capture. |
| `memory_operations (operation_id AUTOINCREMENT, operation_key UNIQUE, intent, trigger, model_id, recipe, operation_json, effects_json, applied_at, rolled_back_at)` | Append-only log of every control-plane operation, for evaluation, replay, and rollback. |

Read-path rule: `LocalStore.read_memories(..., active_only=True)` excludes forgotten rows.
`get()` and `list()` still return them, and `MemoryRecord.forgotten_at` exposes the state for
audit. `SearchHit` never carries a forgotten record.

## A. Fast capture (gate 2)

### Contract

```text
capture(
    content: ContentInput,
    *,
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    context: ObservationContext | None = None,
) -> MemoryRecord

settle(*, limit: int = 100) -> int
pending_captures() -> int
```

`capture()` validates content, materializes media into the asset store, and commits the record,
its assets, its observation context, and one `capture_queue` row in one SQLite transaction. It
calls no model. It returns the same content-addressed `MemoryRecord` that `add()` would return for
the same input, so `capture()` followed by `add()` of the same content is one record. A captured
record is durable, readable through `get()` and `list()`, and not searchable until settled.

`settle()` processes up to `limit` queued records in enqueue order through the enrichment stages
`add()` uses: speech identity, transcription, embedding, the SQLite embedding commit, the outbox
flush, and formation. Settling a record updates `memory_records.content` to the same derived text
`add()` would have stored, inserts embeddings, and deletes the queue row in one transaction. It
returns the number of records settled. A model or storage failure on one record increments
`attempts`, stores `last_error`, leaves the row queued, and raises the mapped error with
`subject=<memory_id>`; earlier records in the batch remain settled.

`add()` and `add_many()` settle a queued record they encounter, so their strong contract holds.
`search()`, `ask()`, `compile()`, `close()`, and opening a store never settle. Time to searchable
is therefore controlled by the host: call `settle()` from an idle loop, a timer, or after each
capture burst. `AsyncMemory` mirrors all three methods with `asyncio.to_thread`.

### Not in scope

No `AsyncCaptureStream` flag, no background thread inside `Memory`, no readiness field on
`MemoryRecord`. A queued record is identified by `pending_captures()` and by the fact that
`search()` does not return it.

### Tests

Captured record is durable and listable but not searchable; `settle()` makes it searchable and
runs formation exactly once; `add()` of captured content settles it; duplicate `capture()` is
idempotent and calls no model; a failing embedder leaves the row queued with `attempts=1` and
`last_error`; a second `Memory` opened on the same directory after a simulated crash still sees
the queue row; async parity; CLI commands derive from `OPERATIONS`.

## B. Agentic memory control plane (gate 3)

### Vocabulary

```python
class MemoryIntent(str, Enum):
    REINFORCE = "reinforce"  # attach independent evidence to a derived memory
    CONSOLIDATE = "consolidate"  # derive a typed memory from several sources
    CORRECT = "correct"  # retire a bad derived inference
    FORGET = "forget"  # cognitive forgetting under policy


class MemoryTrigger(str, Enum):
    MANUAL = "manual"
    EVIDENCE = "evidence"
    FEEDBACK = "feedback"
    CONTRADICTION = "contradiction"
    QUERY_FAILURE = "query_failure"
    PRESSURE = "pressure"
    IDLE = "idle"


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryOperation:
    intent: MemoryIntent
    evidence_ids: tuple[str, ...] = ()  # REINFORCE, CONSOLIDATE: cited sources
    target_ids: tuple[str, ...] = ()  # REINFORCE (one), CORRECT, FORGET
    proposal: FormationProposal | None = None  # CONSOLIDATE only
    rationale: str | None = None  # logged, never interpreted
```

`MemoryOperation.__post_init__` enforces the per-intent shape. `MemoryOperationRecord` is the
logged form: `operation_id`, `operation`, `trigger`, `model_id`, `recipe`, `effects`
(`created_ids`, `changed_ids`), `applied_at`, `rolled_back_at`. `ConsolidationReport` returns
`operations: tuple[MemoryOperationRecord, ...]` and `rejected: tuple[tuple[MemoryOperation, str],
...]`.

### Backend protocol

```python
@runtime_checkable
class ConsolidationBackend(Protocol):
    @property
    def consolidation_model(self) -> str: ...
    @property
    def consolidation_recipe(self) -> str: ...
    def consolidate(
        self, evidence: Sequence[MemoryRecord], *, trigger: MemoryTrigger
    ) -> tuple[MemoryOperation, ...]: ...
    def close(self) -> None: ...
```

`MemoryPlugins.consolidator` and `Memory(consolidator=...)` inject it. `OpenAIModels` implements
it with a structured-output prompt next to the existing formation prompt. The backend only
proposes; it receives the bounded evidence set and may cite only ids from that set.

### Kernel surface

```text
consolidate(
    *,
    evidence_ids: Sequence[str] | None = None,
    query: ContentInput | None = None,
    limit: int = 32,
    trigger: MemoryTrigger = MemoryTrigger.MANUAL,
) -> ConsolidationReport

forget(memory_ids: Sequence[str]) -> MemoryOperationRecord | None
rollback(operation_id: int) -> bool
operations(*, limit: int = 100) -> tuple[MemoryOperationRecord, ...]
```

Evidence gathering: explicit `evidence_ids`, else the active search result for `query`, else the
`limit` newest active records. Forgotten and hidden records are never shown to the backend.

Kernel validation and effects per intent:

| Intent | Kernel checks | Effect | Rollback |
| --- | --- | --- | --- |
| REINFORCE | One derived target; every evidence id in the shown set, existing, not the target, not already linked. | `add_memory_evidence` per pair; confidence and visibility recompute. | Retire those evidence rows and refresh the projection. |
| CONSOLIDATE | Proposal valid; at least one evidence id, all in the shown set; AFFECT cue modality present in some source; spatial frame and anchor match some source. | Derived record with `evidence_ids` = all sources, `model_id`, `recipe`; embeds and commits through the formation path without a `formation_runs` row. | Delete the created derived records; lineage replays as it does today. |
| CORRECT | Targets exist and are derived (`kind != OBSERVATION`). | Retire current versions at transaction time. | Carry a new version with the same interval. |
| FORGET | Targets exist and are not already forgotten. | Set `forgotten_at`. | Clear `forgotten_at` on the ids the operation changed. |

Every operation has `operation_key = sha256(canonical operation JSON + recipe)`. A key already
applied and not rolled back is rejected as a duplicate. Each accepted operation commits in its own
transaction together with its log row. Physical deletion is not an intent; it stays on `delete()`
under host authority. None of these methods is exposed on REST or MCP in this round.

### Tests

Apply and roll back each intent; rejections for evidence outside the shown set, unknown target,
observation kind, duplicate key; forgotten records leave search but stay in `get()`, `list()`, and
`MemoryRecord.forgotten_at`; two independent sources consolidate into one derived record with
noisy-OR confidence; REINFORCE from a second independent source makes a hidden inferred trait
visible; `operations()` lists newest first; async parity; REST and MCP serialize `forgotten_at`.

## C. Context compiler and capability advertisement (gate 4)

### Contract

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ContextBudget:
    max_chars: int = 6000
    max_items: int = 24
    memory_types: frozenset[MemoryType] | None = None
    min_confidence: float = 0.0
    freshness: timedelta | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextConflict:
    lineage_id: str
    subject: str | None
    predicate: str | None
    values: tuple[str, ...]
    memory_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextBundle:
    goal: str
    reference_at: datetime
    actors: tuple[SearchHit, ...]  # ENTITY
    episodes: tuple[SearchHit, ...]  # EPISODIC except AFFECT
    facts: tuple[SearchHit, ...]  # SEMANTIC except ENTITY and TRAIT
    procedures: tuple[SearchHit, ...]  # PROCEDURAL
    affect: tuple[SearchHit, ...]  # AFFECT
    traits: tuple[SearchHit, ...]  # TRAIT
    conflicts: tuple[ContextConflict, ...]
    occurred_from: datetime | None
    occurred_until: datetime | None
    frames: tuple[str, ...]
    omitted: int
    chars: int

    @property
    def hits(self) -> tuple[SearchHit, ...]: ...  # every included hit in rank order
    def render(self) -> str: ...  # deterministic sectioned text with [id] markers
```

```text
compile(
    goal: ContentInput,
    *,
    budget: ContextBudget | None = None,
    reference_at: datetime | None = None,
    scope: RetrievalScope | None = None,
) -> ContextBundle

capabilities() -> MemoryCapabilities
```

`compile()` runs the existing retrieval kernel once with a candidate limit of
`max(100, 3 * max_items)`, filters by `memory_types`, `min_confidence` (a record without typed
context counts as confidence 1.0), and `freshness` measured against `reference_at` using event end,
event start, or creation time, then partitions by memory type and kind. Selection guarantees one
slot per non-empty section in rank order, then fills remaining `max_items` and `max_chars` by
score using the same cost function as `ask()`. Conflicts are hits that share a `lineage_id`, have
kind STATE, RELATION, or TRAIT, and disagree on `value`; the compiler reports them and does not
resolve them. `render()` emits one heading per non-empty section, one line per hit prefixed with
its id and confidence, a conflicts section, and an omitted count. `ask()` is unchanged.

`MemoryCapabilities` is a frozen dataclass: `modalities`, `answer`, `transcribe`, `faces`,
`describe_vision`, `form`, `consolidate`, `decay`. It lets an agent surface advertise what the
instance can do instead of failing on first use.

### Tests

Partition by type and kind; every non-empty section receives one slot before any section receives
two; `max_chars` and `max_items` are never exceeded and `omitted` counts the rest; `freshness`
and `min_confidence` filter; conflicting STATE values in one lineage produce one
`ContextConflict`; `render()` is deterministic and names every included id; forgotten and hidden
records never appear; async parity; CLI derives `compile`.

## Team plan

| Member | Branch | Owns | Reads |
| --- | --- | --- | --- |
| A fast-capture | from foundation commit | `capture`, `settle`, `pending_captures`, `capture_queue` store methods, docs in architecture and omni-streaming pages | this spec, memory.py add path |
| B control-plane | from foundation commit | types above, `ConsolidationBackend`, `OpenAIModels` adapter, `consolidate`, `forget`, `rollback`, `operations`, store methods, docs in memory-types page | this spec, formation path, store bitemporal helpers |
| C compiler | from foundation commit | `src/mindbridge/context.py`, `compile`, `capabilities`, new `docs/context-compilation.md`, docs index | this spec, `_search_prepared`, `_grounding_hits` |

Merge order: A, C, B. After each merge the lead runs the full gate set and scans for duplicated
symbols. REST and MCP exposure of `compile()` is a follow-up task after B lands.

## Rules every member follows

- Keep `Memory` the execution plane. New logic modules are allowed for pure selection and
  validation code; storage writes go through `LocalStore`.
- SQLite commits before Zvec; acknowledge outbox rows only after flush. Never write SQLite from a
  backend.
- Model output is a proposal. The kernel validates every field before persistence.
- Update the owning documentation page and tests in the same commit as the contract.
- Run `ruff format`, `ruff check`, `mypy`, and `pytest -W error` before reporting done.
