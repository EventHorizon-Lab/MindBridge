# Python SDK

## Surface

The supported root-import SDK is re-exported from `mindbridge`. `Memory` is the synchronous
execution boundary, and `AsyncMemory` is its async facade. REST, MCP, and the product CLI expose
subsets of these operations through their own adapters.

There are no tenant, user, request, or benchmark scope parameters. Physical `data_dir` separation
is the isolation boundary. See [architecture](../architecture.md) for storage and consistency and
[configuration](../configuration.md) for backend composition.

## Quick start

Pass already-constructed backends directly:

```python
from mindbridge import JinaOmniEmbedder, Memory

with Memory("./data/assistant", embedder=JinaOmniEmbedder()) as memory:
    record = memory.add("The spare key is in the blue toolbox.")
    hits = memory.search("Where is the spare key?")
```

Or let the bundled declarative configuration construct its adapters:

```python
from mindbridge import Memory

with Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {"provider": "jina-omni"},
    }
) as memory:
    memory.add("Remember this")
```

## Lifecycle and ownership

One physical `data_dir` may have one live `Memory` owner. A second owner fails immediately with
`StorageError(reason="data_dir_in_use")`; use another directory or send work to the existing owner
through REST. A `Memory` is also bound to the process that opened it and must not be used after
`fork`.

`Memory` owns and closes every backend passed to it. A backend can still leave a caller-supplied
provider client open when that adapter documents separate ownership. Use `Memory` as a context
manager, or call `close()`; repeated closes are harmless. See [operations](../operations.md) for
backup, recovery, and telemetry.

## Contract

### Construction

```text
Memory(
    data_dir: str | Path = ".mindbridge",
    *,
    embedder: EmbeddingBackend,
    answerer: GenerationBackend | None = None,
    transcriber: SpeechBackend | TranscriptionBackend | None = None,
    vision_describer: VisionDescriptionBackend | None = None,
    face_analyzer: FaceBackend | None = None,
    former: FormationBackend | None = None,
    consolidator: ConsolidationBackend | None = None,
    index_speech: bool = True,
    index_quantization: IndexQuantization = IndexQuantization.NONE,
    minimum_relevance: float = 0.10,
    ambiguity_margin: float = 0.01,
    evidence_budget_chars: int | None = None,
    decay_half_life_days: float | None = None,
    reinforce_on_answer: bool = True,
    speaker_similarity: float = 0.78,
    speaker_margin: float = 0.05,
    face_similarity: float = 0.363,
    face_margin: float = 0.05,
    identity_link_min_assets: int = 2,
    tracer: opentelemetry.trace.Tracer | None = None,
) -> None
```

`embedder` is required. With a speech-capable `SpeechBackend`, the default `index_speech=True`
stores transcripts and resolved speaker names with new memories; setting it to `False` defers that
analysis until `speech()` is called. `index_quantization` changes only the rebuildable vector
index; supported values are `none`, `fp16`, `int8`, and `rabitq`. The similarity, margin,
relevance, and decay settings are validated when the instance opens. Their behavior and the
supported provider configuration fields live in [configuration](../configuration.md).

A plain `TranscriptionBackend` transcribes supported audio/video during `add` regardless of
`index_speech`; `SpeechBackend` analysis and identity resolution stay behind the explicit flag.

`vision_describer` has no declarative provider and is reachable only through direct construction
or `MemoryPlugins`. `former` proposes typed memories after a source observation commits; omitting
it keeps ordinary add behavior and makes no formation model call. The bundled OpenAI former is
selected by the declarative `formation` slot, which stays off unless it is configured; see
[configuration](../configuration.md#automatic-memory-formation). `consolidator` proposes the
control-plane operations `consolidate()` applies; the bundled OpenAI backend is selected by the
declarative `consolidation` slot, which stays off unless it is configured, and omitting it leaves
`consolidate()` unavailable and every other operation unchanged. See
[configuration](../configuration.md#agentic-memory-management).

Use `index_speech=True` when transcripts and resolved speaker names should become retrieval text.
`evidence_budget_chars=None` grounds `ask()` on exactly `limit` hits. A positive integer may admit
more ranked evidence while its text-equivalent cost fits the budget: 2,000 characters per image,
4,000 per audio asset, and 12,000 per video asset.

`reinforce_on_answer=True` records positive feedback for the hits an answerer actually cites.
Set it to `False` for evaluations that require one question not to change later rankings.

The two other construction boundaries are:

```text
Memory.from_plugins(
    data_dir: str | Path = ".mindbridge",
    *,
    plugins: MemoryPlugins,
    config: MemoryConfig | None = None,
    tracer: Tracer | None = None,
) -> Memory

Memory.from_config(
    config: MindBridgeConfig | Mapping[str, object],
    *,
    tracer: Tracer | None = None,
) -> Memory

resolve_memory_config(
    value: MindBridgeConfig | Mapping[str, object],
) -> MemoryComposition
```

`MemoryPlugins` contains `embedder` plus optional `answerer`, `transcriber`, `vision_describer`,
`face_analyzer`, `former`, and `consolidator`. `MemoryConfig` contains the value-only constructor
settings; `MemorySettings` is its public alias. `MemoryComposition` contains `data_dir`, `plugins`, and
`settings`; call `close()` unless its plugins have been transferred to a `Memory`.

### Content contract

```python
ContentAtom = str | pathlib.Path | Blob | AssetRef
ContentInput = ContentAtom | Sequence[ContentAtom]
```

- `str` is always text, even when it resembles a path or URL.
- `Path` is a local regular media file. Its suffix supplies the media type.
- `Blob(data: bytes, media_type: str, name: str | None = None)` is non-empty inline image, video,
  or audio data.
- `AssetRef(id, ...)` refers to media already stored in the same `data_dir`; `id` is its 64-character
  lowercase SHA-256 identifier. A returned reference includes `modality`, `media_type`,
  `size_bytes`, `sha256`, `name`, and its local `path`.
- An ordered sequence combines text and media into one record. MindBridge does not fetch URLs.

Event timestamps must be timezone-aware. `occurred_end` requires `occurred_at` and must be later
than it. Metadata is a JSON-compatible mapping with non-empty string keys. `MemoryType` is
`semantic`, `episodic`, or `procedural`; see
[memory types, time, and decay](../memory-types-time-and-decay.md).

The stable memory ID covers ordered canonical content, media digests, metadata, event start/end,
memory type, and optional observation context. Repeating the same canonical input returns the same
record without another model call.

`StreamInput` adds per-item values to a stream:

```text
StreamInput(
    content: ContentInput,
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    context: ObservationContext | None = None,
    transcript: str | None = None,
    description: str | None = None,
)
```

### Memory operations

```text
add(
    content: ContentInput,
    *,
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    context: ObservationContext | None = None,
) -> MemoryRecord

add_many(
    contents: Sequence[ContentInput],
    *,
    occurred_at: Sequence[datetime | None] | None = None,
    occurred_end: Sequence[datetime | None] | None = None,
    metadata: Sequence[Mapping[str, object] | None] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    context: Sequence[ObservationContext | None] | None = None,
) -> tuple[MemoryRecord, ...]

add_stream(
    contents: Iterable[ContentInput | StreamInput],
    *,
    capture: bool = False,
) -> Iterator[MemoryRecord]
```

`add` is content-addressed and idempotent. `add_many` uses one model batch and one SQLite
transaction; each optional per-item sequence must have the same length as `contents`. An empty
batch returns `()`. `add_stream` requests one item at a time and makes each yielded record durable
and searchable before requesting the next. If a later item fails, earlier records remain and the
error `subject` identifies `contents[N]`. With `capture=True`, `add_stream` commits each item
through `capture` instead of `add`: every yielded record is durable and readable but has no
vectors, and the host owes the matching `settle` before any of them is searchable.

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

settle(
    *,
    limit: int = 100,
    max_attempts: int = 3,
    memory_ids: Sequence[str] | None = None,
) -> int
pending_captures(
    *,
    limit: int = 100,
    memory_ids: Sequence[str] | None = None,
) -> tuple[PendingCapture, ...]
```

`capture` returns after the SQLite commit and before any model call. It returns the same
content-addressed record `add` returns for the same input, so capturing and then adding the same
content is one memory. A captured record is durable and readable through `get` and `list`
immediately, and invisible to `search`, `ask`, and `compile` until it is settled. It applies the
same embedder-capability check `add` applies, so content `add` would reject raises here instead of
becoming a durable record no `settle` could ever finish. Settling appends the text the models
derive to `content` and never rewrites what the caller supplied; see
[public values](#public-values) for how derived sections are marked.

`settle` runs the deferred stages — speech identity, transcription, embedding, the SQLite embedding
commit, the index flush, and formation — over up to `limit` captured records in enqueue order, and
returns how many it settled. Every record it read is attempted: a model or storage failure on one
leaves it queued with its attempt count and reason while the rest still settle, and the first
failure is raised once the batch is done with `subject` set to that memory ID. A record whose
attempt count has reached `max_attempts` is skipped rather than retried, so one record that can
never settle does not block the queue; it stays queued and visible, and raising `max_attempts`
retries it. `add` and `add_many` settle a queued record they encounter whatever its attempt count,
so their searchable-on-return contract holds. `search`, `ask`, `compile`, `close`, and opening a
store never settle: time to searchable is the host's choice, so call `settle` from an idle loop, a
timer, or after a capture burst.

Pass `memory_ids` to settle only those records. Naming a record is the host asking for it by hand,
so `max_attempts` is ignored for that call: that is how a record parked at the ceiling is retried,
how one is quarantined by never naming it, and how one is settled ahead of the queue. IDs that are
not queued are skipped. One settlement runs at a time per `Memory`; a concurrent `settle` waits
rather than running the same models twice, and then usually finds the queue already drained.

`pending_captures` returns up to `limit` records whose deferred work is not finished, oldest first,
as `PendingCapture` values (`memory_id`, `enqueued_at`, `attempts`, `last_error`, `awaiting`).
`awaiting` is the stage the record is stopped at: `"enrichment"` has no vectors and cannot be
returned by `search`, while `"formation"` is already embedded, indexed, and searchable and owes
only the formation `add` holds a row for between its commit and its model call. Pass `memory_ids`
to ask whether specific records are still waiting: one that is absent from the result is not
pending, which means it is settled or was never stored, and `get` tells the two apart.

```text
search(
    query: ContentInput,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    occurred_from: datetime | None = None,
    occurred_until: datetime | None = None,
    scope: RetrievalScope | None = None,
) -> tuple[SearchHit, ...]

search_with_trace(
    query: ContentInput,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    occurred_from: datetime | None = None,
    occurred_until: datetime | None = None,
    scope: RetrievalScope | None = None,
) -> TracedSearchResult

ask(
    question: ContentInput,
    *,
    limit: int = 5,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    scope: RetrievalScope | None = None,
) -> AnswerResult
```

`reference_at` is the timezone-aware clock for relative time and decay. The current UTC time is
used when it is omitted. `occurred_from` and `occurred_until` are optional timezone-aware,
half-open event-overlap filters; either may be omitted, and two bounds require
`occurred_until > occurred_from`. Records without `occurred_at` do not match a bounded search.

`scope` adds the bitemporal and spatial filters. `valid_at` selects world validity, `known_at`
selects the transaction version known then, and `near` with a non-negative `radius_m` restricts
results to one matching coordinate frame and anchor. SQLite reapplies every scope filter after
candidate retrieval; see
[valid time and transaction time](../memory-types-time-and-decay.md#valid-time-and-transaction-time).

`search_with_trace(...).hits` equals the corresponding `search(...)` result. Its bounded trace
contains identifiers, score components, ranks, and rejection reasons, but no query, content,
metadata, media, vectors, paths, or model output. `ask` requires an answerer and returns only the
retrieved hits the answerer actually used.

```text
compile(
    goal: ContentInput,
    *,
    budget: ContextBudget | None = None,
    reference_at: datetime | None = None,
    scope: RetrievalScope | None = None,
) -> ContextBundle
```

`compile` runs the same retrieval path once with a candidate limit of
`max(100, 3 * budget.max_items)`, then partitions, filters, and budgets the result into a
`ContextBundle`. It calls no generation model, stores no memory, and never resolves a conflict;
like `search`, and through the same helper, it may cache a transcript for spoken query media.
`budget.max_latency_ms` is a deadline checked between stages, never a cancellation, and the bundle
reports `elapsed_ms`, `deadline_exceeded`, and the `unknowns` the request implied but the bundle
does not carry. `ask` is unchanged. [Context compilation](../context-compilation.md) owns the
contract.

```text
capabilities -> MemoryCapabilities  # property
get(memory_id: str) -> MemoryRecord
speech(memory_id: str) -> tuple[SpeakerSegment, ...]
faces(memory_id: str) -> tuple[FaceObservation, ...]
register_speaker(speaker_id: str, name: str, *, relationship: str | None = None) -> None
register_identity(identity_id: str, name: str, *, relationship: str | None = None) -> None
identity(identity_id: str) -> IdentityProfile | None
forget_identity(identity_id: str) -> IdentityErasure
unlink_identity(alias_id: str) -> str | None
reinforce(memory_ids: Sequence[str]) -> int
list(*, limit: int = 100, cursor: str | None = None) -> Page
delete(memory_id: str) -> bool
reindex() -> int
optimize() -> None
close() -> None
```

`get` raises `MemoryNotFoundError` for an unknown ID. `speech` and `faces` return `()` when the
record has no relevant media and require their respective configured backend otherwise.
Registration assigns or replaces a printable name, and optionally a relationship, for an
existing local identity; both raise `IdentityNotFoundError` for an unknown ID. There is no
enrollment call that creates a person from a photograph or a voice sample. Register an identity
the recognizers have already observed instead: add the media, read `faces()` or `speech()` for its
`identity_id`, then name it. Omitting `relationship` on a later call leaves any recorded
relationship intact, so renaming never silently discards it, and there is deliberately no way to
clear one. `identity` resolves an ID through any merge alias and returns what has been registered,
so an observation captured before a merge still reaches the surviving person. `unlink_identity`
reverses one face-and-voice merge, returning the restored ID or `None` when the merge is not
reversible; it resets the pair's accumulated evidence rather than suppressing the pair, so a voice
and face that keep co-occurring are corroborated and merged again. `reinforce`
records explicit positive feedback and returns the number of existing distinct memories updated.
`list` uses an opaque keyset cursor. `delete` is idempotent and reports whether the record existed;
what it removes is spelled out below. `reindex` rebuilds Zvec from authoritative SQLite embeddings
without calling the embedder and returns the number of memories rebuilt. `optimize` merges and
flushes staged index vectors. Repeated `close()` calls are harmless.

#### What `delete` removes

`delete` is physical forgetting, the third and strongest of
[the three forms](../context-os.md#forgetting-is-three-operations). It is not a visibility change:
after it returns, the record's content is not recoverable from MindBridge. Media is
content-addressed and shared, so a blob and its descriptor go only once no remaining memory
references them; deleting one of two memories holding the same file keeps the file.

| Removed with the record | Left behind, and why |
| --- | --- |
| The `memory_records` row, and with it every typed row keyed on it: `memory_semantics`, `memory_versions`, `memory_evidence` for it, `formation_runs`, and its `capture_queue` row | -- |
| Every `embeddings` row, and the matching Zvec vectors, through the durable index outbox `delete` drains before returning | -- |
| Its `memory_assets` links, and then any `media_assets` descriptor and content-addressed blob no other memory still references | A blob a second memory still references, until that memory is deleted too |
| Everything keyed on a removed asset: `speech_analyses`, `speech_segments`, `face_analyses`, `face_observations`, and the cached transcript on the descriptor | -- |
| An `identities` row and its `identity_exemplars` biometric template, once the removed observations were its last and it carries no registered name, relationship, alias, or cross-modal evidence | A *named* or merged person, who is an assertion a caller made rather than a by-product of one recording. `forget_identity` erases a person |
| Derived memories whose last active evidence was this record | A derived memory with other evidence, whose link to this record is retired rather than deleted, keeping its lineage auditable. Its own text may still paraphrase what the record said; that is consolidation, and `delete` on the derived memory removes it |
| -- | `memory_operations` rows naming the id. The operation log is append-only audit history: it records that an operation happened, over which ids, with which proposal and rationale. Rewriting it to hide a deleted id would make `rollback` unsound and the log unable to answer what a deletion followed. It holds ids, a proposal, and a rationale -- never the deleted record's content or its media |

`capabilities` reports what the composition's backends declare rather than what a provider name
suggests, so an agent surface can advertise the instance instead of discovering a missing backend
by failing on first use. `consolidation_model` is set only when a `ConsolidationBackend` is
injected. `GET /healthz` and the MCP server instructions publish the same view.

### Memory management operations

```text
consolidation_candidates(*, limit: int = 32) -> tuple[ConsolidationCandidate, ...]

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

`consolidation_candidates` is the durable trigger: it answers "what needs deliberation?" from
state the store already committed, with no queue, timer, or scheduler behind it. Hand a row's
`memory_ids` straight to `consolidate(evidence_ids=..., trigger=...)`. It needs no backend and
takes no lock, so a host loop may poll it cheaply.

| `trigger` | Row means | `evidence_count` |
| --- | --- | --- |
| `EVIDENCE` | A derived record gained independent evidence no standing operation has weighed. `memory_ids` is that record followed by the new sources. | New evidence links |
| `CONTRADICTION` | One lineage carries two or more current visible claims with different values, the same disagreement `compile()` reports as a `ContextConflict`. It stays listed until a `CORRECT` retires one side. | Distinct conflicting values |
| `FEEDBACK` | A record was confirmed through `reinforce()`, or cited by an `ask()` answer under the default `reinforce_on_answer`, since a standing operation last saw it. | Recorded confirmations |

`QUERY_FAILURE`, `PRESSURE`, and `IDLE` remain labels a caller may pass to `consolidate`: nothing
in the store records a failed query, memory pressure, or an approved idle window, and inventing
bookkeeping for a trigger no host asks for would be a scheduler by another name. A rolled-back
operation weighed nothing, so its evidence becomes due again.

`consolidate` requires a `consolidator` and gathers the evidence set from `evidence_ids`, else the
active search result for `query`, else the newest `limit` active records. Forgotten and hidden
records never reach the backend. The backend proposes `MemoryOperation` values; the kernel
validates each one, applies only the effect its intent allows, and commits it in its own
transaction with its own log row. A pass is therefore not atomic: `ConsolidationReport.operations`
lists exactly what committed and `.rejected` exactly what the kernel refused, and a proposal
refused after an earlier one committed does not undo it. Rejected proposals are returned with a
reason instead of raising, so one bad proposal does not discard the pass. Every operation carries
`operation_key = sha256(canonical operation JSON + recipe)`; a key already applied and not rolled
back is rejected as `"duplicate"`.

| Intent | Kernel checks | Effect | `rollback` |
| --- | --- | --- | --- |
| `REINFORCE` | One derived target in the window whose current version is not retired; every evidence ID shown, existing, not the target, not already linked | Adds independent evidence; confidence and visibility recompute | Retires those evidence rows |
| `CONSOLIDATE` | Valid proposal; at least one shown evidence ID, each still standing; every `target_ids` entry among this proposal's own evidence; affect cue modality and spatial frame present in some source | New derived record citing every source, `forgotten_at` set on any named target, and the lineage supersession the kernel's own rule implies | Deletes the created records, un-forgets the targets, and restores the `superseded` versions |
| `CORRECT` | Every target in the window, existing, and derived (`kind != OBSERVATION`) | Retires current versions at transaction time | Carries a new version with the same interval |
| `FORGET` | Every target in the window, existing, and not already forgotten | Sets `forgotten_at` | Clears `forgotten_at` |

Named targets and cited evidence are bounded by the window the kernel gathered: a proposal naming
a target outside it is rejected as `"target_not_shown"` before any write. The window is the shown
set plus, when the host supplied `evidence_ids`, the IDs it named -- a hidden derived record is
exactly what `REINFORCE` and `CORRECT` exist for and can never appear in the shown set, so naming
it stays the host's decision. A `query` or the default window never widens it.

Lineage supersession is the one effect that reaches outside the window, and it is the kernel's
own deterministic rule rather than a backend choice: a new `STATE` or user-stated `TRAIT` retires
the current version of every other record in the same lineage whose validity it overlaps,
including records nobody showed the backend. The log row names those `(memory_id, version)` pairs
in `MemoryOperationRecord.superseded` and `rollback()` restores exactly them.

Multi-target operations are all or nothing. A proposal whose targets are not all eligible is
rejected whole -- `"unknown_target"`, `"not_derived"`, or `"already_forgotten"` -- so an applied
row's `target_ids` are always the IDs the operation actually acted on. The apply transaction
re-checks the preconditions validation read: a target or cited source that was forgotten,
corrected, deleted, or linked in between makes the proposal `"stale"` with nothing written. There
is no expected-revision token to supply; the in-transaction re-check is the whole guarantee.

`target_ids` on a `CONSOLIDATE` is **consolidation forgetting**: sources the new derived record
replaces in ordinary recall. They are retired in the same transaction that creates the derived
record, stay linked to it as evidence so lineage survives, and are named in the log row's
`forgotten_ids` so the operation reads as one reversible act. It is deliberately not a FORGET
intent, not two unrelated operations, and not `delete()`; the operation log distinguishes the
three by intent, by `forgotten_ids`, and by not appearing at all. One pass may not contradict
itself either: an operation that would retire evidence an earlier accepted operation in the same
pass built on, or that builds on evidence an earlier one retired, is rejected as
`"inconsistent_batch"`.

`forget` is the host entry point for the FORGET intent and needs no backend. The host names the
IDs and is the authority, so no window bounds it. It is cognitive forgetting only: recall skips
the record while `get()`, `list()`, and `MemoryRecord.forgotten_at` keep it for audit. Use
`delete` to remove a record and its media. It is all or nothing like a proposal: an unknown ID
raises `MemoryNotFoundError` the way `get` and `delete` do, and an empty sequence or a set in
which any record is already forgotten returns `None` having changed nothing.

`rollback` reverses one applied operation and returns `False` for an unknown or already-reversed
`operation_id`, and for one a later standing operation has built on. Operations that touched one
lineage reverse newest first: while a second consolidation's derived record supersedes the first
one's, rolling the first one back would leave two current versions in the lineage, so it is
refused and its log row stays standing until the newer one is reversed. `operations` lists the log newest first. Physical deletion is not an intent, and
none of these five operations is exposed on REST or MCP.

The log is sufficient to replay a sequence against a fresh store holding the same sources:
`operation_key` is a pure function of the operation and its recipe, derived record IDs are
content-addressed, and both are reproduced exactly on replay. Two values are not reproduced and
are not meant to be: `applied_at` is re-stamped at replay, and any ID derived from event time
reproduces only if the fresh store's sources carry the same `occurred_at`. There is no public
`apply(operation)`; replay means driving a `ConsolidationBackend` that returns the logged
proposals, which is what `tests/unit/test_memory_control_plane.py` does.

### Cross-modal identity binding

A face and a voice become one identity only after the same pair co-occurs in
`identity_link_min_assets` distinct assets (default `2`). A single asset cannot separate "this
face spoke" from "this face was listening to someone off camera", which is the ordinary case in
egocentric capture: the wearer talks while somebody else fills the frame. Binding on one asset
therefore attaches the wearer's voice to whoever happened to be visible. Set
`identity_link_min_assets=1` to bind on first co-occurrence.

Counting assets raises the price of that mistake without preventing it: a wearer talks to the same
person across many clips, so the wrong pair accumulates as fast as a genuine speaker's. Binding is
therefore also contained. Only a voice-only and a face-only identity may fuse, so an identity that
already holds both modalities absorbs nothing further and one wrong bind cannot cascade into the
next person. A fragment whose modality the identity already holds stays orphaned; merge it
deliberately through the store if the claim is established some other way.

Binding is recorded, not guessed at read time, so it is durable and reversible through
`unlink_identity`. Recognizer yield per asset is reported under the `mindbridge.identity.*` span
attributes, including when nothing was detected.

`forget_identity` erases a person rather than a memory. It removes the profile, every face and
voice exemplar, every merged alias, and the accumulated link evidence, and it rewrites the indexed
documents so a registered name stops being searchable — erasing the rows alone would leave the
name in the search projection. Memories, their content and their media survive, and a transcript
keeps its words with the speaker attribution dropped: forgetting a person is not forgetting the
evening. Face observation rows are deleted because their whole payload is a box plus the identity
claim. The returned `IdentityErasure` reports what was destroyed, which is what an audit needs.

Freed database cells are zero-filled and the write-ahead log is checkpointed, so the stored
vectors are gone from the database and its log. Filesystem snapshots, backups and wear-levelled
storage are not covered. `forget_identity` deliberately does **not** prevent a later encounter
from minting a fresh identity for the same person: recognizing someone as previously-forgotten
would require keeping the template the request destroys. A deployment that needs "never recognize
this person again" needs a retained blocklist, which is the opposite of a deletion.

`ObservationContext(place_id=...)` records the symbolic room-level place a memory was captured
in, `RetrievalScope(place_id=...)` scopes retrieval to it, and `MemoryRecord.place_id` reads it
back. This is a second spatial axis alongside `spatial`'s metric pose: a robot that cannot
localise can still label a room, and "in the kitchen" is the question a household asks. The label
is matched by equality and nothing normalises it beyond rejecting empty or untrimmed text, so a
producer that writes both `kitchen` and `the kitchen` partitions its own store — which is why the
value is readable rather than write-only. A place scope excludes memories with no place; it does
not treat them as being everywhere.

`capabilities` publishes what the composition declared — the modality set each backend accepts,
the embedding model, space and dimension, the optional model identities, and whether speaker
recognition and streaming generation are available. It is a field read: it cannot raise or block.

### AsyncMemory

`AsyncMemory` has the same constructor, class methods, keyword parameters, defaults, and result
values as `Memory`. Finite operations are awaited and `close()` is asynchronous. It mirrors every
listed operation except `forget_identity`; identity erasure currently requires synchronous
`Memory`. Its stream boundary is:

```text
add_stream(
    contents: AsyncIterable[ContentInput | StreamInput],
) -> AsyncIterator[MemoryRecord]
```

The facade runs the synchronous embedded consistency core with `asyncio.to_thread`; it does not
turn synchronous provider clients into native async clients.

`AsyncOmniPrefetch` coalesces evolving query snapshots for one turn:

```text
AsyncOmniPrefetch(
    memory: AsyncMemory,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    occurred_from: datetime | None = None,
    occurred_until: datetime | None = None,
) -> None

latest: PrefetchResult | None
submit(query: ContentInput) -> int
async finalize(query: ContentInput | None = None) -> PrefetchResult
async close() -> None
```

Only one search runs at once and only the newest queued snapshot survives. `finalize` returns the
exact final revision and closes the helper. Snapshots reject mutable `Path` atoms; use `Blob` or
`AssetRef`.

### Async capture streams

The capture helpers use these signatures:

```text
AsyncCaptureStream(
    memory: AsyncMemory,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    max_streams: int = 32,
    capture: bool = False,
) -> None
AsyncAudioStream(
    memory: AsyncMemory,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    context: StreamContext = None,
    reference_at: datetime | None = None,
    max_streams: int = 32,
    capture: bool = False,
) -> None
AsyncVisionStream(
    memory: AsyncMemory,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    context: StreamContext = None,
    reference_at: datetime | None = None,
    max_streams: int = 32,
    capture: bool = False,
) -> None

AsyncCaptureStream.consume(
    events: AsyncIterable[StreamEvent],
) -> AsyncIterator[StreamCommit]
AsyncAudioStream.consume(
    packets: AsyncIterable[AudioStreamPacket],
) -> AsyncIterator[StreamCommit]
AsyncVisionStream.consume(
    packets: AsyncIterable[VisionStreamPacket],
) -> AsyncIterator[StreamCommit]
```

`AsyncCaptureStream` treats `update` as speculative retrieval, `final` as the durable write, and
`cancel` as discard. `AsyncAudioStream` reduces `PCMChunk`, `VADPacket`, `ASRPartial`, and
`AcousticBoundary`; `AsyncVisionStream` reduces `VisionFrame`, `VisionPartial`, and
`SceneBoundary`. `StreamContext` is an `ObservationContext`, a callable sampled once per completed
observation, or `None`. Up to `max_streams` independent `stream_id` values may be active.
`capture=True` commits each final through `capture` instead of `add`, so acknowledgement leaves
the model path and every `StreamCommit` reports `pending_settlement=True`; the host then owes
`settle` before those records are searchable. The default stays the strong `add`. See
[omni streaming and interaction memory](../omni-streaming-and-interaction-memory.md) for event
semantics and complete examples.

### Root import inventory

These are the 103 supported names exported by `mindbridge`:

| Group | Names |
| --- | --- |
| Memory | `Memory`, `AsyncMemory`, `AsyncOmniPrefetch`, `AsyncCaptureStream`, `AsyncAudioStream`, `AsyncVisionStream` |
| Composition | `MindBridgeConfig`, `MemoryComposition`, `MemoryConfig`, `MemorySettings`, `MemoryPlugins`, `resolve_memory_config` |
| Content and records | `ContentAtom`, `ContentInput`, `Blob`, `AssetRef`, `StreamInput`, `MemoryRecord`, `SearchHit`, `AnswerResult`, `Page`, `ObservationContext`, `MemoryContext`, `RetrievalScope`, `SpatialContext`, `SpeakerSegment`, `IdentityProfile`, `IdentityErasure`, `FaceObservation`, `MemoryCapabilities`, `PendingCapture`, `PrefetchResult`, `StreamCommit`, `TracedSearchResult`, `RetrievalTrace`, `RetrievalCandidateTrace`, `FormationProposal`, `ContextBudget`, `ContextBundle`, `ContextConflict`, `ContextUnknown`, `MemoryOperation`, `MemoryOperationRecord`, `ConsolidationReport`, `ConsolidationCandidate` |
| Stream input | `AudioStreamPacket`, `PCMChunk`, `VADPacket`, `ASRPartial`, `AcousticBoundary`, `VisionStreamPacket`, `VisionFrame`, `VisionPartial`, `SceneBoundary`, `StreamEvent` |
| Enums | `Modality`, `MemoryType`, `EvidenceBasis`, `MemoryKind`, `MemoryIntent`, `MemoryTrigger`, `SpatialAnchor`, `ContextUnknownKind`, `AbstentionReason`, `IndexQuantization`, `RetrievalRejection`, `StreamPhase`, `AudioBoundary`, `VisionBoundary`, `EmbedTask` |
| Backend protocols and values | `EmbeddingBackend`, `GenerationBackend`, `StreamingGenerationBackend`, `TranscriptionBackend`, `SpeechBackend`, `VisionDescriptionBackend`, `FaceBackend`, `FormationBackend`, `ConsolidationBackend`, `ModelInput`, `FormationInput`, `SpeechTurn`, `SpeakerEmbedding`, `SpeechAnalysis`, `FaceEmbedding`, `FaceAnalysis` |
| Bundled adapters | `JinaOmniEmbedder`, `SentenceTransformersEmbedder`, `OpenAIModels`, `OpenCVFaceAnalyzer`, `FunASRTranscriber`, `FunASRRecipe`, `DEFAULT_FUNASR_MODEL_ID`, `DEFAULT_FUNASR_RECIPE` |
| Exceptions | `MindBridgeError`, `ValidationError`, `MemoryNotFoundError`, `SpeakerNotFoundError`, `IdentityNotFoundError`, `ModelError`, `ModelOutputTruncatedError`, `StorageError`, `IndexUnavailableError` |

### Public values

`MemoryRecord.content` is the caller's text followed by any text the configured models derived
from the media. Derived sections are appended, never substituted: what the caller supplied stays
byte-identical at the front, and each derived section is introduced by its own marker line --
`[transcript:<asset_id>]`, `[visual description:<asset_id>]`, or
`[speech identities:<asset_id>]` -- so a reader can separate interpretation from evidence and see
which asset it came from. `add` derives before its first write and `settle` derives after
`capture` already committed, so both paths leave the same record; the raw media is never rewritten
and stays in `assets` under its content address.

The principal immutable values are:

| Value | Fields |
| --- | --- |
| `Blob` | `data`, `media_type`, `name` |
| `AssetRef` | `id`, `modality`, `media_type`, `size_bytes`, `sha256`, `name`, `path`; `is_resolved` property |
| `StreamInput` | `content`, `occurred_at`, `occurred_end`, `metadata`, `memory_type`, `context`, `transcript`, `description` |
| `MemoryRecord` | `id`, `content`, `created_at`, `occurred_at`, `occurred_end`, `metadata`, `assets`, `modality`, `memory_type`, `context`, `place_id`, `forgotten_at` |
| `PendingCapture` | `memory_id`, `enqueued_at`, `attempts`, `last_error`, `awaiting` |
| `SearchHit` | all visible memory fields plus `score` |
| `AnswerResult` | `answer`, `hits`, `abstained`, `abstention_reason` |
| `Page` | `items`, `next_cursor` |
| `SpeakerSegment` | `asset_id`, `start_ms`, `end_ms`, `text`, `speaker_id`, `speaker_name`, `identity_score` |
| `IdentityProfile` | `identity_id`, `name`, `relationship` |
| `IdentityErasure` | `identity_id`, `alias_ids`, `face_exemplars`, `voice_exemplars`, `face_observations`, `speech_segments` |
| `FaceObservation` | `asset_id`, `bounding_box`, `identity_id`, `identity_name`, `identity_score`, `observed_at_ms` |
| `SpatialContext` | `frame_id`, `anchor`, `x`, `y`, `z`, `orientation_xyzw`, `position_uncertainty_m` |
| `ObservationContext` | `basis`, `source_id`, `confidence`, `valid_from`, `valid_until`, `spatial`, `place_id` |
| `MemoryContext` | `kind`, `basis`, `confidence`, `valid_from`, `valid_until`, `recorded_at`, `visible`, `retired_at`, `lineage_id`, `source_id`, `subject`, `predicate`, `value`, `evidence_ids`, `supersedes_id`, `model_id`, `recipe`, `spatial`, `cue_modality`, `valence`, `arousal` |
| `RetrievalScope` | `valid_at`, `known_at`, `near`, `radius_m`, `place_id` |
| `ContextBudget` | `max_chars`, `max_items`, `memory_types`, `min_confidence`, `freshness`, `max_latency_ms` |
| `ContextConflict` | `lineage_id`, `subject`, `predicate`, `values`, `memory_ids` |
| `ContextUnknown` | `kind` (a `ContextUnknownKind`), `detail` |
| `ContextBundle` | `goal`, `reference_at`, `budget`, `actors`, `relationships`, `scene`, `episodes`, `facts`, `procedures`, `affect`, `traits`, `conflicts`, `unknowns`, `occurred_from`, `occurred_until`, `frames`, `places`, `omitted`, `chars`, `elapsed_ms`, `deadline_exceeded`; `hits` property and `render()` |
| `MemoryOperation` | `intent`, `evidence_ids`, `target_ids`, `proposal`, `rationale` |
| `MemoryOperationRecord` | `operation_id`, `operation`, `trigger`, `applied_at`, `model_id`, `recipe`, `created_ids`, `changed_ids`, `forgotten_ids`, `superseded`, `rolled_back_at` |
| `ConsolidationCandidate` | `trigger`, `memory_ids`, `evidence_count` |
| `ConsolidationReport` | `operations`, `rejected` as `(MemoryOperation, reason)` pairs |
| `StreamEvent` | `phase`, `item`, `stream_id` |
| `StreamCommit` | `record`, `prefetch`, `retrieval_error`, `stream_id`, `pending_settlement` |
| `PCMChunk` | `data`, `sample_rate_hz`, `channels`, `sample_width_bytes`, `stream_id`, `occurred_at` |
| `VADPacket` | `active`, `stream_id`, `occurred_at` |
| `ASRPartial` | `text`, `stream_id`, `occurred_at` |
| `AcousticBoundary` | `boundary`, `stream_id`, `occurred_at` |
| `VisionFrame` | `image`, `stream_id`, `occurred_at` |
| `VisionPartial` | `text`, `stream_id`, `occurred_at` |
| `SceneBoundary` | `boundary`, `stream_id`, `occurred_at` |
| `PendingCapture` | `memory_id`, `enqueued_at`, `attempts`, `last_error` |
| `PrefetchResult` | positive `revision`, `hits` |
| `TracedSearchResult` | `hits`, `trace` |
| `RetrievalTrace` | `candidates`, `candidate_limit`, `exhaustive`, `ambiguous` |
| `RetrievalCandidateTrace` | `memory_id`, `index_ids`, `dense_relevance`, `dense_confidence`, `lexical_relevance`, `lexical_rerank_bonus`, `lexical_match`, `gate_relevance`, `base_relevance`, `reinforcement_factor`, `temporal_factor`, `retention_factor`, `final_score`, `rank`, `rejected_by` |
| `MemoryCapabilities` | `embedding`, `embedding_model`, `embedding_space`, `embedding_dimension`, `generation`, `transcription`, `vision`, `face`, `formation`, `generation_model`, `transcription_space`, `vision_model`, `face_model`, `formation_model`, `consolidation_model`, `speaker_recognition`, `streaming_generation`; `operations` property and `document()` |

`abstained` is true only when the answerer returns MindBridge's reserved no-evidence sentence. A
provider refusal expressed another way is an ordinary answer unless the adapter maps it to this
protocol signal.

Enum values are:

| Enum | Values |
| --- | --- |
| `Modality` | `text`, `image`, `video`, `audio`, `omni` |
| `MemoryType` | `semantic`, `episodic`, `procedural` |
| `IndexQuantization` | `none`, `fp16`, `int8`, `rabitq` |
| `AbstentionReason` | `no_evidence`, `insufficient_evidence` |
| `RetrievalRejection` | `stale_index`, `occurrence_range`, `missing_memory`, `memory_type`, `minimum_relevance`, `ambiguity`, `limit` |
| `EmbedTask` | `retrieval.query`, `retrieval.document` |
| `MemoryKind` | `observation`, `entity`, `event`, `state`, `relation`, `affect`, `trait`, `response_policy` |
| `MemoryIntent` | `reinforce`, `consolidate`, `correct`, `forget` |
| `MemoryTrigger` | `manual`, `evidence`, `feedback`, `contradiction`, `query_failure`, `pressure`, `idle` |
| `EvidenceBasis` | `observation`, `user_statement`, `model_inference`, `response_feedback` |
| `SpatialAnchor` | `observer`, `subject` |
| `StreamPhase` | `update`, `final`, `cancel` |
| `AudioBoundary` | `start`, `end`, `cancel` |
| `VisionBoundary` | `start`, `end`, `cancel` |

### Backend protocols

Backends are runtime-checkable, thread-safe protocols. Their required methods are:

```text
EmbeddingBackend.embed(
    inputs: Sequence[ModelInput],
    task: EmbedTask = EmbedTask.DOCUMENT,
) -> tuple[tuple[float, ...], ...]

GenerationBackend.answer(
    question: ModelInput,
    hits: Sequence[SearchHit],
) -> AnswerResult

StreamingGenerationBackend.stream_answer(
    question: ModelInput,
    hits: Sequence[SearchHit],
) -> Iterator[str]

TranscriptionBackend.transcribe(
    assets: Sequence[AssetRef],
) -> tuple[str, ...]

SpeechBackend.analyze(
    assets: Sequence[AssetRef],
) -> tuple[SpeechAnalysis, ...]

FaceBackend.analyze(
    assets: Sequence[AssetRef],
) -> tuple[FaceAnalysis, ...]

VisionDescriptionBackend.describe(
    inputs: Sequence[ModelInput],
) -> tuple[str, ...]

FormationBackend.form(
    inputs: Sequence[FormationInput],
) -> tuple[tuple[FormationProposal, ...], ...]

ConsolidationBackend.consolidate(
    evidence: Sequence[MemoryRecord],
    *,
    trigger: MemoryTrigger,
) -> tuple[MemoryOperation, ...]
```

Required properties are `embedding_capabilities`, `embedding_model`, `embedding_space`, and
`embedding_dimension` for embedding; `transcription_capabilities`, `transcription_model`, and
`transcription_space` for transcription and speech; `face_capabilities`, `face_model`,
`face_space`, and `face_analysis_space` for faces; `vision_capabilities` and `vision_model` for
visual description; `formation_capabilities`, `formation_model`, and `formation_space` for
formation; `consolidation_model` and `consolidation_recipe` for consolidation; and
`generation_capabilities` for generation. Every base protocol except the optional
streaming extension implements `close()`.

`form` receives one `FormationInput` per committed source and returns one proposal tuple per
input, in the same order. A former never writes storage: the kernel validates each proposal
against the source modality and spatial frame, assigns identity, links evidence, and commits.
`consolidate` receives the bounded evidence set the kernel chose and may cite only IDs from it;
like a former it proposes and never writes storage.

`ModelInput` contains normalized `text` and resolved `assets`. Speech adapters return
`SpeechAnalysis(turns, speakers)` using `SpeechTurn` and `SpeakerEmbedding`; face adapters return
`FaceAnalysis(faces)` using `FaceEmbedding`.

| Backend value | Fields |
| --- | --- |
| `ModelInput` | `text`, `assets`; derived `modality` and `modalities` properties |
| `SpeechTurn` | `start_ms`, `end_ms`, `text`, `speaker_label` |
| `SpeakerEmbedding` | `speaker_label`, `values` |
| `SpeechAnalysis` | `turns`, `speakers` |
| `FaceEmbedding` | `face_label`, `values`, `bounding_box`, `observed_at_ms` |
| `FaceAnalysis` | `faces` |
| `FormationInput` | `memory_id`, `content`, `context` |
| `FormationProposal` | `kind`, `content`, `basis`, `subject`, `predicate`, `value`, `confidence`, `valid_from`, `valid_until`, `spatial`, `cue_modality`, `valence`, `arousal` |

### Bundled adapters

The public construction signatures are:

```text
SentenceTransformersEmbedder(
    encoder,
    *,
    model_id: str,
    revision: str,
    dimension: int | None = None,
    batch_size: int = 32,
)

SentenceTransformersEmbedder.load(
    model_id: str,
    *,
    revision: str,
    dimension: int | None = None,
    device: str | None = None,
    batch_size: int = 32,
) -> SentenceTransformersEmbedder

JinaOmniEmbedder(
    *,
    dimension: int = 1024,
    device: str | None = None,
    batch_size: int = 32,
)

FunASRTranscriber(
    recipe: FunASRRecipe = DEFAULT_FUNASR_RECIPE,
    *,
    device: str = "auto",
)

FunASRRecipe(
    model_id: str,
    vad_model: str,
    speaker_model: str | None,
    model_revision: str | None = None,
    vad_revision: str | None = None,
    speaker_revision: str | None = None,
    punctuation_model: str | None = None,
    punctuation_revision: str | None = None,
    vad_max_single_segment_ms: int | None = None,
    hub: str = "ms",
    trust_remote_code: bool = False,
)

OpenCVFaceAnalyzer(
    detector_model: str | Path,
    recognizer_model: str | Path,
    *,
    score_threshold: float = 0.9,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    frame_interval_ms: int = 1000,
    max_video_frames: int = 300,
)
```

The direct `SentenceTransformersEmbedder` constructor accepts a caller-owned encoder implementing
`supports`, `get_embedding_dimension`, `encode_query`, and `encode_document`; `load` constructs that
encoder and requires a 40-character immutable commit revision. `JinaOmniEmbedder.load` has the same
keyword parameters as its constructor and eagerly loads the pinned model. Jina loading sets
`trust_remote_code=True` while pinning both model and code revisions. Its weights are CC BY-NC 4.0;
that license covers the weights, not MindBridge. `DEFAULT_FUNASR_MODEL_ID` and
`DEFAULT_FUNASR_RECIPE` publish the default speech recipe.
`FunASRRecipe.auto_model_arguments() -> dict[str, object]` returns its standard FunASR composition.

`OpenAIModels` can fill embedding, generation, transcription, and formation capabilities. Pass the
same object only to the slots it should serve:

```text
OpenAIModels(
    client: OpenAI | None = None,
    *,
    embedding_client: OpenAI | None = None,
    generation_client: OpenAI | None = None,
    transcription_client: OpenAI | None = None,
    embedding_model: str = "text-embedding-3-small",
    embedding_space: str | None = None,
    embedding_dimension: int = 1536,
    embedding_request_format: Literal["input", "messages"] = "input",
    generation_model: str = "gpt-5-mini",
    transcription_model: str = "whisper-1",
    transcription_space: str | None = None,
    transcription_prompt: str | None = None,
    transcription_keywords: Sequence[str] | None = None,
    transcription_languages: Sequence[str] | None = None,
    embedding_capabilities: frozenset[Modality] = frozenset({Modality.TEXT}),
    generation_capabilities: frozenset[Modality] = frozenset({Modality.TEXT}),
    transcription_capabilities: frozenset[Modality] = frozenset({Modality.AUDIO, Modality.VIDEO}),
    generation_seed: int | None = None,
    generation_temperature: float | None = None,
    generation_max_tokens: int | None = None,
    generation_min_video_seconds: float | None = None,
    generation_video_limit: int | None = 8,
    generation_extra_body: Mapping[str, object] | None = None,
)
```

The common `client` fills any operation-specific client left unset. A missing client fails only
when that operation is called. `close()` does not close caller-supplied OpenAI clients. Provider
selection, extras, model identity, and credential behavior are documented once in
[configuration](../configuration.md).

`embedding_request_format` selects an `input` or `messages` request for compatible embedding
endpoints. Transcription prompt, keywords, and languages form the provider hint.
`generation_min_video_seconds` converts shorter videos to four ordered stills when image generation
capability is available; `generation_video_limit` caps retrieved evidence videos but not question
media. `generation_extra_body` is forwarded to the SDK request.

Its operation signatures widen the base protocols only by accepting plain text where useful:

```text
embed(
    inputs: Sequence[ModelInput | str],
    task: EmbedTask = EmbedTask.DOCUMENT,
) -> tuple[tuple[float, ...], ...]
form(inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]
answer(question: ModelInput | str, hits: Sequence[SearchHit]) -> AnswerResult
stream_answer(
    question: ModelInput | str,
    hits: Sequence[SearchHit],
) -> Generator[str, None, tuple[SearchHit, ...]]
transcribe(assets: Sequence[AssetRef]) -> tuple[str, ...]
close() -> None
```

OpenAI requests inline at most 20 MiB of base64 data per media item and 64 MiB in aggregate. Because
base64 expands input, that is roughly 15 MiB per source file and 48 MiB per call on disk. Grounded
answer text is capped at 4 MiB. Answer fitting reserves media capacity for the question, keeps
highest-ranked evidence that fits, and leaves overflow evidence as text when possible; use a
provider-specific upload adapter for larger media.

Named recipes are available through the supported `mindbridge.recipes` submodule, not as a root
re-export:

```text
import mindbridge.recipes as recipes

recipes.names() -> tuple[str, ...]
recipes.slots(name: str) -> tuple[str, ...]
recipes.require_slot(name: str, slot: str) -> None
recipes.probe(name: str) -> str
recipes.describe(name: str) -> dict[str, object]
recipes.embedder(name: str, *, load: bool = False) -> EmbeddingBackend
recipes.answerer(name: str, *, load: bool = False) -> GenerationBackend
recipes.former(name: str, *, load: bool = False) -> FormationBackend
recipes.transcriber(
    name: str,
    *,
    load: bool = False,
) -> SpeechBackend | TranscriptionBackend
```

Each construction function returns an object the caller owns and closes. `load=True` additionally
exercises the recipe's published probe (`weights`, `import`, or `client`).

## Errors and limits

All stable exceptions derive from `MindBridgeError`:

| Exception | `code` | Default `reason` |
| --- | --- | --- |
| `MindBridgeError` | `mindbridge_error` | `None` |
| `ValidationError` | `validation_error` | `input_invalid` |
| `MemoryNotFoundError` | `memory_not_found` | `memory_not_found` |
| `SpeakerNotFoundError` | `speaker_not_found` | `speaker_not_found` |
| `IdentityNotFoundError` | `identity_not_found` | `identity_not_found` |
| `ModelError` | `model_error` | `None` |
| `ModelOutputTruncatedError` | `model_output_truncated` | `output_truncated` |
| `StorageError` | `storage_error` | `None` |
| `IndexUnavailableError` | `index_unavailable` | `None` |

Every error may carry `reason`, `stage`, and `subject`. `retryable` is true only when `reason` is
`connection_failed`, `data_dir_in_use`, `flush_failed`, `index_missing`, `rate_limited`, or
`timeout`; it is never inferred from the message. `ModelOutputTruncatedError` is a deterministic
`ModelError`, and `IndexUnavailableError` is a `StorageError`.

Stable input bounds are:

| Bound | Value |
| --- | --- |
| Ordered parts in one `ContentInput` | 128 |
| Normalized text in one operation, including combined text atoms | 65,536 characters |
| Serialized metadata for one memory | 262,144 UTF-8 bytes |
| One local media asset | 512 MiB |
| `limit` for search, answer, and listing operations | 1 through 100 |
| Identity or speaker name | 255 printable characters |

The 512 MiB value is the local ingestion ceiling, not a promise that every model backend accepts an
asset that large. Backend request limits apply before model work.

Two degradations keep a write alive rather than failing it, and both are recorded on the operation
span. Media the embedding model will not accept inline drops the retrieval key holding it: the
memory is still stored with its media and stays reachable through the keys that did embed, counted
by `mindbridge.embedding.elided_parts`. A memory whose every key is refused still fails, because
nothing would find it. Separately, a video the embedding model cannot fit in its context is embedded
as four ordered stills, counted by `mindbridge.embedding.video_sampled_inputs`; only a rejection
that declares the length constraint triggers it, and the video itself is still stored and still
reaches the answering model on its own route.

`add_many` has no separate SDK item-count cap, but each item is subject to the same content and
metadata bounds. REST and MCP deliberately impose narrower transport limits. A `Memory` is bound
to the process that opened it and cannot be used after `fork`; open a new instance with a different
`data_dir` in the child.
