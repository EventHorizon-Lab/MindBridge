# MindBridge architecture cohesion audit

The embedded architecture broadly follows high cohesion and low coupling. Its main strengths
are explicit plane construction, shared retrieval, authoritative SQLite with a derived index,
and thin protocol adapters. The reviewed snapshot also contained misplaced startup decisions,
unnecessary imports between planes, and obsolete compatibility branches. The changes below
address those specific boundaries and recovery defects without replacing the architecture.

This audit covers product modules, storage families, model and protocol adapters, benchmark
compatibility, and their tests in the 2026-10-26 working tree. Concurrent Zvec and model-provider
changes already in that tree are preserved. File size alone is not treated as an architecture
defect.

## Existing boundaries worth preserving

| Boundary | Evidence and consequence |
| --- | --- |
| Public facades and kernel | [Memory](../../src/mindbridge/memory.py) constructs planes with named dependencies. Kernel imports form an acyclic graph; behavior belongs to named planes rather than a service locator. |
| Shared retrieval | [Retrieval](../../src/mindbridge/kernel/retrieval.py) supplies search, answering, compilation, and control; the clients do not maintain independent ranking implementations. |
| Storage authority | [Projection](../../src/mindbridge/kernel/projection.py) applies committed SQLite rows, flushes Zvec, then acknowledges the outbox. Hydration drops stale IDs; missing indexes rebuild from stored vectors. |
| Table families | [LocalStore](../../src/mindbridge/infrastructure/local/store/local_store.py) owns one pool and directory lock, with eight families over shared transactions. Separate directories provide physical isolation. |
| Model and protocol adapters | Model implementations stay in `models/`; REST and MCP delegate through the SDK. Optional clients load lazily. Product-to-benchmark imports are prohibited by a package test. |

## Problems corrected

### Startup recovery had several owners

Previously `Memory.__init__` made re-embedding and rebuild decisions, `kernel/runtime.py`
reconciled durable metadata, and backend contracts contained historical index recipes. A store
upgrade therefore changed modules responsible for facade wiring, shared resources, and models.

[Recovery](../../src/mindbridge/kernel/recovery.py) now owns startup recovery execution.
[Metadata policy](../../src/mindbridge/kernel/metadata.py) contains pure recipe validation and
upgrade planning. Backend contracts validate model declarations and capture historical-space
declarations once; the recovery plane no longer inspects provider objects for private attributes.
The facade validates index configuration, wires recovery, and invokes it before opening Zvec.

The separation also fixes two correctness defects. A rejected open previously committed missing
metadata and rewrote an equal evidence marker before discovering a later mismatch. Recovery now
validates every marker first and commits updates together. A failure after the first re-embedding
page previously left mixed spaces under the old space marker. A durable target now blocks a
different configuration until the original target completes; final markers and target removal
commit together. Retries reprocess the source records in bounded pages rather than retaining the
whole vector corpus in memory or running model calls inside one long SQLite transaction.

### Pure helpers created unnecessary plane dependencies

`Records` imported `ControlPlane` only for `operation_touches`; `Vision` imported `Formation`
only for `IDENTITY_LINK_RECIPE`. Both now import these values from pure
[control helpers](../../src/mindbridge/control.py). The old modules do not re-export them. This
removes two dependency edges without introducing interfaces or moving stateful behavior.

### Re-embedding lost text excerpt provenance

Re-embedding replaced embedding rows without reconstructing their text selectors. Cascading
deletion therefore removed exact excerpt provenance, and a previously compilable long source
stopped yielding a partial excerpt after a recognized upgrade. Ingestion and re-embedding now
share `text_selector_map`, which consumes only the prepared content it actually needs. Tests
verify the same excerpt content and selector survive both space and legacy recipe upgrades.

### Benchmark defaults duplicated provider request policy

Full-suite verification exposed a second owner of generation controls in the benchmark harness:
its baseline generator, product factory, and declarative configuration injected the same seed
and temperature regardless of the model adapter's supported controls. They now reuse the shared
pure provider policy for those defaults and the output-token parameter name. Result metadata
records the controls actually sent, including null for an omitted control. Explicit invalid
controls still reach validation. Request-shape tests cover generic and reasoning configurations;
these checks make no claim about real-model answer quality.

## Compatibility decisions

| Code | Decision | Reason and impact |
| --- | --- | --- |
| Schema-17 `linked_clauses` log field and rollback path | Removed | Only an incomplete development format populated it, while the store already refuses schema 17. Current schema-18 writers use before/after clause snapshots. Ordinary evidence links and snapshot rollback remain. |
| Benchmark `SEARCH_SPAN` and deprecated JSON metric aliases | Removed | Current consumers use explicit canonical metrics. Result schema advances to 19; the [benchmark reference](../benchmarking.md) lists replacements. Saved result files are not rewritten. |
| `ffmpeg-v1` cache discovery and promotion | Removed | Prepared media is disposable; obsolete entries can be recomputed from source. Current cache reuse and native-frame fallback remain covered. |
| Recognized index and Jina embedding-space upgrades | Retained and isolated | These reopen accepted directories and either reuse canonical vectors or regenerate obsolete keys. Deleting them would strand existing data rather than merely simplify an alias. |
| Old place-aware content-ID lookup | Retained | It preserves idempotency for existing records while checking the explicit place. Removing it could create duplicate records when the same observation is re-added. |
| Flat `memory_evidence` projection | Retained | Current selection, lineage, and dependency checks still consume it. The word “legacy” in comments does not make a maintained projection dead code. |
| `independent_evidence=False` and its pinned recipe | Retained | This remains the default public policy; the alternative is opt-in. Removing one mode would change supported semantics and stored metadata. |
| Pinned Jina loading/video adaptations, endpoint response normalization, streamed answer completion values | Retained | They support current provider behavior and the supported backend contract. They are active adapters, not abandoned SDK aliases. |
| Documented CLI aliases and external response-cache layouts | Retained | These remain consumed console and cache contracts. Removing them requires an intentional interface change, unlike recomputing disposable prepared-media entries. |

No public Python signature, REST route, MCP tool schema, console entry point, dependency, or
SQLite table version changes in this cleanup. `embedding.migration_target` is new durable
metadata, with its retry and downgrade requirements documented in
[operations](../operations.md#restore). A runtime predating the marker cannot enforce
it; finish recovery or restore the backup before downgrading.

## Remaining design tradeoffs

- Naming projection callbacks can invoke embedding through a storage transaction. This keeps
  names, vectors, and evidence atomic, but couples transaction duration to model latency.
  Separating it needs an explicit staged-projection design that preserves erasure and rollback
  atomicity; moving the callback elsewhere alone would not remove the coupling.
- `AsyncMemory.add_stream` coordinates its source separately from synchronous ingestion and
  does not use the synchronous path's bounded projection groups. Unifying those loops must
  preserve cancellation, async backpressure, context propagation, and committed-prefix behavior.
- The compiled native benchmark arm calls `OpenAIModels._answer_request` and adjusts the message
  payload. That ties a benchmark baseline to private provider request structure. A shared request
  builder would need to preserve its frozen prompt and media protocol.
- The async facade does not expose synchronous `forget_identity`. This is API coverage drift;
  adding a public method requires its own contract and documentation rather than a compatibility
  alias hidden in a structural refactor.

These are concrete remaining limits, so this audit does not claim that every module has ideal
cohesion or that all cross-layer coupling has disappeared. The implementation keeps the current
public and durable behavior while removing unnecessary dependencies and obsolete branches.

## Verification

Recovery regressions cover rejected-open immutability, unchanged timestamps on matching opens,
atomic marker rollback, a second-page migration failure, rejection of another target space or
index recipe, and successful retry. Existing tests cover directory ownership, stale-index
hydration, missing-index recovery without model calls, snapshot rollback, media cache reuse,
and benchmark metric calculations. Exact excerpt upgrade regressions and obsolete-cache
recomputation were observed failing before their fixes.

Final verification on Python 3.12.14 passed all **2573 tests** with `uv run --frozen pytest -W error`.
The lockfile, Ruff formatting and lint, strict mypy, and `git diff --check` gates also passed.
The pinned Docker documentation commands passed in the working tree: 85 Markdown files and
791 link checks, with zero errors. These checks validate local contracts and recovery behavior;
they do not establish model answer quality or rerun the full supported Python-version matrix.
