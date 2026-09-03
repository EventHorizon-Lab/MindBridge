# Context compilation

This page owns `compile()`: what a `ContextBudget` bounds, how hits become sections, how conflicts
are reported, and what `render()` emits.

`compile()` selects and structures evidence. It calls no generation model, stores no memory, and
never resolves a conflict. It is not a pure read: it runs the same retrieval path `search()` runs
and makes the same single write that path can make, caching a transcript for spoken query media.
That cache is a cache of the caller's own input, never a new memory, and repeating the call adds
nothing to the store. `ask()` is unchanged and remains the grounded-generation surface; see
[the Python SDK reference](api/python-sdk.md#memory-operations).

## Contract

```text
compile(
    goal: ContentInput,
    *,
    budget: ContextBudget | None = None,
    reference_at: datetime | None = None,
    scope: RetrievalScope | None = None,
) -> ContextBundle
```

`compile()` runs the same retrieval kernel `search()` uses, once, with a candidate limit of
`max(100, 3 * budget.max_items)`. Relative time in the goal text, `reference_at`, and `scope`
behave exactly as they do for `search()`, and SQLite reapplies authoritative visibility: a
forgotten record and a hidden inferred trait never reach a bundle.

## Budget

| Field | Default | Meaning |
| --- | --- | --- |
| `max_chars` | `6000` | Evidence character ceiling, charged with the same cost function `ask()` uses: record text plus a per-modality text equivalent for each media asset |
| `max_items` | `24` | Maximum included memories |
| `memory_types` | `None` | Keep only these `MemoryType` values; `None` keeps every type |
| `min_confidence` | `0.0` | Minimum typed confidence; a record with no typed context counts as `1.0` |
| `freshness` | `None` | Keep only memories anchored within this `timedelta` of `reference_at` |
| `max_latency_ms` | `None` | Deadline in milliseconds; optional stages are skipped once it passes |

The freshness anchor is event end, then event start, then creation time. Every field is validated
on construction: the two maxima and `max_latency_ms` are positive integers, `min_confidence` is in
`[0, 1]`, and `freshness` is a positive `timedelta`.

### The deadline

`max_latency_ms` is a deadline, not a timeout. Nothing is cancelled, no thread is started, and no
work already in flight is interrupted or discarded. The compiler reads the clock once, after
section assembly and before the optional enrichment that follows it, and if the deadline has
already passed it skips that enrichment instead of truncating it halfway. A bundle compiled under
a deadline is therefore a prefix of the bundle without one, never a different bundle.

Today the one optional stage is conflict detection. When it is skipped, `conflicts` is empty and
`unknowns` carries a `stage_skipped` entry naming it, so an empty `conflicts` under a deadline is
never mistaken for agreement. Every bundle reports `elapsed_ms`, measured from before retrieval,
and `deadline_exceeded`, which is true when `elapsed_ms` passed the declared deadline.

## Sections

A typed `MemoryKind` decides the section first; an untyped or generic record falls back to its
`MemoryType`.

| Section | Contents |
| --- | --- |
| `actors` | Kind `entity` |
| `relationships` | Kind `relation` |
| `scene` | Kind `state` |
| `facts` | Type `semantic`, except `entity`, `relation`, `state`, `affect`, and `trait` |
| `episodes` | Type `episodic`, except `affect` |
| `procedures` | Type `procedural` |
| `affect` | Kind `affect` |
| `traits` | Kind `trait` |

`actors`, `relationships`, `scene`, `affect`, and `traits` are keyed on `MemoryKind`. A stored
record is kind `observation` unless a validated `FormationProposal` typed it, so all five sections
stay empty in a composition with neither a `FormationBackend` nor a `ConsolidationBackend`.
`episodes`, `facts`, and `procedures` are keyed on `MemoryType`, which every record carries, and
populate without either.

The bundle also carries no person link. Identity lives asset-keyed in the speech and face tables,
reachable through `speech()` and `faces()`, and no `identity_id` exists on a memory, a hit, or a
`MemoryContext`. Cross-modal person linkage in a bundle needs that edge; it is a known gap rather
than a configuration mistake.

Selection gives every non-empty section one slot in rank order before any section receives a
second, so a small budget still describes the whole scene instead of spending everything on the
top-ranked section. Remaining slots are filled by score. One oversized hit does not close the
bundle: a cheaper lower-ranked candidate can still fit. `omitted` counts every candidate that
passed the filters but did not fit, and `chars` is what the included hits cost.

`ContextBundle` also reports `occurred_from` and `occurred_until` over the included hits, `frames`
(the distinct metric spatial frame IDs, sorted), `places` (the distinct symbolic `place_id`s,
sorted), and `hits` (every included hit in rank order).

## Unknowns

`unknowns` names, in one stable order, what the request implied that the bundle does not carry.
Every entry is a deterministic statement about the compilation, produced without a model call, so
a thin bundle explains itself instead of looking like an empty store.

| `kind` | When it appears |
| --- | --- |
| `budget_excluded` | Counted candidates a bound removed, or that did not fit the budget |
| `section_empty` | A `memory_types` value the request asked for that no included hit carries |
| `scope_empty` | A `scope` was supplied and retrieval matched nothing; the entry names the bounds |
| `modality_unsupported` | The goal carries media this composition's embedder cannot search |
| `stage_skipped` | An optional stage the `max_latency_ms` deadline skipped |

## Conflicts

Included hits that share a `lineage_id`, carry kind `state`, `relation`, or `trait`, and disagree
on `value` produce one `ContextConflict` each. Its `values` and `memory_ids` are aligned: each
distinct value is paired with the highest-ranked included memory asserting it. The compiler reports
the disagreement and leaves resolution to the caller or to a later correction.

Only included hits are compared. A superseded version that bitemporal filtering already excluded
never appears, so `conflicts` is a statement about this bundle and not a belief-revision history;
read `memory_versions` through the control plane for that.

## Rendered text

`render()` is deterministic: the same bundle always produces the same string. It emits a goal
heading, a one-line reading guide, the reference time, the budget it filled, one `##` heading per
non-empty section in the fixed order above, one line per hit, a `## Conflicts` section when the
bundle reports any, a `## Unknowns` section when it carries any, and an omitted trailer only when
something was omitted. Anything omitted is itself a `budget_excluded` unknown, so the trailer never
appears without the `## Unknowns` block above it.

```text
# Context: what should I bring to the workshop?
Each line is one memory: [id] content (confidence; validity).
Reference time: 2026-09-03T12:00:00+00:00
Budget: 132/6000 chars, 3/24 items

## Facts
- [a1b2] the spare key is in the blue toolbox (confidence 1.00)
- [c3d4] Ana works from Berlin (confidence 0.90; valid from 2026-08-01T00:00:00+00:00)

## Episodes
- [e5f6] we walked to the harbour (confidence 1.00)

## Conflicts
- ana location: "berlin" [c3d4] vs "paris" [g7h8]

## Unknowns
- budget_excluded: 4 candidates did not fit 24 items and 6000 chars

Omitted: 4 lower-ranked candidates
```

## Example

```python
from datetime import timedelta

from mindbridge import ContextBudget, Memory, MemoryType

config = {
    "data_dir": "./data/compile-example",
    "embedding": {"provider": "jina-omni"},
    "settings": {"minimum_relevance": 0, "ambiguity_margin": 0},
}

with Memory.from_config(config) as memory:
    memory.add("The spare key is in the blue toolbox.")
    memory.add("We walked to the harbour.", memory_type=MemoryType.EPISODIC)

    bundle = memory.compile(
        "What should I bring to the workshop?",
        budget=ContextBudget(max_chars=2000, max_items=8, freshness=timedelta(days=30)),
    )
    print(bundle.render())
    print(f"{bundle.omitted} candidates did not fit")
```

## Knowing what an instance can do

A caller does not have to compile a bundle to learn what a composition supports. The
`Memory.capabilities` property returns a frozen `MemoryCapabilities` describing the declarations
routing reads: the embedding modalities, model, space and dimension, the modality sets for
generation, transcription, vision, face, and formation with their model identities,
`consolidation_model` when a `ConsolidationBackend` is injected, the `speaker_recognition` and
`streaming_generation` flags, and the derived `operations` set naming which optional operations
those backends can serve (`ask`, `speech`, `transcribe`, `faces`, `describe_vision`, `formation`,
`consolidate`). Its fields are listed in
[the Python SDK reference](api/python-sdk.md#public-values).

`MemoryCapabilities.document()` renders that value as one JSON-ready document, and it is the only
renderer: `GET /healthz` serves it, the MCP server embeds it in its instructions, and
`mindbridge doctor` prints it under `capabilities`. The three surfaces cannot describe one
composition differently.

```python
if memory.capabilities.generation:
    print(memory.ask("Where is the spare key?").answer)
```

`AsyncMemory` mirrors `compile()`. The command line exposes it as `mindbridge compile`, locally and
against `--url`; see [the CLI reference](api/cli.md). Agents reach it as `POST /v1/context` on
[REST](api/rest.md#endpoints) and as the `compile_context` tool on [MCP](api/mcp.md#tools), which
also publishes the capability view as its server instructions. REST publishes the same view from
`GET /healthz`.
