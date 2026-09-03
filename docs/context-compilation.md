# Context compilation

This page owns `compile()`: what a `ContextBudget` bounds, how hits become sections, how conflicts
are reported, and what `render()` emits. It also owns `capabilities()`, the advertisement an agent
surface reads instead of discovering a missing backend by failing on first use.

`compile()` selects and structures evidence. It calls no generation model, writes nothing, and
never resolves a conflict. `ask()` is unchanged and remains the grounded-generation surface; see
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

capabilities() -> MemoryCapabilities
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

The freshness anchor is event end, then event start, then creation time. Every field is validated
on construction: the two maxima are positive integers, `min_confidence` is in `[0, 1]`, and
`freshness` is a positive `timedelta`.

## Sections

A typed `MemoryKind` decides the section first; an untyped or generic record falls back to its
`MemoryType`.

| Section | Contents |
| --- | --- |
| `actors` | Kind `entity` |
| `facts` | Type `semantic`, except `entity` and `trait` |
| `episodes` | Type `episodic`, except `affect` |
| `procedures` | Type `procedural` |
| `affect` | Kind `affect` |
| `traits` | Kind `trait` |

Selection gives every non-empty section one slot in rank order before any section receives a
second, so a small budget still describes the whole scene instead of spending everything on the
top-ranked section. Remaining slots are filled by score. One oversized hit does not close the
bundle: a cheaper lower-ranked candidate can still fit. `omitted` counts every candidate that
passed the filters but did not fit, and `chars` is what the included hits cost.

`ContextBundle` also reports `occurred_from` and `occurred_until` over the included hits, `frames`
(the distinct spatial frame IDs, sorted), and `hits` (every included hit in rank order).

## Conflicts

Included hits that share a `lineage_id`, carry kind `state`, `relation`, or `trait`, and disagree
on `value` produce one `ContextConflict` each. Its `values` and `memory_ids` are aligned: each
distinct value is paired with the highest-ranked included memory asserting it. The compiler reports
the disagreement and leaves resolution to the caller or to a later correction.

## Rendered text

`render()` is deterministic: the same bundle always produces the same string. It emits a goal
heading, a one-line reading guide, the reference time, the budget it filled, one `##` heading per
non-empty section in the fixed order above, one line per hit, a conflicts section, and an omitted
trailer only when something was omitted.

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

## Capabilities

`capabilities()` returns a frozen `MemoryCapabilities`: `modalities` (the embedding contract this
instance accepts) plus the booleans `answer`, `transcribe`, `faces`, `describe_vision`, `form`,
`consolidate`, and `decay`. It reports the configured composition, so an agent surface can publish
what the instance can do before a caller tries it.

```python
capabilities = memory.capabilities()
if capabilities.answer:
    print(memory.ask("Where is the spare key?").answer)
```

`AsyncMemory` mirrors both methods. The command line exposes them as `mindbridge compile` and
`mindbridge capabilities`, locally and against `--url`; see [the CLI reference](api/cli.md).

Both operations reach agents over the transports as well: `POST /v1/context` with
`GET /v1/capabilities` on [REST](api/rest.md#endpoints), and the `compile_context` tool with the
capability view in the server instructions on [MCP](api/mcp.md#tools). Both are read-only views of
existing evidence.
