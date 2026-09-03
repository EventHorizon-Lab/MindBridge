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

There is one retrieval round and no second one. Everything the bundle counts -- `omitted`, and
each `budget_excluded` tally -- is therefore a statement about that window, not about the store:
`omitted == 0` means nothing inside the window was dropped, not that nothing else exists. When the
window came back full and the bundle still lost evidence, `unknowns` carries a
`candidates_exhausted` entry saying so, so the difference is visible rather than inferred.

## Budget

| Field | Default | Meaning |
| --- | --- | --- |
| `max_chars` | `16000` | Rendered-evidence character ceiling: the header, each section heading, each memory's whole rendered line, and a per-modality text equivalent for each media asset. One image part is charged 2000, one audio part 4000, and one video part 12000, so a ceiling below 12000 omits every video record it ranks |
| `max_items` | `24` | Maximum included memories |
| `max_media_items` | `None` | Maximum grounded media parts; `0` compiles a text-only bundle and `None` lets `max_chars` alone decide |
| `memory_types` | `None` | Keep only these `MemoryType` values; `None` keeps every type |
| `min_confidence` | `0.0` | Minimum typed confidence; a record with no typed context counts as `1.0` |
| `freshness` | `None` | Keep only memories anchored within this `timedelta` of `reference_at` |
| `max_latency_ms` | `None` | Deadline in milliseconds; optional stages are skipped once it passes |

The freshness anchor is event end, then event start, then creation time. Every field is validated
on construction: `max_chars`, `max_items` and `max_latency_ms` are positive integers,
`max_media_items` is a non-negative one, `min_confidence` is in `[0, 1]`, and `freshness` is a
positive `timedelta`.

### What a character costs

`max_chars` prices text and media against the same scale, because what a bundle spends is what
the downstream model is charged for, and one grounded media part costs far more than the record
text beside it. `max_media_items` is the second knob for the cases a shared scale cannot express:
"at most two images" and "no media at all" are statements about parts, not about characters, and a
character ceiling can only approximate them. The prices are fixed:

| Charged | Characters |
| --- | --- |
| One memory's rendered line | `len(render_hit_line(hit)) + 1` -- the `- [id]` frame, the squeezed record text, and the confidence and validity suffix |
| The header | its four rendered lines, about 150 characters plus the goal |
| One section heading | its blank line, `## `, and the name |
| One `image` asset | 2000 |
| One `audio` asset | 4000 |
| One `video` asset | 12000 |
| One asset whose modality did not resolve | 4000 |

They are measured, not derived: against this stack an image part came out near five hundred model
tokens and a ten-second video part near three thousand, at the usual four characters per token.
They are coarse on purpose -- the budget is the caller's knob, not these constants.

The default `max_chars` follows from the largest of them. A video memory costs at least 12000, so
a default below that could never buy one, and every video memory would be reported as omitted no
matter how well it ranked. The default is set above the most expensive single part with room for
its record text and a cheaper second part.

### What `min_confidence` compares

`min_confidence` filters a *typed* confidence: the confidence a `FormationBackend` or
`ConsolidationBackend` attached when it inferred something. A record with no typed context has no
inference to discount -- it is a raw observation, evidence in its own right -- so it counts as
`1.0` and passes every bound, including `min_confidence=1.0`. `render()` prints the same `1.00`,
so the filter and the rendered line never disagree. Raising `min_confidence` is therefore a way to
demand better inferences, not a way to exclude observations; filter those by `memory_types`.

### What `max_chars` does and does not bound

`max_chars` bounds the *rendered* evidence, the quantity `ContextBundle.chars` reports and the
quantity the selection charges: the header, every section heading, and every memory line with its
frame. A bundle that reports no compilation diagnostics therefore satisfies
`len(bundle.render()) <= bundle.chars <= max_chars`, and a caller shipping `render()` no longer
has to size against it themselves.

`chars` is an upper bound rather than an exact length -- media parts are charged their text
equivalent, which is far above the zero characters they render as, and the budget line is priced
at its widest -- so the inequality can be slack, never violated.

Four things `render()` appends are outside the ceiling, all of them explanations rather than
grounding: the `## Conflicts` block, the `## Unknowns` block, the `Omitted:` trailer, and the
provisional-actor lines. Suppressing any of them to fit a budget would make a thin bundle look
like an empty store, which is the failure the unknowns exist to prevent, and a provisional actor
is a person the evidence already paid for rather than evidence of its own.

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
| `actors` | Kind `entity`, plus one `ProvisionalActor` per unnamed person in the included evidence |
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

`memory_types` filters the record's own `MemoryType`, and a kind is stored as a type: kind `event`
and `affect` as `episodic`, `response_policy` as `procedural`, everything else as `semantic`. A
section is therefore reachable only for the types its kinds are stored as, so a `{semantic}`
request cannot fill `affect`, `episodes`, or `procedures` however well their records rank. Each
such section is named in `unknowns` as a `section_empty` entry saying which filter emptied it,
rather than leaving the reader to map a dropped type back to the section they lost.

The bundle also carries no person link. Identity lives asset-keyed in the speech and face tables,
reachable through `speech()` and `faces()`, and no `identity_id` exists on a memory, a hit, or a
`MemoryContext`. Cross-modal person linkage in a bundle needs that edge; it is a known gap rather
than a configuration mistake.

### How slots are shared

**Rank decides the top half of `max_items`; the bottom half gives one slot to each section the
top half missed, and only when it is large enough to seat every section the candidates span --
otherwise rank decides the whole bundle.**

Diversity is worth having, but not at the price of rank readability. Reserving a floor slot per
section at a small budget lets a rank-fifty trait evict a rank-two episode, and a bundle that no
longer reads by score is worse than one missing a section. So with eight sections spanned and
`max_items=3` the selection is exactly the top three by rank; the default `max_items=24` leaves
twelve slots for the floor round and every spanned section still gets one.

Remaining slots are filled by score. One oversized hit does not close the bundle: a cheaper
lower-ranked candidate can still fit. `omitted` counts every candidate that passed the filters
but did not fit, and `chars` is what the included hits cost.

This split is a default, not a measured optimum. Gate 4 of [the Context OS plan](context-os.md)
still owes a downstream-utility measurement against the no-memory, full-context, and
retrieval-only baselines; that measurement decides whether the halving point is right or whether
the floor round should go entirely.

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
| `section_empty` | A bundle section a `memory_types` request left empty, naming the section and whether the filter excluded every type it carries or the ranking simply reached none |
| `scope_empty` | A `scope` was supplied and retrieval matched nothing; the entry names the bounds |
| `modality_unsupported` | The goal carries media this composition's embedder cannot search |
| `stage_skipped` | An optional stage the `max_latency_ms` deadline skipped |
| `candidates_exhausted` | Retrieval filled its candidate window and the bundle still lost evidence, so the counts bound the window and not the store |

## Provisional actors

A recognized person whom no visible naming assertion names is reported in `actors` as a
`ProvisionalActor` carrying `identity_id` and the `memory_ids` of the included evidence that
observed them, sorted by identity. An unnamed person present in the room is not a person missing
from context: what an agent may say depends on knowing they are there and that nobody has named
them.

A provisional actor is not a hit. It never appears in `hits`, it takes no item slot, it costs no
characters, and it is dropped when the evidence that observed the person did not make it into the
bundle -- the bundle never reports somebody the reader cannot see the evidence for. `render()`
prints one plainly labelled line per entry, after the ranked actors.

## Conflicts

Candidates that share a `lineage_id`, carry kind `state`, `relation`, or `trait`, and disagree on
`value` produce one `ContextConflict` each. Its `values` and `memory_ids` are aligned: each
distinct value is paired with the highest-ranked candidate asserting it. The compiler reports the
disagreement and leaves resolution to the caller or to a later correction.

Detection reads every candidate the filters kept, not only the ones the budget bought, so running
out of slots cannot make a disagreement disappear. A `memory_id` in a conflict may therefore name
a memory that is not in `hits`; `render()` marks it `not included`, because that memory has no
line of its own anywhere in the text. At least one included memory must take part, so a lineage
the bundle says nothing about is not reported as its disagreement.

A candidate a *filter* removed -- `memory_types`, `min_confidence`, or `freshness` -- is not
compared, and neither is a superseded version bitemporal filtering already excluded. `conflicts`
is a statement about what this request admitted, not a belief-revision history; read
`memory_versions` through the control plane for that.

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
Budget: 132/16000 chars, 3/24 items

## Facts
- [a1b2] the spare key is in the blue toolbox (confidence 1.00)
- [c3d4] Ana works from Berlin (confidence 0.90; valid from 2026-08-01T00:00:00+00:00)

## Episodes
- [e5f6] we walked to the harbour (confidence 1.00)

## Conflicts
- ana location: "berlin" [c3d4] vs "paris" [g7h8, not included]

## Unknowns
- budget_excluded: 4 candidates did not fit 24 items and 16000 chars

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
