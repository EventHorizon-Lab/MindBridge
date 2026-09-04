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
| `max_chars` | `16000` | Evidence character ceiling, charged with the same cost function `ask()` uses: record text plus a per-modality text equivalent for each media asset. One image part is charged 2000, one audio part 4000, and one video part 12000, so a ceiling below 12000 omits every video record it ranks |
| `max_items` | `24` | Maximum included memories |
| `memory_types` | `None` | Keep only these `MemoryType` values; `None` keeps every type |
| `min_confidence` | `0.0` | Minimum typed confidence; a record with no typed context counts as `1.0` |
| `freshness` | `None` | Keep only memories anchored within this `timedelta` of `reference_at` |
| `max_latency_ms` | `None` | Deadline in milliseconds; optional stages are skipped once it passes |

The freshness anchor is event end, then event start, then creation time. Every field is validated
on construction: the two maxima and `max_latency_ms` are positive integers, `min_confidence` is in
`[0, 1]`, and `freshness` is a positive `timedelta`.

### What a character costs

There is one budget, not a text budget and a media budget. `max_chars` prices both against the
same scale, because what a bundle spends is what the downstream model is charged for, and one
grounded media part costs far more than the record text beside it. The prices are fixed:

| Charged | Characters |
| --- | --- |
| Record text | `len(hit.content)` |
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

### What `max_chars` does not bound

`max_chars` bounds evidence, the quantity `ContextBundle.chars` reports. It does not bound
`len(bundle.render())`, which adds a bounded frame around that evidence:

- a four-line header -- the goal, a 61-character reading guide, the reference time, and the
  budget line -- which is roughly 150 characters plus the goal text;
- a blank line and a two-character `##` heading, plus a space, for each non-empty section;
- per included memory, the `- [id]` prefix and the `(confidence 0.00)` suffix, with their
  separating spaces: 23 characters plus the id, plus a validity suffix when it carries one, and
  on an affect entry the provenance and hop marks [affect cues](#affect-cues) lists;
- the optional `## Conflicts`, `## Unknowns`, and `Omitted:` blocks.

A caller who ships `render()` should size against `len(bundle.render())`; the bundle does not
publish a second number for it.

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

The `affect` section carries `AffectCue`, not a plain hit; [affect cues](#affect-cues) below is
its contract. Every other section carries `SearchHit`.

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

## Affect cues

An affect entry asserts how somebody felt, which is the one section where the difference between
a model's guess and a person's own words changes what an agent may say. So `affect` carries
`AffectCue`: every `SearchHit` field, plus `event_ids`, and `render()` prints the provenance on
the line instead of leaving it in `context` for a caller who may never look.

| On the line | Read from | Meaning |
| --- | --- | --- |
| `basis` | `context.basis` | `user_statement` when somebody said it, `model_inference` when a model concluded it |
| `confidence` | `context.confidence` | The typed confidence, the same number `min_confidence` filters on |
| `cue` | `context.cue_modality` | Which modality the cue was read from; formation must name a modality the source actually carries |
| `valence` | `context.valence` | -1 through 1 |
| `arousal` | `context.arousal` | 0 through 1 |
| `from` | `context.evidence_ids` | The observations this cue cites |
| `co-occurring events` | `AffectCue.event_ids` | Events formed from at least one of those same observations |

`cue`, `valence`, `arousal`, and either ID list are omitted from the line when the record does
not carry them. Each list names at most eight IDs and then a `+N more` count, so an affect
line stays bounded by a constant even though these marks are not charged against
`max_chars`. Non-affect sections keep the plain `[id] content (confidence; validity)` line.

```text
## Affect
- [i9j0] the user sounded tense about the deadline (confidence 0.72; basis model_inference; cue audio; valence -0.20; arousal 0.80; from [e5f6]; co-occurring events [k1l2])
```

**`event_ids` is co-occurrence, not cause.** An event is reported for a cue exactly when the two
records share a `memory_evidence` source: both were derived from one thing that was observed.
Nothing in the bundle claims the event caused the feeling, or even that the cue is about the
event, and the compiler asserts no such edge across two different observations. That is why the
field is named for what it is rather than `about` or `triggered_by`.

The hop is one batched store read per `compile`, never one per cue. It runs after selection, for
the affect entries the budget actually bought rather than every candidate retrieval ranked, and
it is skipped entirely once `max_latency_ms` has passed -- optional work, like conflict
detection, and the same `stage_skipped` unknown reports both. It is subject to the same rules
retrieval hydrated the hits under: a retired version, a hidden assertion, a forgotten record, and
anything outside the requested `valid_at`/`known_at` window are all excluded, and `event_ids` is
sorted so two compilations of one store render the same line. Only IDs are
carried: the events are not fetched, they cost no characters and no item slot, and they are not
added to `episodes`. A co-derived event that appears in `episodes` earned that slot from its own
score. Read an event's text with `get()`.

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
Each line is one memory: [id] content (confidence; validity; for affect also basis, cue, valence, arousal, source and co-occurring event ids).
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
