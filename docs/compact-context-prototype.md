# Compact context presentation prototype

`ContextBundle.render()` already exists and remains unchanged. This prototype adds the optional
Python-only `ContextBundle.compact()` method. It returns a frozen `ContextPresentation` containing
the presentation text, its exact character count, and an ordered tuple of typed `ContextSymbol`
values.

The compact text uses request-local `mN` symbols for memory IDs and `iN` symbols for identity IDs.
The returned table retains every stable ID, its `full`, `partial`, or `reference_only` coverage,
and every structural role in which the formatter used it. Identity symbols are always
`reference_only` because an actor line does not deliver a `MemoryRecord`. References exposed by an
evidence list, affect event hop, naming line, or conflict receive symbols even when the referenced
record was not packed. A partial memory symbol carries the exact `TextSpanSelector` already exposed
by its `ContextExcerpt`; selector digests and the matched index ID remain structured fields rather
than being repeated in model text.

The formatter walks typed bundle fields. It does not search or replace stored content, conflict
values, or diagnostic prose. A user's literal `m1` or 64-character hexadecimal string therefore
stays unchanged. Selection, evidence closure, `ContextBudget` accounting, and `ContextBundle.chars`
also stay unchanged; `ContextPresentation.chars` reports the shorter presentation size after the
same bundle was selected.

An adapter must retain the exact `ContextPresentation` paired with its request and call
`presentation.resolve(symbol, namespace=...)` to decode an alias. That method is only an identity
mapping: it proves neither delivery, citation eligibility, support, scope permission, nor which
textual occurrence caused a model to return the token. `resolve_citation(symbol)` is the stricter
path for citations. It rejects identity and `reference_only` symbols, returns a typed full citation
without a selector, and returns a typed partial citation with its exact selector. A partial citation
never satisfies a full-record evidence dependency or becomes an evidence-clause member.

Symbols are not registered globally and are never valid arguments to durable methods such as
`get`, `delete`, or `reinforce`. Two presentations can both contain `m1` with different meanings.
A bare model string does not prove which presentation produced it, so the caller has to preserve
that pairing. Stored source text and the goal are left verbatim. A literal `[m1]` can therefore
coexist with the structural alias, and a goal containing a full stable memory ID does not teach the
model that the matching anchor became `[m1]`. “Lossless” here means that text plus its exact symbol
table is reversible. It does not promise semantic equivalence for arbitrary ID-addressed prompts.
Unknown symbols and namespace mismatches are rejected.

The existing `render()`, `hits`, and `document()` output remain stable. REST, MCP, CLI, and product
answer generation do not expose or select the prototype surface.

When combined with the excerpt experiment, `compact()` renders the selected partial-source lines,
aliases only their parent source memory IDs, and counts full hits plus excerpts in its displayed item
count. It does not alias `matched_index_id`, selector digests, or offsets. Compilation still charges
the original full-hit or excerpt rendering before compaction, so `ContextPresentation.chars` is a
post-selection transport measure and cannot increase the admitted bundle. A full hit and excerpt of
the same parent remain mutually exclusive.

## Frozen conv30 measurement

The measurement recompiled all 72 frozen schema17 witness questions on a fresh physical clone with
the exact recorded query embedding responses: 72 cache hits, zero misses, zero network calls, and
zero generation calls. Every legacy `render()` string matched the previously frozen capture. The
optional presentation reduced 581,016 legacy characters to 435,536, saving 145,480 characters
(25.04%) after selection. It exposed 1,762 symbols: 1,728 delivered IDs and 34 references whose
records were not delivered. Every request retained the same hits and grounding budget; no item was
repacked.
