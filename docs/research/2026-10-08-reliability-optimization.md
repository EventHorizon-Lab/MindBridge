# Memory reliability changes, 2026-10-08

This patch implements deterministic repairs and the instrumentation needed to evaluate the
proposed optimizations. It does not report new benchmark accuracy or promote the historical
proof-search prototypes into the default compiler.

## Implemented behavior

- Formation has an opt-in, read-only recent history window with separate target and history
  aliases. Whole-record character and row limits expose truncation. Validity, retirement and
  event time travel with background text; media is not reopened for history selection.
- Formation proposals must cite the current target and only allowed batch or private-history
  records. Cited evidence is checked for forgetting, latest-version visibility and retirement
  inside the SQLite commit.
  A stale commit fails without derived writes; the original observation stays durable. Joint
  evidence retains only agreed place and whole metadata values.
- The existing one-shot JSON repair preserves string requests as strings rather than splitting
  them into characters. Its instruction requests envelope repair without inventing facts or
  additional witnesses. Model schema requirements and retry count are unchanged.
- Answer evidence includes a source-qualified view of typed state, trait and host policy
  candidates with subject, predicate, value and validity. Overlapping, competing values are
  flagged without choosing the newest. Conditions and exceptions remain in referenced text;
  the view neither interprets arbitrary prose nor grants consent or response-policy authority.
- Corroboration groups proofs by top-level AND clause before independent-pair comparisons.
  Alternatives of one clause cannot corroborate it and no longer consume the comparison budget.
  Complete provenance, capture independence, confidence and truncation semantics are preserved.
- Per-sample telemetry records the last prepared answer request and media fallback. A dedicated
  ICM rejudge command preserves old predictions and scores while applying the corrected judge
  protocol. Original question text is logged separately from formatted generation prompts.

See [configuration](../configuration.md#automatic-memory-formation),
[SDK contracts](../api/python-sdk.md), and
[rejudging](../benchmarking.md#rejudge-saved-icm-predictions) for use and compatibility details.

## Follow-up repairs

A wide OR branch could fill the bounded cache with alternatives of one top-level clause before
a distinct clause arrived. Cache admission now replaces a redundant alternative to retain a
previously unseen top clause when space can be recovered. All indexes contain only retained
proofs, and admission does constant-time bookkeeping. Eviction still marks the result truncated;
it does not guarantee an exact result, the best confidence, or invariance under renaming IDs.

When the work budget expires during a join, completed joins survive. Remaining antecedents must
still be joined, so an unfinished AND prefix cannot become evidence. An independent exhaustive
test enumerates clause assignments on small graphs to check that bounded support and confidence
never exceed the exact result. SDK regressions cover wide OR history, shared captures and rollback.

Formation now checks the latest recorded evidence version inside its commit transaction. A trait
that becomes hidden after an independent support is rolled back cannot seed a new automatic
claim. This leaves host reinforcement of hidden targets available and permits past events as
evidence; visibility is separate from temporal validity. Failure uses the existing formation
model error and preserves the newly captured observation. No additional public or disk contracts
change in these follow-up repairs.

## Compatibility

`Memory` and `MemoryConfig` gain two history settings; `FormationInput` gains defaulted `history`
and `history_truncated` fields. The proposal citation contract now accepts that input's active
history. History stays off by default. `AsyncMemory` forwards the same construction settings.
The OpenAI formation prompt changes its recipe fingerprint; existing formation completion marks
belong to their original recipe. No disk schema, dependencies, REST endpoints or MCP tools change.
Evaluation output schema advances to `18` and the runner recipe to `v17`; prior answer caches are
separated from the new evidence prompt.

## Remaining experimental decisions

The new tests establish local invariants; they do not establish the model's ability to interpret
natural-language retractions or the quality of a recent history window. Compare no history,
recent history, semantic history, and support/counterevidence completion before enabling a new
default. Explicit retirement and forgetting checks do not turn ordinary withdrawal prose into a
control-plane operation.

The per-node proof cache and global propagation limit still make support a conservative lower
bound under truncation. Demand-directed certificates and a separately verifiable selected-proof
protocol remain research work; persistent `evidence_ids` are never rewritten to make a bundle fit.
Preference improvements need paired, held-out evaluation. Media rereading remains on the existing
bounded generation path; an automatic diagnostic reread policy requires evidence that it improves
detail accuracy without increasing identity errors and cost. The rejudge tool has offline tests;
the historical ICM model scoring run has not been repeated by this patch.
