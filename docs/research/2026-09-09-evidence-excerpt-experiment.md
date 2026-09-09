# Evidence excerpt experiment

Date: 2026-09-09

Status: isolated prototype. The accepted schema-17 product snapshot was not modified. No model,
public-QA, or benchmark-gold call informed this implementation.

## Question

Can compile() deliver an exact, inspectable part of a long raw text observation when the complete
record cannot fit, without changing retrieval keys, weakening derived-evidence closure, or
presenting the part as complete support?

## Contract

ContextBundle.hits remains a tuple of complete records. With the explicit
`allow_partial_sources=True` compile option, the additive ContextBundle.excerpts tuple contains
ContextExcerpt values selected only after the matching full raw record fails the existing item and
character fit. The option defaults to false and skips selector reads, preserving the accepted-v3
full-record-only result and query-failure behavior. One excerpt carries:

- source_memory_id and structured matched_index_id;
- partial display content;
- score, source creation and occurrence time, memory type, context, and place;
- a TextSpanSelector with parent-content and embedding-input SHA-256 values, recipe version, and
  ordered TextSpanPiece values;
- each piece's context or body role, half-open Unicode code-point offsets, exact source text, and
  SHA-256 value.

The renderer prints the source memory ID, offsets, occurrence/validity headers, selected text, and
“partial source; omitted text may qualify it”. It does not print the cryptographic digests or
matched_index_id; machine consumers use the structured fields for those checks. Overlapping pieces
merge and every prefix, gap, or suffix becomes a labelled ellipsis.

An excerpt counts as one item. Its heading, warning, source ID, offsets, text, and source headers
are charged against the existing grounding max_chars. Conflict, unknown, and omission diagnostics
retain their existing exclusion from that budget, so max_chars is not a bound on the complete
HTTP, SSE, or final rendered diagnostic payload.

## Eligibility and evidence boundary

Only a dense-matched embedding part written for a pure-text raw observation under schema 18 can
produce an excerpt. Aggregate keys, lexical-only matches, media, transcripts, descriptions,
derived records, and migrated schema-17 embeddings are ineligible. Scope, active and bitemporal
visibility, place, identity resolution, and consent are evaluated on the complete parent before
the excerpt is built. Current pure-text observations ordinarily carry no identity edge, so the
identity and consent rule mostly constrains actor enrichment rather than making their text
unreadable; the prototype does not expand that policy.

An excerpt is not inserted into memory_evidence, an evidence clause, reinforcement, or any durable
citation. It cannot satisfy a derived record's full-source requirement. Full closure is still
admitted atomically. If the same parent is already present as an excerpt, a later closure removes
it only after the full parent and every other required member fit; if that closure fails, the
excerpt remains and the derived assertion is withheld. A full parent and its excerpt never appear
together.

This matters when the selected span says a gate is open but omitted later text withdraws the
statement. The excerpt is useful retrieval context, but it is deliberately insufficient support
for a derived claim that the gate is open. There is no negation scanner or claim that the selected
span is semantically sufficient.

## Persistence and migration

Schema 18 adds:

- embedding_text_selectors with embedding ID, selector position, parent-content digest,
  embedding-input digest, and recipe;
- embedding_text_span_pieces with embedding ID, selector and piece positions, role, code-point
  start and end, and piece digest.

SQLite retains the authoritative parent text once in memory_records. A write verifies the parent
digest, every exact slice and piece digest, and the reconstructed contextual embedding-input
digest before it commits. Hydration repeats those checks against authoritative text; a mismatch
disables only the excerpt. Embedding deletion cascades selector rows.

Schema 17 to 18 migration is transactional and creates empty selector tables. It never guesses
legacy offsets and never re-embeds. Reopening is idempotent. A partial or incompatible table shape
is refused after checking column types and primary keys, selector-to-embedding and composite-piece
foreign keys, check constraints, and the ordered piece index. Failure rolls back any migration
DDL and leaves the schema version and prior tables unchanged. A missing Zvec index rebuilds from
the existing SQLite embeddings and does not call the model or synthesize selectors. An older
schema-17 SDK cannot open the upgraded store; there is no automatic downgrade or physical-delete
recovery.

## Retrieval and benchmark surface

The winning dense part is stable: highest dense relevance, then lexicographically lowest embedding
ID. Parent score, order, ordinary search results, and search traces remain unchanged. If that
winner is an aggregate or otherwise has no verified selector, no excerpt is emitted. Selector
reads are one bounded batched SQLite read over the already scoped and consent-processed candidate
set.

The scratch evaluation schema is version 16. Its `compile_allow_partial_sources` setting defaults
to false and must be explicitly true for the excerpt arm. Compile-arm samples retain full
memory_ids and evidence and add excerpt_source_ids plus excerpt_evidence. The latter is deliberately scored only
by an extractor that opts in; it must not credit a whole gold parent merely because one span was
delivered. compile_bundle_items counts full hits plus excerpts and compile_bundle_excerpts reports
the partial count. Generation receives the rendered excerpt as text, while native-media binding
continues to use complete hits only.

## Falsification results

The frozen accepted-v3 module and the schema-18 path were run in separate physical stores with
bytecode disabled. The paired receipt covers repeated-key deduplication, deterministic maximum-key
sampling, combining characters, emoji, CRLF and strip boundaries, an oversized aggregate fallback,
and final object_part renumbering. The record content and ID, every embedding request byte, stored
embedding ID and part, and default full-record compiled rendering were exactly equal. Only schema
version and selector rows differed.

Focused tests also cover excerpts-only delivery, roomy full-record delivery, parent mutual
exclusion, the adversarial omitted withdrawal, failed derived closure, place scope, deletion,
parent and selector tampering, schema migration and partial-shape rollback, index rebuild without
embedding, and REST, MCP, and CLI serialization.

## Integration revision v2

Independent review of the first frozen prototype found that excerpt-only delivery could still emit
a `SECTION_EMPTY` diagnostic saying no memory of its type was included. The v2 integration counts
an excerpt's parent memory type only when deciding whether that type is absent. It does not place
the excerpt into a full-hit section or let it satisfy evidence closure. The same review reproduced
schema-17 migration accepting pre-existing exact-column selector tables without their required
constraints; v2 performs the structural checks described above before advancing the version.

The v2 scratch integration also adds the independently frozen compact-ID prototype as an opt-in
Python presentation. Memory symbols carry `full`, `partial`, or `reference_only` coverage. Partial
symbols retain their exact selector; identities are reference-only because their actor lines do
not deliver a memory record. `resolve()` only decodes request-local aliases. The stricter
`resolve_citation()` rejects identity and reference-only symbols and returns a typed full or
partial citation; a partial citation cannot enter durable support or compiler closure.

Compaction aliases the excerpt parent ID but not its matched index ID, offsets, or digest fields.
It happens after compilation, does not repack the bundle, and does not change grounding budget
accounting. Stored content and goals remain verbatim. Literal alias-shaped text can therefore be
ambiguous, and a prompt containing a full runtime memory ID need not reveal that its matching
anchor is now `mN`. Reversibility is limited to the presentation text plus its exact symbol table;
no arbitrary-prompt semantic or answer-quality equivalence is claimed.

## Integration revision v3

The second independent review accepted the selector, migration, default-off, and compact citation
invariants but found that REST and MCP coerced numeric and string lookalikes into the partial-source
boolean. The final revision makes this opt-in strict on both transports and tests actual REST and
MCP calls with `1` and `"true"`: each returns a validation error before invoking `Memory.compile`.
The direct SDK already rejected non-boolean values. No selector, compilation, budget, storage, or
benchmark algorithm changed in this revision.

## Limits

- Code-point offsets are exact for the stored Python string but may split a grapheme cluster.
- Offsets are representation-bound. The parent digest prevents applying them to normalized,
  edited, or otherwise different text; this prototype does not claim W3C Annotation Model
  conformance.
- Legacy and explicit embedding-recipe migration rows remain selector-free. Reconstructing their
  offsets would manufacture provenance.
- An aggregate winning the dense route yields no excerpt even if another stored chunk has a
  selector. Expanding the trace or changing parent ranking was outside this experiment.
- Partial raw text can omit a qualifier, contradiction, retraction, speaker turn, or object
  boundary. The type and warning expose that risk but cannot solve it.
- The prototype establishes structural integrity and compatibility on synthetic fixtures. It
  makes no downstream accuracy, token-saving, or quality claim.
