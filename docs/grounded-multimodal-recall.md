# Grounded multimodal recall

The existing public SDK can combine native retrieval with bounded semantic queries. Enable
`settings.recall_planning` and `settings.recall_query_expansion` explicitly. A single `ask` shares
at most two additional semantic queries across all rounds. Repeated queries reuse their results;
successful routes remain available when a later query depends on an earlier bridge fact.

The original `limit` window remains the grounding floor. Supplemental semantic routes take turns
offering novel rows before the replaceable original tail spends the remaining budget. Required
rows are reserved before optional rows are priced. Predicate row/media limits and omission
reporting remain enforced. A complete predicate read is not automatically the question's complete
answer set: anchors and neighbors must still satisfy the question and its counting unit.

When expansion is enabled, a strict abstention can trigger another bounded planning round.
The planner receives at most 6000 characters of untrusted evidence summaries, with at most 1500
characters per record and explicit truncation/omission counts. Streaming releases only the final
selected round. Ordinary planning retains its existing policy when expansion is disabled.

Sequence plans order the entire selected union after packing, using event time and SQLite
insertion order for ties. Similarity rank, record IDs and creation timestamps do not substitute
for the source's same-time order. Ordering does not add records or establish a complete history.

## Optional query vision and answer style

`query_vision` defaults to `False`. With an image-capable vision backend configured, it adds the
query image's description and visible text as independent search routes while retaining native
image retrieval. Descriptions live only inside the operation. They create no memory, asset,
identity or durable description row. Description/embedding failures preserve native routes;
text-only queries add no vision work. Enabling it can add vision and embedding latency.

`generation.answer_style` accepts `default` or `concise`, default `default`. Concise answers
include only requested entities, relations, values or list items, and only output a date or
duration when requested. Both existing answer policies and their insufficient-evidence behavior
remain in force. Stored instructions are untrusted even when stored beside useful facts.

## Request-local media transport

OpenAI embedding, generation, formation and vision configurations accept `image_max_pixels`,
an integer from 1 through 64,000,000 or `None` (default). Explicit bounds inspect image headers,
verify source integrity, and re-encode/resize only when the pixel or provider byte limit requires
it. Originals, identifiers and SQLite records are unchanged. Repeated source assets are converted
once per request; actual transmitted parts, including repeats where required, are priced.

Explicit image transport recipes participate in embedding, formation and caption identities.
Changing a store-pinned embedding recipe requires a fresh directory and re-ingestion. A derived
Zvec rebuild alone never recomputes stored vectors. With no pixel bound, the previous image
transport behavior remains in place.

`vision.video_frames` accepts integers from 2 through 64, default 4. Non-default frame counts
participate in the caption recipe. Configure `vision.modalities: [image, video]` explicitly if
video captions are wanted; a frame setting alone does not enable video description or backfill
an existing corpus.

Oversized videos can be represented as bounded ordered stills when the backend accepts images.
Media evidence retains its original record `source_modality`, attachment modalities, and explicit
sampling/omission labels. Sampled stills are views of one video, not independent photos; motion,
audio and exact timing are not supplied. Image resizing can lose fine detail. Valid source SHA,
decode bounds, total text bytes and actual inline-media bytes remain enforced. Byte and pixel
limits do not prove that every mixed request fits a provider's token window. Plain-text records
keep their existing payload shape without an extra source-modality field.

## Derived evidence

Formation remains opt-in. A proposal that includes its canonical fact as a member of its own
support conjunction is rejected whole; removing only that member would authorize a different
claim. Independent valid proposals still proceed. This guard does not claim better semantic
state formation or enable formation/consolidation by default.

See the [Python API](api/python-sdk.md), [configuration](configuration.md), and
[quality evidence](research/2026-10-29-grounded-recall-quality.md).
