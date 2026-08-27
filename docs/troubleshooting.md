# Troubleshooting

Start with the public exception or REST/MCP error code, then inspect the matching boundary.
MindBridge fails early rather than opening unsafe durable state or silently dropping media.

## Data directory is already in use

Symptom: constructing `Memory` raises `StorageError` saying the directory is already in use.

1. Resolve `data_dir` to an absolute path and inspect every process or test fixture.
2. Close the existing `Memory` before opening another owner.
3. Give concurrent applications and benchmark cases different directories.
4. Do not delete `.mindbridge.lock` while an owner may still run.

The file can remain after shutdown. Ownership is the operating-system lock, not file presence.

## Store metadata mismatch

Symptom: startup raises `StorageError` mentioning an embedding, transcription, face, or index
metadata key.

The active embedding model, vector-space ID, dimension, transcription space, face space, or index
recipe differs from the values that created the store. Restore the original configuration, or
create a new directory and re-ingest. Editing SQLite metadata would mix incompatible vectors or
cached analyses and is unsupported.

Local adapters derive a new `space_id` when adapter, model, immutable revision, dimension, or input
recipe changes. Use that adapter with a new data directory and re-encode source content. For an
explicit combined HTTP backend, set a new `MINDBRIDGE_EMBEDDING_SPACE` as well.
Likewise, change `MINDBRIDGE_TRANSCRIPTION_SPACE` and use a new directory whenever the ASR model,
preprocessing, language, prompt, or decoding recipe can change transcript text. Do not reuse one
directory merely because the transcription endpoint accepts the new model.

## Unsupported SQLite schema

Symptom: startup reports an unsupported version, unversioned schema, or missing tables.

Confirm the path belongs to this MindBridge release and was not pointed at an unrelated SQLite
database. Restore a known-good complete backup. Do not edit `PRAGMA user_version` or manually
create tables. Supported v1 through v4 local-schema migrations run automatically; old PostgreSQL
data does not.

## Model operation failed

Python raises `ModelError`; REST returns 502; MCP returns `model_error`.

Check the failing operation independently:

| Operation | URL/key variables | Route |
| --- | --- | --- |
| Default embedding | local Sentence Transformers runtime | `encode_query` / `encode_document` |
| Combined HTTP embedding | `MINDBRIDGE_EMBEDDING_*` | `/v1/embeddings` |
| Generation | `MINDBRIDGE_GENERATION_*` | `/v1/chat/completions` |
| Transcription | `MINDBRIDGE_TRANSCRIPTION_*` | `/v1/audio/transcriptions` |
| Face recognition | local InsightFace/ONNX Runtime | `FaceAnalysis.get` |

HTTP operation-specific variables fall back to `OPENAI_API_KEY` and `OPENAI_BASE_URL`. For local
embedding, confirm `mindbridge[local]` is installed, the immutable model revision is available,
device memory is sufficient, media decoders are installed, and the requested dimension is native
or advertised as Matryoshka-trained. All backends must return one finite vector of the declared
dimension per input.

For face analysis, confirm `mindbridge[face]` is installed, the requested InsightFace pack is
available, OpenCV decodes the media, and the requested ONNX execution provider is active. Low
detection score, embedding norm, or face size is filtered rather than enrolled; calibrate those
controls on representative inputs.

MindBridge sanitizes provider response bodies and credentials. Reproduce the request directly
against a non-production endpoint when provider diagnostics are required.

If the error says inline model media exceeds 64 MiB, the aggregate raw media for one built-in
embedding or generation call exceeded the `data` transport boundary. A large video should use
`file` transport with a trusted co-located backend, or a custom backend that streams/uploads files.
Splitting it into multiple memories changes retrieval semantics and is not an automatic feature.

## Configured model does not support a modality

Symptom: `ModelError` says embedding, generation, or transcription does not support an input
modality.

The generic Sentence Transformers adapter discovers embedding support from the model; it cannot be
enabled with an environment variable. For HTTP generation/transcription or a combined HTTP
embedding backend, capability lists describe the endpoint rather than enabling a model feature.

Audio fallback requires transcription support. When embedding/generation lacks audio, MindBridge
uses transcript text and retains supported image/video parts. It will not remove an unsupported
image or video to force a fallback. Choose a VLM/omni backend or a different input.

## Remote media URL is rejected

Check all of these conditions:

- The URL uses HTTPS and has no credentials or fragment.
- Its exact hostname is listed in `MINDBRIDGE_ALLOWED_URL_HOSTS`.
- Every redirect remains on an allowed hostname.
- DNS resolves only to public addresses.
- The connection can reach the validated public IP while using the original hostname for TLS SNI.
- Response `Content-Type` exactly matches a concrete MIME hint, or matches the requested media
  family when using `image/*`, `video/*`, or `audio/*`.
- Content length and streamed bytes remain within the ingestion limit.

The default host allowlist is empty. `localhost`, private IPs, link-local addresses, wildcard
hosts, and `file:` URLs are intentionally rejected. Prefer Python `Path` for local media.

## Local media is rejected or missing

`Path` must identify a regular readable file. MindBridge avoids following a final symlink where the
platform supports that open flag. If MIME type cannot be inferred from a common suffix, use
`Blob(data, media_type, name)` or `URL` with an explicit media type.

After ingestion, records refer to the immutable copy under `data_dir/assets`, not the source path.
If a returned record exists but its CAS file is missing or has the wrong size, restore the complete
data directory from backup. Reindex cannot reconstruct original media bytes.

If identical bytes were added under different filenames, seeing the first authoritative non-empty
CAS name on every returned reference is expected. Names belong to the digest, not to each memory;
use record text or metadata for per-memory labels.

On POSIX, startup forces the top-level `data_dir` mode to `0700`. If a group account loses access,
run the service under the owning account or choose a correctly owned directory; do not depend on a
group-writable shared store.

## Index unavailable

Python raises `IndexUnavailableError`; REST returns 503.

Confirm the process can write `data_dir/zvec`, the filesystem has free space, and no other process
owns the directory. If derived state is damaged:

1. Stop MindBridge.
2. Back up the complete data directory, including SQLite and assets.
3. Move `zvec/` aside; do not alter SQLite or `assets/`.
4. Start MindBridge once and allow bounded rebuild from stored embeddings.
5. Verify known text and media searches before discarding the old index copy.

Do not remove `zvec/` while an instance is running.

## Add raised after the model succeeded

An add can commit authoritative SQLite/CAS state and then fail while flushing Zvec. The call raises
because the record is not yet searchable, but its ID and embedding remain durable. Repair the index
condition and retry the same logical input. Pending work is replayed without another historical
embedding call.

## A deleted result appears in Zvec

Public search hydrates candidates from SQLite and drops stale IDs, so a deleted record must not be
returned even if derived state is temporarily stale. If it appears through the public API, capture
a minimal reproduction and MindBridge version; do not query Zvec directly as a product read path.

## Invalid cursor

List cursors are opaque keyset values. Pass `Page.next_cursor` back unchanged. A trimmed, decoded,
re-encoded, or constructed cursor raises `ValidationError` or REST 422.

## REST request fails before reaching Memory

- `401 authentication_error`: send `Authorization: Bearer <MINDBRIDGE_API_KEY>`.
- `413 request_too_large`: keep JSON below 8 MiB or ingest large media through Python `Path` or an
  allowed HTTPS URL. The later built-in `data` model call still has a separate 64 MiB raw aggregate
  ceiling.
- `422 validation_error`: inspect `issues`; local paths, unknown fields, ambiguous media sources,
  invalid base64, and naive datetimes are rejected.

The inbound service key is not a model provider key. `/healthz` requires neither.

## Multiple server workers fail

Run exactly one Uvicorn/Gunicorn process for a data directory. Separate processes cannot share the
embedded Zvec/CAS boundary. See [deployment](deployment.md#why-one-worker-is-mandatory).

## Ask has no useful evidence

Inspect `AnswerResult.hits` or REST/MCP `hits`. Confirm relevant memories were stored, the query
capabilities match its modality, and the embedding space is correct. Then improve the input text or
media. There is no server-side metadata filter, chunking control, per-asset vector, or learned
reranker to tune in the current release. For a temporal query, also inspect `occurred_at`,
`memory_type`, and the effective `reference_at`.

For backup and recovery, see [operations](operations.md).
