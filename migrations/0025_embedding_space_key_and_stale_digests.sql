BEGIN;

-- Two pieces of fallout from migration 0021, which dropped the revision columns.

-- 1. `space_id` is missing from the key that replaced the revision-keyed one.
--
-- 0021 replaced UNIQUE (tenant_id, object_type, object_id, model_id, model_revision, task)
-- with the same key minus the revision. But the revision was the component that let one
-- object hold two vectors at once, and that is a documented operational state, not an
-- accident: `docs/configuration.md` and `docs/troubleshooting.md` both tell an operator that
-- vectors in several spaces are accepted while a re-embedding is in progress, and
-- `unreachable_embedded_object_types` exists to report when one is only half done.
--
-- Under the narrowed key that workflow cannot run. Setting a new MINDBRIDGE_EMBEDDING_SPACE_ID
-- and re-embedding under the same model_id collides on the key, so the INSERT's
-- ON CONFLICT DO NOTHING fires, the follow-up SELECT filters on the new space_id and matches
-- nothing, and the write fails with "embedding conflict could not be resolved" -- an error
-- naming a conflict it could not see rather than the constraint that caused it. Adding
-- space_id restores the coexistence the revision used to provide, and does it with the column
-- that actually names a search space.

-- The key is being widened, so nothing that fits the current key can collide under the new
-- one and this dedup cannot delete a row on a database already migrated past 0021. It is here
-- for the database that has not been: 0021's own dedup ran before its narrower key existed,
-- so a row pair that differed only by a revision this migration now distinguishes by space_id
-- would survive 0021 and land here. Keeping the earliest is the same choice 0021 made, and for
-- the same reason: a re-encode wrote the later vector, and `upsert_embedding` treats an
-- existing vector for an ID as the authoritative one.
--
-- FORCE ROW LEVEL SECURITY applies the tenant policy to the table owner too and no tenant is
-- set during a migration, so an owner who is not a superuser would match no rows at all.
-- Lifted for the statement exactly as 0007 and 0021 lift it for theirs.
ALTER TABLE embeddings NO FORCE ROW LEVEL SECURITY;

DELETE FROM embeddings AS duplicate
USING embeddings AS surviving
WHERE duplicate.tenant_id = surviving.tenant_id
  AND duplicate.object_type = surviving.object_type
  AND duplicate.object_id = surviving.object_id
  AND duplicate.model_id = surviving.model_id
  AND duplicate.space_id = surviving.space_id
  AND duplicate.task = surviving.task
  AND (duplicate.created_at, duplicate.embedding_id)
      > (surviving.created_at, surviving.embedding_id);

ALTER TABLE embeddings FORCE ROW LEVEL SECURITY;

ALTER TABLE embeddings DROP CONSTRAINT embeddings_object_model_task_key;

ALTER TABLE embeddings
    ADD CONSTRAINT embeddings_object_model_space_task_key
    UNIQUE (tenant_id, object_type, object_id, model_id, space_id, task);

-- 2. Idempotency digests that can never match again.
--
-- `_request_digest` hashes the whole request, so removing a field from `ObserveRequest`
-- changed the digest of a request whose bytes did not change. `Observation.idempotency_key`
-- hashes only (tenant_id, device_id, boot_id, sequence), so the key is stable while its
-- digest moved: an edge device retrying an observation the server already accepted now fails
-- `stored_digest != content_digest` and gets 409 "idempotency key already stores different
-- content", forever, for a byte-identical resend -- which is the exact case DUPLICATE exists
-- to serve.
--
-- The digest cannot be recomputed here. It is a sha256 over Python's `json.dumps` of the
-- request, and reproducing that byte-for-byte in SQL would make the escaping rules of one
-- serializer load-bearing in a migration. Dropping the claims is the honest repair: a claim
-- is a cache that lets a retry skip work it already did, so losing one costs a reprocess, and
-- the reprocess is idempotent because `write_observation` still dedupes on the derived
-- `observation_id` and returns DUPLICATE. A wrong 409 is not recoverable; a redundant
-- reprocess is.
--
-- Scoped to 'observe' because that is the only operation whose payload carried the removed
-- field -- `identity_observations` exists on `ObserveRequest` and on nothing else -- so a
-- `remember` claim still digests exactly what it always did and is left alone.
ALTER TABLE idempotency_keys NO FORCE ROW LEVEL SECURITY;

DELETE FROM idempotency_keys WHERE operation = 'observe';

ALTER TABLE idempotency_keys FORCE ROW LEVEL SECURITY;

-- The observation row carries the same digest, so purging the claim is only half the repair:
-- `write_observation` also compares `observations.content_digest`, and would raise
-- "device sequence already stores different observation content" for the same resend.
--
-- There is nothing to recompute it from either, so the column says so instead. NULL means
-- "written by a recipe that no longer exists, and therefore not comparable" -- the one thing a
-- stale digest cannot express about itself. `write_observation` accepts a resend whose stored
-- digest is NULL and writes the current digest back, so the guard is restored after one
-- resend rather than dropped. Scoped by `jsonb_array_length` because `identity_observations`
-- is where the removed field lived: an observation with no identity spans digests exactly what
-- it always did, and its guard is left alone.
ALTER TABLE observations ALTER COLUMN content_digest DROP NOT NULL;

ALTER TABLE observations NO FORCE ROW LEVEL SECURITY;

UPDATE observations
SET content_digest = NULL
WHERE jsonb_array_length(identity_observations) > 0;

ALTER TABLE observations FORCE ROW LEVEL SECURITY;

-- 3. Let the runtime role read the migration ledger.
--
-- `write_embedding_on_connection` re-keys a vector stranded by 0021's change to the
-- `embedding_id` recipe, and bounds that to rows older than when 0021 was applied so a later
-- ID disagreement still raises. Reading that timestamp needs a grant: migration 0005 grants
-- per-table access only to tables carrying a `tenant_id`, and `schema_migrations` has none,
-- so `mindbridge_runtime` could not read it. The integration fixture connects as the owner,
-- which would have hidden this until it failed in a deployment with "permission denied for
-- table schema_migrations". SELECT only -- the ledger is written by migrations, not by the
-- runtime.
GRANT SELECT ON TABLE schema_migrations TO mindbridge_runtime;

INSERT INTO schema_migrations (version) VALUES (25);

COMMIT;
