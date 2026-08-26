BEGIN;

-- `embedding_id` now hashes `space_id` in every recipe, and this records which recipe wrote
-- each row so the ones written by the previous one can be re-keyed rather than stranded.
--
-- Migration 0025 widened the vectors unique key with `space_id` to restore the re-embedding
-- workflow `docs/configuration.md` documents: one object holding two vectors while a new
-- MINDBRIDGE_EMBEDDING_SPACE_ID is filled in. That was necessary and not sufficient. The
-- table is also keyed PRIMARY KEY (tenant_id, embedding_id), `embedding_id` is
-- content-addressed, and five of the six recipes deriving it did not hash `space_id` -- so
-- the second vector derived the *same* ID as the first, collided on the primary key, and the
-- write failed with "embedding conflict could not be resolved", naming a conflict it could
-- not see. Only `kernel.py`'s memory-record recipe already included `space_id`, which is why
-- 0025 appeared to work: the one object type anybody re-embedded by hand was the one type
-- whose ID already varied.
--
-- Claims, events, entities, evidence spans and consolidated summaries now derive their IDs
-- through `derive_embedding_id`, where `model_id`, `space_id` and `task` are keyword-only and
-- required. That is the part that stops this recurring: five call sites omitted `space_id`
-- independently because each spelled the argument list out by hand.

-- Every row currently on disk was written by recipe 1, whatever its object type. Rows whose
-- ID happens to be unchanged under recipe 2 -- every memory-record vector `kernel.py` wrote --
-- keep working either way: their ID still matches, so the writer never reaches the re-key and
-- never reads this column. Labelling them 1 is therefore accurate about what wrote them and
-- harmless to what happens next.
ALTER TABLE embeddings
    ADD COLUMN embedding_id_recipe smallint NOT NULL DEFAULT 1;

ALTER TABLE embeddings
    ADD CONSTRAINT embeddings_embedding_id_recipe_positive CHECK (embedding_id_recipe >= 1);

-- The default is for the backfill only. A writer states which recipe produced the ID it is
-- inserting, so a row can never silently acquire the current recipe's number by being written
-- with an older one.
ALTER TABLE embeddings
    ALTER COLUMN embedding_id_recipe DROP DEFAULT;

-- What this replaces. `_adopt_embedding_id` bounded its re-key by
-- `created_at < (SELECT applied_at FROM schema_migrations WHERE version = 21)`, which was
-- wrong in both directions. `created_at` is supplied by the caller -- `kernel.py` passes the
-- memory's own creation time, not the write's -- so a replayed or backfilled record could
-- claim an amnesty it had no right to and have a genuine content disagreement silently
-- re-keyed instead of raised. And a per-migration-number bound cannot survive a second recipe
-- change: this one would have needed the number 26 hard-coded beside 21, and the next another.
-- A recipe number the writer records is a fact about the write, so the comparison is exact and
-- the mechanism is reusable.
--
-- `mindbridge_runtime` no longer reads `schema_migrations`, so the grant 0025 added for it is
-- now unused. It is deliberately left in place: during a rolling upgrade the previous release
-- is still running against this schema and still issues that read, and revoking it here would
-- break every instance that has not restarted yet. Revoke it in a later migration, once no
-- deployment can be running code from before this one.

INSERT INTO schema_migrations (version) VALUES (26);

COMMIT;
