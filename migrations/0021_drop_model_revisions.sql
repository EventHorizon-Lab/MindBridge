BEGIN;

-- Every derived record carried the revision of the model that produced it, alongside that
-- model's id, and every vector carried its embedding space's revision alongside the space id.
-- Nothing read either one: recall matches on the space id, the writers compare model identity,
-- and no query, index, or report ever selected a revision. Dropping the four columns also
-- narrows the two keys they were part of.

ALTER TABLE events DROP COLUMN model_revision;

ALTER TABLE claims DROP COLUMN model_revision;

-- A memory record has a model only when something derived it, so the table paired the two
-- columns in one CHECK. Dropping the column drops that constraint with it; the surviving half
-- of the pairing has to be restated.
ALTER TABLE memory_records DROP COLUMN model_revision;

ALTER TABLE memory_records
    ADD CONSTRAINT memory_records_model_id_not_empty
    CHECK (model_id IS NULL OR model_id <> '');

-- Dropping these takes their UNIQUE constraint and both space indexes with them, because each
-- one names a revision column. The replacements below are the same keys without it: one vector
-- per object, task, and model rather than per revision of that model.
ALTER TABLE embeddings
    DROP COLUMN model_revision,
    DROP COLUMN space_revision;

-- Two vectors of the same object that differed only by revision now collide. Keeping the
-- earliest is the one choice that cannot invent a row: a re-encode wrote the later vector, and
-- `upsert_embedding` already treats an existing vector for an ID as the authoritative one.
-- FORCE ROW LEVEL SECURITY applies the tenant policy to the table owner too, and no tenant is
-- set during a migration, so an owner who is not a superuser would see -- and so delete --
-- nothing here. Lifted for the statement the same way migration 0007 lifts it for its backfill.
ALTER TABLE embeddings NO FORCE ROW LEVEL SECURITY;

DELETE FROM embeddings AS duplicate
USING embeddings AS surviving
WHERE duplicate.tenant_id = surviving.tenant_id
  AND duplicate.object_type = surviving.object_type
  AND duplicate.object_id = surviving.object_id
  AND duplicate.model_id = surviving.model_id
  AND duplicate.task = surviving.task
  AND (duplicate.created_at, duplicate.embedding_id)
      > (surviving.created_at, surviving.embedding_id);

ALTER TABLE embeddings FORCE ROW LEVEL SECURITY;

ALTER TABLE embeddings
    ADD CONSTRAINT embeddings_object_model_task_key
    UNIQUE (tenant_id, object_type, object_id, model_id, task);

CREATE INDEX embeddings_space_search_idx
    ON embeddings (tenant_id, space_id, task, object_type);

CREATE INDEX embeddings_object_lookup_idx
    ON embeddings (tenant_id, object_type, object_id, space_id, task);

-- The edge sends identity spans as JSON and they are stored as sent. The reader validates that
-- document against a model that forbids unknown fields, so a leftover model_revision key would
-- fail every observation that has one rather than being ignored.
ALTER TABLE observations NO FORCE ROW LEVEL SECURITY;

UPDATE observations
SET identity_observations = (
        SELECT coalesce(jsonb_agg(identity - 'model_revision' ORDER BY ordinality), '[]'::jsonb)
        FROM jsonb_array_elements(identity_observations)
             WITH ORDINALITY AS element(identity, ordinality)
    )
WHERE EXISTS (
    SELECT 1
    FROM jsonb_array_elements(identity_observations) AS element(identity)
    WHERE element.identity ? 'model_revision'
);

ALTER TABLE observations FORCE ROW LEVEL SECURITY;

INSERT INTO schema_migrations (version) VALUES (21);

COMMIT;
