BEGIN;

-- Model and embedding-space revisions are gone. Every model and dataset MindBridge uses is
-- pulled by name from its provider, so a revision column recorded a pin nothing enforced:
-- the value arrived from configuration, was written verbatim, and no read path ever compared
-- it against what the provider actually served. A provenance field that cannot detect the
-- drift it exists to detect is a column to drop, not a column to keep filling in.
--
-- `model_id` stays. Which model produced a derived record is still answerable; which byte-exact
-- checkout of it produced the record is not, and was not before this either.
--
-- Dropping a column also drops every constraint and index that names it, so the UNIQUE key on
-- `embeddings` and both space-keyed indexes are recreated below without it.

ALTER TABLE events DROP COLUMN model_revision;

ALTER TABLE claims DROP COLUMN model_revision;

-- The dropped CHECK required model_id and model_revision to be present or absent together.
-- Only the nullable-but-never-empty half of it survives the column.
ALTER TABLE memory_records
    DROP COLUMN model_revision,
    ADD CONSTRAINT memory_records_model_id_not_empty
        CHECK (model_id IS NULL OR model_id <> '');

ALTER TABLE embeddings
    DROP COLUMN model_revision,
    DROP COLUMN space_revision;

-- One vector per object per model per task, as before minus the revision.
ALTER TABLE embeddings
    ADD CONSTRAINT embeddings_one_vector_per_object_model_task
        UNIQUE (tenant_id, object_type, object_id, model_id, task);

-- Recreated from 0007 and 0009 with `space_revision` removed. `embeddings_space_search_idx`
-- is the index 0018 showed serves every RLS-scoped search, so it keeps leading with
-- tenant_id.
CREATE INDEX embeddings_space_search_idx
    ON embeddings (tenant_id, space_id, task, object_type);

CREATE INDEX embeddings_object_lookup_idx
    ON embeddings (tenant_id, object_type, object_id, space_id, task);

INSERT INTO schema_migrations (version) VALUES (21);

COMMIT;
