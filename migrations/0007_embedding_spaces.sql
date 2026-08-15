BEGIN;

ALTER TABLE embeddings
    ADD COLUMN space_id text,
    ADD COLUMN space_revision text;

ALTER TABLE embeddings NO FORCE ROW LEVEL SECURITY;

UPDATE embeddings
SET space_id = model_id,
    space_revision = model_revision;

ALTER TABLE embeddings FORCE ROW LEVEL SECURITY;

ALTER TABLE embeddings
    ALTER COLUMN space_id SET NOT NULL,
    ALTER COLUMN space_revision SET NOT NULL,
    ADD CONSTRAINT embeddings_space_id_not_empty CHECK (space_id <> ''),
    ADD CONSTRAINT embeddings_space_revision_not_empty CHECK (space_revision <> '');

CREATE INDEX embeddings_space_search_idx
    ON embeddings (tenant_id, space_id, space_revision, task, object_type);

INSERT INTO schema_migrations (version) VALUES (7);

COMMIT;
