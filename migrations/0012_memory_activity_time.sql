BEGIN;

UPDATE memory_records
SET last_accessed_at = created_at
WHERE last_accessed_at < created_at;

ALTER TABLE memory_records
    ADD CONSTRAINT memory_records_access_after_creation
        CHECK (last_accessed_at IS NULL OR last_accessed_at >= created_at);

INSERT INTO schema_migrations (version) VALUES (12);

COMMIT;
