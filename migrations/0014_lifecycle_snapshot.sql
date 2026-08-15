BEGIN;

ALTER TABLE memory_records
    ADD COLUMN lifecycle_changed_at timestamptz;

UPDATE memory_records AS memory
SET lifecycle_changed_at = GREATEST(
    now(),
    memory.created_at,
    memory.last_accessed_at,
    memory.superseded_at,
    (
        SELECT max(feedback.created_at)
        FROM memory_feedback AS feedback
        WHERE feedback.tenant_id = memory.tenant_id
          AND feedback.memory_id = memory.memory_id
    )
);

ALTER TABLE memory_records
    ALTER COLUMN lifecycle_changed_at SET NOT NULL,
    ADD CONSTRAINT memory_records_lifecycle_change_after_creation
        CHECK (lifecycle_changed_at >= created_at);

INSERT INTO schema_migrations (version) VALUES (14);

COMMIT;
