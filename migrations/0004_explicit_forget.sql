BEGIN;

ALTER TABLE memory_records
    DROP CONSTRAINT memory_records_supersedes_memory_fk,
    ADD CONSTRAINT memory_records_supersedes_memory_fk
        FOREIGN KEY (tenant_id, supersedes_memory_id)
        REFERENCES memory_records (tenant_id, memory_id)
        ON DELETE SET NULL (supersedes_memory_id);

ALTER TABLE deletion_tombstones
    ADD COLUMN error_code text,
    ADD CONSTRAINT deletion_tombstones_target_type_check
        CHECK (target_type IN ('memory_record', 'observation')),
    ADD CONSTRAINT deletion_tombstones_completion_check
        CHECK ((propagation_state = 'complete') = (completed_at IS NOT NULL)),
    ADD CONSTRAINT deletion_tombstones_failure_check
        CHECK ((propagation_state = 'failed') = (error_code IS NOT NULL));

CREATE INDEX deletion_tombstones_pending_idx
    ON deletion_tombstones (propagation_state, requested_at)
    WHERE propagation_state IN ('pending', 'propagating', 'failed');

INSERT INTO schema_migrations (version) VALUES (4);

COMMIT;
