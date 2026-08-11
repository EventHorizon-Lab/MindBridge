BEGIN;

ALTER TABLE memory_records
    ADD COLUMN salience double precision NOT NULL DEFAULT 0.5
        CHECK (salience BETWEEN 0 AND 1),
    ADD COLUMN strength double precision NOT NULL DEFAULT 0.5
        CHECK (strength NOT IN ('Infinity'::float8, '-Infinity'::float8, 'NaN'::float8)),
    ADD COLUMN useful_access_count bigint NOT NULL DEFAULT 0
        CHECK (useful_access_count >= 0),
    ADD COLUMN positive_feedback_count bigint NOT NULL DEFAULT 0
        CHECK (positive_feedback_count >= 0),
    ADD COLUMN negative_feedback_count bigint NOT NULL DEFAULT 0
        CHECK (negative_feedback_count >= 0),
    ADD COLUMN last_accessed_at timestamptz,
    ADD COLUMN supersedes_memory_id text,
    ADD COLUMN superseded_at timestamptz,
    ADD CONSTRAINT memory_records_supersedes_memory_fk
        FOREIGN KEY (tenant_id, supersedes_memory_id)
        REFERENCES memory_records (tenant_id, memory_id) ON DELETE RESTRICT,
    ADD CONSTRAINT memory_records_not_self_superseding
        CHECK (supersedes_memory_id IS NULL OR supersedes_memory_id <> memory_id);

CREATE UNIQUE INDEX memory_records_one_successor_idx
    ON memory_records (tenant_id, supersedes_memory_id)
    WHERE supersedes_memory_id IS NOT NULL;

CREATE INDEX memory_records_recallable_timeline_idx
    ON memory_records (tenant_id, occurred_at DESC, memory_id)
    WHERE superseded_at IS NULL;

ALTER TABLE memory_feedback
    ADD COLUMN recall_trace_id text,
    ADD COLUMN corrected_memory_id text,
    ADD COLUMN resulting_state text CHECK (
        resulting_state IN ('active', 'strengthened', 'cold', 'compressed')
    ),
    ADD COLUMN resulting_strength double precision CHECK (
        resulting_strength NOT IN ('Infinity'::float8, '-Infinity'::float8, 'NaN'::float8)
    ),
    ADD CONSTRAINT memory_feedback_corrected_memory_fk
        FOREIGN KEY (tenant_id, corrected_memory_id)
        REFERENCES memory_records (tenant_id, memory_id) ON DELETE SET NULL (corrected_memory_id),
    ADD CONSTRAINT memory_feedback_result_pair_check CHECK (
        (resulting_state IS NULL) = (resulting_strength IS NULL)
    );

ALTER TABLE idempotency_keys
    DROP CONSTRAINT idempotency_keys_operation_check,
    ADD CONSTRAINT idempotency_keys_operation_check
        CHECK (operation IN ('observe', 'remember', 'feedback', 'forget'));

INSERT INTO schema_migrations (version) VALUES (3);

COMMIT;
