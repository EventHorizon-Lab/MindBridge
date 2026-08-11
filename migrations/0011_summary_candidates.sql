BEGIN;

CREATE INDEX memory_records_summary_candidates_idx
    ON memory_records (tenant_id, memory_id)
    WHERE superseded_at IS NULL
      AND memory_type IN ('episodic', 'semantic')
      AND verification_status IN ('verified', 'attested');

CREATE UNIQUE INDEX relations_one_memory_summary_parent_idx
    ON relations (tenant_id, target_id)
    WHERE source_type = 'memory_record'
      AND relation_type = 'contains'
      AND target_type = 'memory_record';

INSERT INTO schema_migrations (version) VALUES (11);

COMMIT;
