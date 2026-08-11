BEGIN;

CREATE INDEX claims_consolidation_candidates_idx
    ON claims (tenant_id, claim_id)
    WHERE superseded_at IS NULL;

CREATE INDEX relations_claim_consolidation_idx
    ON relations (tenant_id, source_id, relation_type, target_id)
    WHERE source_type = 'claim' AND target_type = 'claim';

INSERT INTO schema_migrations (version) VALUES (10);

COMMIT;
