BEGIN;

-- Entity resolution asks "has this pair already been judged?" once per candidate pair, and
-- the answer decides whether it spends a model call. Claim consolidation earned its own
-- partial index for the same read; this is that index for the entity verdicts. Both verdict
-- types share it because the question is settled-or-not, not which way it was settled.
CREATE INDEX relations_entity_resolution_idx
    ON relations (tenant_id, source_id, target_id, relation_type)
    WHERE source_type = 'entity'
      AND target_type = 'entity'
      AND relation_type IN ('same_as', 'not_same_as');

INSERT INTO schema_migrations (version) VALUES (20);

COMMIT;
