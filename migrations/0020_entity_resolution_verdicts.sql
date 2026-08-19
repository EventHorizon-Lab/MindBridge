BEGIN;

-- The adjudicator is required to name the cue its verdict rested on, and until now that
-- answer was validated and then dropped: a same_as edge asserted two records were one
-- person durably, with nothing an operator could read back when the merge turned out wrong.
--
-- The cue lives here rather than on relations because it is a property of the adjudication,
-- not of the edge. Every other relation kind -- claim consolidation, summary parents, the
-- semantic graph mirror -- would carry the column as NULL forever, and widening the shared
-- edge row would widen the strict relation writer's column-by-column conflict comparison
-- along with it.
--
-- Keyed on relation_id, which entity_resolution derives from the pair and deliberately not
-- from the verdict. That makes "one pair owns exactly one verdict row" a primary key rather
-- than a convention, and it is what lets --entity-readjudicate replace a verdict instead of
-- appending a second, contradicting one.
CREATE TABLE entity_resolution_verdicts (
    tenant_id text NOT NULL,
    relation_id text NOT NULL,
    -- The direction is deliberately not duplicated here. relation_type on the edge already
    -- carries it, derived from the same boolean; a second copy could only ever drift, and
    -- nothing would say which one an audit should believe.
    confidence double precision NOT NULL
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    discriminating_cue text NOT NULL CHECK (discriminating_cue <> ''),
    -- Distinct from relations.created_at, which only moves when the direction flips. A
    -- re-judgement that reaches the same answer for a different reason still lands a new
    -- cue, and this is the column that dates it.
    decided_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, relation_id),
    -- A justification must not outlive the edge it justifies.
    FOREIGN KEY (tenant_id, relation_id)
        REFERENCES relations (tenant_id, relation_id) ON DELETE CASCADE
);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE entity_resolution_verdicts TO mindbridge_runtime;
ALTER TABLE entity_resolution_verdicts ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_resolution_verdicts FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON entity_resolution_verdicts
    USING (tenant_id = current_setting('mindbridge.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('mindbridge.tenant_id', true));

INSERT INTO schema_migrations (version) VALUES (20);

COMMIT;
