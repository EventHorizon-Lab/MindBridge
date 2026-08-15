BEGIN;

ALTER TABLE claims
    ADD COLUMN claim_type text NOT NULL DEFAULT 'fact'
        CHECK (claim_type IN ('fact', 'state', 'intent', 'relation')),
    ADD COLUMN prompt_version text NOT NULL DEFAULT 'legacy_v1'
        CHECK (prompt_version <> '');

ALTER TABLE claims
    ALTER COLUMN claim_type DROP DEFAULT,
    ALTER COLUMN prompt_version DROP DEFAULT;

CREATE INDEX claims_validity_idx
    ON claims (tenant_id, valid_from DESC, claim_id)
    WHERE superseded_at IS NULL;
CREATE INDEX entities_canonical_name_idx
    ON entities (tenant_id, entity_type, lower(canonical_name))
    WHERE canonical_name IS NOT NULL;

INSERT INTO schema_migrations (version) VALUES (8);

COMMIT;
