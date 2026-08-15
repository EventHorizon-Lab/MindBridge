BEGIN;

ALTER TABLE observations
    ADD COLUMN identity_observations jsonb NOT NULL DEFAULT '[]'::jsonb,
    ADD CONSTRAINT observations_identity_observations_array
        CHECK (jsonb_typeof(identity_observations) = 'array');

INSERT INTO schema_migrations (version) VALUES (6);

COMMIT;
