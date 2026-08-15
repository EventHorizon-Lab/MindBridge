BEGIN;

ALTER TABLE claims
    DROP CONSTRAINT claims_verification_status_check,
    ADD CONSTRAINT claims_verification_status_check
        CHECK (verification_status IN ('verified', 'attested', 'unverified'));

ALTER TABLE memory_records
    DROP CONSTRAINT memory_records_verification_status_check,
    ADD CONSTRAINT memory_records_verification_status_check
        CHECK (verification_status IN ('verified', 'attested', 'unverified'));

INSERT INTO schema_migrations (version) VALUES (2);

COMMIT;
