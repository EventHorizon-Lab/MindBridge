BEGIN;

ALTER TABLE observations
    ADD COLUMN input_text text;

ALTER TABLE observations
    ADD CONSTRAINT observations_input_text_valid CHECK (
        input_text IS NULL
        OR (
            char_length(input_text) BETWEEN 1 AND 2048
            AND btrim(input_text) <> ''
        )
    );

INSERT INTO schema_migrations (version) VALUES (28);

COMMIT;
