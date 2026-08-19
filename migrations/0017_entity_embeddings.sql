BEGIN;

-- Entities become a retrieval entry point, not only a second-hop bridge between events.
-- Named entities are also keyed globally per tenant from this point on, so one person or
-- object is one node that every mentioning event attaches to; the previous per-event keys
-- stay valid rows and simply keep their own, narrower reach until they are rebuilt.
ALTER TABLE embeddings
    DROP CONSTRAINT embeddings_object_type_check;

ALTER TABLE embeddings
    ADD CONSTRAINT embeddings_object_type_check CHECK (
        object_type IN ('evidence_span', 'event', 'claim', 'memory_record', 'entity')
    );

INSERT INTO schema_migrations (version) VALUES (17);

COMMIT;
