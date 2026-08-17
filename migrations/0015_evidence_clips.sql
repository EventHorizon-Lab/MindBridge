BEGIN;

-- Derived media records which original object it was cut from, so deletion and
-- provenance can both follow the link back to the raw evidence.
ALTER TABLE media_objects
    ADD COLUMN derived_from_media_object_id text,
    ADD CONSTRAINT media_objects_derived_from_fkey
        FOREIGN KEY (tenant_id, derived_from_media_object_id)
        REFERENCES media_objects (tenant_id, media_object_id) ON DELETE RESTRICT,
    ADD CONSTRAINT media_objects_derived_from_not_self
        CHECK (derived_from_media_object_id IS DISTINCT FROM media_object_id);

-- One evidence span maps to one clip per encoder window, so audio longer than
-- the model's window keeps its tail instead of being silently truncated. The
-- mapping is stored rather than derived because identical clip content
-- deduplicates onto a single media object.
CREATE TABLE evidence_clips (
    tenant_id text NOT NULL,
    evidence_id text NOT NULL,
    ordinal smallint NOT NULL CHECK (ordinal >= 0),
    media_object_id text NOT NULL,
    start_ms bigint NOT NULL CHECK (start_ms >= 0),
    end_ms bigint NOT NULL CHECK (end_ms >= start_ms),
    created_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, evidence_id, ordinal),
    FOREIGN KEY (tenant_id, evidence_id)
        REFERENCES evidence_spans (tenant_id, evidence_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, media_object_id)
        REFERENCES media_objects (tenant_id, media_object_id) ON DELETE RESTRICT
);

CREATE INDEX evidence_clips_media_object_idx
    ON evidence_clips (tenant_id, media_object_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE evidence_clips TO mindbridge_runtime;
ALTER TABLE evidence_clips ENABLE ROW LEVEL SECURITY;
ALTER TABLE evidence_clips FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON evidence_clips
    USING (tenant_id = current_setting('mindbridge.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('mindbridge.tenant_id', true));

INSERT INTO schema_migrations (version) VALUES (15);

COMMIT;
