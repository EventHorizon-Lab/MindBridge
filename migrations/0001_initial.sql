BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE schema_migrations (
    version integer PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO schema_migrations (version) VALUES (1);

CREATE TABLE media_objects (
    tenant_id text NOT NULL,
    media_object_id text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('image', 'video', 'audio')),
    uri text NOT NULL CHECK (uri <> ''),
    sha256 char(64) NOT NULL CHECK (sha256 ~ '^[0-9a-fA-F]{64}$'),
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    duration_ms bigint CHECK (duration_ms >= 0),
    codec_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    retention_class text NOT NULL DEFAULT 'standard',
    created_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, media_object_id),
    UNIQUE (tenant_id, sha256)
);

CREATE TABLE observations (
    tenant_id text NOT NULL,
    observation_id text NOT NULL,
    device_id text NOT NULL,
    boot_id text NOT NULL,
    sequence bigint NOT NULL CHECK (sequence >= 0),
    sensor text NOT NULL CHECK (sensor IN ('camera', 'microphone', 'gaze', 'imu', 'robot_state')),
    occurred_at timestamptz NOT NULL,
    ended_at timestamptz NOT NULL CHECK (ended_at >= occurred_at),
    observed_at timestamptz NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    clock_offset_ms integer NOT NULL DEFAULT 0,
    upload_status text NOT NULL DEFAULT 'complete' CHECK (
        upload_status IN ('manifest', 'uploading', 'complete', 'failed')
    ),
    content_digest char(64) NOT NULL CHECK (content_digest ~ '^[0-9a-f]{64}$'),
    PRIMARY KEY (tenant_id, observation_id),
    UNIQUE (tenant_id, device_id, boot_id, sequence)
);

CREATE TABLE observation_media (
    tenant_id text NOT NULL,
    observation_id text NOT NULL,
    media_object_id text NOT NULL,
    ordinal smallint NOT NULL CHECK (ordinal >= 0),
    PRIMARY KEY (tenant_id, observation_id, media_object_id),
    UNIQUE (tenant_id, observation_id, ordinal),
    FOREIGN KEY (tenant_id, observation_id)
        REFERENCES observations (tenant_id, observation_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, media_object_id)
        REFERENCES media_objects (tenant_id, media_object_id) ON DELETE RESTRICT
);

CREATE TABLE evidence_spans (
    tenant_id text NOT NULL,
    evidence_id text NOT NULL,
    observation_id text NOT NULL,
    media_object_id text NOT NULL,
    start_ms bigint NOT NULL CHECK (start_ms >= 0),
    end_ms bigint NOT NULL CHECK (end_ms >= start_ms),
    frame_start bigint,
    frame_end bigint,
    x_min integer,
    y_min integer,
    x_max integer,
    y_max integer,
    audio_track integer CHECK (audio_track >= 0),
    created_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, evidence_id),
    FOREIGN KEY (tenant_id, observation_id)
        REFERENCES observations (tenant_id, observation_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, media_object_id)
        REFERENCES media_objects (tenant_id, media_object_id) ON DELETE RESTRICT,
    CHECK (
        (frame_start IS NULL AND frame_end IS NULL)
        OR (
            frame_start IS NOT NULL
            AND frame_end IS NOT NULL
            AND frame_start >= 0
            AND frame_end >= frame_start
        )
    ),
    CHECK (
        (x_min IS NULL AND y_min IS NULL AND x_max IS NULL AND y_max IS NULL)
        OR (
            x_min IS NOT NULL
            AND y_min IS NOT NULL
            AND x_max IS NOT NULL
            AND y_max IS NOT NULL
            AND x_min >= 0
            AND y_min >= 0
            AND x_max > x_min
            AND y_max > y_min
        )
    )
);

CREATE TABLE events (
    tenant_id text NOT NULL,
    event_id text NOT NULL,
    parent_event_id text,
    hierarchy_level text NOT NULL DEFAULT 'event' CHECK (hierarchy_level IN ('event', 'episode')),
    description text NOT NULL CHECK (description <> ''),
    salience double precision NOT NULL CHECK (salience BETWEEN 0 AND 1),
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('candidate', 'active', 'superseded')),
    occurred_at timestamptz NOT NULL,
    ended_at timestamptz NOT NULL CHECK (ended_at >= occurred_at),
    model_id text NOT NULL CHECK (model_id <> ''),
    model_revision text NOT NULL CHECK (model_revision <> ''),
    created_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, event_id),
    FOREIGN KEY (tenant_id, parent_event_id)
        REFERENCES events (tenant_id, event_id) ON DELETE SET NULL (parent_event_id)
);

CREATE TABLE event_observations (
    tenant_id text NOT NULL,
    event_id text NOT NULL,
    observation_id text NOT NULL,
    PRIMARY KEY (tenant_id, event_id, observation_id),
    FOREIGN KEY (tenant_id, event_id)
        REFERENCES events (tenant_id, event_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, observation_id)
        REFERENCES observations (tenant_id, observation_id) ON DELETE RESTRICT
);

CREATE TABLE event_evidence (
    tenant_id text NOT NULL,
    event_id text NOT NULL,
    evidence_id text NOT NULL,
    PRIMARY KEY (tenant_id, event_id, evidence_id),
    FOREIGN KEY (tenant_id, event_id)
        REFERENCES events (tenant_id, event_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, evidence_id)
        REFERENCES evidence_spans (tenant_id, evidence_id) ON DELETE RESTRICT
);

CREATE TABLE entities (
    tenant_id text NOT NULL,
    entity_id text NOT NULL,
    entity_type text NOT NULL CHECK (
        entity_type IN ('person', 'object', 'place', 'device', 'organization', 'topic')
    ),
    canonical_name text,
    created_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, entity_id)
);

CREATE TABLE entity_mentions (
    tenant_id text NOT NULL,
    mention_id text NOT NULL,
    entity_id text NOT NULL,
    event_id text,
    evidence_id text,
    confidence double precision NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    created_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, mention_id),
    UNIQUE NULLS NOT DISTINCT (tenant_id, entity_id, event_id, evidence_id),
    FOREIGN KEY (tenant_id, entity_id)
        REFERENCES entities (tenant_id, entity_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, event_id)
        REFERENCES events (tenant_id, event_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, evidence_id)
        REFERENCES evidence_spans (tenant_id, evidence_id) ON DELETE CASCADE,
    CHECK (event_id IS NOT NULL OR evidence_id IS NOT NULL)
);

CREATE TABLE claims (
    tenant_id text NOT NULL,
    claim_id text NOT NULL,
    supersedes_claim_id text,
    statement text NOT NULL CHECK (statement <> ''),
    confidence double precision NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    verification_status text NOT NULL CHECK (verification_status IN ('verified', 'unverified')),
    valid_from timestamptz NOT NULL,
    valid_to timestamptz CHECK (valid_to IS NULL OR valid_to >= valid_from),
    model_id text NOT NULL CHECK (model_id <> ''),
    model_revision text NOT NULL CHECK (model_revision <> ''),
    created_at timestamptz NOT NULL,
    superseded_at timestamptz,
    PRIMARY KEY (tenant_id, claim_id),
    FOREIGN KEY (tenant_id, supersedes_claim_id)
        REFERENCES claims (tenant_id, claim_id) ON DELETE SET NULL (supersedes_claim_id)
);

CREATE TABLE claim_evidence (
    tenant_id text NOT NULL,
    claim_id text NOT NULL,
    evidence_id text NOT NULL,
    PRIMARY KEY (tenant_id, claim_id, evidence_id),
    FOREIGN KEY (tenant_id, claim_id)
        REFERENCES claims (tenant_id, claim_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, evidence_id)
        REFERENCES evidence_spans (tenant_id, evidence_id) ON DELETE RESTRICT
);

CREATE TABLE memory_records (
    tenant_id text NOT NULL,
    memory_id text NOT NULL,
    memory_type text NOT NULL CHECK (
        memory_type IN ('perceptual', 'working', 'episodic', 'semantic', 'procedural', 'prospective')
    ),
    summary text NOT NULL CHECK (summary <> ''),
    verification_status text NOT NULL CHECK (verification_status IN ('verified', 'unverified')),
    state text NOT NULL CHECK (state IN ('active', 'strengthened', 'cold', 'compressed')),
    occurred_at timestamptz NOT NULL,
    ended_at timestamptz NOT NULL CHECK (ended_at >= occurred_at),
    model_id text,
    model_revision text,
    content_digest char(64) NOT NULL CHECK (content_digest ~ '^[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, memory_id),
    CHECK (
        (model_id IS NULL AND model_revision IS NULL)
        OR (
            model_id IS NOT NULL
            AND model_revision IS NOT NULL
            AND model_id <> ''
            AND model_revision <> ''
        )
    )
);

CREATE TABLE memory_evidence (
    tenant_id text NOT NULL,
    memory_id text NOT NULL,
    evidence_id text NOT NULL,
    PRIMARY KEY (tenant_id, memory_id, evidence_id),
    FOREIGN KEY (tenant_id, memory_id)
        REFERENCES memory_records (tenant_id, memory_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, evidence_id)
        REFERENCES evidence_spans (tenant_id, evidence_id) ON DELETE RESTRICT
);

CREATE TABLE relations (
    tenant_id text NOT NULL,
    relation_id text NOT NULL,
    source_type text NOT NULL,
    source_id text NOT NULL,
    relation_type text NOT NULL,
    target_type text NOT NULL,
    target_id text NOT NULL,
    created_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, relation_id),
    UNIQUE (tenant_id, source_type, source_id, relation_type, target_type, target_id)
);

CREATE TABLE embeddings (
    tenant_id text NOT NULL,
    embedding_id text NOT NULL,
    object_type text NOT NULL CHECK (
        object_type IN ('evidence_span', 'event', 'claim', 'memory_record')
    ),
    object_id text NOT NULL,
    model_id text NOT NULL CHECK (model_id <> ''),
    model_revision text NOT NULL CHECK (model_revision <> ''),
    task text NOT NULL CHECK (task <> ''),
    dimension integer NOT NULL CHECK (dimension = 1024),
    normalized boolean NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, embedding_id),
    UNIQUE (tenant_id, object_type, object_id, model_id, model_revision, task)
);

CREATE TABLE memory_feedback (
    tenant_id text NOT NULL,
    feedback_id text NOT NULL,
    memory_id text,
    feedback_type text NOT NULL CHECK (feedback_type IN ('useful', 'wrong', 'missing', 'correction')),
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, feedback_id),
    FOREIGN KEY (tenant_id, memory_id)
        REFERENCES memory_records (tenant_id, memory_id) ON DELETE SET NULL (memory_id)
);

CREATE TABLE deletion_tombstones (
    tenant_id text NOT NULL,
    tombstone_id text NOT NULL,
    target_type text NOT NULL,
    target_id text NOT NULL,
    propagation_state text NOT NULL CHECK (
        propagation_state IN ('pending', 'propagating', 'complete', 'failed')
    ),
    requested_at timestamptz NOT NULL,
    completed_at timestamptz,
    PRIMARY KEY (tenant_id, tombstone_id),
    UNIQUE (tenant_id, target_type, target_id)
);

CREATE TABLE jobs (
    tenant_id text NOT NULL,
    job_id text NOT NULL,
    job_type text NOT NULL CHECK (job_type IN ('process_observation')),
    state text NOT NULL CHECK (state IN ('pending', 'running', 'succeeded', 'failed')),
    payload jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(payload) = 'object'),
    attempt integer NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    error_code text,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, job_id),
    CHECK ((state = 'pending' AND attempt = 0) OR (state <> 'pending' AND attempt > 0)),
    CHECK ((state = 'failed') = (error_code IS NOT NULL))
);

CREATE TABLE idempotency_keys (
    tenant_id text NOT NULL,
    operation text NOT NULL CHECK (operation IN ('observe', 'remember')),
    idempotency_key text NOT NULL,
    content_digest char(64) NOT NULL CHECK (content_digest ~ '^[0-9a-f]{64}$'),
    resource_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, operation, idempotency_key)
);

CREATE INDEX observations_timeline_idx
    ON observations (tenant_id, occurred_at DESC, observation_id);
CREATE INDEX observations_timeline_brin_idx ON observations USING brin (occurred_at);
CREATE INDEX evidence_media_time_idx
    ON evidence_spans (tenant_id, media_object_id, start_ms, end_ms);
CREATE INDEX events_timeline_idx ON events (tenant_id, occurred_at DESC, event_id);
CREATE INDEX entity_mentions_event_idx ON entity_mentions (tenant_id, event_id);
CREATE INDEX relations_source_idx ON relations (tenant_id, source_type, source_id);
CREATE INDEX relations_target_idx ON relations (tenant_id, target_type, target_id);
CREATE INDEX memory_records_timeline_idx
    ON memory_records (tenant_id, occurred_at DESC, memory_id);
CREATE INDEX memory_records_summary_fts_idx
    ON memory_records USING gin (to_tsvector('simple', summary));
CREATE INDEX embeddings_vector_hnsw_idx
    ON embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX jobs_pending_idx
    ON jobs (state, created_at) WHERE state IN ('pending', 'failed');

COMMIT;
