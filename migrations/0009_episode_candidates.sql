BEGIN;

CREATE INDEX events_episode_candidates_idx
    ON events (tenant_id, event_id)
    WHERE hierarchy_level = 'event' AND status = 'active' AND parent_event_id IS NULL;

CREATE INDEX entity_mentions_event_entity_idx
    ON entity_mentions (tenant_id, event_id, entity_id);

CREATE INDEX embeddings_object_lookup_idx
    ON embeddings (tenant_id, object_type, object_id, space_id, space_revision, task);

INSERT INTO schema_migrations (version) VALUES (9);

COMMIT;
