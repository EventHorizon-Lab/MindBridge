BEGIN;

-- One object embedded in pieces needs one row per piece, and the vectors key had no room for
-- the piece.
--
-- An evidence span is cut into encoder-sized clips before it is embedded. `audio_windows`
-- splits anything longer than `AUDIO_WINDOW_MS`, so a 70-second audio span becomes three clips
-- of different sound with three different vectors, while `object_id` stays the span for all of
-- them -- `recall` reads that column straight back as an `EvidenceId` to load the span, so it
-- has to. Under UNIQUE (tenant_id, object_type, object_id, model_id, space_id, task) the second
-- clip conflicted with the first, `write_embedding_on_connection` compared their vectors, found
-- them different, and raised "embedding version already stores different vector content".
--
-- That raise lands inside `commit_observation_processing`, which writes every derived record of
-- one observation in a single transaction and iterates the embeddings sequentially. So the
-- failure was never the loss of one vector: one audio evidence span over 30 seconds rolled back
-- the whole observation -- its events, entities, claims, memories, and the first clip's own
-- vector with them. Verified against PostgreSQL 18.4 by writing three clips of one span through
-- the real writer: the second raises and the span ends with zero rows.
--
-- `object_part` is that piece's ordinal, which `embedding_id` already hashed. The ID was
-- therefore always distinct per clip and no ID changes here; what was missing was the key
-- agreeing with it, so no re-key and no `embedding_id_recipe` bump belongs in this migration.

-- The backfill is unconditionally correct, and the transaction above is why. A multi-clip span
-- aborted its own commit, so no database can hold a row for a clip other than the first: every
-- evidence-span vector that exists was written as ordinal 0, and every other object type is
-- embedded whole. There is nothing here to disambiguate.
ALTER TABLE embeddings
    ADD COLUMN object_part smallint NOT NULL DEFAULT 0;

ALTER TABLE embeddings
    ADD CONSTRAINT embeddings_object_part_non_negative CHECK (object_part >= 0);

-- The default existed for the backfill. A writer states which part it is inserting, as it does
-- for `embedding_id_recipe`, so a second clip cannot silently claim to be the first.
ALTER TABLE embeddings
    ALTER COLUMN object_part DROP DEFAULT;

-- Widening a key cannot make existing rows collide, so there is nothing to resolve before
-- applying this: every existing row takes part 0 and the key it satisfied is a prefix of the
-- new one. No dedup runs here for the same reason 0026's did not.
ALTER TABLE embeddings DROP CONSTRAINT embeddings_object_model_space_task_key;

ALTER TABLE embeddings
    ADD CONSTRAINT embeddings_object_part_model_space_task_key
    UNIQUE (tenant_id, object_type, object_id, model_id, space_id, task, object_part);

INSERT INTO schema_migrations (version) VALUES (27);

COMMIT;
