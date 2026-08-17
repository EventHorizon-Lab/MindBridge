BEGIN;

-- Every observation must carry image, video, or audio evidence, so 'gaze', 'imu' and
-- 'robot_state' never had an ingestion path. Narrow the constraint to the two sensors the
-- contract can actually accept. This fails loudly if any historical row used the others;
-- resolve such rows explicitly rather than widening the constraint again.
ALTER TABLE observations
    DROP CONSTRAINT observations_sensor_check;

ALTER TABLE observations
    ADD CONSTRAINT observations_sensor_check CHECK (sensor IN ('camera', 'microphone'));

INSERT INTO schema_migrations (version) VALUES (16);

COMMIT;
