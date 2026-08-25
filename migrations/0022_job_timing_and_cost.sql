BEGIN;

-- A job row carried created_at and updated_at and nothing else about when it ran. updated_at
-- moves on every state change, so queue wait and work time could not be separated: the claim
-- that starts an attempt is itself one of those changes. `process_observation` works around
-- this by measuring queue lag only on attempt 1, where updated_at at claim time happens to be
-- the start; every retry, and every question about a run in progress, was unanswerable. During
-- the 2026-08-21 evaluation two agents attributed worker time to different tenants from these
-- rows and neither could be right, because the rows do not say.
--
-- started_at is stamped by the claim, so it always describes the attempt currently recorded:
-- started_at - created_at is the wait, and updated_at - started_at is the attempt that ran.
ALTER TABLE jobs ADD COLUMN started_at timestamptz;

ALTER TABLE jobs
    ADD CONSTRAINT jobs_started_after_creation
    CHECK (started_at IS NULL OR started_at >= created_at);

-- What the attempts cost. The generator reports its usage at the adapter boundary and it went
-- to OTLP only, so a deployment with no collector -- that evaluation included -- could not
-- answer "what did this benchmark cost". These two accumulate across attempts rather than
-- being replaced, because a failed attempt is paid for as surely as a successful one: 28 of the
-- 61 write failures there were model_output_invalid, raised after the tokens were spent. NULL
-- therefore means "finished before this column existed", and 0 means "spent nothing".
ALTER TABLE jobs
    ADD COLUMN input_tokens bigint CHECK (input_tokens >= 0),
    ADD COLUMN output_tokens bigint CHECK (output_tokens >= 0);

INSERT INTO schema_migrations (version) VALUES (22);

COMMIT;
