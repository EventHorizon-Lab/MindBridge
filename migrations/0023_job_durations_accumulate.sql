BEGIN;

-- Migration 0022 added `started_at` so that `started_at - created_at` was the wait and
-- `updated_at - started_at` was the attempt. Both are only true of a job that ran exactly once
-- and has finished, which is not the interesting case.
--
-- The claim stamps `started_at` and `updated_at` in the same statement, so a *running* job
-- reports `updated_at - started_at = 0`. The ledger's whole purpose is answering "who is
-- consuming the worker", and it ordered by that column -- so the jobs actually holding the
-- worker contributed nothing to their tenant's total and sorted last.
--
-- A retry then moves `started_at` forward. After it, `started_at - created_at` spans creation to
-- the *latest* claim, which includes the earlier attempt's processing and its backoff, while
-- `updated_at - started_at` covers only the final attempt. So one row mixed a cumulative token
-- count -- 0022 deliberately accumulates those, because a failed attempt is paid for -- with a
-- last-attempt duration, and understated worker time by every attempt but the last.
--
-- These two accumulate on the same principle the token columns already use: add this attempt's
-- share as it closes, rather than deriving a difference from two timestamps that have each been
-- overwritten. NULL means "this job last moved before the column existed"; 0 means "measured,
-- and it was that fast".
ALTER TABLE jobs
    ADD COLUMN queue_wait_seconds double precision CHECK (queue_wait_seconds >= 0),
    ADD COLUMN work_seconds double precision CHECK (work_seconds >= 0);

INSERT INTO schema_migrations (version) VALUES (23);

COMMIT;
