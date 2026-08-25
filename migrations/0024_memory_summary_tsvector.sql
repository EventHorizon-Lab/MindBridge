BEGIN;

-- Lexical recall re-derived the tsvector of every candidate summary, twice per row, on every
-- call. 0019 indexed the expression `to_tsvector('mindbridge_text', summary)`, but the only
-- query that could read that index cannot: `search_memories` ORs the tsquery match with
-- `strpos(lower(summary), lower(query)) > 0`, an arm with no index to serve it, so the planner
-- has to visit every row of the tenant regardless and applies both arms as a filter. The ORDER
-- BY then calls `ts_rank_cd` on the same expression, and an expression index stores no value to
-- rank with, so the scan lexed each summary a second time.
--
-- The index was therefore never read while still being maintained on every write. Measured
-- across two complete nine-benchmark evaluations: `memory_records_summary_fts_idx` had 0 scans
-- in both, occupying 32 MB and 13 MB, while `memory_records_pkey` served 6.9 million.
--
-- Storing the vector makes both the filter and the ranking a column read. On a 68,682-row
-- table whose largest tenant holds 12,784 memories, the recall query's median fell from 298 ms
-- to 34.6 ms, and one API process went from 15.3 to 85.5 recalls per second at 16-way
-- concurrency with p50 latency dropping from 1131 ms to 184 ms. The stored column added about
-- 21 MB, less than the 32 MB the dropped index returns.
--
-- This is not the end of the plan the shape suggests. A GIN index on the stored column was
-- measured here and the planner did not choose it: at 12,784 rows a sequential scan over a
-- cheap filter wins, and the `strpos` arm would keep it winning anyway. A deployment whose
-- single tenant grows far past that should index the column AND make the substring arm
-- indexable in the same change -- `lower(summary) LIKE '%' || <escaped> || '%'` against
-- `gin (lower(summary) gin_trgm_ops)`, which measured identical results to `strpos` on
-- caller text containing `%`, `_`, backslashes and quotes. Dropping the substring arm instead
-- changes what recall returns and is not an optimisation: it loses substring hits inside longer
-- tokens, so `atm` returned 6 rows where the current query returns 10.
--
-- ADD COLUMN ... GENERATED rewrites the table under ACCESS EXCLUSIVE -- 5.2 s for 150 MB here.
ALTER TABLE memory_records
    ADD COLUMN summary_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('mindbridge_text', summary)) STORED;

DROP INDEX memory_records_summary_fts_idx;

INSERT INTO schema_migrations (version) VALUES (24);

COMMIT;
