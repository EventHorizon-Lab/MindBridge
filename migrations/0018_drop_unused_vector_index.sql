BEGIN;

-- The HNSW index was never read. Row-level security injects
-- `tenant_id = current_setting('mindbridge.tenant_id')` into every query, and
-- `embeddings_space_search_idx` leads with `tenant_id`, so the planner always has a
-- predicate selective enough to reach one tenant's vectors directly and sort them
-- exactly. Measured on 200,000 vectors across 40 tenants, driven through the real
-- `mindbridge_runtime` role and RLS: `embeddings_vector_hnsw_idx` had 0 scans and
-- occupied 1,196 MB, while `embeddings_space_search_idx` served all 25 scans from
-- 1,648 kB. Disabling bitmap and sequential scans did not reach the HNSW index
-- either; only bypassing RLS as a superuser did, which no deployment path does.
--
-- It was not free. Maintaining the graph on insert cost 18.8x: 5,000 vectors landed
-- in 2.19s with the index and 0.12s without it.
--
-- This is a consequence of the multi-tenant shape, not a claim that approximate
-- search is useless. Exact scan cost grows linearly with ONE tenant's vector count
-- -- about 5 ms at 1,000 rows and 51 ms at 11,000 -- so a deployment that ever
-- concentrates millions of vectors in a single tenant should add the index back:
--
--   CREATE INDEX CONCURRENTLY embeddings_vector_hnsw_idx
--       ON embeddings USING hnsw (embedding vector_cosine_ops);
--
-- Whoever does that must keep `SET LOCAL hnsw.iterative_scan = strict_order` in
-- `_postgres_embeddings.search_embeddings`. It is inert while no HNSW index exists,
-- and it is the reason a filtered search returns a full LIMIT rather than silently
-- fewer rows once one does.
DROP INDEX embeddings_vector_hnsw_idx;

INSERT INTO schema_migrations (version) VALUES (18);

COMMIT;
