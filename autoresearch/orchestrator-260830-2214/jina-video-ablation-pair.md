# Jina video metadata paired ablation

Verdict: **ACCEPT**. Quality is the first-priority gate; any task regression rejects v6.

| Task | v4 accuracy | v6 accuracy | Delta | TTFT delta | Generation tokens delta | Retrieval set Jaccard | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| egolife-n10 | 40.0000% | 40.0000% | +0.0000 pp | -0.3164% | -4.2142% | 0.7946 | accept |
| m3-l1 | 46.6667% | 46.6667% | +0.0000 pp | +0.0005% | -0.0657% | 1.0000 | accept |

All pairs passed identity/order/input-hash/configuration checks, use distinct data-root paths, have zero errors/ingest/cache, and report complete usage. Store metadata differs only at `embedding.space_id`; M3 also has distinct captured device/inode identities.

Evidence limitation: v4 Ego's device/inode was not captured before cleanup. Its hash-bound sidecar is explicitly marked as a manual transcription of the live SQLite integrity, lock, count, and metadata audit.

Node-level latency, TTFT, tokens by module, artifact hashes, embedding-space identities, and retrieval-ID overlap are preserved in the JSON companion.

## Paired score flips

| Task | Direction | Sample | Question type | v4 → v6 | ID overlap | Top-1 equal |
| --- | --- | --- | --- | ---: | ---: | --- |
| — | — | No paired score flips | — | — | — | — |
