# Memory types, temporal reasoning, and decay

MindBridge supports semantic, episodic, and procedural memory as explicit roles on the same
durable record. It also supports event-time-aware retrieval and optional, non-destructive memory
decay. These features share the existing SQLite and Zvec path; they do not create separate stores,
workers, or logical scopes.

## Support boundary

| Capability | Current support | Deliberate boundary |
| --- | --- | --- |
| Semantic memory | `MemoryType.SEMANTIC`, the default | The caller classifies content; MindBridge does not extract facts automatically |
| Episodic memory | `MemoryType.EPISODIC` plus optional `occurred_at` | No automatic episode segmentation or reflection |
| Procedural memory | `MemoryType.PROCEDURAL` for instructions and reusable routines | Stored procedures are evidence, not executable code |
| Temporal reasoning | ISO dates and common English/Chinese relative calendar expressions | No unrestricted natural-language temporal theorem prover |
| Memory decay | Optional search-time soft reranking with bounded retrieval reinforcement | No automatic deletion, archival, or rewriting |

Before these contracts were added, MindBridge had semantic similarity retrieval and persisted
`occurred_at`, but every record was otherwise untyped, event time did not affect retrieval, and no
access history or decay factor existed.

## Python example

```python
from datetime import datetime, timezone

from mindbridge import Config, Memory, MemoryType

config = Config(decay_half_life_days=30)

with Memory("./data/agent", config=config) as memory:
    memory.add(
        "The deployment failed because the token had expired.",
        memory_type=MemoryType.EPISODIC,
        occurred_at=datetime(2026, 8, 20, 9, tzinfo=timezone.utc),
    )
    memory.add(
        "Refresh the token, retry once, then escalate.",
        memory_type=MemoryType.PROCEDURAL,
    )

    episode = memory.search(
        "What failed last week?",
        memory_type=MemoryType.EPISODIC,
        reference_at=datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
    )
```

`add_many(..., memory_type=...)` applies one role to the complete batch. `search` and `ask` accept
an optional role filter. A different non-semantic role produces a different stable identity for
otherwise identical content; the default semantic identity remains compatible with records made
before this feature.

## Temporal retrieval

`occurred_at` is semantic event time. `created_at` remains storage time and is not substituted for
an absent event time during temporal matching.

`reference_at` resolves relative expressions in its timezone. When omitted, MindBridge uses the
current UTC time. The deterministic parser recognizes:

- one or two ISO dates such as `2026-08-20` or `2026-08-20 ... 2026-08-22`;
- `today`, `yesterday`, `tomorrow`, and their common Chinese equivalents;
- last, this, next, or past week, including `上周`, `本周`, and `下周`;
- last, this, or next month and year, including common Chinese equivalents;
- `N days ago`, `N 天前`, and rolling `past N days`, `过去 N 天`, or `最近 N 天`.

For a detected temporal expression, Zvec first retrieves candidates inside the event-time range.
MindBridge also fetches a bounded fallback pool so a sparse or missing timeline does not turn into
an unexplained empty result. SQLite then rechecks event times, ranks matched events first, applies
the normal relevance score, and hydrates only authoritative records. `ask` passes the resolved
reference time to the generation model so relative-date wording is not interpreted against the
provider's clock.

This is intentionally a bounded temporal retrieval layer. Complex relation chains such as “the
meeting two releases after the migration” require application-supplied normalization or a future
measured temporal planner.

## Decay and reinforcement

Decay is off by default. Enable it with `Config(decay_half_life_days=...)` or
`MINDBRIDGE_DECAY_HALF_LIFE_DAYS`.

MindBridge over-fetches at least 50 candidates, computes a factor at search time, sorts by adjusted
relevance, clamps public scores to `[0, 1]`, and returns the requested limit. The durable memory is
never filtered or deleted by decay.

For each candidate:

```text
anchor = last_accessed_at or occurred_at or updated_at
strength = 1 + log2(1 + min(access_count, 20))
retention = 2 ^ (-age / (half_life * strength))
decay_factor = 0.3 + 1.2 * retention
adjusted_score = relevance * decay_factor
```

Only returned hits are reinforced. SQLite caps `access_count` at 20 and moves
`last_accessed_at` forward using the real retrieval time; a simulated `reference_at` never writes a
future access timestamp. Reinforcement can change later ranking, so MCP search and answer tools are
conservatively marked as non-read-only. Disabling decay leaves access history unused.

## Research basis and architecture decision

The design follows the smallest common mechanism supported by the literature:

- [CoALA](https://arxiv.org/abs/2309.02427) separates episodic experiences, semantic knowledge,
  and procedural action knowledge, while warning that procedural writes can directly change agent
  behavior.
- [Generative Agents](https://arxiv.org/abs/2304.03442) combines relevance, recency, and importance
  during retrieval rather than deleting old observations.
- [MemoryBank](https://arxiv.org/abs/2305.10250) adapts an Ebbinghaus-style exponential retention
  curve and strengthens memories after recall.
- [LongMemEval](https://arxiv.org/abs/2410.10813) treats temporal reasoning as a distinct long-term
  memory ability and motivates time-aware query restriction.
- [TReMu](https://aclanthology.org/2025.findings-acl.972/) and
  [Temporal Semantic Memory](https://aclanthology.org/2026.findings-acl.1496/) show why an event
  timeline and semantic event time matter more than dialogue or storage order.
- [Zep's temporal knowledge graph](https://arxiv.org/abs/2501.13956),
  [A-MEM](https://arxiv.org/abs/2502.12110), and
  [MIRIX](https://arxiv.org/abs/2507.07957) explore richer graph, linked-note, and multi-store
  architectures.
- Mem0 documents comparable conventions for
  [memory roles](https://mem0.ai/blog/semantic-vs-episodic-vs-procedural-memory-in-ai-agents-a-complete-comparison),
  [temporal reasoning](https://docs.mem0.ai/platform/features/temporal-reasoning), and
  [soft memory decay](https://docs.mem0.ai/platform/features/memory-decay).

MindBridge adopts explicit type, event, and access fields plus query-time filtering/reranking. It
does not adopt a graph database, autonomous consolidation agents, an LLM classification call, or a
procedure executor. Those layers would add new failure modes and dependencies without a measured
requirement in the embedded SDK. Add them only when a benchmark shows that typed aggregate
retrieval cannot meet a concrete workload.

Memory type and metadata are not isolation controls. One physical `data_dir` remains one memory
domain and one live MindBridge owner.
