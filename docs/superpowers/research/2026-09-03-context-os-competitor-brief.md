# Context OS competitor brief: fast capture, memory control plane, context compiler

Research date 2026-09-03. This answers only questions raised by
[the round 1 design](../specs/2026-09-03-context-os-round-1-design.md).
Snapshots inspected: ByteDance-Seed/m3-agent `0e3e419` (`master`, 2026-02-12); MemTensor/MemOS
`28dfb4e` (`main`, 2026-09-01); letta-ai/letta `archive` branch `56ba9c2` (2026-08-13, the retired
V1 Python server) and letta-ai/letta-code `680cba4` (2026-09-02); mem0ai/mem0 `9a7924b`
(2026-09-02); getzep/graphiti `11538f6` (2026-09-01); agiresearch/A-mem and wangyu-ustc/Mem-alpha
read at `main` through the GitHub contents API on 2026-09-03.

## 1. M3-Agent: memorization operations, retrieval payload, stopping rule

Memorization is four operations in [`mmagent/memory_processing.py::process_memories`](https://github.com/ByteDance-Seed/m3-agent/blob/master/mmagent/memory_processing.py):

- Episodic: always `insert_memory` → `VideoGraph.add_text_node` plus one `add_edge` per parsed entity. No dedup, no conflict check.
- Semantic: compare against `get_connected_nodes(type=['semantic'])` whose entity set is a superset of the new one; `similarity > 0.85` calls `reinforce_node` (+1 on every incident edge), `similarity < negative_threshold` calls `weaken_node` (−1, edges deleted at weight ≤ 0), else insert.
- `negative_threshold = 0` requires a negative cosine, so with a modern text embedder the contradiction path is dead code — and its branch also sets `create_new_node = False`, contradicting the comment above it, so a detected contradiction retires nothing and records nothing.
- Read-time conflict handling: `videograph.get_entity_info(drop_threshold=0.9)` collapses semantic pairs above 0.9 by higher incident edge weight; `fix_collisions(mode='eq_only'|'argmax'|'dropout')` picks one face/voice equivalence per entity by max edge weight, breaking ties on `random.random() < 0.5`.

What "control" retrieves is deliberately thin (`mmagent/retrieve.py::search`): a dict
`{"CLIP_<id>": [memory strings]}` sorted by clip id, JSON-dumped into a user turn prefixed
`"Searched knowledge: "`. Strings only — no confidence, node ids, provenance, or timestamps beyond
the clip id. Budget per step is `configs/processing_config.json` `topk: 2` clips (all text nodes of
each clip); the character-id path uses `mem_wise=True, topk=20` capped on memory count.
`data["currenr_clips"]` suppresses already-shown clips, and an empty result appends
`"(The search result is empty. Please try searching from another perspective.)"`.

Stopping is a parsed action, not a learned halt: `m3_agent/control.py` matches
`r"Action: \[(.*)\].*Content: (.*)"` and ends on `[Answer]`. The hard cap is `total_round: 5`; the
last round appends `"(The Action of this round must be [Answer]. If there is insufficient
information, you can make reasonable guesses.)"`. The paper
([arXiv:2508.09736](https://arxiv.org/abs/2508.09736)) trains `M3-Agent-Control` with DAPO from
`control-32b-prompt`, binary GPT-4o-judged reward, group-normalised advantage, H = 5 rounds, and a
prompt rule that each new query differ from the previous.

## 2. MemOS: scheduler loop, lifecycle states, context assembly

The loop is a labelled task queue, not a planner. `src/memos/mem_scheduler/schemas/task_schemas.py` holds the whole vocabulary — `query`, `answer`, `add`, `mem_read`, `mem_organize`, `mem_dream`, `mem_update`, `mem_archive`, `api_mix_search`, `pref_add`, `mem_feedback` — each with a handler under `mem_scheduler/task_schedule_modules/handlers/` and a `TaskPriorityLevel` 1–3; `GeneralScheduler` builds a dispatch map from `SchedulerHandlerRegistry`. Budgets sit in `schemas/general_schemas.py`: `DEFAULT_CONSUME_BATCH = 3`, `DEFAULT_CONSUME_INTERVAL_SECONDS = 0.01`, `DEFAULT_THREAD_POOL_MAX_WORKERS = 50`, `DEFAULT_WORKING_MEM_MONITOR_SIZE_LIMIT = 30`, `DEFAULT_ACTIVATION_MEM_MONITOR_SIZE_LIMIT = 20`, `DEFAULT_TOP_K = 5`, `DEFAULT_WEIGHT_VECTOR_FOR_RANKING = [0.9, 0.05, 0.05]`.

Lifecycle is two orthogonal fields in `src/memos/memories/textual/item.py`:
`TextualMemoryMetadata.status: Literal["activated", "resolving", "archived", "deleted"]` (where
`resolving` means "updating with conflicting/duplicating new memories"), and
`TreeNodeTextualMemoryMetadata.memory_type` ∈ {`WorkingMemory`, `LongTermMemory`, `UserMemory`,
`OuterMemory`, `ToolSchemaMemory`, `ToolTrajectoryMemory`, `RawFileMemory`, `SkillMemory`,
`PreferenceMemory`, `Context`}. Versioning is rollback-shaped: `version: int`,
`history: list[ArchivedTextualMemory]`, `evolve_to`, `covered_history`, and
`ArchivedTextualMemory.update_type: Literal["conflict", "duplicate", "extract", "unrelated",
"feedback"]` — a reason code per archived version.

Conflict resolution is `tree_text_memory/organize/handler.py::NodeHandler`: `detect` (`EMBEDDING_THRESHOLD = 0.8`, then an LLM judgement), `resolve` (LLM fusion over only `["key", "background", "confidence", "updated_at"]`), `_hard_update` (newest `updated_at` wins if the LLM cannot resolve), and `_resolve_in_graph`, which sets both parents `status = "archived"`, inherits their edges onto the merged node, and adds `MERGED_TO` edges. Background restructuring is a separate priority queue: `organize/reorganizer.py::QueueMessage` with `op: Literal["add", "remove", "merge", "update", "end"]` and `op_priority = {"add": 2, "remove": 2, "merge": 1, "end": 0}`.

"Context" is not compiled; it is whatever survives in `WorkingMemory`. `mem_scheduler/base_mixins/memory_ops.py::replace_working_memory` drops every `mode:fast` item, reranks against `query_monitors` history, then LLM-filters unrelated and redundant items (`memory_manage_modules/memory_filter.py`). `MemCube` (`mem_cube/general.py`) is only the container of `text_mem`, `act_mem`, `para_mem`, `pref_mem`. `MemOperator` appears nowhere in `src/` at this snapshot — paper vocabulary only (unverified in code).

## 3. Letta/MemGPT: context window compilation

`letta-ai/letta` `main` is now a landing page; source moved to `letta-ai/letta-code`, and the V1
code below is the `archive` branch. `compile_in_context_memory` no longer exists. Compilation is two
functions. `letta/schemas/memory.py::Memory.compile(tool_usage_rules, sources, max_files_open,
llm_config, client_skills)` (async wrapper `compile_async` offloads to a thread) writes, in order:

1. `<memory_blocks>`, preamble "The following memory blocks are currently engaged in your core memory unit:", each block as `<{label}>` → `<description>` → `<metadata>` (`read_only=true`, `chars_current=`, `chars_limit=`) → `<value>`. Variants `_render_memory_blocks_standard`, `_render_memory_blocks_line_numbered` (only `sleeptime_agent`/`memgpt_v2_agent`/`letta_v1_agent` on Anthropic endpoints), `_render_memory_blocks_git`; `react_agent`/`workflow_agent` get no blocks.
2. `<tool_usage_rules>` from `ToolRulesSolver.compile_tool_rule_prompts()`.
3. `<directories>` with `<file_limits>` (`current_files_open=`, `max_files_open=`) and one `<file status="open|closed">` per file block carrying the same chars pair. `<memory_filesystem>` and `<available_skills>` render separately; skills are request-scoped and deliberately not persisted into the compiled system prompt.

`letta/prompts/prompt_generator.py::compile_system_message_async` then appends
`compile_memory_metadata_block(...)` as `memory_with_sources + "\n\n" + memory_metadata_string` and
substitutes the whole thing for the protected `{CORE_MEMORY}` variable, appending it if the system
prompt omits the variable. That block is the out-of-context inventory: `<memory_metadata>` with
`AGENT_ID`, `CONVERSATION_ID`, `System prompt last recompiled: <local time>`, "N previous messages
… stored in recall memory", "N total memories you created are stored in archival memory (use tools
to access them)", "Available archival memory tags: …".

Budgets: `CORE_MEMORY_BLOCK_CHAR_LIMIT = 100000` per block, surfaced to the model as `chars_limit` rather than truncated silently; `DEFAULT_MAX_FILES_OPEN = 5` (`MAX_FILES_OPEN_LIMIT = 1000`); `RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE = 5`. History trimming is separate: `SummarizationMode.STATIC_MESSAGE_BUFFER` or `PARTIAL_EVICT_MESSAGE_BUFFER` with `message_buffer_limit = 10`, `message_buffer_min = 3`, `partial_evict_summarizer_percentage = 0.30`, and `services/summarizer/thresholds.get_compaction_trigger_threshold` firing at 90% of the context window for the GPT-5 family, 100% otherwise.

## 4. Mem0 v3 and Zep/Graphiti: operation vocabulary and the context block

Mem0's ADD/UPDATE/DELETE/NONE vocabulary is now dead code. `mem0/configs/prompts.py` still defines
`DEFAULT_UPDATE_MEMORY_PROMPT` ("ADD … UPDATE … DELETE … NONE: Make no change") and
`get_update_memory_messages`, but at `9a7924b` the only callers are `tests/configs/test_prompts.py`.
Both live add paths (`mem0/memory/main.py:942` and `:2604`) use `ADDITIVE_EXTRACTION_PROMPT`, whose
own text says "Your sole operation is ADD". Supersession moved to link chains: the extractor emits
`linked_memory_ids`, and `mem0/client/main.py::delete` takes `delete_linked: bool`, "also delete the
older memories this one superseded (the v3 `linked_memory_ids` chain), transitively … the
delete-side counterpart of `latest_only`". OSS default is still `MemoryConfig.version = "v1.1"`;
platform migration notes record `async_mode` as removed because ingestion is "async by default".

Graphiti keeps a decision vocabulary but constrains it to a selection over a bounded candidate set. `graphiti_core/prompts/dedupe_edges.py::EdgeDuplicate` is two integer index lists: `duplicate_facts` (indices from `EXISTING FACTS` only) and `contradicted_facts` (either list), over one continuous idx space, with "NEVER mark facts with key differences as duplicates" and the rule that a fact can be both duplicate and contradicted. Deterministic code writes: `utils/maintenance/edge_operations.py::resolve_extracted_edge` sets `resolved_edge.invalid_at = candidate.valid_at` and `expired_at = utc_now()` — never a delete.

The context block is one template, `graphiti_core/search/search_helpers.py::search_results_to_context_string`:
a preamble ("Facts are considered valid between their valid_at and invalid_at dates. Facts with an
invalid_at date of 'Present' are considered valid."), then `<FACTS>` (`fact`, `valid_at`,
`invalid_at`), `<ENTITIES>` (`entity_name`, `summary`), `<EPISODES>` (`source_description`,
`content`), `<COMMUNITIES>` (`community_name`, `summary`). No ids, confidence, or scores. Zep's
hosted block adds a user summary on top: `thread.get_user_context()` returns `<USER_SUMMARY>` then
`<FACTS>` with dates inline as `"fact (2024-11-14 02:13:19+00:00 - present)"`; the `mode`
("summarized") parameter was removed for latency, and section membership plus per-section caps now
come from context templates — `%{user_summary}`, `%{thread_summaries}`, `%{entities}`,
`%{edges limit=10}`, `%{episodes}` ([retrieving context](https://help.getzep.com/retrieving-context),
[context types](https://blog.getzep.com/zep-context-types/)).

## 5. Anthropic and OpenAI public guidance on compiled context

Anthropic, [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents):
organise prompts "into distinct sections (like `<background_information>`, `<instructions>`, `## Tool guidance`, `## Output description`, etc)" using "XML tagging or Markdown headers to delineate these sections", "striving for the minimal set of information that fully outlines your expected behavior". Retrieval should be just-in-time — agents "maintain lightweight identifiers (file paths, stored queries, web links, etc.) and use these references to dynamically load data into context at runtime using tools" — with a hybrid that retrieves "some data up front for speed". Metadata is named as signal: "Folder hierarchies, naming conventions, and timestamps all provide important signals that help both humans and agents understand how and when to utilize information." Long-horizon primitives are compaction ("distills the contents of a context window in a high-fidelity manner"), structured note-taking persisted outside the window, and sub-agent isolation.

OpenAI, [Context engineering for personalization](https://developers.openai.com/cookbook/examples/agents_sdk/context_personalization)
(Agents SDK cookbook, 2026-01-05), is the closest published analogue of a compiled bundle with
provenance and freshness. State is `global_memory = {"notes": []}` and
`session_memory = {"notes": []}`; each note is `{"text", "last_update_date" (YYYY-MM-DD),
"keywords" (1–3 tags)}`. Capture is a tool, `save_memory_note()`, admitting only facts that are
"Durable: likely to remain true across trips" and "Actionable: changes recommendations or
constraints", rejecting speculation, instructions, and sensitive PII. Consolidation is a separate
LLM pass, `consolidate_memory()`, dropping ephemeral notes containing "this time"/"this trip",
removing exact duplicates, merging near-duplicates into one canonical version, and on conflict
"keep[ing] the one with the most recent `last_update_date`", preferring `SESSION_NOTES` on ties,
then clearing `session_memory["notes"]`. Rendering is per-section top-k by recency —
`render_global_memories_md()` at k = 6, `render_session_memories_md()` at k = 8 — inside a
`<memories>` delimiter with `GLOBAL memory:` / `SESSION memory:` sub-headings, ordered instructions
→ profile (YAML frontmatter) → memories → memory policy. Session notes are re-injected only when a
trim event sets `inject_session_memories_next_turn = True`.

## 6. Proposal-validation and rollback patterns worth borrowing

ABot-AgentOS ([arXiv:2607.10350](https://arxiv.org/abs/2607.10350)) is the only reviewed system with a stated accept criterion. Its loop is Diagnose → Propose → Compile → Gate over "lifecycle-managed JSON DSL records" carrying target layer, triggering condition, permitted action, safety constraints, provenance, validation result, and version id. The gate is `Accept(a) = 𝕀[ΔS_target(a) ≥ τ_gain ∧ ΔS_reg(a) ≥ −τ_reg]` — a minimum target gain *and* a bounded regression — with the leakage rule "No evo-asset generated from split t is used during inference on split t", and rollback to the previous snapshot when full-set or per-skill behaviour degrades. Its write-path maintenance is Merge (only where provenance, temporal context, and identity evidence are compatible), Supersede via temporal edges "rather than deleted immediately", and selective writing.

A-MEM ([arXiv:2502.12110](https://arxiv.org/abs/2502.12110), `agentic_memory/memory_system.py`) is
the counter-example. Its evolution prompt returns `{"should_evolve", "actions": ["strengthen",
"update_neighbor"], "suggested_connections", "tags_to_update", "new_context_neighborhood",
"new_tags_neighborhood"}`, and `update_neighbor` assigns `notetmp.tags` and `notetmp.context` in
place on the k nearest neighbours. `MemoryNote.evolution_history` is declared, initialised and
serialised but never appended to anywhere in the file, so an LLM rewrite of a neighbour's meaning is
unrecoverable. Consolidation fires on a bare counter, `evo_cnt % evo_threshold == 0`, `evo_threshold = 100`.

Mem-α ([arXiv:2509.25911](https://arxiv.org/abs/2509.25911), [repo](https://github.com/wangyu-ustc/Mem-alpha)) gives a metric shape rather than a safety shape. Tools are `new_memory_insert(memory_type, content)`, `memory_update(memory_type, new_content, memory_id)`, `memory_delete(memory_type, memory_id)`, `memory_search(...)` over core/episodic/semantic. Validation is deterministic and pre-commit: `_ensure_memory_type_enabled`, `_content_exists` skips a duplicate insert, and core memory "cannot be inserted. Use memory_update". Two prompt modes exist — capture, and a consolidation mode instructed to do "Redundancy Elimination" via `memory_delete`/`memory_update` and "Information Synthesis" via `memory_update`/`memory_insert`. Training is GRPO, `max_turns=5`, reward = task accuracy + `compression_ratio_weight=0.05` + `function_content_reward_weight=0.1`.

Letta's sleeptime agent is the nearest thing to a two-phase commit: `BASE_SLEEPTIME_TOOLS = ["memory_replace", "memory_insert", "memory_rethink", "memory_finish_edits"]`, with an explicit terminal tool — but its trigger is the timer MindBridge already rejects, `turns_counter % group.sleeptime_agent_frequency == 0` (`letta/groups/sleeptime_multi_agent_v3.py`). The [Memory in the Age of AI Agents survey](https://arxiv.org/abs/2512.13564) §5.2 taxonomises evolution as consolidation, updating (External Memory Update, Model Editing) and forgetting (time-, frequency-, importance-driven) but records no validation, verification, or rollback mechanism at all: the auditable-operation-log position is not covered ground.

## 7. Fast/slow separation: acknowledge before indexing, and readiness

MemOS is the only reviewed system with a first-class raw-then-enriched record. `AddRequest` in
`src/memos/api/product_models.py` has two independent switches: `async_mode: Literal["async",
"sync"] = "async"` ("enqueue background add (non-blocking)") and `mode: Literal["fast", "fine"] |
None`, "(Internal) Add mode used only when `async_mode='sync'`". A fast add writes a real item
marked `TextualMemoryMetadata.is_fast = True` — "carrying raw memory contents that haven't been
edited by llms yet" — tagged `mode:fast`, plus `working_binding: str`, "the working memory id
binding of the (fast) memory", stamped in `tree_text_memory/organize/manager.py` as
`f"[working_binding:{working_id}] direct built from raw inputs"`. Readiness is expressed by
`evolve_to: list[str]`, "recording which new memory nodes it 'evolves' to after llm extraction",
and by exclusion: `replace_working_memory` filters every `mode:fast` item before reranking, and
`is_fast`/`evolve_to`/`working_binding`/`version`/`history` all appear in
`retrieve/recall._LIGHTWEIGHT_VECTOR_RETURN_FIELDS`, so a reader can tell a raw record from a
settled one. There is no readiness endpoint and no dedicated `status` value — `status` stays
`activated` — and no documented time-to-searchable target (unverified).

Everyone else acknowledges early and exposes no readiness. Graphiti's server (`server/graph_service/routers/ingest.py`) returns `HTTP_202_ACCEPTED` with `Result(message='Messages added to processing queue', success=True)` from an `AsyncWorker` wrapping a plain in-process `asyncio.Queue`; `stop()` drains it with `get_nowait()`, so unprocessed work is lost at shutdown, and `routers/retrieve.py` has no status route. Zep's hosted docs say only that "Requests to add data to the same graph are completed sequentially … processing may be slow for large datasets", with no readiness field or latency figure. Mem0 v3 made async the only behaviour (`async_mode` deleted as a parameter) without adding a readiness surface. Letta stays synchronous on the write path and moves *interpretation* off-line into sleeptime agents; its readiness signal to the model is the `<memory_metadata>` counts, not per-record state.

The general name for MemOS's shape is write-behind (write-back) indexing: commit the authoritative
row, acknowledge, let the derived index catch up, and keep a per-record marker so readers know which
rows are still catching up. No reviewed system publishes a time-to-searchable objective.

## Recommendations for MindBridge A/B/C

Fast capture (A):

1. Borrow MemOS `is_fast` as a record-level marker, not a new status — a fast row is authoritative evidence that is not yet enriched, and no second table is needed.
2. Borrow `evolve_to`/`working_binding` as the readiness link: the fast row names the derived rows that replaced it, which is the lineage `capture_queue` needs to be inspectable.
3. Reject Graphiti's in-process `asyncio.Queue`; its `get_nowait()` shutdown drain is exactly the loss `capture_queue` in SQLite exists to prevent.
4. Expose `pending_captures` as a count plus per-record marker (Letta's `<memory_metadata>` style), not a global ready flag — no reviewed system has one.
5. Set the time-to-searchable objective from our own hardware runs; there is no external baseline published to match.

Control plane (B):

1. Borrow Graphiti's proposal shape — indices into a shown bounded candidate list, not free-text ids — which removes hallucinated ids as a failure class and matches the "all in the shown set" validation already specified.
2. Borrow M3-Agent's shown-set suppression (`currenr_clips`) so a later loop pass cannot re-cite evidence already presented.
3. Borrow MemOS `ArchivedTextualMemory.update_type` as a reason code on `memory_operations` beside `intent`, making the log queryable by *why* and not only by *what*.
4. Borrow MemOS `_resolve_in_graph` discipline for any future MERGE: archive both parents, inherit links, add an explicit `MERGED_TO` edge — reversible by construction.
5. Borrow ABot's `ΔS_target ≥ τ_gain ∧ ΔS_reg ≥ −τ_reg` plus its no-leakage split rule as the loop's *evaluation* gate before defaults change, not as a per-operation gate.
6. Borrow Mem-α's composite reward as slow-loop metrics — task accuracy plus a compression ratio — since consolidation precision alone cannot catch a loop that consolidates nothing.
7. Reject A-MEM `update_neighbor`: in-place LLM rewrite of neighbours with a declared but never-written `evolution_history` is precisely what CORRECT-by-new-version prevents.
8. Reject M3-Agent edge-weight voting: at this snapshot the contradiction branch is unreachable and also suppresses the correction, so frequency is not truth and a dead conflict path is worse than none.
9. Reject Letta's `memory_finish_edits` two-phase protocol and counter triggers (`evo_cnt % 100`, `turns_counter % frequency`); one validated atomic batch and the typed `MemoryTrigger` set already cover both.
10. Read Mem0's retreat to additive-only as a warning, not a model: keep CORRECT as a retirement that adds a version, never as an LLM-authored overwrite.

Context compiler (C):

1. Borrow Graphiti's per-fact `valid_at`/`invalid_at` rendering and its one-line preamble on how to read them; `render()` already emits ids and confidence, and MindBridge stores the bounds already.
2. Borrow Letta's in-band budget disclosure — chars used against chars limit, plus an out-of-context inventory with an access hint — rather than only an `omitted` integer.
3. Borrow Zep and OpenAI per-section caps (`%{edges limit=10}`; k = 6 / k = 8) as the tie-breaker inside `max_items`, but make the cap explicit only if a benchmark shows one section starving another.
4. Borrow OpenAI's freshness-wins rule for the *ordering* of a reported conflict while keeping the compiler's refusal to resolve conflicts.
5. Reject Zep's "summarized" context mode: it was removed upstream for latency, which argues against any LLM pass inside `compile()`.
6. Reject M3-Agent's retrieve-until-answer loop for round 1; if an iterative compiler ever appears, borrow only its hard round cap and forced terminal action on the last round.
