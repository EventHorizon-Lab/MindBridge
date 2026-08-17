# Dataset -> Pipeline Schema Reference

Source of truth for six loader implementations. Each section covers: raw
files, how to build `{role, content, timestamp}` messages, the retrieval
scope (= one `user_id`), the questions, the exact payload keys the vendored
pipeline reads, and gotchas. All paths are relative to the repo root unless
absolute.

Pipelines were read directly from
`benchmarks/aml/pipelines/{locomo-refined,longmemeval-s,personamem,beam,clbench,scriptmem}/`
(read-only, not edited).

---

## 1. LoCoMo (`benchmarks/aml/pipelines/locomo-refined/pipeline.py`)

### Files
- `.benchmarks/locomo/data/locomo10.json` — a JSON list of 10 samples.
  Present and fully usable (10 samples, ~199 QA pairs each in the example
  checked).

### Sample shape
```python
sample = {
  "sample_id": "conv-26",
  "conversation": {
    "speaker_a": "Caroline", "speaker_b": "Melanie",
    "session_1_date_time": "1:56 pm on 8 May, 2023",
    "session_1": [
      {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel! ..."},
      ...
    ],
    "session_2_date_time": "...", "session_2": [...],
    ...  # session_N / session_N_date_time repeat, N = 1..however many exist
  },
  "qa": [
    {"question": "...", "answer": "...", "evidence": ["D1:3"], "category": 2}
  ],
  "event_summary": {...}, "observation": {...}, "session_summary": {...},
}
```

### Messages
Flatten by walking `session_N` keys in numeric order (sessions are not
necessarily contiguous — discover them by regex `session_(\d+)$`, sorted by
N, and pull the matching `session_N_date_time`):

```python
import re

conv = sample["conversation"]
session_nums = sorted(int(m.group(1)) for k in conv if (m := re.fullmatch(r"session_(\d+)", k)))
messages = []
for n in session_nums:
    date_time = conv.get(f"session_{n}_date_time")  # e.g. "1:56 pm on 8 May, 2023"
    for turn in conv[f"session_{n}"]:
        role = "user" if turn["speaker"] == conv["speaker_a"] else "assistant"
        messages.append(
            {
                "role": role,
                "content": turn["text"],
                "timestamp": date_time,  # SESSION-level, not per-turn
            }
        )
```
- Real timestamps exist but only at **session granularity** (free-text like
  `"1:56 pm on 8 May, 2023"`, not ISO-8601) — every turn in a session shares
  the same timestamp string. `dia_id` (e.g. `"D1:3"`) is the evidence-pointer
  unit, not a timestamp.
- `speaker_a`/`speaker_b` are the two participant names; role mapping is
  arbitrary (there's no inherent "user" vs "assistant" — LoCoMo is a
  peer dialogue, not a chatbot transcript). Keep the original speaker name
  alongside role if the harness needs it, since the answer prompt template
  addresses both speakers by name.

### Retrieval scope
One **sample** (`sample_id`, e.g. `"conv-26"`) = one scope = one `user_id`.
All ~199 QA pairs for that sample share the same conversation haystack.

### Questions
`sample["qa"]` is a list of `{question, answer, evidence, category}`.
**No id field** — must be synthesized, e.g. `f"{sample_id}:qa{index:04d}"`.
`category` is an int (1-5); not read by the pipeline.

### Payload keys the pipeline reads
`answer()` (via `render_answer_prompt`) and `evaluate()` (via
`render_accuracy_prompt` / `gold_answer`) read, per record:

| Key | Required | Notes |
|---|---|---|
| `id` | required | `rows()`/`answer()`/`evaluate()` all key off this; must be present and match between input file and generated-answers file |
| `question` | required | raw string, inserted verbatim |
| `speaker_1_name` | optional | default `"speaker 1"` |
| `speaker_1_memories` | optional | falls back to `retrieved_context`, then `memories`, then `""` |
| `speaker_2_name` | optional | default `"speaker 2"` |
| `speaker_2_memories` | optional | default `""` (no fallback to shared context) |
| `retrieved_context` / `memories` | optional | used only as the fallback for `speaker_1_memories` when that key is absent |
| one of `gold_answer` / `golden_answer` / `reference_answer` / `correct_answer` | **required for `evaluate`** | first key found wins; raises `ValueError` if none present |

`memory_text()` accepts str, list (joined with `\n`), or dict (JSON-dumped) —
so `speaker_1_memories` can be a list of memory strings, not just one blob.

### Gotchas
- Dataset's gold key is **`answer`**, but the pipeline looks for
  `gold_answer`/`golden_answer`/`reference_answer`/`correct_answer` — the
  loader must rename `qa["answer"]` into one of those four keys (recommend
  `gold_answer`) when building the input JSONL.
  Do not confuse with `qa["question"]`, which the pipeline reads directly.
- No `id` field exists in the raw QA objects — must construct one
  (`f"{sample_id}:qa{index:04d}"` is a safe convention, and keeps ids stable
  across reruns).
- `evaluate()` requires the `answers` file's id set to exactly equal the
  `input` file's id set (`set(items) != set(answers)` raises `SystemExit`) —
  any synthesized id scheme must be applied identically at both answer- and
  eval-time.
- Session numbering is not guaranteed contiguous/sorted by string order
  (`session_10` sorts before `session_2` lexically) — sort numerically.

---

## 2. LongMemEval (`benchmarks/aml/pipelines/longmemeval-s/pipeline.py`)

### Files present
`.benchmarks/longmemeval/` contains three **fully materialized JSON files**
(not git-lfs pointers — verified with `file`, they are plain ASCII JSON):
- `longmemeval_s` (278 MB, 500 questions, ~40-60 haystack sessions/question) — **this is the "S" split the vendored pipeline directory is named for; use this one.**
- `longmemeval_m` (2.7 GB, larger haystacks)
- `longmemeval_oracle` (15 MB, 500 questions, only 2-3 gold sessions per question — a reduced/oracle-retrieval variant, not the full haystack)

All three share the same JSON schema. Present and usable as-is; no
extraction/download step needed.

### Sample shape (one element of the top-level JSON list)
```python
{
    "question_id": "e47becba",
    "question_type": "single-session-user",  # or multi-session / temporal-reasoning / knowledge-update / single-session-assistant / single-session-preference
    "question": "What degree did I graduate with?",
    "answer": "Business Administration",
    "question_date": "2023/05/30 (Tue) 23:40",
    "haystack_dates": [
        "2023/05/20 (Sat) 02:21",
        "2023/05/20 (Sat) 02:57",
        ...,
    ],  # one date per haystack session, same order/index as haystack_session_ids
    "haystack_session_ids": ["sharegpt_yywfIrx_0", "85a1be56_1", ...],
    "haystack_sessions": [
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...,
        ],  # session 0's turns
        [...],  # session 1's turns
        ...,
    ],
    "answer_session_ids": ["answer_280352e9"],  # which haystack_session_ids are gold-relevant
}
```
Question-type distribution across `longmemeval_s` (500 total):
`multi-session` 133, `temporal-reasoning` 133, `knowledge-update` 78,
`single-session-user` 70, `single-session-assistant` 56,
`single-session-preference` 30.

### Messages
```python
messages = []
for session, session_id, date in zip(
    q["haystack_sessions"], q["haystack_session_ids"], q["haystack_dates"]
):
    for turn in session:
        messages.append(
            {
                "role": turn["role"],  # "user" or "assistant" already
                "content": turn["content"],
                "timestamp": date,  # SESSION-level again, format "2023/05/20 (Sat) 02:21"
            }
        )
```
- Real timestamps exist, at session granularity, format
  `"YYYY/MM/DD (Dow) HH:MM"`. Every turn within a session shares that
  session's date. `question_date` (same format) is the time the question is
  asked — useful as an "as-of" anchor for relative-time questions.
- Sessions are NOT shared across questions — verified `haystack_session_ids`
  for question 0 and question 1 have **zero overlap** — each question owns
  its own independent haystack.

### Retrieval scope
One **question** (`question_id`) = one scope = one `user_id`. Unlike
LoCoMo/CL-bench, the "conversation" here is actually a bag of distinct
distractor + gold sessions (a synthetic long-term-memory haystack), not one
continuous dialogue — ingest all of a question's `haystack_sessions` as that
question's memory corpus.

### Questions
Same JSON records carry both the haystack and the question (`question`,
`answer`, `question_id`) — there's no separate question file.

### Payload keys the pipeline reads
Identical contract to LoCoMo-refined (this pipeline file is a byte-for-byte
copy per its own header comment "uses exactly the same answer and
evaluation contracts"). Same table as LoCoMo section 1 above: `id`
(required), `question` (required), `speaker_1_name`/`speaker_1_memories`/
`speaker_2_name`/`speaker_2_memories` (optional, with `retrieved_context`/
`memories` fallback), and one of `gold_answer`/`golden_answer`/
`reference_answer`/`correct_answer` (required for evaluate).

### Gotchas
- Dataset's id key is **`question_id`**, pipeline wants **`id`** — rename.
- Dataset's gold key is **`answer`**, pipeline wants `gold_answer` (or one
  of its 3 aliases) — rename.
- LongMemEval is single-user (no `speaker_b`), so `speaker_2_name`/
  `speaker_2_memories` should simply be omitted (template default renders an
  empty section for "speaker 2" — harmless but slightly odd cosmetically;
  consider passing a single set of memories under `speaker_1_memories` only).
- `longmemeval_m` is 2.7 GB — do not `json.load()` it eagerly in a hot path;
  if it's ever used, stream/parse incrementally. `longmemeval_s` (278 MB)
  is the one aligned with this pipeline directory's name and is the
  practical choice.
- `longmemeval_oracle` is a *different* (reduced) haystack per question
  (only the gold sessions) — do not accidentally substitute it for `_s` when
  the task calls for testing real retrieval over a full haystack.

---

## 3. PersonaMem (`benchmarks/aml/pipelines/personamem/pipeline_v1.py` and `pipeline_v2.py`)

**v1 and v2 are different datasets** (different generation runs, different
schemas, different question design — v1 is 4-option MCQ over persona-aware
chat; v2 supports both MCQ and open-ended "generative" grading and adds
richer preference metadata). They are not two splits of one dataset — do
not merge them.

### 3a. PersonaMem v1 (`.benchmarks/personamem-v1/`)

#### Files
- `questions_32k.csv`, `questions_128k.csv`, `questions_1M.csv` — one CSV
  per context-length variant, 589 rows each (20 personas) for the 32k file
  (verified).
- `shared_contexts_32k.jsonl`, `shared_contexts_128k.jsonl`,
  `shared_contexts_1M.jsonl` — each **line** is a single-key JSON object:
  `{shared_context_id: [ {role, content}, ... ]}` (183 messages in the
  example checked).

Columns of `questions_*.csv`:
```
persona_id, question_id, question_type, topic, context_length_in_tokens,
context_length_in_letters, distance_to_ref_in_blocks, distance_to_ref_in_tokens,
num_irrelevant_tokens, distance_to_ref_proportion_in_context,
user_question_or_message, correct_answer, all_options,
shared_context_id, end_index_in_shared_context
```

#### Messages
```python
import json

shared = {}  # shared_context_id -> list[{"role","content"}]
with open("shared_contexts_32k.jsonl") as f:
    for line in f:
        shared.update(json.loads(line))

row = ...  # one CSV row
context_messages = shared[row["shared_context_id"]][: int(row["end_index_in_shared_context"])]
```
- Messages have only `role`/`content` — **no timestamps at all**; omit the
  `timestamp` field (or set `None`) when building the flattened list.
- Content strings are already prefixed with literal `"User: "` /
  `"Assistant: "` text inside the string itself (in addition to the `role`
  key) — this is upstream's own formatting; keep it as-is, don't strip it,
  since the official prompt was built assuming it's there.
- `end_index_in_shared_context` truncates the shared context to exactly the
  slice that question was generated against — this becomes `context_messages`
  for the pipeline, per its own docstring.

#### Retrieval scope
One **`shared_context_id`** is a full simulated chat history for one
persona; multiple questions reuse slices of the same `shared_context_id`.
Practically, one `persona_id` (equivalently, one `shared_context_id`, they
are 1:1 per context-length file) = one scope = one `user_id`.

#### Questions
`questions_*.csv`, keyed by `question_id`. `correct_answer` is a gold-option
marker like `"(c)"`. `all_options` is a **plain string** already formatted
as the four lettered options block (verified: it is literally a string
column, not JSON — matches the pipeline's docstring requirement exactly, do
NOT `json.loads`/reconstruct it into a list).

#### Payload keys the pipeline reads
From the pipeline's own docstring and code:

| Key | Required | Notes |
|---|---|---|
| `id` | required (`row_id`) | falls back to `question_id`, then `qid`, then row index |
| `context_messages` | required | must be a `list`; falls back to `context` key; raises `TypeError` otherwise |
| `question` | required | plain string |
| `all_options` | required | must be a `str` (raises `TypeError` if not) — the **original** options block text, not a reconstructed list |
| `correct_answer` | required for evaluate | gold option marker, e.g. `"(a)"` or `"a"`; stripped of `()` and whitespace, lowercased |

#### Gotchas
- Use `question_id` (dataset's native id) directly as `id` — `row_id()`
  already checks `id`/`question_id`/`qid` in that order, so **no rename
  needed**, just make sure the CSV's `question_id` value ends up on the
  built record.
- `all_options` must stay a string — do not split it into a Python list;
  the official inference/evaluation contract needs the verbatim options
  block appended after the instruction text.
- No conversation ends up scoped by `sample_id`/`conv-N` like LoCoMo —
  scope here is per-persona (`shared_context_id`).

### 3b. PersonaMem v2 (`.benchmarks/personamem-v2/`)

#### Files
- `benchmark/text/benchmark.csv` (5000 rows, 200 personas) — plus
  `train.csv`/`val.csv` splits of the same schema.
- `benchmark/multimodal/*.csv` — a separate multimodal variant, out of scope
  here unless requested.
- `data/chat_history_32k/*.json` and `data/chat_history_128k/*.json` — one
  file per **(persona, generation run)** pair (there can be more than one
  timestamped file per `personaN.json` suffix — e.g. two different
  `chat_history_25091*_persona0.json` files exist; **use the exact filename
  from the CSV's link column**, don't glob-guess by persona number).
- `data/raw_data/*.json` — raw persona generation data (referenced by
  `raw_persona_file` column; not required by the pipeline).
- `column_descriptions.md` — authoritative column documentation, already
  cross-checked below.

Columns of `benchmark.csv` (28 total; relevant ones):
```
persona_id, chat_history_32k_link, chat_history_128k_link, raw_persona_file,
short_persona, expanded_persona, user_query, correct_answer,
incorrect_answers, topic_query, preference, topic_preference,
conversation_scenario, pref_type, related_conversation_snippet, who,
updated, prev_pref, sensitive_info, total_tokens_in_chat_history_32k, ...
```

Chat history file shape:
```python
{
    "metadata": {
        "total_messages": 237,
        "final_token_count": 31983,
        "persona_id": 0,
        "input_filename": "...",
    },
    "chat_history": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...,
    ],
}
```

#### Messages
```python
import json, os

chat_path = os.path.join(".benchmarks/personamem-v2", row["chat_history_32k_link"])
history = json.load(open(chat_path))["chat_history"]
messages = [{"role": m["role"], "content": m["content"], "timestamp": None} for m in history]
```
- No per-message timestamps — omit.
- `user_query` CSV field is a **Python-repr string** (single-quoted dict,
  e.g. `"{'role': 'user', 'content': '...'}"`), not JSON — matches the
  pipeline's own `ast.literal_eval` fallback in `user_query_text()`. Verified
  by direct inspection.
- `incorrect_answers` CSV field **is valid JSON** (double-quoted list of
  strings) — verified with `json.loads`; matches the pipeline's
  `json.loads(incorrect_answers)` path.

#### Retrieval scope
One **`persona_id`** (equivalently one chat-history file) = one scope = one
`user_id`. 200 personas x up to 25 questions each = 5000 rows.

#### Questions
`benchmark.csv` rows. No `id`/`question_id`/`qid`/`sample_id` column exists
at all — `row_id()` will fall through to the enumerate index unless the
loader adds an explicit id (recommended: `f"persona{persona_id}-{row_index}"`
or similar, since the raw CSV provides nothing else unique per row).

#### Payload keys the pipeline reads
Two answer modes (`--mode mcq` / `--mode generative`) and two eval commands
(`evaluate-mcq` / `evaluate-narrow`):

| Key | Required | Notes |
|---|---|---|
| `id` | required (`row_id`) | checks `id`/`question_id`/`qid`/`sample_id`, else row index — **dataset provides none of these**, loader must synthesize |
| `chat_history` | required | falls back to `messages`, then `context_messages`; must be a list, else `TypeError` |
| `user_query` | required | falls back to `question`, then `query`; if the string looks like a dict literal (`starts with "{"`), it is `ast.literal_eval`'d and `.content`/`.text` extracted |
| `correct_answer` | required (MCQ mode) | plain string, becomes one MCQ option |
| `incorrect_answers` | required (MCQ mode) | list, or a JSON string of a list; raises `TypeError` if empty/missing |
| `persona_id` | optional | used only to seed the deterministic MCQ option shuffle (`random.seed(hash(f"{persona_id}_{query}"))`) — if omitted, shuffle seed changes but scoring is still self-consistent since mapping is stored alongside the generated answer |
| `preference` / `target_preference` | required for `evaluate-narrow` (generative judge) | chooses the narrow-positive vs narrow-negative judge prompt based on whether it starts with `"do not"` |

#### Gotchas
- No id column in the CSV at all — must synthesize one and keep it
  consistent between the `answer` and `evaluate-*` steps (same id-assignment
  order/logic both times, e.g. row index in file-read order).
- `chat_history_32k_link` (and `_128k_link`) are **file-specific**, not just
  persona-specific — a persona can have multiple chat-history snapshots;
  always resolve via the CSV's link column, never reconstruct the filename.
  Confirmed two on-disk files differ only by timestamp for `persona0`.
- MCQ-mode option shuffling is **deterministic and keyed by
  `persona_id` + query text** — if the loader passes a different/missing
  `persona_id` than upstream would have, the *order* of options differs
  from any published reference numbers, but correctness scoring is still
  internally consistent (the mapping used for grading is generated the same
  way it's graded).

---

## 4. BEAM (`benchmarks/aml/pipelines/beam/pipeline.py`)

### Files
`.benchmarks/beam/` is the full upstream research repo (git checkout, not
just a data dump). The actual per-conversation data lives under:
```
.benchmarks/beam/chats/{100K,500K,1M,10M}/{conv_id}/
    chat.json                        # the conversation transcript, batched
    user_messages.json                # user-only turns, batched (redundant view)
    topic.json                        # conversation metadata (id, category, title, theme, subtopics)
    probing_questions/probing_questions.json   # the questions, keyed by category
    labels.txt, relationships.txt, main_spec.txt, plan.txt, plan_new.txt  # generation scaffolding, not needed
    (binary generation-time cache files, not needed)
.benchmarks/beam/chats/{size}/topics.json      # index of all conv_id -> topic metadata for that size
```
Conversation counts per size (verified): 100K=20, 500K=35, 1M=35, 10M=10
conversation directories.

### `chat.json` shape
```python
[
    {
        "batch_number": 1,
        "turns": [
            [
                {
                    "role": "user",
                    "id": 0,
                    "time_anchor": "March-15-2024",
                    "index": "1,1",
                    "question_type": "main_question",
                    "content": "...",
                },
                {"role": "assistant", "id": 1, "content": "..."},
            ],
            ...,  # more turn-pairs in this batch
        ],
    },
    {"batch_number": 2, "turns": [...]},
    ...,
]
```

### Messages
```python
messages = []
for batch in chat_json:
    for turn_pair in batch["turns"]:
        for turn in turn_pair:
            messages.append(
                {
                    "role": turn["role"],
                    "content": turn["content"],
                    "timestamp": turn.get(
                        "time_anchor"
                    ),  # only present on some (user) turns; free text like "March-15-2024"
                }
            )
```
- Timestamps are **free-text month-day-year anchors** (`"March-15-2024"`),
  present inconsistently — typically only on the `user` turn that opens a
  batch, not on every turn. Treat as a coarse per-batch anchor; carry
  forward the last-seen anchor onto assistant turns if the harness needs
  every message timestamped, or leave assistant-turn timestamps `None`.
- `content` strings for early turns may contain a literal `"->-> 1,1"`
  suffix (an internal index marker from the generation harness) — harmless
  to keep verbatim since it's part of what the official pipeline would see
  too, but worth stripping if it pollutes retrieval embeddings.

### `probing_questions.json` shape
```python
{
    "abstention": [
        {
            "question": "...",
            "ideal_response": "...",
            "difficulty": "medium",
            "abstention_type": "missing_detail",
            "why_unanswerable": "...",
            "plan_reference": "Batch 3, Bullet 2",
            "rubric": ["..."],
        },
        ...,
    ],
    "contradiction_resolution": [...],
    "event_ordering": [...],
    "information_extraction": [...],
    "instruction_following": [...],
    "knowledge_update": [...],
    "multi_session_reasoning": [...],
    "preference_following": [...],
    "summarization": [...],
    "temporal_reasoning": [...],
}
```
The top-level **dict key is the category** and is exactly what upstream's
own `answer_generation.py` uses as `question_type`/category when iterating
(`for key in data.keys(): questions = data[key]`) — confirmed by reading
`src/answer_probing_questions/answer_generation.py`.

### Retrieval scope
One **conversation directory** (`{size}/{conv_id}`) = one scope = one
`user_id`. All of that conversation's `probing_questions.json` entries
(across all 10 categories) query the same haystack.

### Questions
`probing_questions/probing_questions.json`, grouped by category key. **No
id field on individual questions** — must synthesize, e.g.
`f"{size}-{conv_id}-{category}-{index:04d}"`.

### Payload keys the pipeline reads
`answer()` renders `ANSWER_GENERATION_FOR_RAG` via `render_answer_prompt` /
`context_text`; `evaluate()` renders the rubric judge via `rubric_items`:

| Key | Required | Notes |
|---|---|---|
| `id` | required (`rows()` enforces unique, non-empty ids — raises `ValueError` otherwise) | must synthesize (see above) |
| `question` | required | plain string |
| `context` / `retrieved_context` / `memories` | one required | first found wins, else falls back to assembling `speaker_1_memories`/`speaker_2_memories` (see LoCoMo-style keys), else raises `ValueError` |
| `rubric_nuggets` / `rubrics` / `rubric` | required | must be a non-empty list; each item is either a string or a dict with `rubric_criteria`/`criterion`/`text` — dataset's `rubric` key is already a **list of plain strings**, matches directly |
| `question_type` / `category` | optional | used to decide whether to additionally compute `event_ordering` metrics (Kendall-tau + F1 against the rubric list) — set this to the `probing_questions.json` category key (e.g. `"event_ordering"`) |

### Gotchas
- No `id` on raw questions — synthesize one, and it must be non-empty and
  globally unique across the whole answer/eval file pair (`rows()` raises if
  not).
- The dataset's `rubric` key already matches one of the pipeline's fallback
  names (`rubric_nuggets`→`rubrics`→`rubric`) — no rename needed, just make
  sure the key survives into the built record verbatim as `"rubric"`.
  Note `rubric` here is a list of plain strings (not dicts), which the
  pipeline handles directly.
- `event_ordering` category questions get special extra scoring
  (`align_with_llm` + `kendall_tau_b`) that expects the **generated answer**
  to be a newline-separated list of event snippets comparable against the
  rubric list — this only fires when `question_type == "event_ordering"`, so
  the loader must propagate the category string faithfully.
- `context` construction is entirely the loader's/harness's job (BEAM's raw
  data gives you the conversation to ingest, not pre-retrieved memories) —
  the pipeline only consumes whatever `context`/`speaker_*_memories` the
  harness produced after running its own retrieval over the ingested
  conversation.
- Timestamps are inconsistent/missing on many turns; don't assume every
  message has one.

---

## 5. CL-Bench (`benchmarks/aml/pipelines/clbench/pipeline.py`)

### Files
- `.benchmarks/clbench/CL-bench.jsonl` — 1899 lines, present and usable.

### Record shape
```python
{
    "messages": [
        {"role": "system", "content": "<persona/system prompt>"},
        {"role": "user", "content": "<huge reference document, ending in the actual question>"},
        # OR, for multi-turn records (verified: 2/4/6/8/10/12-message variants exist):
        # [system, user(doc), assistant(intro turn), user(short question)]
        # (message-count distribution sampled from first 500 lines: {2: 278, 4: 125, 6: 82, 8: 14, 10: 2, 12: 1})
    ],
    "rubrics": ["The response should ...", "The response should ...", ...],  # list of plain strings
    "metadata": {
        "task_id": "...",
        "context_id": "...",
        "context_category": "Rule System Application",
        "sub_category": "Game Mechanics",
    },
}
```
**Critically: the raw dataset has no separate `question` field.** The
actual question is the tail of the final `user` message's content (verified
example: a 158,789-character user message ending in
`"...This is my first time playing, what do Sightings Cards do?"`). There is
also no `qa_type`, `options`, or `system_prompt` top-level field, and no
`retrieval`/`msp_retrieval`/`speaker_*_retrieval` dict — those are all
pipeline-input concepts the loader/harness must construct.

### Messages / question split
```python
msgs = record["messages"]
system_prompt = next((m["content"] for m in msgs if m["role"] == "system"), "")
history = [m for m in msgs if m["role"] != "system"]  # everything except system
question = history[-1]["content"]  # the final user turn IS the question
conversation_history = history[:-1]  # ingest these as the memory corpus
# -> flatten conversation_history into {"role","content","timestamp": None} — no timestamps exist anywhere in this dataset
```

### Retrieval scope
One **JSONL record** (keyed by `metadata.task_id`) = one scope = one
`user_id`. Each record is an independent long-document-plus-question task;
there is no cross-record shared conversation.

### Questions
Same record supplies both the "conversation to remember" and the question
(the final user turn). `metadata.task_id` is the natural question id.

### Payload keys the pipeline reads
`build_answer_prompt()` / `collected_memories()` / `official_rubrics()`:

| Key | Required | Notes |
|---|---|---|
| `idx` / `id` / `question_id` / `task_id` | required (`row_id`), else falls back to `metadata.task_id`/`metadata.id`, else row index | dataset's `metadata.task_id` matches directly via the fallback chain — **no rename needed if `metadata` is preserved on the built record** |
| `system_prompt` | optional, default `""` | loader must extract from `messages` (see above) — raw record has no top-level `system_prompt` |
| `question` | required (used inside `format_structured_question`) | loader must extract from the last user turn — raw record has no top-level `question` |
| `qa_type` | optional | only `"single_choice"`/`"multi_select"`/`"ordering"` trigger option-lettering; CL-Bench raw data never has this key, so it always falls through to plain-question rendering — fine, no action needed |
| `options` | optional | list of strings; CL-Bench raw data never has this key — fine |
| `retrieval` (dict with `selected: [{created_at, text}, ...]`) | one of these four is read, first match wins | else `msp_retrieval`, else `speaker_a_retrieval` + `speaker_b_retrieval` (both optional, concatenated) — **entirely harness-produced**, not in raw dataset |
| `rubrics` | required for evaluate | list of strings or dicts with `rubric_criteria`; falls back to `metadata.rubrics` if top-level `rubrics` is absent — dataset's top-level `rubrics` (list of plain strings) matches directly, **no rename needed** |

### Gotchas
- **No `question` field exists in the raw dataset** — it must be sliced out
  of the last `user` message's content. Do not attempt to answer using the
  full 150K-character user message as "the question"; only its trailing
  sentence is the actual query the rubric grades against.
- **No `system_prompt` field exists** either — extract from the `system`
  role message.
- Preserve `metadata` verbatim on the built record so `row_id()`'s
  `metadata.task_id` fallback keeps working, or explicitly set `idx`/`id`
  to `metadata["task_id"]`.
- Multi-turn records (4/6/8+ messages) exist — don't hardcode "exactly 2
  messages"; always take "last user message" as the question and everything
  before it as history.
- No timestamps anywhere in this dataset.

---

## 6. ScriptMem (`benchmarks/aml/pipelines/scriptmem/pipeline.py`)

### Files
- `.benchmarks/scriptmem/data/public/questions.jsonl` — 457 questions, present and usable.
- `.benchmarks/scriptmem/data/public/conversations.jsonl` — 4 lines, **one per source work**, but each line's `conversation` field is only a placeholder: `{"format_example": {"speakers": [...], "session_1_date_time": "Unknown", "session_1": [{"type": "dialogue"|"narration", "speaker": ..., "dia_id": ..., "text": "Synthetic example ... schema."}]}}`.
- `.benchmarks/scriptmem/data/public/manifest.json` — corpus stats + file map.
- `.benchmarks/scriptmem/data/public/submission_template.json` — the exact shape `evaluate()` expects a submission in.
- `.benchmarks/scriptmem/data/raw/{angry,enemy,friends,man_earth}.json` — the `DATASET_FILES` the official `evaluate()`/`load_gold_records()` reads gold answers from. **Verified**: every sample's `conversation` field in all four raw files is **also** just `{"format_example": {...}}` — i.e. this download does **not** contain the real script transcripts (Friends / 12 Angry Men / The Man from Earth / An Enemy of the People).

### CRITICAL GOTCHA: no real conversation text is present
`.benchmarks/scriptmem/README.md` states explicitly: *"Please note ScriptMem
releases task-specific questions, options, reference answers, metadata, and
evaluation code; the original source texts are not included in this
repository."* This is confirmed at the data level — both
`data/public/conversations.jsonl` and every `conversation` field inside
`data/raw/*.json` contain only the literal placeholder object
`{"format_example": {...}}` with `"text": "Synthetic example utterance
showing the dialogue schema."` — **there is no real dialogue to ingest for
any of the 4 source works.** The questions, gold answers, and grading
machinery are all present and byte-exact; a MindBridge loader can compute
scores against a *real* memory corpus only if the actual script transcripts
(Friends episodes, 12 Angry Men, The Man from Earth, An Enemy of the People)
are sourced separately (likely copyrighted, out-of-repo). Absent that, the
best a loader can do is run the pipeline in a "conversation absent" mode
(e.g. skip ingestion, answer from the question's embedded option text only)
— which is not a meaningful memory-system test.

### `raw/{dataset}.json` shape (used for gold + qa_id construction)
```python
[
    {
        "sample_id": "conv-0",  # or omitted -> f"{source}-{sample_index}"
        "conversation": {"format_example": {...}},  # placeholder only, see above
        "qa": [
            {
                "qa_type": "single_choice" | "multi_select" | "ordering",
                "question": "...",
                "answer": "A. ..." or ["A. ...", "C. ..."],
            },
            ...,
        ],
    }
]
```

### `public/questions.jsonl` shape (already denormalized, one line per QA)
```python
{
    "qa_id": "angry:conv-0#q0000",
    "question_id": "angry:conv-0#q0000",
    "conversation_id": "angry:conv-0",
    "source": "angry",
    "sample_id": "conv-0",
    "qa_index": 0,
    "qa_type": "single_choice",
    "question": "...",
    "option": ["A. ...", "B. ...", ...],
    "answer": "B. ...",
    "answer_letters": ["B"],
}
```

### `load_gold_records(data_dir)` — exact id derivation (read directly from pipeline.py)
```python
DATASET_FILES = ("angry.json", "enemy.json", "friends.json", "man_earth.json")


def dataset_name(filename):  # "angry.json" -> "angry"
    return filename[:-5]


for filename in DATASET_FILES:
    source = dataset_name(filename)  # "angry", "enemy", "friends", "man_earth"
    data = json.loads((data_dir / filename).read_text())
    for sample_index, sample in enumerate(data):
        sample_id = (
            sample.get("sample_id") or f"{source}-{sample_index}"
        )  # dataset already sets "conv-0" for all 4 files
        for qa_index, qa in enumerate(sample.get("qa", [])):
            qa_id = f"{source}:{sample_id}#q{qa_index:04d}"  # e.g. "angry:conv-0#q0000"
```
So `dataset` = the raw filename stem (one of the 4 `DATASET_FILES` stems,
exactly), `sample_id` = the sample's own `sample_id` field (verified: always
present, always `"conv-0"` for all 4 current files — one sample per source
work, since each source work is treated as a single mega-conversation), and
`qa_index` = the QA's 0-based position within that sample's `qa` list, zero
padded to 4 digits. `public/questions.jsonl`'s own `qa_id` field already
matches this exact format (verified byte-identical construction) — **a
loader can just read `qa_id` straight from `questions.jsonl` instead of
recomputing it**, as long as it doesn't touch `qa_index` numbering (it must
stay 0-based, source-file order, not re-sorted).

### Messages
Not obtainable from this download for real content (see CRITICAL GOTCHA).
If/when real transcripts are sourced separately, the intended flattening
(per the `format_example` schema) would be:
```python
conv = sample["conversation"]  # once real, keyed like session_N / session_N_date_time (LoCoMo-style)
messages = []
for session_key in sorted(k for k in conv if re.fullmatch(r"session_\d+", k)):
    date = conv.get(f"{session_key}_date_time")   # e.g. "Unknown" or a real date/time string
    for turn in conv[session_key]:
        if turn["type"] != "dialogue":
            continue  # or include narration turns as system/narrator content, per harness design
        messages.append({"role": "user" or "assistant" (map by turn["speaker"]),
                          "content": turn["text"], "timestamp": date})
```
Timestamps, if the real data followed the placeholder's schema, would be
per-session free text (`"session_1_date_time"`), sometimes literally
`"Unknown"`.

### Retrieval scope
One **source work** (`dataset`/`source`, one of `angry`/`enemy`/`friends`/
`man_earth`) = one `sample_id` (`"conv-0"` in every current file) = one
scope = one `user_id`. All ~90-174 questions for that work share the one
mega-conversation.

### Questions
`public/questions.jsonl`, one line per QA, already carrying `qa_id` and
`conversation_id` to key against a scope.

### Payload keys the pipeline reads
`answer()` uses `render_answer_prompt` (a locally-authored prompt, *not*
part of ScriptMem's official repo per the module docstring); `evaluate()`
is the **official** exact-match scorer, driven by `evaluate_official()`:

| Key | Required | Where |
|---|---|---|
| `id` (for `answer()`'s bookkeeping only — dedup/resume) | optional | falls back to `qa_id`/`question_id`/`qid`, else row index |
| `speaker_1_name`/`speaker_1_memories`/`speaker_2_name`/`speaker_2_memories` | optional | used only by the local (non-official) `render_answer_prompt` for generating an answer |
| `question` | required for `render_answer_prompt` | plain string |
| **For `evaluate_official()` (official, gold-comparison path):** | | |
| gold side: `sample_id` (or synthesized), `qa[].qa_type`, `qa[].question`, `qa[].answer` | required | read straight from `data/raw/*.json` via `load_gold_records`, **not** from any "input" JSONL |
| submission side: a JSON list of `{"dataset": ..., "qa_results": [{"qa_id": ..., "predicted_answer": ...}, ...]}` (see `submission_template.json`) — or the more flexible `load_submission()` parser which also accepts `results`/`predictions`/`answers` list keys and several prediction-field aliases (`predicted_answer`/`prediction`/`answer`/`response`) | required | the `--submission` CLI arg to `evaluate` |

`score_item()` requires `qa_type` to be exactly one of `single_choice` /
`multi_select` / `ordering` (raises `ValueError` otherwise) and compares
letter-sets/sequences extracted by `predicted_letters()` against
`gold_letters()` (which parses each gold `answer` string/list entry via
regex `^\s*([A-F])\.` — i.e. **gold answers are option-lettered strings like
`"B. ..."`**, and only the leading letter is used for scoring).

### Gotchas
- **No real conversation/script text is present in this download** — the
  single biggest gotcha of all six benchmarks; see CRITICAL GOTCHA above.
  Confirm with the project owner whether transcripts will be sourced
  separately before building a full ingest+retrieve loader; otherwise this
  benchmark can only exercise the QA/scoring machinery, not memory recall.
- `qa_id` format is exactly
  `f"{dataset_filename_stem}:{sample_id}#q{qa_index:04d}"` — `dataset` comes
  from the **filename**, not any field inside the JSON; `sample_id` comes
  from the sample's own field (with a fallback pattern that is currently
  unused since all 4 files set it explicitly to `"conv-0"`); `qa_index` is
  strictly the 0-based position in file order — do not derive it from
  `question_id` string parsing or re-sort by any other key.
- `evaluate()`'s official path (`evaluate_official`) reads gold **directly
  from `data/raw/*.json`**, bypassing `data/public/questions.jsonl`
  entirely — a loader building a submission file should use `qa_id` values
  that exactly match what `load_gold_records` derives (or just copy `qa_id`
  from `questions.jsonl`, which is verified identical).
- Gold `answer` can be a **list** of option-lettered strings (multi-select /
  ordering) or a single string (single-choice) — `gold_letters()` handles
  both via `isinstance(answer, list)`.
- The answer-generation prompt in this pipeline (`CHOICE_ANSWER_TEMPLATE`)
  is explicitly **not** ScriptMem's official prompt (per module docstring:
  "It does not define this memory-search answer prompt... must label the
  answer prompt as user-specified") — only the `evaluate()` path is
  official/verifiable against upstream.

---

## Cross-benchmark summary

| Benchmark | Raw data usable as-is | Real conv. text present | Timestamp granularity | id/gold key rename needed |
|---|---|---|---|---|
| LoCoMo | Yes | Yes | per-session, free text | Yes — both `id` (synthesize) and gold (`answer`→`gold_answer`) |
| LongMemEval | Yes (use `_s`) | Yes | per-session, `YYYY/MM/DD (Dow) HH:MM` | Yes — both `id` (`question_id`→`id`) and gold (`answer`→`gold_answer`) |
| PersonaMem v1 | Yes | Yes (shared contexts) | none | No (native `question_id` matches `row_id` fallback); keep `all_options` as string |
| PersonaMem v2 | Yes | Yes (chat_history files) | none | Yes — no id column at all, must synthesize |
| BEAM | Yes | Yes | per-batch, free text, inconsistent | Yes — no id on questions, must synthesize; native `rubric` key already matches |
| CL-Bench | Yes | Yes (but question is embedded in doc, not separate) | none | Extract `question`/`system_prompt` from `messages`; native `metadata.task_id`/`rubrics` already match fallbacks |
| ScriptMem | Questions/gold yes; **conversations no** | **No — placeholder only** | n/a (real schema unknown/unseen) | qa_id format must be reproduced exactly (see section 6) |
