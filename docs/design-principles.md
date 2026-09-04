# Product goals and design principles

MindBridge's product direction is an **Agentic Native Embodied Omni-Modal Memory System**: a memory
layer designed as a native part of intelligent systems that perceive, reason, and act. Its
reference product is an emotional companion robot. The same memory contract should also serve
desktop robots, personal work assistants, chatbots, and voice agents without requiring a separate
architecture.

MindBridge is the memory system, not the robot, agent framework, sensor stack, or foundation model.
It turns ordered text, image, video, and audio observations into durable memory that an agent can
search, question, and trace back to evidence.

> This page defines product direction and the criteria for future design decisions. It is not a
> claim that every target is implemented in the current release. See
> [product capabilities](product-capabilities.md) for what ships today,
> [architecture](architecture.md) for its invariants, and
> [plugin architecture](plugin-architecture.md) for today's extension boundary. Where this page and
> an owning reference disagree, the reference wins and this page is the page to fix.

## What the direction means

| Term | Commitment |
| --- | --- |
| Agentic Native | Memory semantics are a first-class agent capability, not a thin wrapper over a vector database. |
| Embodied | Time, source media, identity signals, and other evidence from the physical world remain meaningful throughout storage and retrieval. |
| Omni-Modal | Every supported modality works independently or in an arbitrary ordered combination under one memory contract. |
| Memory System | MindBridge owns memory semantics, retrieval orchestration, and durable local consistency while models and transports remain replaceable. |

The household companion is the reference product because it exercises the hardest requirements:
long-lived personal context, mixed sensory input, low perceived latency, intermittent connectivity,
sensitive data, and heterogeneous hardware. Other product forms are supported by generalizing this
same path, not by adding product-specific forks.

MindBridge pursues four product outcomes:

| Outcome | Commitment |
| --- | --- |
| Stronger | Target state-of-the-art memory quality independently across text-only, image-text, video, audio, and omni routes. |
| Faster | Measure and reduce the complete ingestion, retrieval, and answer path experienced by the agent. |
| Leaner | Reduce model, compute, memory, storage, energy, and operational cost without trading away quality or durability. |
| Developer- and agent-friendly | Keep the API concise, typed, discoverable, structured, and consistent across every public surface. |

## Design goals

### One memory surface for every supported modality

Applications should use the same memory operations regardless of which modalities are present.
Users control the input composition and ordering; MindBridge chooses a route from the actual atomic
modalities and the configured backend capabilities.

| Available input | Intended route |
| --- | --- |
| Text only | Text embedding, retrieval, and generation |
| Image with text | Native image-text route |
| Video, with optional text | Native video-text route |
| Audio, with optional text | Native audio-text route, or explicit transcription fallback when required |
| Multiple media families | Omni route that retains every supported evidence type |

The product-direction default is an omni-capable embedding and generation composition when the
deployment can run it. Omni-first does not mean always invoking the largest model: a text-only
request should remain on the text path, and absent modalities must not add work. Unsupported media
must never be discarded silently.

Input form belongs to the user: capture formats, sensor records, interaction events, and application
objects may be defined freely outside MindBridge. They normalize into canonical content at an
adapter boundary so the core API remains small, typed, and validatable instead of accepting
unvalidated `Any` values.

### End-to-end memory and search speed

Performance is the elapsed time experienced by the agent, not an isolated model or database
microbenchmark. The complete path includes any configured ASR, embedding, face or emotion
recognition, media preparation, durable writes, index visibility, retrieval, and grounded
generation.

Each relevant deployment should report at least:

- Ingestion latency from accepted input to durable, searchable memory, plus sustained throughput.
- Search latency at p50, p95, and p99, query throughput, and retrieval quality at the same settings.
- ASR real-time factor and inference latency for configured perception capabilities.
- End-to-end answer latency and time to first token when the selected generation backend streams.
- CPU, GPU or NPU utilization, memory, storage growth, and edge-device power where applicable.

An optimization counts only when it improves a measured bottleneck without weakening retrieval
quality, durability, or modality coverage. Concurrency, batching, quantization, caching, and native
runtimes are tools to apply after the end-to-end trace identifies the limiting stage.

### Portable deployment, stable semantics

MindBridge targets device classes rather than one vendor or accelerator. Representative targets
include RK3588-class systems, NVIDIA Jetson Orin, Intel/OpenVINO devices, Apple silicon Macs,
RTX 4090/5090 workstations, and L20-class servers.

Platform-specific runtimes, compiled artifacts, batch sizes, precision, and calibration thresholds
may vary. Content semantics, memory identity, embedding-space compatibility, failure behavior, and
durability rules may not. A platform adapter should remain thin and exist only after that platform
has a measured requirement.

Runtime selection may be explicit or automatic. Explicit user configuration always wins. An
automatic selector must be deterministic from declared capabilities and observed device facts,
must expose the selected model/runtime and reason, and must fail clearly rather than silently
changing modality coverage or quality. Hardware-sensitive batch, concurrency, and calibration
controls remain overrideable.

### State-of-the-art quality across modality routes

MindBridge aims for state-of-the-art memory quality in text, image-text, video-text, audio-text, and
omni settings. This is a research and engineering target, not an unqualified release claim. A strong
text score cannot stand in for multimodal quality, and a model demo cannot stand in for memory-system
quality.

A quality claim must identify the dataset and revision, official split and evaluator, input route,
model and runtime revisions, retrieval settings, hardware, and measured latency and resource cost.
Results must come through the public product path and be replayable. Benchmark-specific shortcuts
that do not improve a real agent scenario are out of scope. See [benchmarking](benchmarking.md) for
the current harness and reporting rules.

### Developer- and agent-friendly interfaces

A developer should reach the first durable add/search loop in minutes. An agent should be able to
choose and call a MindBridge operation from its schema without interpreting prose or scraping
terminal output. Both consumers use the same memory semantics.

The [Mem0 open-source documentation](https://docs.mem0.ai/open-source/overview) is a developer
experience reference: it demonstrates a short add/search path, consistent memory verbs,
progressive component configuration, and dedicated agent integrations. MindBridge should match that
time-to-first-success while retaining its own boundaries. In particular, it does not copy hidden
provider construction, logical `user_id` scoping, or metadata-based isolation; model clients remain
explicit and one physical `data_dir` remains one memory domain.

The SDK, MCP, product CLI, and REST API share one execution plane: the application-composed
`Memory` instance and its domain operations. The SDK exposes that plane directly. Other surfaces may
normalize transport input and serialize output, but they must not implement their own modality
routing, retrieval, persistence, provider selection, defaults, or error policy. See
[architecture](architecture.md#public-and-trust-boundaries).

For developers:

- The common path is one explicit `Memory`, followed by `add`, `search`, and optionally `ask`;
  continuous sources use `add_stream`. The [command line](api/cli.md) is the same path without an
  editor: one composition flag, one command.
- The common path is a small vocabulary — `add`, `add_stream`, `search`, `ask`, `get`, `list`, and
  `delete` — and a caller who needs nothing else should never have to learn anything else. That is
  a property of the *default* path, not the size of the surface: the full inventory is larger and
  lives in [the Python SDK reference](api/python-sdk.md#memory-operations). Growth beyond the common
  path is justified per operation, not by convenience. `speech` and `faces` read a stored record but
  invoke a model and persist identity state, so they are separate operations rather than variants of
  `get`; batching (`add_many`), diagnostics (`search_with_trace`), identity naming, explicit
  feedback (`reinforce`), and index maintenance (`reindex`, `optimize`) are each their own verb for
  the same reason. So are the three operations that split acknowledgement from enrichment
  (`capture`, `settle`, `pending_captures`) and the four that manage stored memory rather than add
  to it (`consolidate`, `forget`, `rollback`, `operations`): each changes when work happens or what
  authority applies it, not just the shape of a result. `compile` is the one that pays for itself
  on the common path, because an agent assembling its own prompt otherwise re-derives structure and
  budget from a flat hit list. An operation that only reshapes an existing result is not one of
  them.
- The SDK is the canonical capability inventory. MCP and the product CLI expose the same product
  operations unless a transport limitation is documented explicitly; a gap is implementation work,
  not permission to create different behavior.
- Python, REST, MCP, CLI, and future adapters preserve the same IDs, field meanings, pagination,
  idempotency, defaults, and error semantics for every operation they share.
- Inputs and outputs are typed, defaults are safe, advanced configuration is progressively
  disclosed, and examples run against the current public API.
- A caller does not need to understand SQLite, the media store, the outbox, or Zvec to remember and
  retrieve content.

For agents:

- The schema is the contract: MCP exposes typed tool schemas, REST exposes OpenAPI, and the product
  CLI emits one stable machine-readable JSON document per invocation.
- Tool names and descriptions are self-contained and state when to call an operation, its side
  effects, result limits, and whether retrying it is safe.
- Results and failures are structured. An agent never has to parse a human log line to recover an
  ID, cursor, error code, or grounding source.
- Outputs are bounded and pageable. Stable IDs, opaque cursors, deterministic ordering, and
  idempotent writes make retries and multi-step plans predictable.
- The product CLI is non-interactive and composable: data goes to stdout, diagnostics go to stderr,
  exit codes are stable, and stdin can carry generated input without shell quoting tricks.
- MCP and the product CLI derive their operations from the SDK. Transport-specific encoding may
  differ, but capability, side effects, defaults, and results remain aligned. Today the CLI reaches
  every SDK operation and MCP reaches a documented subset; see
  [current release and direction](#current-release-and-direction).
- The CLI surface is two console scripts, not one command tree: `mindbridge` for product
  operations and `mindbridge-bench` for benchmarks. Product commands dispatch to `Memory`;
  benchmark commands exercise the public SDK and never define an alternate product path. The split
  is required, not stylistic: the packaging guard scans string constants, so a single dispatcher
  could not name the benchmark package even to import it lazily, and product modules must not
  import benchmark modules.
- Published documentation should provide stable deep links, downloadable schemas, and an
  agent-readable index such as `llms.txt` when a documentation site exists.

Agent integrations may add optional lifecycle hooks or skills that retrieve context before a turn
and store outcomes afterward. Such automation must make writes observable, preserve the distinction
between user statements and agent-generated conclusions, and remain outside the core memory
semantics. `AsyncOmniPrefetch` implements the narrow Python-side speculative-recall lifecycle;
capture, turn detection, derived-memory extraction, and agent prompting remain application work.
MCP and the `mindbridge` CLI are the implemented transport-facing agent surfaces.

### Developer-friendly extensibility

Applications should be able to replace inference implementations and add optional perception or
reasoning capabilities without forking memory semantics. Examples include text, visual, or speech
emotion recognition and face recognition. These are optional derived capabilities, not new core
modalities and not requirements imposed on every deployment.

An extension is successful when omitting it leaves the normal API unchanged, configuring it is
explicit, and its cost and failure behavior are visible. A rich extension ecosystem should come
from a few stable capability contracts rather than a large framework of task-specific hooks.

## Design principles

### Route by capability, not by provider name

Every backend declares the atomic modalities and operation it actually supports. Routing uses those
declarations, never a model-name heuristic. The same provider may implement several narrow
operations, and different providers may be composed for embedding, generation, transcription, or
future analysis.

Fallbacks must preserve information. For example, when audio is transcribed for a text-only
operation, supported image or video evidence stays on its native route. If no valid route remains,
MindBridge fails before inference.

### Keep memory semantics independent of inference runtimes

Models and runtimes will change faster than stored memory. Stable operation contracts isolate that
change while durable model and embedding-space identities prevent incompatible representations from
being mixed.

Fun-ASR-Nano illustrates the desired runtime range: the upstream ecosystem documents
[PyTorch and vLLM paths](https://github.com/modelscope/FunASR/blob/main/docs/vllm_guide.md) and a
[llama.cpp/GGUF edge runtime](https://github.com/modelscope/FunASR/tree/main/runtime/llama.cpp).
The current `FunASRTranscriber` uses `funasr.AutoModel` only. Supporting another runtime requires a
separate, tested adapter behind the same speech contract; upstream availability alone is not a
MindBridge support claim.

### Prefer explicit configuration and observable automation

Applications own provider clients, credentials, and deployment policy. MindBridge accepts explicit
backend objects and validates their capabilities and durable identities. Future automatic selection
is a policy over the same explicit candidates, not a hidden provider registry.

Selected models, revisions, runtimes, precision, and route-affecting settings belong in benchmark
artifacts and operational telemetry. Automatic fallback must not make two equivalent requests
irreproducible or silently migrate an embedding space.

### Reuse the ecosystem before writing infrastructure

Blind novelty and duplicate infrastructure are maintenance failures. For a general capability, use
this order:

1. The provider's official SDK or runtime.
2. A portable, maintained ecosystem such as Hugging Face or Sentence Transformers.
3. A target platform's native acceleration stack.
4. An already-installed mature dependency or the Python standard library.
5. The smallest local adapter only when the preceding choices leave a demonstrated gap.

This is why OpenAI-compatible calls use the official OpenAI SDK, local Jina inference uses Sentence
Transformers, FunASR execution stays behind the upstream runtime, and benchmark datasets use
`huggingface_hub` where the dataset is published there. MindBridge should innovate in memory
semantics, routing, retrieval, and consistency, not in another HTTP client, downloader, codec, model
loader, or generic queue.

Reuse is not automatic approval. A dependency must still fit the supported Python and hardware
matrix, license and security requirements, performance envelope, and maintenance boundary.

### Add extensions as narrow, optional capabilities

The extension surface is the set of explicit backend protocols listed in
[architecture](architecture.md#model-boundary); that page owns the inventory and this one
does not repeat it. Applications can pass those objects directly to `Memory`; `Memory.from_config`
provides a typed convenience layer for the bundled implementations and delegates to the same
constructor. Explicit objects remain the third-party plugin mechanism; there is no global runtime
plugin registry.

`StreamingGenerationBackend` is the shape a narrow capability should take. It adds one method, stays
optional, is selected by a structural check rather than a provider name, and leaves the
`GenerationBackend` contract unchanged for backends that do not implement it.

Face analysis demonstrates the admission rule: its public contract arrived with a concrete local
OpenCV implementation and end-to-end identity use case. A future public capability such as emotion
analysis must likewise declare accepted modalities, typed output, provenance, configuration,
resource lifecycle, concurrency behavior, privacy boundary, and failure mapping. It must not create
provider branches inside `Memory` or treat metadata as an execution or isolation mechanism. The
rule, and the one protocol currently in breach of it, are stated in
[plugin architecture](plugin-architecture.md#admission-rule).

### Preserve evidence, privacy, and durability

Embodied memory contains sensitive media and identity-derived representations. Local processing and
storage are the preferred baseline; a remote model call must be an explicit deployment choice.
Optional biometric or emotion plugins must not export source media or derived identities implicitly.

SQLite and the content-addressed media store remain authoritative, while Zvec is a rebuildable search
projection. A durable write commits before the projection changes, and a faster path is invalid if it
can acknowledge data that cannot be recovered. Performance work must preserve these invariants.

### Keep the product path small and measurable

The best design is the smallest one that satisfies a current, measured need. Do not add a registry,
factory, service, queue, cache, or compatibility layer for a hypothetical deployment. Add one thin
boundary when a real second implementation proves the existing boundary insufficient.

Quality and performance benchmarks exercise the public SDK. A benchmark adapter may translate a
dataset, but it must not become an alternate memory implementation. Code, model, dataset, and runtime
revisions must be reproducible before a result guides a default.

## Current release and direction

Every "current release" cell below is a statement about code that exists today, and each defers to
an owning reference for detail. A cell that stops being true is a defect in this page.

| Concern | Current release | Product direction |
| --- | --- | --- |
| Input | Ordered text, image, video, and audio through `ContentInput`; lazy completed observations through `StreamInput`/`add_stream`; deferred enrichment through `capture`/`settle`; async speculative omni recall | More capture formats normalize into the same canonical contract |
| Embedding | Caller explicitly supplies a backend; Jina v5 Omni is the bundled omni adapter | Omni-capable recommended composition with route-specific execution |
| Generation | Optional caller-supplied backend with explicit capabilities | Omni-capable recommended composition where the deployment supports it |
| Speech runtime | Built-in FunASR adapter uses `AutoModel` | Additional measured runtime adapters, selected explicitly or by observable policy |
| Extensions | The explicit model protocols in [architecture](architecture.md#model-boundary), including `ConsolidationBackend` for the memory control plane, one of them optional; no registry. Declarative configuration builds a subset of the slots; the rest are object injection only | Optional domain capabilities after a real implementation establishes the contract |
| Hardware | Runs where Python, dependencies, and the selected models are supported | Verified device-class matrix with published quality, latency, and resource evidence |
| Developer interfaces | Typed Python API, OpenAPI-documented REST adapter, and a JSON-only product CLI over the same composition | Same small vocabulary and time-to-first-success across supported transports |
| Execution plane | Python SDK, REST, MCP, and the `mindbridge` CLI all dispatch to one `Memory`; none of them implements its own routing, persistence, or defaults | Every surface reaches the operations the SDK publishes, with no transport gap left undocumented |
| Agent interfaces | Fifteen typed MCP tools and a `mindbridge` CLI whose commands are the SDK operations kebab-cased plus `doctor`. The common path — `add_memory`, `search_memories`, `ask_memory`, `compile_context`, `get_memory`, `list_memories`, `delete_memory` — plus the embodied and identity operations an agent driving a robot needs: `analyze_speech`, `analyze_faces`, `register_speaker`, `register_identity`, `get_identity`, `unlink_identity`, `forget_identity`, `reinforce_memories`. Erasure is exposed, so a privacy request reaches an agent surface, and the composition's capability view is the server instructions rather than a tool. What MCP still does not expose is batching, streaming, diagnostics, index maintenance, deferred capture, and the memory control plane; each is listed with its reason in [the MCP reference](api/mcp.md#operations-without-a-tool) | SDK-derived MCP and CLI capability parity, machine-readable schemas, and lifecycle integrations |

This distinction is deliberate: goals guide what to build next, while current API and deployment
documentation remain the source of truth for what users can run now.

## Decision checklist

Before adding or changing a model, runtime, plugin, or performance path, answer:

- Which real agent scenario and modality route improves?
- What is the fallback when a requested modality is unsupported?
- Which end-to-end quality, latency, throughput, and resource measurements justify the change?
- Which model, runtime, data, and embedding-space identities must remain reproducible?
- Which official SDK, upstream runtime, or existing dependency was evaluated first?
- Does the change preserve evidence, privacy, durability, and explicit configuration?
- Can a developer discover the common path and an agent execute it from structured contracts alone?
- Does every public surface dispatch to the same `Memory` operation instead of duplicating policy?
- Can the same result be achieved with less code or one existing extension contract?
- Is the new contract reachable from the product path a user actually calls, or only from a narrower
  one? An exported protocol with no reachable caller is an unfinished change, not an extension
  point.
