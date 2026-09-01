# Plugin architecture

## Decision

MindBridge uses a stable memory kernel with explicit, typed computation plugins. It does not use a
global registry, arbitrary pipeline hooks, automatic package installation, or a second execution
plane.

The kernel owns memory identity and semantics, capability routing, validation, SQLite and media-CAS
durability, the index outbox, SQLite hydration, and final context construction. A plugin may perform
inference, but it cannot own or bypass those rules.

## Implemented composition

The supported plugins are the narrow `EmbeddingBackend`, `GenerationBackend`,
`TranscriptionBackend`, `SpeechBackend`, `VisionDescriptionBackend`, `FaceBackend`, and
`FormationBackend` protocols. The declarative entry point constructs bundled implementations
without exposing those runtime details:

```python
from mindbridge import Memory

with Memory.from_config(
    {
        "data_dir": "./data",
        "embedding": {"provider": "jina-omni"},
        "speech": {"provider": "funasr"},
        "settings": {"index_speech": True},
    }
) as memory:
    memory.add("Remember this")
```

`Memory.from_config` validates a closed catalog of bundled providers, constructs owned adapters, and
then delegates through `Memory.from_plugins` to the normal constructor. Custom and third-party
implementations remain explicit object injection. All paths share capability validation, durable
identity checks, storage, and execution. Omitting an optional plugin adds no model call or optional
dependency. Reusing one object for several protocols is supported by the object API.

Composition is fixed for one `Memory` lifetime. Changing an adapter requires closing and reopening
the instance; changing the embedder also requires a compatible durable space or a new directory.
There is no mid-operation hot swap.

## Kernel and plugin boundary

Current computation plugins own model and runtime details such as device, precision, batching,
thresholds intrinsic to that model, or a caller-supplied remote SDK client. Formation backends own
only inference: they receive committed, resolved observations and return typed proposals. The
kernel validates source modality and spatial binding, assigns deterministic identity, records
evidence and bitemporal versions, resolves state conflicts, and commits the search outbox. Potential
Calibrated affect perception, visual grounding, and reranking belong on the plugin side only after
a concrete implementation proves a narrow contract. Caption, OCR, or detector fallback now uses
the narrow `VisionDescriptionBackend`; the stream kernel still owns routing and durability.

`OpenAIModels` implements `FormationBackend` with the configured generation model. It sends model
content but not stable memory/CAS IDs or exact spatial values, and every proposal remains
`model_inference`. A trusted custom former is injected explicitly:

```python
with Memory("./data", embedder=embedder, former=former) as memory:
    source = memory.add(observation, context=observation_context)
```

The former cannot write storage or acknowledge formation. Derived memories, evidence links,
embeddings, formation completion, and outbox work commit atomically in SQLite. Omitting `former`
keeps ordinary add behavior and avoids every formation model call. A custom `formation_space` must
identify its capability set as part of the recipe; unsupported observations are not marked complete
and remain eligible after an adapter upgrade.

SQLite, the media CAS, the durable outbox, and Zvec are not public plugins. They have one supported
implementation, and abstracting them now would add indirection without a second product use case.
Metadata remains application data, never an execution, isolation, or authorization boundary.

The optional `server` and `mcp` dependencies are packaging boundaries, not plugins. REST, MCP, and
the product CLI remain thin transports over the application-composed `Memory`.

`Memory.add_stream` is also not a plugin boundary. It repeats the existing durable `add` operation
over caller-segmented observations. `AsyncOmniPrefetch` is a thin lifecycle helper over the same
`AsyncMemory.search`. `AsyncCaptureStream` accepts complete `UPDATE` snapshots, exact `FINAL`
observations, and `CANCEL` boundaries associated by `stream_id`. `AsyncAudioStream` is a thin
canonical adapter over that reducer for PCM chunks, VAD state, complete ASR hypotheses, and acoustic
boundaries. `AsyncVisionStream` applies the same pattern to encoded image frames, complete visual
descriptions, and scene boundaries while retaining one bounded keyframe. Neither owns model
execution or a second persistence plane; provider-specific packet decoding, device capture, pixel
conversion, and other sensor protocols remain application adapters.

## Embodied integration boundary

MindBridge may improve embodied memory representation, retrieval, provenance, temporal validity,
and failure diagnosis without becoming a robot AgentOS. Planner, skill-runner, verifier,
edge/cloud routing, robot-control, benchmark harness, and model-training contracts remain outside
the product boundary. Procedural memory is evidence and is never executed as code.

Typed entity, relation, temporal, and spatial semantics remain additive to authoritative raw
records. A graph projection must earn its complexity through MindBridge's own measured retrieval
results. A paper's architecture or benchmark score is not by itself a reason to add a graph
database.

Interaction memory uses the existing semantic, episodic, and procedural roles. Affect, trait, and
response-policy are typed `MemoryKind` values within those roles, not new stores or executable
instructions. A former may propose them, while the kernel retains evidence provenance and enforces
visibility and correction rules. Plugins may not write around durability or reinterpret metadata as
routing, isolation, or trust.

## Trade-offs and revisit triggers

Typed declarative construction makes bundled adapters concise; explicit construction keeps custom
dependencies, lifecycle, privacy, and hardware selection visible. Both require reopening `Memory`
to change composition.

Add package entry-point discovery only when an independently distributed plugin requires it. Add a
new public capability only with a concrete adapter plus lifecycle, failure, privacy, provenance, and
product-path tests. Add richer evidence or relationship storage only when the executable embodied
loop demonstrates that existing records cannot represent the required state without loss.
