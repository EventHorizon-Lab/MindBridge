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
`TranscriptionBackend`, `SpeechBackend`, and `FaceBackend` protocols. The declarative entry point
constructs bundled implementations without exposing those runtime details:

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
thresholds intrinsic to that model, or a caller-supplied remote SDK client. Potential OCR, emotion,
visual-grounding, reranking, and semantic-extraction capabilities belong on this side only after a
concrete implementation proves a narrow contract.

SQLite, the media CAS, the durable outbox, and Zvec are not public plugins. They have one supported
implementation, and abstracting them now would add indirection without a second product use case.
Metadata remains application data, never an execution, isolation, or authorization boundary.

The optional `server` and `mcp` dependencies are packaging boundaries, not plugins. REST, MCP, and
the product CLI remain thin transports over the application-composed `Memory`.

`Memory.add_stream` is also not a plugin boundary. It repeats the existing durable `add` operation
over caller-segmented observations. `AsyncOmniPrefetch` is a thin lifecycle helper over the same
`AsyncMemory.search`; it neither performs perception nor owns a second execution plane. Capture,
ASR partials, frame selection, and finality stay with the application.

## Embodied integration boundary

MindBridge may improve embodied memory representation, retrieval, provenance, temporal validity,
and failure diagnosis without becoming a robot AgentOS. Planner, skill-runner, verifier,
edge/cloud routing, robot-control, benchmark harness, and model-training contracts remain outside
the product boundary. Procedural memory is evidence and is never executed as code.

New entity, relation, or spatial projections must remain additive to authoritative records and earn
their complexity through MindBridge's own measured retrieval results. A paper's architecture or
benchmark score is not by itself a reason to replace the flat durable representation.

Interaction memory uses the existing semantic, episodic, and procedural roles. Any
emotion, trait, or response-policy inference stays outside the kernel or behind a future concrete
typed analysis plugin. Derived records retain evidence provenance and use ordinary `Memory` writes;
plugins may not write around durability or reinterpret metadata as routing, isolation, or trust.

## Trade-offs and revisit triggers

Typed declarative construction makes bundled adapters concise; explicit construction keeps custom
dependencies, lifecycle, privacy, and hardware selection visible. Both require reopening `Memory`
to change composition.

Add package entry-point discovery only when an independently distributed plugin requires it. Add a
new public capability only with a concrete adapter plus lifecycle, failure, privacy, provenance, and
product-path tests. Add richer evidence or relationship storage only when the executable embodied
loop demonstrates that existing records cannot represent the required state without loss.
