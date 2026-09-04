# Plugin architecture

## Decision

MindBridge uses a stable memory kernel with explicit, typed computation plugins. It does not use a
global registry, arbitrary pipeline hooks, automatic package installation, or a second execution
plane.

This keeps the product rich without making the API sprawling: applications define their own input
and capture forms at the edge, while replaceable model plugins implement narrow capabilities behind
the same memory operations.

The kernel owns memory identity and semantics, capability routing, validation, SQLite and media-CAS
durability, the index outbox, SQLite hydration, and final context construction. A plugin may perform
inference, but it cannot own or bypass those rules.

## Implemented composition

The supported plugins are the narrow backend protocols inventoried in
[architecture](architecture.md#model-boundary). That page owns the list; this page owns
the rules for adding to it. The declarative entry point constructs bundled implementations without
exposing runtime details:

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
dependency. Reusing one object for several protocols is supported by the object API, and the bundled
OpenAI adapter does exactly that.

Declarative configuration reaches a slot when there is a bundled implementation for its closed
provider catalog to name; [configuration](configuration.md#declarative-configuration) owns the
current list of accepted keys. A protocol with no bundled implementation deliberately has no key,
and is composed by passing an object to `Memory`, `Memory.from_plugins`, or `MemoryPlugins` — see
[the Python SDK reference](api/python-sdk.md#backend-protocols) for the signatures.

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

A plugin whose output the kernel persists must declare that output's *space* — `embedding_space`,
`transcription_space`, `face_space`, `formation_space`, `vision_space`. The kernel caches derived
work per asset and space, so the space is what stops two recipes from sharing one result; the
plugin owns it because only the plugin knows what its recipe is. It is not the model name: a
describer that edits its prompt, or a transcriber that changes its keyword list, produces different
output from the same model and must return a different space. Declaring a stable space over a
changed recipe is the one failure the kernel cannot detect, because a stale derived value is
indistinguishable from a fresh one once it is inside a searchable document.

The optional `server` and `mcp` dependencies are packaging boundaries, not plugins. REST, MCP, and
the product CLI remain thin transports over the application-composed `Memory`.

`Memory.add_stream` is also not a plugin boundary. It repeats the existing durable `add` operation
over caller-segmented observations. `AsyncOmniPrefetch` is a thin lifecycle helper over the same
`AsyncMemory.search`; it neither performs perception nor owns a second execution plane. Capture,
ASR partials, frame selection, and finality stay with the application.

## Admission rule

A new public capability protocol is admitted only together with all of:

1. A concrete implementation in `src/`, not a test double.
2. A reachable caller on a product path a user invokes, not only on an adjacent or optional one.
3. Declared accepted modalities and typed output.
4. Provenance, configuration, resource lifecycle, concurrency behavior, privacy boundary, and
   failure mapping.
5. Tests that exercise the capability through the public product path.

Face analysis is the worked example: `FaceBackend` arrived with a local OpenCV implementation, an
end-to-end identity use case, and product-path tests. A capability that satisfies the type system
but nothing else is exported surface area with no user, and it costs the same to maintain as one
that works.

### Outstanding violation

`VisionDescriptionBackend` breached criterion 1 for as long as no implementation shipped in `src/`
and its only implementor in this repository was a test fake. `OpenAIModels.describe` and the
`vision` configuration key close it; the entry stays because the sequence is the point — a protocol
admitted ahead of its implementation spent releases as exported surface with no user.

Criterion 2 no longer applies to it either. The describer used to be reachable only from the
asynchronous vision capture stream; `add` and `add_many` now call a configured one for every
embedder whenever a visual asset has no description yet, so the capability reaches the path a user
actually invokes.

The absence of a `vision_describer` key in declarative configuration is **deliberate, not the same
gap**. Adding configuration for a protocol with no bundled implementation would deepen this
violation rather than fix it: there would be nothing for a closed provider catalog to name. The key
becomes appropriate when an implementation exists, and not before. Object injection remains the
route in the meantime, which is the correct shape for a capability whose only implementations are
the caller's own.

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
