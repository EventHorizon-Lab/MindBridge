# Extension status

MindBridge v0.2 does not expose a runtime plugin registry. The supported implementation is one
direct path from `Memory` to SQLite, Zvec, and explicit operation backends.

This is deliberate:

- There is one authoritative storage implementation.
- There is one derived search implementation.
- `SentenceTransformersEmbedder` covers standard local models; `JinaOmniEmbedder` isolates Jina's
  provider-specific input contract.
- The official OpenAI SDK adapter covers compatible cloud endpoints without a provider registry;
  other services use their provider SDK behind the narrow operation protocols.
- Constructor injection already supplies the lifecycle boundary for another implementation.

Application code can extend behavior by composition:

```python
from mindbridge import EmbeddingBackend, Memory


class ProjectMemory:
    def __init__(self, path: str, embedder: EmbeddingBackend) -> None:
        self.memory = Memory(path, embedder=embedder)

    def remember_decision(self, text: str) -> str:
        return self.memory.add(text, metadata={"kind": "decision"}).id

    def close(self) -> None:
        self.memory.close()
```

Composition must not rely on metadata as a security boundary. Independent applications or memory
domains still require independent directories.

The optional `server` and `mcp` dependencies are packaging boundaries, not runtime plugins. The
REST and MCP adapters are thin transports over `Memory`; they do not provide alternative storage
or retrieval implementations.

A future extension point should be introduced only with a concrete implementation and contract
tests for lifecycle, failure mapping, durability, and performance. Until then, private constructor
arguments and infrastructure adapters are internal and may change without compatibility promises.
