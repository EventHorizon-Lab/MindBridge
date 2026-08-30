# Performance and token observability

MindBridge emits OpenTelemetry spans for each end-to-end memory operation and the material stages
inside it. The library depends only on `opentelemetry-api`; without an application-configured SDK,
instrumentation is a no-op. Install the small SDK extra when traces need to be collected:

```bash
uv add "mindbridge[observability]"
```

Configure any OpenTelemetry exporter before constructing `Memory`. This local example writes spans
to stdout; production applications can attach their existing exporter instead:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from mindbridge import JinaOmniEmbedder, Memory

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

with Memory("./data", embedder=JinaOmniEmbedder()) as memory:
    memory.add("Remember this")
    memory.search("Remember")
```

`Memory(..., tracer=provider.get_tracer("application.memory"))` can inject a tracer from a
non-global provider. `AsyncMemory` accepts the same argument and preserves context through its
worker thread. No span records memory text, media bytes, asset IDs, paths, metadata, or model
responses.

## Span structure

The operation span is the elapsed time visible to the caller. Child spans identify the work that
can be optimized independently:

| Kind | Spans |
| --- | --- |
| End to end | `mindbridge.add`, `mindbridge.add_many`, `mindbridge.search`, `mindbridge.ask`, `mindbridge.get`, `mindbridge.speech`, `mindbridge.register_speaker`, `mindbridge.reinforce`, `mindbridge.list`, `mindbridge.delete`, `mindbridge.reindex`, `mindbridge.optimize` |
| Input and storage | `mindbridge.content.prepare`, `mindbridge.storage.lookup`, `mindbridge.storage.write`, `mindbridge.storage.hydrate` |
| Retrieval | `mindbridge.index.sync`, `mindbridge.index.search`, `mindbridge.retrieval.rank` |
| Models | `mindbridge.model.embedding`, `mindbridge.model.transcription`, `mindbridge.model.generation` |

End-to-end operation spans contain their stage and model spans, so those levels are intentionally
not additive. Material stage spans are siblings; compare them directly when locating a bottleneck.

Model spans use OpenTelemetry GenAI attributes where the ecosystem defines them:

- `gen_ai.operation.name` and `gen_ai.request.model` identify the model boundary.
- `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` contain provider-reported totals.
- `gen_ai.response.time_to_first_chunk` measures request-to-first-stream-chunk time.
- `gen_ai.response.finish_reasons` carries the provider's stop reason, so a `length` truncation is
  countable rather than inferred from a failure.
- `mindbridge.model.time_to_first_token` measures model-span start to the first non-empty text
  delta. It exists only for a streaming generation backend.
- `mindbridge.model.module` is `embedding`, `transcription`, `generation`, or benchmark-only
  `judge`.

Each end-to-end operation span rolls up its descendant model calls into
`mindbridge.token_usage.total_tokens`. `mindbridge.token_usage.complete` is true only when every
token-metered request supplied a usable total; request counters and known modality totals remain
available as an exact lower bound when it is false.

The bundled `OpenAIModels` generation span also reports what its inline media budget removed:
`mindbridge.grounding.media_elided_hits` counts retrieved hits whose media did not fit, and
`mindbridge.grounding.dropped_hits` counts hits left out of the request entirely. Both are zero on
a request that sent every retrieved hit intact. `AnswerResult.hits` still returns the retrieved
hits, so these counters are how a shrunken grounding payload becomes visible.

`OpenAIModels` implements the optional `StreamingGenerationBackend` protocol. `Memory.ask()`
consumes its deltas internally and still returns one `AnswerResult`, while the generation span
retains both total model latency and TTFT. A non-streaming custom answerer keeps total model latency
but cannot truthfully emit TTFT.

## Token accounting

Token values come only from the model response. MindBridge does not estimate tokens from character
count, file size, media duration, or a tokenizer belonging to another model.

Exact provider modality details are stored under:

```text
mindbridge.token_usage.input_tokens.text
mindbridge.token_usage.input_tokens.image
mindbridge.token_usage.input_tokens.video
mindbridge.token_usage.input_tokens.audio
mindbridge.token_usage.output_tokens.<modality>
```

When a provider reports a total but cannot separate all requested modalities, the unresolved
remainder is recorded as `.unattributed`. This preserves the exact total without inventing a visual
or audio split. OpenAI transcription token details are mapped to text/audio; duration-billed
responses additionally expose `mindbridge.token_usage.audio_seconds`. The bundled local Jina,
Sentence Transformers, and FunASR adapters are not token-billed and do not synthesize tokenizer
usage.

`mindbridge.model.request_count`, `mindbridge.token_usage.expected_request_count`, and
`mindbridge.token_usage.reported_request_count` make missing usage visible. A failed request or a
custom provider that omits usage therefore makes an aggregate incomplete instead of silently
contributing zero. These count calls issued to the provider, not work items submitted, so a
batching backend such as Sentence Transformers or FunASR reports one request for a whole batch.

## Benchmark output

`mindbridge-bench eval` uses a private OpenTelemetry provider and bounded online aggregation. Every
task row in `results.json` includes:

```json
{
  "performance": {
    "duration_seconds": {
      "total": 12.5,
      "average": 1.25,
      "mindbridge": 10.0,
      "judge": 2.5
    },
    "nodes": {
      "mindbridge.model.generation": {
        "count": 10,
        "total_seconds": 6.0,
        "average_ms": 600.0,
        "ttft_ms": {"count": 10, "average": 80.0}
      }
    },
    "token_usage": {
      "complete": true,
      "total_tokens": 1500,
      "average_tokens": 150.0,
      "reported_total_tokens": 1500,
      "input_by_modality": {"text": 1000, "image": 200},
      "output_by_modality": {"text": 300},
      "by_module": {}
    }
  }
}
```

`total` is task execution wall time plus that task's judge window; `average` divides it by the
selected question count. Node totals are diagnostic accumulated span time and can overlap under
concurrency. Token totals include MindBridge model calls and benchmark judge calls. Cached answers
do not consume new model tokens. `total_tokens` and `average_tokens` are `null` when any
token-metered response omitted usage; `reported_total_tokens` remains the exact lower bound.
