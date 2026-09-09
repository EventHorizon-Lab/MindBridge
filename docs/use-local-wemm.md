# Use a local WeMM embedding service

MindBridge can use an OpenAI-compatible local WeMM embedding service through the same public
configuration used for hosted providers. Keep the embedding service local and configure generation
separately when a remote Qwen endpoint answers questions.

```yaml
data_dir: /absolute/path/to/a-new-mindbridge-store

embedding:
  provider: openai
  base_url: http://127.0.0.1:18871/v1
  api_key: EMPTY
  model: tencent/WeMM-Embedding-2B
  dimension: 2048
  modalities: [text, image, video]
  request_format: messages
  timeout: 120.0
  max_retries: 0

generation:
  provider: openai
  base_url: http://your-qwen-endpoint.example/v1
  api_key: replace-me
  model: Qwen3.8-27B
  modalities: [text, image, video]
  video_limit: 8
  extra_body:
    chat_template_kwargs:
      enable_thinking: false

speech:
  provider: funasr
  device: cuda

settings:
  index_speech: true
```

Open the configuration with the public SDK:

```python
from mindbridge import Memory

with Memory.from_config("/absolute/path/to/config.yaml") as memory:
    # Add records and ask questions through this one physical memory directory.
    pass
```

Give every live instance a different `data_dir`. The directory is the durable memory domain and
has one owner at a time.

The local WeMM service in this recipe has a 2048-vector `messages` contract. It accepts text,
images, and videos; it does not accept audio embeddings. Configure FunASR to turn audio and video
audio into searchable text, then keep `index_speech` enabled. This adds ASR work on ingestion.

The local service limits one request to 128 samples and 64 MiB, each inline media object to 20 MiB,
and native video input to at least two real frames. It internally microbatches eight samples. Use
the direct service URL above for normal deployments; an experiment cache proxy is not part of the
public SDK configuration.

The local server runtime, model checkpoint, and service script are machine-managed dependencies;
they are not installed by MindBridge. Use the official
[WeMM-Embedding-2B model card](https://huggingface.co/tencent/WeMM-Embedding-2B) to obtain the
checkpoint and its serving prerequisites, then configure the service to expose the `messages`,
2048-dimension contract shown above. Keep its runtime and model files outside the MindBridge
environment, and keep the service's logs and cache in a fresh deployment directory when restarting
it.

## Optional typed-memory formation

Formation is off unless the configuration includes a `formation` slot. It uses a separate
OpenAI-compatible completion client on the write path, so choose the endpoint's output budget and
reasoning settings explicitly:

```yaml
formation:
  provider: openai
  base_url: http://your-qwen-endpoint.example/v1
  api_key: replace-me
  model: Qwen3.8-27B
  modalities: [text]
  temperature: 0.0
  seed: 0
  max_tokens: 1024
  timeout: 120.0
  max_retries: 0
  extra_body:
    chat_template_kwargs:
      enable_thinking: false
```

`add_many` makes one formation request for the supplied `Sequence[ContentInput]`. If an endpoint
needs smaller requests, choose that application-level batch size before calling the public method:

```python
from mindbridge import Memory

with Memory.from_config("/absolute/path/to/config.yaml") as memory:
    observations = (
        "Ada said she prefers tea.",
        "Ada brought a blue mug to the meeting.",
    )

    for start in range(0, len(observations), 2):
        memory.add_many(observations[start : start + 2])

    # After addressing a formation failure, resume its committed sources.
    pending = memory.pending_captures()
    memory.settle(memory_ids=[item.memory_id for item in pending])
```

MindBridge does not impose a formation batch size. Set the batch size and token limit for the
endpoint you operate. A source is committed before its formation call, so a formation failure does
not discard the source. Inspect `memory.pending_captures()` and, after addressing the failure,
resume the returned memory IDs with `memory.settle`. Settlement can retry model work; its attempt
and selection controls are documented with the
[public memory operations](api/python-sdk.md#memory-operations).
