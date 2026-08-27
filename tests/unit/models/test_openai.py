"""Contract tests for the bundled OpenAI-compatible adapters."""

import base64
import json
import re
import sys
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from types import ModuleType
from typing import cast

import httpx
import pytest
from openai import AsyncOpenAI

import mindbridge.edge.identity_diarization as identity_diarization
from mindbridge import configuration
from mindbridge.application.capabilities import OutputSchema
from mindbridge.core import (
    EmbeddingSpaceReference,
    MediaKind,
    ModelOutputError,
    ModelReference,
    ModelRequestError,
    ModelUnavailableError,
)
from mindbridge.edge.identity_inference import CAMPPLUS_MODEL
from mindbridge.models import (
    EmbedRequest,
    EmbedTask,
    GenerateRequest,
    MediaPart,
    ModelInput,
    TextPart,
)
from mindbridge.models.openai import (
    AudioFallbackGenerator,
    OpenAIEmbedder,
    OpenAIGenerator,
    create_generator,
    normalize_base_url,
)
from mindbridge.models.plugins import close_model
from mindbridge.telemetry import model_token_usage, operation_span

MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"


async def test_text_query_uses_typed_embedding_sdk() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        assert request.url.path == "/api/v1/embeddings"
        assert payload == {
            "input": ["Query: where is the tool?"],
            "model": MODEL_ID,
            "dimensions": 1_024,
            "encoding_format": "float",
        }
        return _embedding_response()

    embedder = _embedder(respond)
    try:
        result = await embedder.embed(
            EmbedRequest(
                inputs=(ModelInput((TextPart("where is the tool?"),)),),
                task=EmbedTask.QUERY,
            )
        )
    finally:
        await embedder.close()

    assert result.embeddings[0].values == (1.0,) + (0.0,) * 1_023


async def test_memory_document_uses_jina_document_prompt() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        assert payload == {
            "input": ["Document: Caroline plans to become a counselor."],
            "model": MODEL_ID,
            "dimensions": 1_024,
            "encoding_format": "float",
        }
        return _embedding_response(model=MODEL_ID)

    embedder = _embedder(respond)
    try:
        result = await embedder.embed(
            EmbedRequest(
                inputs=(ModelInput((TextPart("Caroline plans to become a counselor."),)),),
                task=EmbedTask.DOCUMENT,
            )
        )
    finally:
        await embedder.close()

    assert result.embeddings[0].values == (1.0,) + (0.0,) * 1_023


async def test_text_document_embedder_batches_and_restores_index_order() -> None:
    second = [0.0, 1.0] + [0.0] * 1_022

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        assert payload["input"] == ["Document: first event", "Document: second claim"]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": second},
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [1.0] + [0.0] * 1_023,
                    },
                ],
                "model": MODEL_ID,
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    embedder = _embedder(respond)
    try:
        result = await embedder.embed(
            EmbedRequest(
                inputs=(
                    ModelInput((TextPart("first event"),)),
                    ModelInput((TextPart("second claim"),)),
                ),
                task=EmbedTask.DOCUMENT,
            )
        )
    finally:
        await embedder.close()

    assert tuple(item.values for item in result.embeddings) == (
        (1.0,) + (0.0,) * 1_023,
        tuple(second),
    )


async def test_multimodal_query_preserves_native_av_parts() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        samples = cast(list[list[dict[str, object]]], payload["input"])
        messages = samples[0]
        content = cast(list[dict[str, object]], messages[0]["content"])
        assert request.url.path == "/api/v1/embeddings"
        assert payload["model"] == MODEL_ID
        assert payload["dimensions"] == 1_024
        assert content[0] == {"type": "text", "text": "Query: find this moment"}
        assert {item["type"] for item in content} == {
            "text",
            "image_url",
            "video_url",
            "audio_url",
        }
        audio = next(item for item in content if item["type"] == "audio_url")
        assert cast(dict[str, str], audio["audio_url"])["url"].endswith("/media_audio")
        video = next(item for item in content if item["type"] == "video_url")
        assert video == {
            "type": "video_url",
            "video_url": {"url": "https://objects.example.test/media_video"},
        }
        return _embedding_response()

    embedder = _embedder(respond)
    query = ModelInput(
        (
            TextPart("find this moment"),
            *(
                _media_part(kind, suffix)
                for kind, suffix in (
                    (MediaKind.IMAGE, "image"),
                    (MediaKind.VIDEO, "video"),
                    (MediaKind.AUDIO, "audio"),
                )
            ),
        )
    )
    try:
        await embedder.embed(EmbedRequest(inputs=(query,), task=EmbedTask.QUERY))
    finally:
        await embedder.close()


@pytest.mark.parametrize(
    ("model", "embedding", "match"),
    [
        ("wrong-model", [1.0] + [0.0] * 1_023, "model"),
        (MODEL_ID, [1.0, 0.0], "dimension"),
        (MODEL_ID, [0.5] + [0.0] * 1_023, "L2-normalized"),
    ],
)
async def test_invalid_embedding_output_is_rejected(
    model: str,
    embedding: list[float],
    match: str,
) -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return _embedding_response(model=model, embedding=embedding)

    embedder = _embedder(respond)
    try:
        with pytest.raises(ModelOutputError, match=match):
            await embedder.embed(
                EmbedRequest(
                    inputs=(ModelInput((TextPart("find it"),)),),
                    task=EmbedTask.QUERY,
                )
            )
    finally:
        await embedder.close()


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (400, ModelRequestError),
        (408, ModelUnavailableError),
        (409, ModelUnavailableError),
        (429, ModelUnavailableError),
        (500, ModelUnavailableError),
    ],
)
async def test_only_transient_provider_failures_are_retryable(
    status_code: int,
    error_type: type[RuntimeError],
) -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"error": {"message": "provider detail", "type": "request_error"}},
        )

    embedder = _embedder(respond)
    try:
        with pytest.raises(error_type, match="embedding request failed"):
            await embedder.embed(
                EmbedRequest(
                    inputs=(ModelInput((TextPart("find it"),)),),
                    task=EmbedTask.QUERY,
                )
            )
    finally:
        await embedder.close()


async def test_a_dropped_completion_stream_is_retryable() -> None:
    """A provider that closes a long multimodal response mid-stream raises inside the stream
    iterator, past the SDK error handler, so an observation died permanently on a transient
    network fault instead of being retried."""

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _truncated_completion_response()

    generator = _generator(respond)
    try:
        with pytest.raises(ModelUnavailableError, match="generation request failed"):
            await generator.generate(
                GenerateRequest(
                    system_prompt="Answer from evidence.",
                    input=ModelInput((TextPart("where is the tool?"),)),
                    max_output_tokens=64,
                )
            )
    finally:
        await generator.close()


async def test_a_dropped_stream_still_charges_what_it_had_already_spent() -> None:
    """A truncated response was still billed, and a failed attempt's charge reaches the job row."""

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _truncated_completion_response(with_usage=True)

    generator = _generator(respond)
    try:
        async with operation_span("mindbridge.test.observation"):
            with pytest.raises(ModelUnavailableError):
                await generator.generate(
                    GenerateRequest(
                        system_prompt="Answer from evidence.",
                        input=ModelInput((TextPart("where is the tool?"),)),
                        max_output_tokens=64,
                    )
                )
            usage = model_token_usage()
    finally:
        await generator.close()

    assert usage == {"input": 11, "output": 1}


async def test_generation_reports_the_configured_model_not_the_serving_fingerprint() -> None:
    """A per-request serving fingerprint must not become the model identity."""

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _completion_response("serving-fingerprint-01")

    generator = _generator(respond)
    try:
        result = await generator.generate(
            GenerateRequest(
                system_prompt="Answer from evidence.",
                input=ModelInput((TextPart("where is the tool?"),)),
                max_output_tokens=64,
            )
        )
    finally:
        await generator.close()

    assert result.text == "on the workbench"
    assert result.model_reference == ModelReference(model_id="qwen3.8-max")


async def test_vlm_fallback_lazily_transcribes_av_and_preserves_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = 0
    transcribed: list[str] = []
    requests: list[list[dict[str, object]]] = []

    class Pipeline:
        def generate(self, **kwargs: object) -> list[dict[str, object]]:
            path = Path(cast(str, kwargs["input"]))
            transcribed.append(path.suffix)
            assert path.read_bytes() == b"media"
            assert kwargs["return_spk_res"] is True
            assert kwargs["return_spk_center"] is True
            assert kwargs["sentence_timestamp"] is True
            return [
                {
                    "text": "hello there. goodbye. again.",
                    "sentence_info": [
                        {"start": 0, "end": 800, "text": "hello there.", "spk": 0},
                        {"start": 1_000, "end": 1_900, "text": "goodbye.", "spk": 1},
                        {"start": 2_000, "end": 2_500, "text": "again.", "spk": 0},
                    ],
                    "spk_embedding_center": [[1.0, 0.0], [0.0, 1.0]],
                }
            ]

    def auto_model(**kwargs: object) -> Pipeline:
        nonlocal loaded
        loaded += 1
        assert kwargs["device"] == "cpu"
        assert kwargs["spk_model"] == CAMPPLUS_MODEL.model_id
        return Pipeline()

    funasr = ModuleType("funasr")
    funasr.AutoModel = auto_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "funasr", funasr)
    monkeypatch.setattr(identity_diarization, "select_torch_device", lambda _device: "cpu")

    async def respond(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], json.loads(request.content))
        messages = cast(list[dict[str, object]], payload["messages"])
        requests.append(cast(list[dict[str, object]], messages[1]["content"]))
        return _completion_response("serving-fingerprint-01")

    encoded = base64.b64encode(b"media").decode("ascii")
    generator = AudioFallbackGenerator(_generator(respond), asr_device="cpu")
    try:
        await generator.generate(
            GenerateRequest(
                system_prompt="Answer from evidence.",
                input=ModelInput((TextPart("text only"),)),
                max_output_tokens=64,
            )
        )
        assert loaded == 0
        await generator.generate(
            GenerateRequest(
                system_prompt="Answer from evidence.",
                input=ModelInput(
                    (
                        TextPart("inspect this audiovisual input"),
                        MediaPart(
                            MediaKind.VIDEO,
                            f"data:video/mp4;base64,{encoded}",
                            source_uri="s3://memory/clip.mp4",
                        ),
                        MediaPart(
                            MediaKind.AUDIO,
                            f"data:audio/wav;base64,{encoded}",
                            source_uri="s3://memory/clip.wav",
                        ),
                    )
                ),
                max_output_tokens=64,
            )
        )
    finally:
        await generator.close()

    assert loaded == 1
    assert transcribed == [".mp4", ".wav"]
    assert [part["type"] for part in requests[0]] == ["text"]
    assert [part["type"] for part in requests[1]] == ["text", "video_url", "text", "text"]
    transcript_parts = [part for part in requests[1] if part["type"] == "text"][1:]
    assert all("asr_transcript" in cast(str, part["text"]) for part in transcript_parts)
    assert all(
        "speaker A | 00:00:00.000-00:00:00.800 | hello there." in cast(str, part["text"])
        and "speaker B | 00:00:01.000-00:00:01.900 | goodbye." in cast(str, part["text"])
        and "speaker A | 00:00:02.000-00:00:02.500 | again." in cast(str, part["text"])
        for part in transcript_parts
    )


async def test_generator_factory_selects_native_or_transcribed_audio() -> None:
    config = {
        "api_key": "unit-test-key",
        "endpoint": "https://generator.example.test/v1",
        "model_id": "test-model",
    }
    native = create_generator(config)
    fallback = create_generator({**config, "audio_mode": "transcribe"})
    try:
        assert isinstance(native, OpenAIGenerator)
        assert isinstance(fallback, AudioFallbackGenerator)
        with pytest.raises(ValueError, match="audio_mode"):
            create_generator({**config, "audio_mode": "guess"})
    finally:
        await close_model(native)
        await close_model(fallback)


async def test_transcribing_generator_carries_its_engine_and_recipe_to_funasr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`audio_mode="transcribe"` is a whole speech pipeline behind two config keys. Without
    this, `asr_engine` and `asr_recipe` could be dropped on the floor and every existing test
    would still pass -- the transcription test only sets `asr_device`.
    """
    arguments: dict[str, object] = {}

    class Pipeline:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            return [{"text": ""}]

    def auto_model(**kwargs: object) -> Pipeline:
        arguments.update(kwargs)
        return Pipeline()

    funasr = ModuleType("funasr")
    funasr.AutoModel = auto_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "funasr", funasr)
    monkeypatch.setattr(identity_diarization, "select_torch_device", lambda _device: "cpu")

    config = {
        "api_key": "unit-test-key",
        "endpoint": "https://generator.example.test/v1",
        "model_id": "video-vlm",
        "audio_mode": "transcribe",
        "asr_engine": "automodel",
        "asr_recipe": "paraformer",
        "asr_device": "cpu",
    }
    generator = create_generator(config)
    encoded = base64.b64encode(b"media").decode("ascii")
    try:
        await generator.generate(
            GenerateRequest(
                system_prompt="Answer from evidence.",
                input=ModelInput((MediaPart(MediaKind.AUDIO, f"data:audio/wav;base64,{encoded}"),)),
                max_output_tokens=64,
            )
        )
    except Exception:
        # The HTTP call is not the subject; the recipe reaching FunASR is.
        pass
    finally:
        await close_model(generator)

    # The named recipe, not the default: `paraformer` is the only one carrying a punctuation
    # model, so its presence proves the composition travelled rather than just a model id.
    assert arguments["model"] == identity_diarization.FUNASR_ASR_MODEL_ID
    assert arguments["punc_model"] == identity_diarization.FUNASR_PUNCTUATION_MODEL_ID

    # An unusable engine, or an engine that cannot serve the recipe, is refused while the
    # generator is being built. The pipeline still loads lazily, but a worker that accepts
    # traffic and then fails on the first clip carrying speech is worse than one that will not
    # start -- and both answers are known from two strings.
    with pytest.raises(ValueError, match="unknown speech engine"):
        create_generator({**config, "asr_engine": "llama.cpp"})
    with pytest.raises(ValueError, match="cannot run"):
        create_generator({**config, "asr_engine": "vllm", "asr_recipe": "sensevoice"})


async def test_the_shipped_sample_config_can_actually_be_uncommented(tmp_path: Path) -> None:
    """`mindbridge.toml` is the shipped example, so its commented lines have to work together.

    Two drafts of this block did not: one pinned the engine as a side effect of switching
    transcription on, and the next paired the vLLM engine with a model it cannot serve. Both
    read fine and both would have failed for whoever uncommented them.
    """
    generator = create_generator(
        {**_uncommented_sample_generator_config(tmp_path), "api_key": "unit-test-key"}
    )
    try:
        assert isinstance(generator, AudioFallbackGenerator)
    finally:
        await close_model(generator)


def _uncommented_sample_generator_config(directory: Path) -> dict[str, object]:
    """Read `[generator]` from the shipped sample with every commented key switched on.

    Parsed by `mindbridge.configuration` rather than by a TOML library imported here: it already
    carries the `tomli` backport that the floor of the version matrix needs, and going through
    the real loader is the point of the exercise anyway.
    """
    sample = Path("mindbridge.toml").read_text(encoding="utf-8")
    section = sample[sample.index("[generator]") :]
    section = section[: section.index("\n[", 1)]
    commented = re.findall(r"^# ([a-z_]+ = .*)$", section, re.M)
    assert commented, "the sample lost its commented generator keys; this guard now tests nothing"

    live = sample
    for line in commented:
        live = live.replace(f"# {line}", line)
    live = live.replace('audio_mode = "native"', 'audio_mode = "transcribe"')
    located = directory / "mindbridge.toml"
    located.write_text(live, encoding="utf-8")

    document = configuration._configuration_document({}, located)
    assert document is not None
    encoded = (
        configuration._flattened_scalars(document) | configuration._flattened_plugins(document, {})
    )["MINDBRIDGE_GENERATOR_CONFIG_JSON"]
    return cast(dict[str, object], json.loads(encoded))


async def test_cumulative_stream_usage_is_charged_once_not_once_per_chunk() -> None:
    """A server sending usage on every chunk multiplied the recorded bill by the chunk count.

    OpenAI sends one final usage chunk, but vLLM's `continuous_usage_stats` -- and several
    gateways by default -- repeat the running total on each one. The token account accumulates
    whatever it is given, so what reached the job row and `mindbridge.model.tokens` was the sum
    of every partial total rather than the charge.
    """

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _continuous_usage_completion_response()

    generator = _generator(respond)
    try:
        async with operation_span("mindbridge.test.observation"):
            result = await generator.generate(
                GenerateRequest(
                    system_prompt="Answer from evidence.",
                    input=ModelInput((TextPart("where is the tool?"),)),
                    max_output_tokens=64,
                )
            )
            usage = model_token_usage()
    finally:
        await generator.close()

    assert result.text == "on the workbench"
    # The last chunk's totals, which are the whole request's charge.
    assert usage == {"input": 11, "output": 3}


async def test_schema_constrained_generation_asks_for_the_named_shape() -> None:
    formats: list[object] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        formats.append(cast(dict[str, object], json.loads(request.content))["response_format"])
        return _completion_response("serving-fingerprint-01")

    generator = _generator(respond)
    try:
        await generator.generate(_schema_request())
    finally:
        await generator.close()

    assert formats == [
        {
            "type": "json_schema",
            "json_schema": {
                "name": "verdict",
                "schema": {
                    "additionalProperties": False,
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "type": "object",
                },
                "strict": True,
            },
        }
    ]


async def test_an_endpoint_without_schema_support_degrades_once_and_stays_degraded() -> None:
    """A 400 is otherwise permanent, so an older endpoint would fail every observation.

    The capability is latched for the process: rediscovering it costs one rejected request
    per generation, and an endpoint that cannot compile a schema will not start to.
    """
    formats: list[object] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        response_format = cast(dict[str, object], json.loads(request.content))["response_format"]
        formats.append(response_format)
        if cast(dict[str, object], response_format)["type"] == "json_schema":
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "response_format json_schema is not supported",
                        "type": "invalid_request_error",
                    }
                },
            )
        return _completion_response("serving-fingerprint-01")

    generator = _generator(respond)
    try:
        first = await generator.generate(_schema_request())
        second = await generator.generate(_schema_request())
    finally:
        await generator.close()

    assert first.text == second.text == "on the workbench"
    assert [cast(dict[str, object], item)["type"] for item in formats] == [
        "json_schema",
        "json_object",
        "json_object",
    ]


async def test_an_ordinary_bad_request_is_not_mistaken_for_missing_schema_support() -> None:
    """Falling back on any 400 would silently pay for a second call on every real failure."""
    attempts = 0

    async def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            400,
            json={"error": {"message": "image exceeds the pixel budget", "type": "invalid"}},
        )

    generator = _generator(respond)
    try:
        with pytest.raises(ModelRequestError, match="generation request failed"):
            await generator.generate(_schema_request())
    finally:
        await generator.close()

    assert attempts == 1


def _schema_request() -> GenerateRequest:
    return GenerateRequest(
        system_prompt="Answer from evidence.",
        input=ModelInput((TextPart("where is the tool?"),)),
        max_output_tokens=64,
        output_schema=OutputSchema(
            name="verdict",
            json_schema=json.dumps(
                {
                    "additionalProperties": False,
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "type": "object",
                },
                sort_keys=True,
            ),
        ),
    )


def _generator(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> OpenAIGenerator:
    return OpenAIGenerator(
        AsyncOpenAI(
            api_key="unit-test-key",
            base_url=normalize_base_url("https://vlm.example.test/api/v1/chat/completions"),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
        ),
        ModelReference(model_id="qwen3.8-max"),
    )


def _completion_response(fingerprint: str) -> httpx.Response:
    event = {
        "id": "completion_01",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen3.8-max",
        "system_fingerprint": fingerprint,
        "choices": [
            {
                "index": 0,
                "delta": {"content": "on the workbench"},
                "finish_reason": "stop",
            }
        ],
    }
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
    )


def _continuous_usage_completion_response() -> httpx.Response:
    """Three chunks, each repeating the running total, as `continuous_usage_stats` does."""
    events = [
        {
            "id": "completion_01",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "qwen3.8-max",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": completion_tokens,
                "total_tokens": 11 + completion_tokens,
            },
        }
        for content, finish, completion_tokens in (
            ("on ", None, 1),
            ("the ", None, 2),
            ("workbench", "stop", 3),
        )
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"{body}data: [DONE]\n\n",
    )


def _truncated_completion_response(*, with_usage: bool = False) -> httpx.Response:
    """One valid chunk, then the peer disappears in the middle of the body."""
    event: dict[str, object] = {
        "id": "completion_01",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen3.8-max",
        "choices": [{"index": 0, "delta": {"content": "on the "}, "finish_reason": None}],
    }
    if with_usage:
        event["usage"] = {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12}

    async def body() -> AsyncIterator[bytes]:
        yield f"data: {json.dumps(event)}\n\n".encode()
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )

    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body(),
    )


async def test_embedding_reports_its_usage_into_the_token_account() -> None:
    """Serving the encoder makes embedding a metered call, so its bill has to land somewhere.

    Only the generator reported usage, so an embedding endpoint's charge was invisible -- which
    matters exactly when the recommended configuration moves embedding off the local GPU and
    onto a metered endpoint. An embedding charges entirely on its input; there is no completion
    half to report.
    """

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [1.0] + [0.0] * 1_023},
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0] + [0.0] * 1_022},
                ],
                "model": MODEL_ID,
                "usage": {"prompt_tokens": 42, "total_tokens": 42},
            },
        )

    embedder = _embedder(respond)
    try:
        async with operation_span("mindbridge.test.observation"):
            await embedder.embed(
                EmbedRequest(
                    inputs=(
                        ModelInput((TextPart("first event"),)),
                        ModelInput((TextPart("second claim"),)),
                    ),
                    task=EmbedTask.DOCUMENT,
                )
            )
            usage = model_token_usage()
    finally:
        await embedder.close()

    assert usage == {"input": 42}


async def test_multimodal_embedding_charges_the_whole_batch() -> None:
    """A multimodal batch crosses the wire once and restores provider index order."""
    requests: list[httpx.Request] = []
    vectors = (
        [1.0] + [0.0] * 1_023,
        [0.0, 1.0] + [0.0] * 1_022,
        [0.0, 0.0, 1.0] + [0.0] * 1_021,
    )

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload: dict[str, object] = json.loads(request.content)
        samples = cast(list[list[dict[str, object]]], payload["input"])
        assert len(samples) == 3
        assert [
            cast(dict[str, str], cast(list[dict[str, object]], sample[0]["content"])[1][kind])[
                "url"
            ]
            for sample, kind in zip(samples, ("video_url", "image_url", "audio_url"), strict=True)
        ] == [
            "https://objects.example.test/media_01",
            "https://objects.example.test/media_02",
            "https://objects.example.test/media_03",
        ]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": index, "embedding": vectors[index]}
                    for index in (2, 0, 1)
                ],
                "model": MODEL_ID,
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        )

    embedder = _embedder(respond)
    try:
        async with operation_span("mindbridge.test.observation"):
            result = await embedder.embed(
                EmbedRequest(
                    inputs=(
                        ModelInput((_media_part(MediaKind.VIDEO, "01"),)),
                        ModelInput((_media_part(MediaKind.IMAGE, "02"),)),
                        ModelInput((_media_part(MediaKind.AUDIO, "03"),)),
                    ),
                    task=EmbedTask.DOCUMENT,
                )
            )
            usage = model_token_usage()
    finally:
        await embedder.close()

    assert len(requests) == 1
    assert tuple(item.values for item in result.embeddings) == tuple(map(tuple, vectors))
    assert usage == {"input": 3}


def _embedder(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> OpenAIEmbedder:
    client = AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_base_url("https://embedding.example.test/api/v1/embeddings"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return OpenAIEmbedder(
        client,
        ModelReference(model_id=MODEL_ID),
        space_reference=EmbeddingSpaceReference(space_id="jina-space"),
    )


def _media_part(kind: MediaKind, suffix: str) -> MediaPart:
    extension = {MediaKind.IMAGE: "jpg", MediaKind.VIDEO: "mp4", MediaKind.AUDIO: "wav"}[kind]
    return MediaPart(
        kind=kind,
        url=f"https://objects.example.test/media_{suffix}",
        source_uri=f"s3://memory/tenants/tenant_01/query.{extension}",
    )


def _embedding_response(
    *,
    model: str = MODEL_ID,
    embedding: list[float] | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": embedding or [1.0] + [0.0] * 1_023,
                }
            ],
            "model": model,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )
