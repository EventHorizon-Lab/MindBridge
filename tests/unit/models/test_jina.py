"""Tests for the local SentenceTransformers boundary."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from mindbridge.core import (
    EmbeddingSpaceReference,
    MediaKind,
    ModelOutputError,
    ModelReference,
    ModelRequestError,
    ModelUnavailableError,
)
from mindbridge.models import EmbedRequest, EmbedTask, MediaPart, ModelInput, TextPart
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDER_MODEL_ID,
    DEFAULT_EMBEDDER_REVISION,
    embedder_revision_for,
    sentence_transformers_media_embedder_config,
)
from mindbridge.models.jina import SentenceTransformersEmbedder, _EmbedderConfig


class Matrix:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def tolist(self) -> list[list[float]]:
        return self._values


class RecordingEncoder:
    """Minimal current SentenceTransformer API with recorded batch inputs."""

    def __init__(
        self,
        values: list[list[float]],
        *,
        modalities: tuple[str, ...] = ("text", "image", "video", "message"),
        native_dimension: int = 2,
        matryoshka_dimensions: tuple[int, ...] = (),
    ) -> None:
        self.values = values
        self.modalities = modalities
        self.native_dimension = native_dimension
        self.config = SimpleNamespace(
            is_matryoshka=bool(matryoshka_dimensions),
            matryoshka_dimensions=matryoshka_dimensions,
        )
        self.calls: list[tuple[str, list[object], int | None]] = []

    def supports(self, modality: str | tuple[str, ...]) -> bool:
        if isinstance(modality, tuple):
            return "message" in self.modalities and all(
                item in self.modalities for item in modality
            )
        return modality in self.modalities

    def get_sentence_embedding_dimension(self) -> int:
        return self.native_dimension

    def encode_query(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> Matrix:
        self.calls.append(("query", sentences, truncate_dim))
        return Matrix(self.values)

    def encode_document(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> Matrix:
        self.calls.append(("document", sentences, truncate_dim))
        return Matrix(self.values)


def test_default_jina_pins_remote_code_without_omni_loader_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    encoder = RecordingEncoder([[1.0, 0.0]])

    def sentence_transformer(model_id: str, **kwargs: object) -> RecordingEncoder:
        calls.append((model_id, kwargs))
        return encoder

    monkeypatch.setattr(
        "mindbridge.models.jina.import_module",
        lambda _name: SimpleNamespace(SentenceTransformer=sentence_transformer),
    )
    monkeypatch.setattr("mindbridge.models.jina.select_torch_device", lambda _device: "cuda")

    SentenceTransformersEmbedder.load(dimension=2)

    assert calls == [
        (
            DEFAULT_EMBEDDER_MODEL_ID,
            {
                "revision": DEFAULT_EMBEDDER_REVISION,
                "trust_remote_code": True,
                "device": "cuda",
                "model_kwargs": {"code_revision": DEFAULT_EMBEDDER_REVISION},
                "config_kwargs": {"code_revision": DEFAULT_EMBEDDER_REVISION},
            },
        )
    ]


def test_other_models_are_unpinned_untrusted_and_require_a_new_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def sentence_transformer(model_id: str, **kwargs: object) -> RecordingEncoder:
        calls.append((model_id, kwargs))
        return RecordingEncoder([[1.0, 0.0]])

    monkeypatch.setattr(
        "mindbridge.models.jina.import_module",
        lambda _name: SimpleNamespace(SentenceTransformer=sentence_transformer),
    )
    monkeypatch.setattr("mindbridge.models.jina.select_torch_device", lambda _device: "cpu")

    with pytest.raises(ValueError, match="new space_id"):
        SentenceTransformersEmbedder.load(model_id="Qwen/Qwen3-VL-Embedding-2B", dimension=2)

    SentenceTransformersEmbedder.load(
        model_id="Qwen/Qwen3-VL-Embedding-2B",
        space_reference=EmbeddingSpaceReference(space_id="qwen-vl-2b-v1"),
        dimension=2,
    )

    assert calls == [
        (
            "Qwen/Qwen3-VL-Embedding-2B",
            {
                "revision": None,
                "trust_remote_code": False,
                "device": "cpu",
                "model_kwargs": None,
                "config_kwargs": None,
            },
        )
    ]


def test_an_overridden_model_id_does_not_inherit_the_jina_pin() -> None:
    config = sentence_transformers_media_embedder_config(
        {"MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID": "Qwen/Qwen3-VL-Embedding-2B"}
    )
    validated = _EmbedderConfig.model_validate(config)

    assert embedder_revision_for(validated.model_id, validated.model_revision) is None
    assert embedder_revision_for(DEFAULT_EMBEDDER_MODEL_ID, None) == DEFAULT_EMBEDDER_REVISION


async def test_standard_multimodal_dict_and_message_inputs_are_batched() -> None:
    encoder = RecordingEncoder([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    embedder = SentenceTransformersEmbedder(
        encoder,
        ModelReference(model_id="Qwen/Qwen3-VL-Embedding-2B"),
        space_reference=EmbeddingSpaceReference(space_id="qwen-vl-2b-v1"),
        dimension=2,
    )

    result = await embedder.embed(
        EmbedRequest(
            inputs=(
                ModelInput(
                    (
                        TextPart("describe"),
                        MediaPart(MediaKind.IMAGE, "image.jpg"),
                    )
                ),
                ModelInput(
                    (
                        MediaPart(MediaKind.IMAGE, "first.jpg"),
                        MediaPart(MediaKind.IMAGE, "second.jpg"),
                    )
                ),
                ModelInput((MediaPart(MediaKind.VIDEO, "clip.mp4"),)),
            ),
            task=EmbedTask.DOCUMENT,
        )
    )

    assert tuple(item.values for item in result.embeddings) == (
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
    )
    assert encoder.calls == [
        (
            "document",
            [
                {"text": "describe", "image": "image.jpg"},
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "first.jpg"},
                            {"type": "image", "image": "second.jpg"},
                        ],
                    }
                ],
                {"video": "clip.mp4"},
            ],
            None,
        )
    ]


async def test_model_modalities_are_checked_before_encoding() -> None:
    encoder = RecordingEncoder([[1.0, 0.0]], modalities=("text", "image"))
    embedder = SentenceTransformersEmbedder(
        encoder,
        ModelReference(model_id="image-only-multimodal"),
        space_reference=EmbeddingSpaceReference(space_id="image-space"),
        dimension=2,
    )
    await embedder.embed(
        EmbedRequest(
            inputs=(ModelInput((MediaPart(MediaKind.IMAGE, "sample.jpg"),)),),
            task=EmbedTask.DOCUMENT,
        )
    )

    with pytest.raises(ModelRequestError, match="does not support audio"):
        await embedder.embed(
            EmbedRequest(
                inputs=(ModelInput((MediaPart(MediaKind.AUDIO, "sample.wav"),)),),
                task=EmbedTask.DOCUMENT,
            )
        )
    assert encoder.calls[0][1] == [{"image": "sample.jpg"}]
    assert len(encoder.calls) == 1


async def test_default_jina_keeps_its_legacy_multipart_input_only() -> None:
    encoder = RecordingEncoder([[1.0, 0.0]], modalities=("text",))
    embedder = SentenceTransformersEmbedder(
        encoder,
        ModelReference(model_id=DEFAULT_EMBEDDER_MODEL_ID),
        dimension=2,
    )

    await embedder.embed(
        EmbedRequest(
            inputs=(ModelInput((TextPart("describe"), MediaPart(MediaKind.IMAGE, "image.jpg"))),),
            task=EmbedTask.DOCUMENT,
        )
    )

    assert encoder.calls[0][1] == [("describe", "image.jpg")]


def test_load_accepts_only_native_or_advertised_matryoshka_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _load_with(RecordingEncoder([[1.0, 0.0]], native_dimension=2), monkeypatch, 2)
    shrunk = _load_with(
        RecordingEncoder(
            [[1.0, 0.0]],
            native_dimension=4,
            matryoshka_dimensions=(2, 4),
        ),
        monkeypatch,
        2,
    )

    assert native._truncate_dim is None
    assert shrunk._truncate_dim == 2

    with pytest.raises(ModelUnavailableError, match="not an advertised Matryoshka dimension"):
        _load_with(RecordingEncoder([[1.0, 0.0]], native_dimension=4), monkeypatch, 2)


async def test_query_and_document_methods_and_output_validation() -> None:
    encoder = RecordingEncoder([[1.0, 0.0]])
    embedder = SentenceTransformersEmbedder(
        encoder,
        ModelReference(model_id="text-model"),
        space_reference=EmbeddingSpaceReference(space_id="text-space"),
        dimension=2,
    )
    request = (ModelInput((TextPart("screwdriver"),)),)

    await embedder.embed(EmbedRequest(inputs=request, task=EmbedTask.QUERY))
    await embedder.embed(EmbedRequest(inputs=request, task=EmbedTask.DOCUMENT))

    assert [call[0] for call in encoder.calls] == ["query", "document"]

    broken = SentenceTransformersEmbedder(
        RecordingEncoder([[1.0, 1.0]]),
        ModelReference(model_id="text-model"),
        space_reference=EmbeddingSpaceReference(space_id="text-space"),
        dimension=2,
    )
    with pytest.raises(ModelOutputError, match="L2-normalized"):
        await broken.embed(EmbedRequest(inputs=request, task=EmbedTask.QUERY))


class GatedEncoder(RecordingEncoder):
    def __init__(self) -> None:
        super().__init__([[1.0, 0.0]])
        self.entered = threading.Event()
        self.release = threading.Event()

    def encode_document(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> Matrix:
        self.calls.append(("document", sentences, truncate_dim))
        self.entered.set()
        assert self.release.wait(timeout=10)
        return Matrix(self.values)


async def test_query_lane_stays_free_while_documents_encode() -> None:
    encoder = GatedEncoder()
    embedder = SentenceTransformersEmbedder(
        encoder,
        ModelReference(model_id="text-model"),
        space_reference=EmbeddingSpaceReference(space_id="text-space"),
        dimension=2,
    )
    document = EmbedRequest(
        inputs=(ModelInput((TextPart("bulk document"),)),),
        task=EmbedTask.DOCUMENT,
    )
    first = asyncio.create_task(embedder.embed(document))
    await asyncio.to_thread(encoder.entered.wait, 10)
    second = asyncio.create_task(embedder.embed(document))
    await asyncio.sleep(0.05)

    query = await asyncio.wait_for(
        embedder.embed(
            EmbedRequest(
                inputs=(ModelInput((TextPart("where?"),)),),
                task=EmbedTask.QUERY,
            )
        ),
        timeout=5,
    )
    assert query.embeddings[0].values == (1.0, 0.0)
    assert [call[0] for call in encoder.calls].count("document") == 1

    encoder.release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=10)


class OmniModule:
    def __init__(self, processor: object) -> None:
        self.processor = processor


class OmniEncoder(RecordingEncoder):
    def __init__(self, processor: object) -> None:
        super().__init__([[1.0, 0.0]])
        self._modules = [OmniModule(processor)]

    def __iter__(self) -> object:
        return iter(self._modules)


def test_default_jina_refuses_a_missing_media_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ModelUnavailableError, match="only embed text"):
        _load_with(OmniEncoder(None), monkeypatch, 2)


def _load_with(
    encoder: RecordingEncoder,
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
) -> SentenceTransformersEmbedder:
    monkeypatch.setattr(
        "mindbridge.models.jina.import_module",
        lambda _name: SimpleNamespace(SentenceTransformer=lambda *_args, **_kwargs: encoder),
    )
    monkeypatch.setattr("mindbridge.models.jina.select_torch_device", lambda _device: "cuda")
    return SentenceTransformersEmbedder.load(dimension=dimension)
