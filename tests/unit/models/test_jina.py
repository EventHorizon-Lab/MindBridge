"""Tests for the local SentenceTransformers boundary."""

import asyncio
import threading
from collections.abc import Callable
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


def test_default_jina_loads_the_pinned_snapshot_without_omni_loader_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What pins the weights is the local directory, not the `revision` kwarg beside it.

    `SentenceTransformer(repo_id, revision=...)` hands `revision` to a custom module's `load`
    as a keyword that module may ignore, and the bundled one -- `load(cls, input_path,
    **kwargs)` -- does ignore it, leaving `from_pretrained` to resolve the bare repository id
    against its default branch. So asserting the kwargs were passed passes while nothing is
    pinned; asserting the snapshot directory is what gets loaded is the pin itself.
    """
    calls: list[tuple[str, dict[str, object]]] = []
    encoder = RecordingEncoder([[1.0, 0.0]])

    def sentence_transformer(model_path: str, **kwargs: object) -> RecordingEncoder:
        calls.append((model_path, kwargs))
        return encoder

    monkeypatch.setattr(
        "mindbridge.models.jina.import_module",
        _importer(sentence_transformer, calls.append),
    )
    monkeypatch.setattr("mindbridge.models.jina.select_torch_device", lambda _device: "cuda")

    SentenceTransformersEmbedder.load(dimension=2)

    assert calls == [
        (
            "snapshot_download",
            {"repo_id": DEFAULT_EMBEDDER_MODEL_ID, "revision": DEFAULT_EMBEDDER_REVISION},
        ),
        (
            "/models/pinned",
            {
                "revision": DEFAULT_EMBEDDER_REVISION,
                "trust_remote_code": True,
                "device": "cuda",
                "model_kwargs": {"code_revision": DEFAULT_EMBEDDER_REVISION},
                "config_kwargs": {"code_revision": DEFAULT_EMBEDDER_REVISION},
            },
        ),
    ]


def test_trusted_remote_code_requires_a_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opting a repository into remote code without a commit is a moving execution target.

    `trust_remote_code` runs that repository's Python in every worker on every restart, so an
    unpinned opt-in re-reads whatever its default branch holds -- with no configuration change,
    and with `space_id` unchanged, so nothing downstream can see it happened.
    """
    monkeypatch.setattr(
        "mindbridge.models.jina.import_module",
        _importer(lambda *_args, **_kwargs: RecordingEncoder([[1.0, 0.0]])),
    )
    monkeypatch.setattr("mindbridge.models.jina.select_torch_device", lambda _device: "cpu")

    with pytest.raises(ValueError, match="requires a pinned"):
        SentenceTransformersEmbedder.load(
            model_id="someone/custom-code",
            trust_remote_code=True,
            space_reference=EmbeddingSpaceReference(space_id="custom-code-v1"),
            dimension=2,
        )

    SentenceTransformersEmbedder.load(
        model_id="someone/custom-code",
        revision="0123456789abcdef0123456789abcdef01234567",
        trust_remote_code=True,
        space_reference=EmbeddingSpaceReference(space_id="custom-code-v1"),
        dimension=2,
    )


def test_other_models_are_unpinned_untrusted_and_require_a_new_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def sentence_transformer(model_path: str, **kwargs: object) -> RecordingEncoder:
        calls.append((model_path, kwargs))
        return RecordingEncoder([[1.0, 0.0]])

    monkeypatch.setattr(
        "mindbridge.models.jina.import_module",
        _importer(sentence_transformer, calls.append),
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
        ("snapshot_download", {"repo_id": "Qwen/Qwen3-VL-Embedding-2B", "revision": None}),
        (
            "/models/pinned",
            {
                "revision": None,
                "trust_remote_code": False,
                "device": "cpu",
                "model_kwargs": None,
                "config_kwargs": None,
            },
        ),
    ]


def test_an_overridden_model_id_does_not_inherit_the_jina_pin() -> None:
    config = sentence_transformers_media_embedder_config(
        {"MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID": "Qwen/Qwen3-VL-Embedding-2B"}
    )
    validated = _EmbedderConfig.model_validate(config)

    assert embedder_revision_for(validated.model_id, validated.model_revision) is None
    assert embedder_revision_for(DEFAULT_EMBEDDER_MODEL_ID, None) == DEFAULT_EMBEDDER_REVISION


def test_the_embedder_plugin_keeps_an_operators_existing_pin() -> None:
    """`model_revision` in a config object is a pin, so it must not be ignored as retired.

    `PluginConfigModel` ignores the names migration 0021 retired so an operator's existing
    `*_CONFIG_JSON` does not stop a process from starting. This one name went on meaning
    something for this plugin, and silently replacing a pin with the default would be a worse
    failure than the strictness that tolerance relaxes.
    """
    config = _EmbedderConfig.model_validate(
        {"model_revision": "0123456789abcdef0123456789abcdef01234567", "space_revision": "gone"}
    )

    assert config.model_revision == "0123456789abcdef0123456789abcdef01234567"


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
    encoder = OmniEncoder(object())
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


async def test_the_same_legacy_model_under_another_name_keeps_its_media_support() -> None:
    """A mirror, a local snapshot directory, or the non-retrieval repository name is the model.

    The legacy shape was detected by comparing the configured model id against one string, so
    every other name for the same weights became a text-only encoder: the worker started clean
    and then refused each image, video, and audio embed at request time.
    """
    encoder = OmniEncoder(object())
    embedder = SentenceTransformersEmbedder(
        encoder,
        ModelReference(model_id="mirror.internal/jina-embeddings-v5-omni-small-retrieval"),
        space_reference=EmbeddingSpaceReference(space_id="jina-mirror-v1"),
        dimension=2,
    )

    await embedder.embed(
        EmbedRequest(
            inputs=(ModelInput((MediaPart(MediaKind.IMAGE, "image.jpg"),)),),
            task=EmbedTask.DOCUMENT,
        )
    )

    assert encoder.calls[0][1] == ["image.jpg"]


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


async def test_jina_skips_model_call_for_empty_batch() -> None:
    """An empty batch has no model cost or ambiguous output shape."""
    encoder = RecordingEncoder([])
    embedder = SentenceTransformersEmbedder(
        encoder,
        ModelReference(model_id="text-model"),
        space_reference=EmbeddingSpaceReference(space_id="text-space"),
        dimension=2,
    )

    assert (await embedder.embed(EmbedRequest(inputs=(), task=EmbedTask.DOCUMENT))).embeddings == ()
    assert encoder.calls == []


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
    """A module that predates the modality API: a processor slot and no declared modalities.

    `SentenceTransformer.modalities` is `getattr(self[0], "modalities", ["text"])`, so upstream
    reports this model as text-only however many media it can actually encode.
    """

    def __init__(self, processor: object) -> None:
        super().__init__([[1.0, 0.0]], modalities=("text",))
        self._modules = [OmniModule(processor)]

    def __iter__(self) -> object:
        return iter(self._modules)

    def __getitem__(self, index: int) -> object:
        return self._modules[index]


def test_default_jina_refuses_a_missing_media_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ModelUnavailableError, match="only embed text"):
        _load_with(OmniEncoder(None), monkeypatch, 2)


def test_jina_accepts_a_model_that_carries_its_media_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard rejects an empty processor slot, not every model it cannot introspect."""
    assert _load_with(OmniEncoder(object()), monkeypatch, 2) is not None


def test_jina_refuses_to_start_without_its_audio_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing audio support must fail readiness, not the first production audio request."""

    def import_dependency(name: str) -> object:
        if name == "librosa":
            raise ImportError(name)
        return _importer(lambda *_args, **_kwargs: OmniEncoder(object()))(name)

    monkeypatch.setattr("mindbridge.models.jina.import_module", import_dependency)

    with pytest.raises(ModelUnavailableError, match="cloud-models"):
        SentenceTransformersEmbedder.load(dimension=2)


def _importer(
    sentence_transformer: object,
    record: Callable[[tuple[str, dict[str, object]]], object] | None = None,
) -> Callable[[str], object]:
    """Stub the three dependencies `load` imports, recording the snapshot download."""

    def snapshot_download(**kwargs: object) -> str:
        if record is not None:
            record(("snapshot_download", kwargs))
        return "/models/pinned"

    def import_dependency(name: str) -> object:
        if name == "sentence_transformers":
            return SimpleNamespace(SentenceTransformer=sentence_transformer)
        if name == "huggingface_hub":
            return SimpleNamespace(snapshot_download=snapshot_download)
        return SimpleNamespace()

    return import_dependency


def _load_with(
    encoder: RecordingEncoder,
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
) -> SentenceTransformersEmbedder:
    monkeypatch.setattr(
        "mindbridge.models.jina.import_module",
        _importer(lambda *_args, **_kwargs: encoder),
    )
    monkeypatch.setattr("mindbridge.models.jina.select_torch_device", lambda _device: "cuda")
    return SentenceTransformersEmbedder.load(dimension=dimension)
