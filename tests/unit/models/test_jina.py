"""Tests for the Jina Sentence Transformers boundary."""

from types import SimpleNamespace

import pytest

from mindbridge.core import MediaKind, ModelOutputError, ModelReference, ModelUnavailableError
from mindbridge.models import EmbedRequest, EmbedTask, MediaPart, ModelInput, TextPart
from mindbridge.models.defaults import require_matryoshka_dimension
from mindbridge.models.jina import JinaEmbedder


class Matrix:
    """Minimal ndarray-shaped result used by the adapter boundary."""

    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def tolist(self) -> list[list[float]]:
        return self._values


class RecordingEncoder:
    """Records whether query/document semantics reach the official API."""

    def __init__(self, values: list[list[float]]) -> None:
        self.values = values
        self.calls: list[str] = []

    def encode_query(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int,
    ) -> Matrix:
        self.calls.append("query")
        return Matrix(self.values)

    def encode_document(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int,
    ) -> Matrix:
        self.calls.append("document")
        return Matrix(self.values)


def test_jina_pins_remote_code_to_the_model_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def sentence_transformer(model_path: str, **kwargs: object) -> RecordingEncoder:
        calls.append((model_path, kwargs))
        return RecordingEncoder([[1.0, 0.0]])

    def snapshot_download(**kwargs: object) -> str:
        calls.append(("snapshot_download", kwargs))
        return "/models/pinned"

    monkeypatch.setattr(
        "mindbridge.models.jina.import_module",
        lambda name: (
            SimpleNamespace(SentenceTransformer=sentence_transformer)
            if name == "sentence_transformers"
            else SimpleNamespace(snapshot_download=snapshot_download)
        ),
    )
    monkeypatch.setattr("mindbridge.models.jina.select_torch_device", lambda _device: "cuda")

    JinaEmbedder.load(revision="pinned-revision", dimension=2)

    assert calls == [
        (
            "snapshot_download",
            {
                "repo_id": "jinaai/jina-embeddings-v5-omni-small-retrieval",
                "revision": "pinned-revision",
            },
        ),
        (
            "/models/pinned",
            {
                "revision": "pinned-revision",
                "trust_remote_code": True,
                "device": "cuda",
                "model_kwargs": {
                    "modality": "omni",
                    "code_revision": "pinned-revision",
                },
                "config_kwargs": {"code_revision": "pinned-revision"},
            },
        ),
    ]


async def test_jina_uses_distinct_query_and_document_methods() -> None:
    """Retrieval prefixes remain owned by Sentence Transformers."""
    encoder = RecordingEncoder([[1.0, 0.0]])
    embedder = JinaEmbedder(
        encoder,
        ModelReference(model_id="jina", revision="revision"),
        dimension=2,
    )

    query = await embedder.embed(
        EmbedRequest(
            inputs=(ModelInput((TextPart("Where is the screwdriver?"),)),),
            task=EmbedTask.QUERY,
        )
    )
    document = await embedder.embed(
        EmbedRequest(
            inputs=(ModelInput((MediaPart(MediaKind.IMAGE, "file:///image.jpg"),)),),
            task=EmbedTask.DOCUMENT,
        )
    )

    assert tuple(item.values for item in query.embeddings) == ((1.0, 0.0),)
    assert document.embeddings == query.embeddings
    assert encoder.calls == ["query", "document"]


async def test_jina_rejects_invalid_model_output() -> None:
    """Malformed upstream vectors cannot enter the semantic index."""
    embedder = JinaEmbedder(
        RecordingEncoder([[1.0, 1.0]]),
        ModelReference(model_id="jina", revision="revision"),
        dimension=2,
    )

    with pytest.raises(ModelOutputError, match="L2-normalized"):
        await embedder.embed(
            EmbedRequest(
                inputs=(ModelInput((TextPart("query"),)),),
                task=EmbedTask.QUERY,
            )
        )


async def test_jina_skips_model_call_for_empty_batch() -> None:
    """An empty batch has no model cost or ambiguous output shape."""
    encoder = RecordingEncoder([])
    embedder = JinaEmbedder(
        encoder,
        ModelReference(model_id="jina", revision="revision"),
        dimension=2,
    )

    assert (await embedder.embed(EmbedRequest(inputs=(), task=EmbedTask.DOCUMENT))).embeddings == ()
    assert encoder.calls == []


def test_matryoshka_validation_accepts_trained_widths_and_rejects_others() -> None:
    """A deployment may shrink vectors, but only to a width Jina actually trained."""
    assert require_matryoshka_dimension(256) == 256
    assert require_matryoshka_dimension(1_024) == 1_024

    with pytest.raises(ValueError, match="embedding dimension must be one of"):
        require_matryoshka_dimension(500)


class OmniModule:
    """One SentenceTransformer submodule, with or without its media processor."""

    def __init__(self, processor: object) -> None:
        self.processor = processor


class OmniEncoder(RecordingEncoder):
    """An encoder that exposes its submodules the way SentenceTransformer does."""

    def __init__(self, processor: object) -> None:
        super().__init__([[1.0, 0.0]])
        self._modules = [OmniModule(processor)]

    def __iter__(self) -> object:
        return iter(self._modules)


def _load_with(encoder: object, monkeypatch: pytest.MonkeyPatch) -> JinaEmbedder:
    monkeypatch.setattr(
        "mindbridge.models.jina.import_module",
        lambda name: (
            SimpleNamespace(SentenceTransformer=lambda *_a, **_k: encoder)
            if name == "sentence_transformers"
            else SimpleNamespace(snapshot_download=lambda **_k: "/models/pinned")
        ),
    )
    monkeypatch.setattr("mindbridge.models.jina.select_torch_device", lambda _device: "cuda")
    return JinaEmbedder.load(revision="pinned-revision", dimension=2)


def test_jina_refuses_a_model_whose_media_processor_failed_to_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A text-only degradation is refused at load, not discovered on the first frame.

    Jina assigns `None` to the processor inside a bare `except Exception`, so without this the
    model loads, embeds text, and raises an opaque TypeError once a perception call has already
    been spent on the clip it cannot encode.
    """
    with pytest.raises(ModelUnavailableError, match="only embed text"):
        _load_with(OmniEncoder(None), monkeypatch)


def test_jina_accepts_a_model_that_carries_its_media_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard rejects an empty processor slot, not every model it cannot introspect."""
    assert _load_with(OmniEncoder(object()), monkeypatch) is not None
