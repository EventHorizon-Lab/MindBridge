"""Tests for the Jina Sentence Transformers boundary."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from mindbridge.core import MediaKind, ModelOutputError, ModelReference, ModelUnavailableError
from mindbridge.models import EmbedRequest, EmbedTask, MediaPart, ModelInput, TextPart
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDER_MODEL_ID,
    DEFAULT_EMBEDDER_REVISION,
    embedder_revision_for,
    jina_media_embedder_config,
    require_matryoshka_dimension,
)
from mindbridge.models.jina import JinaEmbedder, _EmbedderConfig


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


def test_jina_pins_the_weights_and_the_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pin has to reach both the download and the code that gets executed.

    `trust_remote_code=True` means this process runs Python from the model repository, so an
    unpinned load resolves whatever is on its default branch. Migration 0021 removed the
    revision *column*, which nothing read; these are loader arguments, which the Hub resolves
    against a content-addressed commit.
    """
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

    JinaEmbedder.load(dimension=2)

    assert calls == [
        (
            "snapshot_download",
            {
                "repo_id": "jinaai/jina-embeddings-v5-omni-small-retrieval",
                "revision": DEFAULT_EMBEDDER_REVISION,
            },
        ),
        (
            "/models/pinned",
            {
                "revision": DEFAULT_EMBEDDER_REVISION,
                "trust_remote_code": True,
                "device": "cuda",
                "model_kwargs": {
                    "modality": "omni",
                    "code_revision": DEFAULT_EMBEDDER_REVISION,
                },
                "config_kwargs": {"code_revision": DEFAULT_EMBEDDER_REVISION},
            },
        ),
    ]


def test_jina_can_load_an_unpinned_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another repository is loaded unpinned, because this pin could not resolve against it.

    `DEFAULT_EMBEDDER_REVISION` is a commit of `DEFAULT_EMBEDDER_MODEL_ID`. Applying it to a
    repository the caller named instead is a `RevisionNotFoundError` at load, naming a sha that
    appears nowhere in their configuration -- so the pin is resolved from the model id rather
    than defaulted onto whatever repository turns up.
    """
    calls: list[tuple[str, dict[str, object]]] = []

    def sentence_transformer(model_path: str, **kwargs: object) -> RecordingEncoder:
        calls.append((model_path, kwargs))
        return RecordingEncoder([[1.0, 0.0]])

    def snapshot_download(**kwargs: object) -> str:
        calls.append(("snapshot_download", kwargs))
        return "/models/unpinned"

    monkeypatch.setattr(
        "mindbridge.models.jina.import_module",
        lambda name: (
            SimpleNamespace(SentenceTransformer=sentence_transformer)
            if name == "sentence_transformers"
            else SimpleNamespace(snapshot_download=snapshot_download)
        ),
    )
    monkeypatch.setattr("mindbridge.models.jina.select_torch_device", lambda _device: "cpu")

    JinaEmbedder.load(model_id="someone/else", dimension=2)

    assert calls[0] == ("snapshot_download", {"repo_id": "someone/else", "revision": None})


def test_an_overridden_model_id_alone_does_not_inherit_the_bundled_pin() -> None:
    """One variable pointing the encoder at another repository must not pin it to this one's sha.

    `MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID` and `MINDBRIDGE_MEDIA_EMBEDDER_MODEL_REVISION` are
    documented as two independent optional variables. Defaulting the second onto whatever the
    first names made setting only the first a worker that cannot start.
    """
    named_elsewhere = jina_media_embedder_config(
        {"MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID": "jinaai/jina-embeddings-v4"}
    )
    validated = _EmbedderConfig.model_validate(named_elsewhere)

    assert embedder_revision_for(validated.model_id, validated.model_revision) is None
    assert embedder_revision_for(DEFAULT_EMBEDDER_MODEL_ID, None) == DEFAULT_EMBEDDER_REVISION
    assert embedder_revision_for("jinaai/jina-embeddings-v4", "deadbeef" * 5) == "deadbeef" * 5, (
        "an explicit pin always wins, whichever repository it is for"
    )


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


async def test_jina_uses_distinct_query_and_document_methods() -> None:
    """Retrieval prefixes remain owned by Sentence Transformers."""
    encoder = RecordingEncoder([[1.0, 0.0]])
    embedder = JinaEmbedder(
        encoder,
        ModelReference(model_id="jina"),
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


class GatedEncoder(RecordingEncoder):
    """A document encode that parks inside the worker thread until it is released."""

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
        truncate_dim: int,
    ) -> Matrix:
        self.calls.append("document")
        self.entered.set()
        assert self.release.wait(timeout=10)
        return Matrix(self.values)


async def test_jina_keeps_a_query_lane_free_while_documents_encode() -> None:
    """An interactive recall must not queue behind bulk ingest.

    Reads and writes shared one semaphore, so an empty recall -- no memories retrieved and no
    generation at all -- took 57-71 s under write load against a 12.4 s idle baseline: the query
    embed was waiting for whichever document batch held the slot. The document lane stays
    bounded by `max_concurrency`; only the query lane is separate.
    """
    encoder = GatedEncoder()
    embedder = JinaEmbedder(
        encoder,
        ModelReference(model_id="jina"),
        dimension=2,
        max_concurrency=1,
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
                inputs=(ModelInput((TextPart("Where is the screwdriver?"),)),),
                task=EmbedTask.QUERY,
            )
        ),
        timeout=5,
    )

    assert query.embeddings[0].values == (1.0, 0.0)
    # The document lane is still one at a time: the second batch has not been handed to the
    # encoder while the first holds the only document slot.
    assert encoder.calls.count("document") == 1

    encoder.release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=10)
    assert encoder.calls.count("document") == 2


async def test_jina_rejects_invalid_model_output() -> None:
    """Malformed upstream vectors cannot enter the semantic index."""
    embedder = JinaEmbedder(
        RecordingEncoder([[1.0, 1.0]]),
        ModelReference(model_id="jina"),
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
        ModelReference(model_id="jina"),
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
    return JinaEmbedder.load(dimension=2)


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


def test_jina_refuses_to_start_without_its_audio_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing audio support must fail readiness, not the first production audio request."""

    def import_dependency(name: str) -> object:
        if name == "sentence_transformers":
            return SimpleNamespace(SentenceTransformer=lambda *_a, **_k: OmniEncoder(object()))
        if name == "huggingface_hub":
            return SimpleNamespace(snapshot_download=lambda **_k: "/models/pinned")
        raise ImportError(name)

    monkeypatch.setattr("mindbridge.models.jina.import_module", import_dependency)

    with pytest.raises(ModelUnavailableError, match="cloud-models"):
        JinaEmbedder.load(dimension=2)
