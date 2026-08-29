"""Focused tests for the generic Sentence Transformers embedding boundary."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import mindbridge.models.sentence_transformers as sentence_transformers
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import EmbeddingBackend, EmbedTask, ModelInput
from mindbridge.models.sentence_transformers import SentenceTransformersEmbedder
from mindbridge.types import AssetRef, Modality

REVISION = "a" * 40


class Matrix:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def tolist(self) -> object:
        return self.values


class RecordingEncoder:
    def __init__(
        self,
        values: list[list[float]],
        *,
        supported: tuple[str, ...] = ("text", "image", "video", "message"),
        native_dimension: int = 2,
        matryoshka_dimensions: tuple[int, ...] = (),
    ) -> None:
        self.values = values
        self.supported = supported
        self.native_dimension = native_dimension
        self.config = SimpleNamespace(
            is_matryoshka=bool(matryoshka_dimensions),
            matryoshka_dimensions=matryoshka_dimensions,
        )
        self.calls: list[tuple[str, list[object], int | None]] = []
        self.batch_sizes: list[int] = []
        self.closed = 0

    def supports(self, modality: str | tuple[str, ...]) -> bool:
        requested = (modality,) if isinstance(modality, str) else modality
        return all(value in self.supported for value in requested)

    def get_embedding_dimension(self) -> int:
        return self.native_dimension

    def encode_query(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> Matrix:
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        self.batch_sizes.append(batch_size)
        self.calls.append(("query", sentences, truncate_dim))
        return Matrix(self.values)

    def encode_document(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> Matrix:
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        self.batch_sizes.append(batch_size)
        self.calls.append(("document", sentences, truncate_dim))
        return Matrix(self.values)

    def close(self) -> None:
        self.closed += 1


def test_standard_dict_and_message_inputs_are_batched(tmp_path: Path) -> None:
    encoder = RecordingEncoder([[3.0, 4.0], [0.0, 2.0], [1.0, 0.0]])
    embedder = SentenceTransformersEmbedder(
        encoder,
        model_id="org/omni",
        revision=REVISION,
        dimension=2,
        batch_size=7,
    )
    first = _asset(tmp_path, "first", Modality.IMAGE, "image/png")
    second = _asset(tmp_path, "second", Modality.IMAGE, "image/jpeg")
    video = _asset(tmp_path, "clip", Modality.VIDEO, "video/mp4")

    vectors = embedder.embed(
        (
            ModelInput(text="describe", assets=(first,)),
            ModelInput(assets=(first, second)),
            ModelInput(assets=(video,)),
        )
    )

    assert isinstance(embedder, EmbeddingBackend)
    assert embedder.embedding_capabilities == {
        Modality.TEXT,
        Modality.IMAGE,
        Modality.VIDEO,
    }
    assert vectors[0] == pytest.approx((0.6, 0.8))
    assert vectors[1] == pytest.approx((0.0, 1.0))
    assert vectors[2] == pytest.approx((1.0, 0.0))
    assert encoder.calls == [
        (
            "document",
            [
                {"text": "describe", "image": str(first.path)},
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(first.path)},
                            {"type": "image", "image": str(second.path)},
                        ],
                    }
                ],
                {"video": str(video.path)},
            ],
            None,
        )
    ]
    assert encoder.batch_sizes == [7]


def test_query_and_document_methods_remain_distinct() -> None:
    encoder = RecordingEncoder([[1.0, 0.0]])
    embedder = SentenceTransformersEmbedder(
        encoder,
        model_id="org/text",
        revision=REVISION,
    )
    batch = (ModelInput(text="where"),)

    embedder.embed(batch, EmbedTask.QUERY)
    embedder.embed(batch, EmbedTask.DOCUMENT)

    assert [call[0] for call in encoder.calls] == ["query", "document"]
    with pytest.raises(ValidationError, match="task"):
        embedder.embed(batch, "classification")  # type: ignore[arg-type]


def test_instance_encode_isolated_from_class_level_remote_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DelegatingEncoder(RecordingEncoder):
        def encode(
            self,
            sentences: list[object],
            *,
            batch_size: int,
            convert_to_numpy: bool,
            normalize_embeddings: bool,
            truncate_dim: int | None,
        ) -> Matrix:
            del convert_to_numpy, normalize_embeddings
            self.batch_sizes.append(batch_size)
            self.calls.append(("native", sentences, truncate_dim))
            return Matrix(self.values)

        def encode_query(
            self,
            sentences: list[object],
            *,
            batch_size: int,
            convert_to_numpy: bool,
            normalize_embeddings: bool,
            truncate_dim: int | None,
        ) -> Matrix:
            return self.encode(
                sentences,
                batch_size=batch_size,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
                truncate_dim=truncate_dim,
            )

        encode_document = encode_query

    encoder = DelegatingEncoder([[1.0, 0.0]])
    embedder = SentenceTransformersEmbedder(
        encoder,
        model_id="org/text",
        revision=REVISION,
    )

    def contaminated(*_args: object, **_kwargs: object) -> Matrix:
        raise AssertionError("class-level provider patch leaked into a generic encoder")

    monkeypatch.setattr(DelegatingEncoder, "encode", contaminated)

    assert embedder.embed((ModelInput(text="memory"),)) == ((1.0, 0.0),)
    assert encoder.calls[0][0] == "native"


def test_capabilities_are_checked_before_model_execution(tmp_path: Path) -> None:
    encoder = RecordingEncoder([[1.0, 0.0]], supported=("text", "image"))
    embedder = SentenceTransformersEmbedder(
        encoder,
        model_id="org/image",
        revision=REVISION,
    )
    audio = _asset(tmp_path, "speech", Modality.AUDIO, "audio/wav")

    with pytest.raises(ModelError, match="does not support audio"):
        embedder.embed((ModelInput(assets=(audio,)),))

    assert encoder.calls == []


def test_native_or_advertised_matryoshka_dimensions_define_the_space() -> None:
    native_encoder = RecordingEncoder([[1.0, 0.0, 0.0, 0.0]], native_dimension=4)
    short_encoder = RecordingEncoder(
        [[3.0, 4.0]],
        native_dimension=4,
        matryoshka_dimensions=(2, 4),
    )
    native = SentenceTransformersEmbedder(
        native_encoder,
        model_id="org/text",
        revision=REVISION,
    )
    short = SentenceTransformersEmbedder(
        short_encoder,
        model_id="org/text",
        revision=REVISION,
        dimension=2,
    )
    same = SentenceTransformersEmbedder(
        RecordingEncoder(
            [[3.0, 4.0]],
            native_dimension=4,
            matryoshka_dimensions=(2, 4),
        ),
        model_id="org/text",
        revision=REVISION,
        dimension=2,
    )
    other_revision = SentenceTransformersEmbedder(
        RecordingEncoder(
            [[3.0, 4.0]],
            native_dimension=4,
            matryoshka_dimensions=(2, 4),
        ),
        model_id="org/text",
        revision="b" * 40,
        dimension=2,
    )

    assert native.embedding_dimension == 4
    assert short.embedding_dimension == 2
    assert short.embedding_space == same.embedding_space
    assert native.embedding_space != short.embedding_space
    assert short.embedding_space != other_revision.embedding_space
    assert short.embed((ModelInput(text="document"),))[0] == pytest.approx((0.6, 0.8))
    assert short_encoder.calls[0][2] == 2
    with pytest.raises(ModelError, match="advertised Matryoshka"):
        SentenceTransformersEmbedder(
            RecordingEncoder([[1.0, 0.0]], native_dimension=4),
            model_id="org/text",
            revision=REVISION,
            dimension=2,
        )


def test_injected_encoder_requires_an_immutable_revision_and_positive_batch() -> None:
    with pytest.raises(ValidationError, match="immutable"):
        SentenceTransformersEmbedder(
            RecordingEncoder([[1.0, 0.0]]),
            model_id="org/text",
            revision="main",
        )
    with pytest.raises(ValidationError, match="batch_size"):
        SentenceTransformersEmbedder(
            RecordingEncoder([[1.0, 0.0]]),
            model_id="org/text",
            revision=REVISION,
            batch_size=0,
        )


def test_standard_loader_passes_the_immutable_revision_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    encoder = RecordingEncoder([[1.0, 0.0]])

    def factory(model_path: str, **kwargs: object) -> RecordingEncoder:
        calls.append((model_path, kwargs))
        return encoder

    def importer(name: str) -> object:
        if name == "sentence_transformers":
            return SimpleNamespace(SentenceTransformer=factory)
        raise ImportError(name)

    monkeypatch.setattr(sentence_transformers, "import_module", importer)
    loaded = SentenceTransformersEmbedder.load(
        "org/custom",
        revision=REVISION,
        dimension=2,
        device="cpu",
        batch_size=9,
    )

    assert loaded.embedding_model == "org/custom"
    assert calls == [
        (
            "org/custom",
            {
                "revision": REVISION,
                "trust_remote_code": False,
                "device": "cpu",
            },
        ),
    ]
    loaded.embed((ModelInput(text="memory"),))
    assert encoder.batch_sizes == [9]


def test_optional_dependency_and_mutable_revision_fail_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def missing(name: str) -> object:
        imported.append(name)
        raise ImportError(name)

    monkeypatch.setattr(sentence_transformers, "import_module", missing)
    with pytest.raises(ValidationError, match="immutable"):
        SentenceTransformersEmbedder.load("org/custom", revision="main")
    assert imported == []

    with pytest.raises(ModelError, match="local extra"):
        SentenceTransformersEmbedder.load("org/text", revision=REVISION)
    assert imported == ["sentence_transformers"]


class GatedEncoder(RecordingEncoder):
    def __init__(self) -> None:
        super().__init__([[1.0, 0.0]])
        self.entered = threading.Event()
        self.release = threading.Event()

    def _wait(self, kind: str, sentences: list[object], truncate_dim: int | None) -> Matrix:
        self.calls.append((kind, sentences, truncate_dim))
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test timed out")
        return Matrix(self.values)

    def encode_query(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> Matrix:
        return self._wait("query", sentences, truncate_dim)

    def encode_document(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> Matrix:
        return self._wait("document", sentences, truncate_dim)


def test_one_lock_serializes_query_and_document_calls() -> None:
    encoder = GatedEncoder()
    embedder = SentenceTransformersEmbedder(
        encoder,
        model_id="org/text",
        revision=REVISION,
    )
    batch = (ModelInput(text="memory"),)

    with ThreadPoolExecutor(max_workers=2) as pool:
        document = pool.submit(embedder.embed, batch, EmbedTask.DOCUMENT)
        assert encoder.entered.wait(timeout=5)
        query = pool.submit(embedder.embed, batch, EmbedTask.QUERY)
        assert len(encoder.calls) == 1
        encoder.release.set()
        assert document.result(timeout=5)[0] == (1.0, 0.0)
        assert query.result(timeout=5)[0] == (1.0, 0.0)

    assert sorted(call[0] for call in encoder.calls) == ["document", "query"]


@pytest.mark.parametrize(
    "values",
    (
        [[1.0]],
        [[0.0, 0.0]],
        [[float("nan"), 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
    ),
)
def test_invalid_model_outputs_are_rejected(values: list[list[float]]) -> None:
    embedder = SentenceTransformersEmbedder(
        RecordingEncoder(values),
        model_id="org/text",
        revision=REVISION,
        dimension=2,
    )
    with pytest.raises(ModelError, match="invalid response"):
        embedder.embed((ModelInput(text="memory"),))


def test_close_is_idempotent_and_stops_new_calls() -> None:
    encoder = RecordingEncoder([[1.0, 0.0]])
    embedder = SentenceTransformersEmbedder(
        encoder,
        model_id="org/text",
        revision=REVISION,
    )

    embedder.close()
    embedder.close()

    assert encoder.closed == 1
    with pytest.raises(ModelError, match="closed"):
        embedder.embed((ModelInput(text="memory"),))


def _asset(
    directory: Path,
    asset_id: str,
    modality: Modality,
    media_type: str,
) -> AssetRef:
    return AssetRef(
        id=asset_id,
        modality=modality,
        media_type=media_type,
        size_bytes=1,
        sha256="0" * 64,
        path=directory / asset_id,
    )
