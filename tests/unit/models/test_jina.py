"""Focused tests for the pinned official Jina Omni adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mindbridge.models.jina as jina
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import EmbeddingBackend, EmbedTask, ModelInput
from mindbridge.models.jina import (
    DEFAULT_JINA_MODEL_ID,
    DEFAULT_JINA_REVISION,
    JinaOmniEmbedder,
)
from mindbridge.types import AssetRef, Modality


class _Matrix:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def tolist(self) -> object:
        return self.values


class _Encoder:
    def __init__(self, *, dimension: int = 2, processor: object | None = object()) -> None:
        self.dimension = dimension
        self.module = SimpleNamespace(processor=processor)
        self.config = SimpleNamespace(is_matryoshka=True, matryoshka_dimensions=(2, 32))
        self.calls: list[tuple[str, list[object], int | None, int]] = []

    def __getitem__(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return self.module

    def supports(self, _modality: str | tuple[str, ...]) -> bool:
        return True

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _Matrix:
        del convert_to_numpy, normalize_embeddings
        self.calls.append(("encode", sentences, truncate_dim, batch_size))
        return _Matrix([[3.0, 4.0] for _ in sentences])

    def encode_query(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _Matrix:
        del convert_to_numpy, normalize_embeddings
        self.calls.append(("query", sentences, truncate_dim, batch_size))
        return _Matrix([[3.0, 4.0] for _ in sentences])

    def encode_document(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _Matrix:
        del convert_to_numpy, normalize_embeddings
        self.calls.append(("document", sentences, truncate_dim, batch_size))
        return _Matrix([[3.0, 4.0] for _ in sentences])


def test_loaded_jina_uses_official_query_document_methods_and_native_parts(
    tmp_path: Path,
) -> None:
    encoder = _Encoder()
    embedder = jina._LoadedJinaOmniEmbedder(encoder, dimension=2, batch_size=6)
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png")

    vectors = embedder.embed(
        (
            ModelInput(text="describe", assets=(image,)),
            ModelInput(assets=(image,)),
        ),
        EmbedTask.QUERY,
    )

    assert isinstance(embedder, EmbeddingBackend)
    assert embedder.embedding_capabilities == {
        Modality.TEXT,
        Modality.IMAGE,
        Modality.VIDEO,
        Modality.AUDIO,
    }
    assert vectors[0] == pytest.approx((0.6, 0.8))
    assert vectors[1] == pytest.approx((0.6, 0.8))
    method, prepared, truncate_dim, batch_size = encoder.calls[0]
    assert (method, truncate_dim, batch_size) == ("query", None, 6)
    fused = prepared[0]
    assert isinstance(fused, tuple)
    assert str(fused[0]) == "describe"
    assert fused[1] == str(image.path)
    assert prepared[1] == str(image.path)


def test_application_text_cannot_trigger_jina_url_or_path_autodetection(tmp_path: Path) -> None:
    local = tmp_path / "existing.png"
    local.write_bytes(b"not an image")
    texts = ("https://127.0.0.1/private.png", str(local))
    encoder = _Encoder()
    embedder = jina._LoadedJinaOmniEmbedder(encoder, dimension=2)

    embedder.embed(tuple(ModelInput(text=text) for text in texts))

    method, prepared, _truncate_dim, _batch_size = encoder.calls[0]
    assert method == "document"
    assert all(not isinstance(value, str) for value in prepared)
    assert tuple(map(str, prepared)) == texts


def test_public_adapter_has_fixed_identity_and_does_not_load_until_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[dict[str, object]] = []
    monkeypatch.setattr(jina, "find_spec", lambda _name: SimpleNamespace())

    def unexpected_load(**kwargs: object) -> jina._LoadedJinaOmniEmbedder:
        loads.append(kwargs)
        raise AssertionError("empty embed or close loaded weights")

    monkeypatch.setattr(jina, "_load_jina", unexpected_load)
    backend = JinaOmniEmbedder(dimension=32, device="cpu", batch_size=7)

    assert isinstance(backend, EmbeddingBackend)
    assert backend.embedding_model == DEFAULT_JINA_MODEL_ID
    assert backend.embedding_dimension == 32
    assert backend.embed(()) == ()
    backend.close()
    assert loads == []
    with pytest.raises(ModelError, match="closed"):
        backend.embed((ModelInput(text="memory"),))


def test_loader_pins_revision_and_isolates_provider_class_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    instances: list[_Encoder] = []
    unit_vector = [1.0, *(0.0 for _ in range(31))]

    def provider_encode(
        encoder: _Encoder,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _Matrix:
        del convert_to_numpy, normalize_embeddings
        encoder.calls.append(("provider-encode", sentences, truncate_dim, batch_size))
        return _Matrix([unit_vector for _ in sentences])

    def provider_query(
        encoder: _Encoder,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _Matrix:
        encoder.calls.append(("provider-query", sentences, truncate_dim, batch_size))
        return encoder.encode(
            sentences,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            truncate_dim=truncate_dim,
        )

    def provider_document(
        encoder: _Encoder,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _Matrix:
        encoder.calls.append(("provider-document", sentences, truncate_dim, batch_size))
        return encoder.encode(
            sentences,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            truncate_dim=truncate_dim,
        )

    class PatchingEncoder(_Encoder):
        def __init__(self, model: str, **kwargs: object) -> None:
            calls.append((model, kwargs))
            super().__init__(dimension=32)
            instances.append(self)
            for name, method in (
                ("encode", provider_encode),
                ("encode_query", provider_query),
                ("encode_document", provider_document),
            ):
                setattr(type(self), name, method)

    original = {
        name: getattr(PatchingEncoder, name)
        for name in ("encode", "encode_query", "encode_document")
    }

    def importer(name: str) -> object:
        if name == "sentence_transformers":
            return SimpleNamespace(SentenceTransformer=PatchingEncoder)
        if name == "librosa":
            return SimpleNamespace()
        raise ImportError(name)

    monkeypatch.setattr(jina, "find_spec", lambda _name: SimpleNamespace())
    monkeypatch.setattr(jina, "import_module", importer)
    monkeypatch.setattr(jina, "_jina_methods", None)
    backend = JinaOmniEmbedder.load(dimension=32, device="cpu", batch_size=5)

    assert DEFAULT_JINA_REVISION == "e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4"
    assert calls == [
        (
            DEFAULT_JINA_MODEL_ID,
            {
                "revision": DEFAULT_JINA_REVISION,
                "trust_remote_code": True,
                "device": "cpu",
                "model_kwargs": {"code_revision": DEFAULT_JINA_REVISION},
                "config_kwargs": {"code_revision": DEFAULT_JINA_REVISION},
            },
        )
    ]
    assert all(getattr(PatchingEncoder, name) is method for name, method in original.items())
    assert backend.embedding_model == DEFAULT_JINA_MODEL_ID
    assert backend.embed((ModelInput(text="memory"),), EmbedTask.QUERY)[0] == tuple(unit_vector)
    assert [call[0] for call in instances[0].calls] == ["provider-query", "provider-encode"]


def test_jina_readiness_validation_stays_at_the_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ModelError, match="media processor"):
        jina._LoadedJinaOmniEmbedder(_Encoder(processor=None), dimension=2)

    monkeypatch.setattr(jina, "find_spec", lambda _name: None)
    with pytest.raises(ModelError, match="local extra"):
        JinaOmniEmbedder()
    with pytest.raises(ValidationError, match="one of"):
        JinaOmniEmbedder(dimension=2)


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
