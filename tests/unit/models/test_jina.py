"""Focused tests for the pinned Jina Omni compatibility adapter."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

import mindbridge.models.jina as jina
import mindbridge.models.sentence_transformers as sentence_transformers
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import EmbeddingBackend, EmbedTask, ModelInput
from mindbridge.models.jina import (
    DEFAULT_JINA_MODEL_ID,
    DEFAULT_JINA_REVISION,
    JinaOmniEmbedder,
)
from mindbridge.models.sentence_transformers import SentenceTransformersEmbedder
from mindbridge.types import AssetRef, Modality

UNIT_32 = [1.0, *(0.0 for _ in range(31))]
VECTOR_32 = tuple(UNIT_32)


class Matrix:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def tolist(self) -> object:
        return self.values


class LegacyEncoder:
    def __init__(
        self,
        values: list[list[float]],
        *,
        processor: object | None = object(),
        native_dimension: int = 2,
        matryoshka_dimensions: tuple[int, ...] = (),
    ) -> None:
        self.values = values
        self.module = SimpleNamespace(processor=processor)
        self.native_dimension = native_dimension
        self.config = SimpleNamespace(
            is_matryoshka=bool(matryoshka_dimensions),
            matryoshka_dimensions=matryoshka_dimensions,
        )
        self.calls: list[tuple[str, list[object], int | None]] = []
        self.batch_sizes: list[int] = []

    def __getitem__(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return self.module

    def supports(self, modality: str | tuple[str, ...]) -> bool:
        return modality == "text"

    def get_embedding_dimension(self) -> int:
        return self.native_dimension

    def encode(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> Matrix:
        self.calls.append(("native-encode", sentences, truncate_dim))
        self.batch_sizes.append(batch_size)
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
        self.calls.append(("query", sentences, truncate_dim))
        self.batch_sizes.append(batch_size)
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
        self.calls.append(("document", sentences, truncate_dim))
        self.batch_sizes.append(batch_size)
        return Matrix(self.values)


def test_jina_advertises_legacy_capabilities_and_uses_tuple_inputs(tmp_path: Path) -> None:
    encoder = LegacyEncoder([[3.0, 4.0], [0.0, 2.0]])
    embedder = jina._LoadedJinaOmniEmbedder(
        encoder,
        text_encode=encoder.encode,
        dimension=2,
        batch_size=6,
    )
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png")

    vectors = embedder.embed(
        (
            ModelInput(text="describe", assets=(image,)),
            ModelInput(assets=(image,)),
        ),
        EmbedTask.QUERY,
    )

    assert isinstance(embedder, EmbeddingBackend)
    assert embedder.capabilities == {
        Modality.TEXT,
        Modality.IMAGE,
        Modality.VIDEO,
        Modality.AUDIO,
    }
    assert vectors[0] == pytest.approx((0.6, 0.8))
    assert vectors[1] == pytest.approx((0.0, 1.0))
    method, prepared, truncate_dim = encoder.calls[0]
    assert method == "query"
    assert truncate_dim is None
    fused = prepared[0]
    assert isinstance(fused, tuple)
    assert fused[1] == str(image.path)
    assert prepared[1] == str(image.path)
    text = fused[0]
    assert not isinstance(text, str)
    assert str(text) == "describe"
    assert encoder.batch_sizes == [6]


def test_jina_keeps_url_and_path_shaped_text_out_of_media_autodetection(
    tmp_path: Path,
) -> None:
    local = tmp_path / "existing.png"
    local.write_bytes(b"not an image")
    texts = ("https://127.0.0.1/private.png", str(local))
    encoder = LegacyEncoder([[1.0, 0.0], [1.0, 0.0]])
    embedder = jina._LoadedJinaOmniEmbedder(
        encoder,
        text_encode=encoder.encode,
        dimension=2,
    )

    embedder.embed(tuple(ModelInput(text=text) for text in texts))

    prepared = encoder.calls[0][1]
    assert prepared == [{"text": f"Document: {text}"} for text in texts]


def test_jina_reuses_matryoshka_and_recipe_logic() -> None:
    encoder = LegacyEncoder(
        [[3.0, 4.0]],
        native_dimension=4,
        matryoshka_dimensions=(2, 4),
    )
    jina_embedder = jina._LoadedJinaOmniEmbedder(
        encoder,
        text_encode=encoder.encode,
        dimension=2,
    )
    generic = SentenceTransformersEmbedder(
        LegacyEncoder(
            [[3.0, 4.0]],
            native_dimension=4,
            matryoshka_dimensions=(2, 4),
        ),
        model_id=DEFAULT_JINA_MODEL_ID,
        revision=DEFAULT_JINA_REVISION,
        dimension=2,
    )

    assert jina_embedder.embed((ModelInput(text="memory"),))[0] == pytest.approx((0.6, 0.8))
    assert encoder.calls[0][2] == 2
    assert jina_embedder.space_id != generic.space_id


def test_public_default_has_fixed_identity_without_loading(
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
    assert backend.model_id == DEFAULT_JINA_MODEL_ID
    assert backend.dimension == 32
    assert backend.capabilities == {
        Modality.TEXT,
        Modality.IMAGE,
        Modality.VIDEO,
        Modality.AUDIO,
    }
    assert (
        backend.space_id
        == JinaOmniEmbedder(
            dimension=32,
            device="cuda",
            batch_size=1,
        ).space_id
    )
    assert backend.embed(()) == ()
    backend.close()
    backend.close()
    assert loads == []
    with pytest.raises(ModelError, match="closed"):
        backend.embed((ModelInput(text="memory"),))


def test_first_public_embed_loads_once_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jina, "find_spec", lambda _name: SimpleNamespace())
    entered = threading.Event()
    release = threading.Event()
    calls: list[dict[str, object]] = []
    encoder = LegacyEncoder([UNIT_32], native_dimension=32)
    loaded = jina._LoadedJinaOmniEmbedder(
        encoder,
        text_encode=encoder.encode,
        dimension=32,
        batch_size=11,
    )

    def load(**kwargs: object) -> jina._LoadedJinaOmniEmbedder:
        calls.append(kwargs)
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test timed out")
        return loaded

    monkeypatch.setattr(jina, "_load_jina", load)
    backend = JinaOmniEmbedder(dimension=32, device="cpu", batch_size=11)
    request = (ModelInput(text="memory"),)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(backend.embed, request)
        assert entered.wait(timeout=5)
        second = pool.submit(backend.embed, request)
        release.set()
        assert first.result(timeout=5) == (VECTOR_32,)
        assert second.result(timeout=5) == (VECTOR_32,)

    assert calls == [{"dimension": 32, "device": "cpu", "batch_size": 11}]
    assert encoder.batch_sizes == [11, 11]


class PatchingSentenceTransformer(LegacyEncoder):
    patch_on_init: ClassVar[bool] = True
    instances: ClassVar[list[PatchingSentenceTransformer]] = []
    load_calls: ClassVar[list[tuple[str, dict[str, object]]]] = []
    patch_entered: ClassVar[threading.Event | None] = None
    patch_release: ClassVar[threading.Event | None] = None
    method_lookup: ClassVar[threading.Event | None] = None

    def __init__(self, model_path: str, **kwargs: object) -> None:
        super().__init__([UNIT_32], native_dimension=32)
        self.instances.append(self)
        self.load_calls.append((model_path, kwargs))
        if self.patch_on_init:
            cls = type(self)
            for name, method in {
                "encode": _jina_encode,
                "encode_query": _jina_query,
                "encode_document": _jina_document,
            }.items():
                setattr(cls, name, method)
            if cls.patch_entered is not None:
                cls.patch_entered.set()
            if cls.patch_release is not None and not cls.patch_release.wait(timeout=5):
                raise RuntimeError("test timed out")

    def __getattribute__(self, name: str) -> object:
        if name in {"encode", "encode_query", "encode_document"}:
            event = type(self).method_lookup
            if event is not None:
                event.set()
        return super().__getattribute__(name)

    def encode(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> Matrix:
        self.calls.append(("native-encode", sentences, truncate_dim))
        self.batch_sizes.append(batch_size)
        return Matrix(self.values)


def _jina_encode(
    encoder: PatchingSentenceTransformer,
    sentences: list[object],
    **_kwargs: object,
) -> Matrix:
    encoder.calls.append(("jina-encode", sentences, None))
    return Matrix(encoder.values)


def _jina_query(
    encoder: PatchingSentenceTransformer,
    sentences: list[object],
    **kwargs: object,
) -> Matrix:
    truncate_dim = kwargs.get("truncate_dim")
    batch_size = kwargs.get("batch_size")
    assert truncate_dim is None or isinstance(truncate_dim, int)
    assert isinstance(batch_size, int)
    encoder.calls.append(("jina-query", sentences, truncate_dim))
    encoder.batch_sizes.append(batch_size)
    return Matrix(encoder.values)


def _jina_document(
    encoder: PatchingSentenceTransformer,
    sentences: list[object],
    **kwargs: object,
) -> Matrix:
    truncate_dim = kwargs.get("truncate_dim")
    batch_size = kwargs.get("batch_size")
    assert truncate_dim is None or isinstance(truncate_dim, int)
    assert isinstance(batch_size, int)
    encoder.calls.append(("jina-document", sentences, truncate_dim))
    encoder.batch_sizes.append(batch_size)
    return Matrix(encoder.values)


def test_default_load_pins_and_isolates_jina_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    module = SimpleNamespace(SentenceTransformer=PatchingSentenceTransformer)
    original = {
        name: getattr(PatchingSentenceTransformer, name)
        for name in ("encode", "encode_query", "encode_document")
    }
    monkeypatch.setattr(PatchingSentenceTransformer, "patch_on_init", True)
    monkeypatch.setattr(PatchingSentenceTransformer, "instances", [])
    monkeypatch.setattr(PatchingSentenceTransformer, "load_calls", [])
    monkeypatch.setattr(jina, "_jina_methods", None)
    monkeypatch.setattr(jina, "find_spec", lambda _name: SimpleNamespace())

    def download(**kwargs: object) -> str:
        calls.append(("snapshot_download", kwargs))
        return "/models/jina-pinned"

    def importer(name: str) -> object:
        if name == "sentence_transformers":
            return module
        if name == "huggingface_hub":
            return SimpleNamespace(snapshot_download=download)
        if name == "librosa":
            return SimpleNamespace()
        raise ImportError(name)

    monkeypatch.setattr(jina, "import_module", importer)
    monkeypatch.setattr(sentence_transformers, "import_module", importer)

    loaded = JinaOmniEmbedder.load(dimension=32, device="cpu", batch_size=5)

    assert loaded.model_id == DEFAULT_JINA_MODEL_ID
    assert DEFAULT_JINA_REVISION == "1f9ba7a04283c80cecdfe7a98f9d9c6f09796ffb"
    assert calls == [
        (
            "snapshot_download",
            {"repo_id": DEFAULT_JINA_MODEL_ID, "revision": DEFAULT_JINA_REVISION},
        ),
    ]
    assert PatchingSentenceTransformer.load_calls == [
        (
            "/models/jina-pinned",
            {
                "revision": DEFAULT_JINA_REVISION,
                "trust_remote_code": True,
                "device": "cpu",
                "model_kwargs": {"code_revision": DEFAULT_JINA_REVISION},
                "config_kwargs": {"code_revision": DEFAULT_JINA_REVISION},
            },
        )
    ]
    assert all(
        getattr(PatchingSentenceTransformer, name) is method for name, method in original.items()
    )
    assert loaded.embed((ModelInput(text="jina"),), EmbedTask.QUERY) == (VECTOR_32,)
    assert PatchingSentenceTransformer.instances[0].calls[-1] == (
        "native-encode",
        [{"text": "Query: jina"}],
        None,
    )
    assert PatchingSentenceTransformer.instances[0].batch_sizes == [5]

    monkeypatch.setattr(PatchingSentenceTransformer, "patch_on_init", False)
    generic = SentenceTransformersEmbedder.load(
        "org/generic",
        revision="b" * 40,
        dimension=32,
    )
    assert generic.embed((ModelInput(text="generic"),)) == (VECTOR_32,)
    assert PatchingSentenceTransformer.instances[1].calls[-1][0] == "document"


def test_jina_load_cannot_leak_into_a_concurrent_generic_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace(SentenceTransformer=PatchingSentenceTransformer)
    monkeypatch.setattr(PatchingSentenceTransformer, "patch_on_init", False)
    generic_encoder = PatchingSentenceTransformer("generic")
    generic = SentenceTransformersEmbedder(
        generic_encoder,
        model_id="org/generic",
        revision="c" * 40,
        dimension=32,
    )
    generic_encoder.calls.clear()
    entered = threading.Event()
    release = threading.Event()
    lookup = threading.Event()
    started = threading.Event()
    monkeypatch.setattr(PatchingSentenceTransformer, "patch_on_init", True)
    monkeypatch.setattr(PatchingSentenceTransformer, "patch_entered", entered)
    monkeypatch.setattr(PatchingSentenceTransformer, "patch_release", release)
    monkeypatch.setattr(PatchingSentenceTransformer, "method_lookup", lookup)
    monkeypatch.setattr(jina, "_jina_methods", None)
    monkeypatch.setattr(jina, "find_spec", lambda _name: SimpleNamespace())

    def download(**_kwargs: object) -> str:
        return "/models/jina-pinned"

    def importer(name: str) -> object:
        if name == "sentence_transformers":
            return module
        if name == "huggingface_hub":
            return SimpleNamespace(snapshot_download=download)
        if name == "librosa":
            return SimpleNamespace()
        raise ImportError(name)

    def generic_embed() -> tuple[tuple[float, ...], ...]:
        started.set()
        return generic.embed((ModelInput(text="generic"),))

    monkeypatch.setattr(jina, "import_module", importer)
    with ThreadPoolExecutor(max_workers=2) as pool:
        loading = pool.submit(JinaOmniEmbedder.load, dimension=32)
        assert entered.wait(timeout=5)
        embedding = pool.submit(generic_embed)
        assert started.wait(timeout=5)
        assert not lookup.wait(timeout=0.1)
        try:
            assert embedding.result(timeout=1) == (VECTOR_32,)
        finally:
            release.set()
        assert loading.result(timeout=5).model_id == DEFAULT_JINA_MODEL_ID

    assert generic_encoder.calls[-1][0] == "document"


def test_jina_fails_readiness_without_processor_or_audio_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = LegacyEncoder([[1.0, 0.0]], processor=None)
    with pytest.raises(ModelError, match="media processor"):
        jina._LoadedJinaOmniEmbedder(
            encoder,
            text_encode=encoder.encode,
            dimension=2,
        )

    def missing(_name: str) -> object:
        raise ImportError

    monkeypatch.setattr(jina, "find_spec", lambda _name: SimpleNamespace())
    monkeypatch.setattr(jina, "import_module", missing)
    with pytest.raises(ValidationError, match="batch_size"):
        JinaOmniEmbedder.load(dimension=32, batch_size=0)
    with pytest.raises(ModelError, match="local extra"):
        JinaOmniEmbedder.load(dimension=32)


def test_public_default_preflights_the_local_extra_without_importing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def imported(name: str) -> object:
        imports.append(name)
        raise AssertionError("preflight imported an optional dependency")

    monkeypatch.setattr(jina, "find_spec", lambda _name: None)
    monkeypatch.setattr(jina, "import_module", imported)

    with pytest.raises(ModelError, match="local extra"):
        JinaOmniEmbedder()
    assert imports == []

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
