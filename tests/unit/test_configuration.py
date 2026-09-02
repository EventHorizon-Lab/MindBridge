from __future__ import annotations

import inspect
import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import openai
import pytest
from _feature_support import TinyEmbedder
from pydantic import ValidationError as PydanticValidationError

import mindbridge.configuration as configuration
import mindbridge.recipes as recipes_module
from mindbridge import (
    AnswerResult,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryConfig,
    MemoryKind,
    MemoryPlugins,
    MemorySettings,
    MindBridgeConfig,
    Modality,
    ObservationContext,
    SearchHit,
)
from mindbridge.configuration import resolve_memory_config
from mindbridge.exceptions import ValidationError
from mindbridge.models.base import FormationBackend, GenerationBackend, ModelInput
from mindbridge.types import IndexQuantization


def test_declarative_config_is_typed_strict_and_keeps_local_policy_separate(
    tmp_path: Path,
) -> None:
    config = MindBridgeConfig.model_validate(
        {
            "data_dir": tmp_path,
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimension": 768,
            },
            "generation": {
                "provider": "openai",
                "model": "gpt-5-mini",
                "temperature": 0.1,
            },
            "speech": {"provider": "funasr", "device": "cuda:1"},
            "face": {
                "provider": "opencv",
                "detector_model": tmp_path / "yunet.onnx",
                "recognizer_model": tmp_path / "sface.onnx",
            },
            "settings": {"index_speech": True, "index_quantization": "fp16"},
        }
    )

    assert isinstance(config.embedding, configuration.OpenAIEmbeddingConfig)
    assert isinstance(config.generation, configuration.OpenAIGenerationConfig)
    assert isinstance(config.speech, configuration.FunASRSpeechConfig)
    assert isinstance(config.face, configuration.OpenCVFaceConfig)
    assert config.settings == MemoryConfig(
        index_speech=True,
        index_quantization=IndexQuantization.FP16,
    )
    assert MemorySettings is MemoryConfig

    with pytest.raises(PydanticValidationError, match="extra_forbidden"):
        MindBridgeConfig.model_validate(
            {
                "embedding": {"provider": "openai", "unexpected": True},
            }
        )


@pytest.mark.parametrize(
    "config",
    (
        {"embedding": {"provider": "openai", "dimension": True}},
        {
            "embedding": {"provider": "openai"},
            "generation": {"provider": "openai", "temperature": True},
        },
        {
            "embedding": {"provider": "openai"},
            "settings": {"index_speech": "false"},
        },
    ),
)
def test_declarative_config_does_not_coerce_boolean_or_numeric_fields(
    config: dict[str, object],
) -> None:
    with pytest.raises(PydanticValidationError):
        MindBridgeConfig.model_validate(config)


def test_declarative_memory_settings_validate_every_range() -> None:
    invalid = {
        "minimum_relevance": 2,
        "ambiguity_margin": -1,
        "decay_half_life_days": 0,
        "speaker_similarity": 2,
        "speaker_margin": -1,
        "face_similarity": 2,
        "face_margin": -1,
    }

    with pytest.raises(PydanticValidationError) as failure:
        MindBridgeConfig.model_validate(
            {
                "embedding": {"provider": "openai"},
                "settings": invalid,
            }
        )

    assert {tuple(issue["loc"]) for issue in failure.value.errors()} == {
        ("settings", field) for field in invalid
    }


def test_local_policy_defaults_stay_pinned_to_their_recorded_provenance() -> None:
    # Every default here is public contract, and two of them are calibration constants whose
    # provenance is recorded next to the field in `plugins.py`. Asserting the whole value means a
    # silent change to any single default fails here rather than only in a benchmark months later.
    defaults = MemoryConfig()

    assert defaults == MemoryConfig(
        index_speech=True,
        index_quantization=IndexQuantization.NONE,
        minimum_relevance=0.10,
        ambiguity_margin=0.01,
        evidence_budget_chars=None,
        decay_half_life_days=None,
        reinforce_on_answer=True,
        speaker_similarity=0.78,
        speaker_margin=0.05,
        face_similarity=0.363,
        face_margin=0.05,
        identity_link_min_assets=2,
    )
    # `0.10` gates the same relevance `SearchHit.score` reports and reproduces the effective floor
    # of the pre-rescale `0.55`. A future reader tidying it to a rounder number would silently
    # tighten retrieval, so pin it separately from the whole-value assertion above.
    assert defaults.minimum_relevance == 0.10
    # `face_similarity` is upstream SFace's `_threshold_cosine` verbatim. `speaker_similarity` has
    # no upstream provenance and is deliberately NOT the pinned CAM++ recipe's published
    # `yesOrno_thr` of 0.31: that number is calibrated for a single embedding pair, while the
    # matcher accepts on a `max` over up to 20 exemplars, where it is a lower bound rather than an
    # operating point. Erring high fragments identities; erring low merges two people.
    assert defaults.face_similarity == 0.363
    assert defaults.speaker_similarity == 0.78
    assert defaults.speaker_similarity > 0.31
    assert defaults.speaker_margin > 0
    # The kernel and the declarative path must read the same defaults; a divergence would make
    # `Memory(...)` and `Memory.from_config({})` two different products.
    assert (
        MindBridgeConfig.model_validate({"embedding": {"provider": "openai"}}).settings
        == MemoryConfig()
    )
    assert inspect.signature(Memory.__init__).parameters["index_speech"].default is True
    assert inspect.signature(Memory.__init__).parameters["minimum_relevance"].default == 0.10


def test_a_former_is_declaratively_reachable_but_never_implicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, object]] = []

    # Stands in for `OpenAIModels`, which is one object implementing every completion role.
    class _Models(TinyEmbedder):
        generation_capabilities = frozenset({Modality.TEXT})
        formation_capabilities = frozenset({Modality.TEXT})
        formation_model = "gpt-5-mini"
        formation_space = "gpt-5-mini:mindbridge-formation-v1:test"

        def answer(self, question: ModelInput, hits: Sequence[SearchHit]) -> AnswerResult:
            raise AssertionError("composition must not call the model")

        def form(
            self,
            inputs: Sequence[FormationInput],
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(() for _value in inputs)

    def build(**values: object) -> object:
        built.append(values)
        return _Models()

    monkeypatch.setattr(recipes_module, "_owned_openai_models", build)
    base = {"data_dir": tmp_path, "embedding": {"provider": "openai"}}

    # Configuring generation must not enable formation: a former is a model call per observation on
    # the write path, so it stays opt-in even when a bundled generation backend is already present.
    assert MindBridgeConfig.model_validate(base).formation is None
    with_generation = configuration.resolve_memory_config(
        {**base, "generation": {"provider": "openai"}}
    )
    try:
        assert with_generation.plugins.answerer is not None
        assert with_generation.plugins.former is None
    finally:
        with_generation.close()

    # Asking for one reaches `MemoryPlugins.former`, through its own build, with its own knobs.
    built.clear()
    composition = configuration.resolve_memory_config(
        {**base, "formation": {"provider": "openai", "model": "gpt-5", "temperature": 0.0}}
    )
    try:
        assert composition.plugins.former is not None
        assert composition.plugins.answerer is None
    finally:
        composition.close()

    # The former is built through its own call with its own knobs, not aliased off the embedder.
    assert len(built) == 2
    assert built[-1] == {
        "generation_model": "gpt-5",
        "generation_capabilities": frozenset({Modality.TEXT}),
        "generation_temperature": 0.0,
    }
    # `video_limit` shapes an answer, not a formation proposal, so it is not part of this slot.
    with pytest.raises(PydanticValidationError, match="extra_forbidden"):
        MindBridgeConfig.model_validate(
            {**base, "formation": {"provider": "openai", "video_limit": 4}}
        )
    # `vision_describer` has no bundled implementation, so it deliberately has no key at all.
    with pytest.raises(PydanticValidationError, match="extra_forbidden"):
        MindBridgeConfig.model_validate({**base, "vision_describer": {"provider": "openai"}})


# One reply carrying the two capabilities the audit said only a former can supply: a bitemporal
# trait with a validity window, and an affect proposal with valence and arousal.
_LOCAL_FORMATION_REPLY = json.dumps(
    {
        "items": [
            {
                "observation_id": "observation_0",
                "proposals": [
                    {
                        "kind": "trait",
                        "content": "Ada drinks tea in the morning.",
                        "subject": "Ada",
                        "predicate": "prefers",
                        "value": "tea",
                        "confidence": 0.8,
                        "valid_from": "2026-09-01T08:00:00+00:00",
                        "valid_until": "2026-09-02T08:00:00+00:00",
                    },
                    {
                        "kind": "affect",
                        "content": "Ada sounded relieved.",
                        "subject": "Ada",
                        "value": "relief",
                        "confidence": 0.6,
                        "cue_modality": "text",
                        "valence": 0.5,
                        "arousal": 0.2,
                    },
                ],
            }
        ]
    }
)


@contextmanager
def _local_openai_server(reply: str) -> Iterator[tuple[str, list[dict[str, object]]]]:
    """Serve the slice of chat completions a former needs, over a real localhost socket.

    A local llama.cpp, vLLM, or Ollama endpoint is reached through `base_url` and nothing else, so
    a stub that replaced the SDK client would prove only that the configuration field parses. This
    is the smallest thing that proves the wire path: stdlib server, no network, no model.
    """
    received: list[dict[str, object]] = []

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            received.append({"path": self.path, "request": request})
            body = json.dumps(
                {
                    "id": "local-completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": reply},
                        }
                    ],
                }
            ).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", received
    finally:
        server.shutdown()
        worker.join(timeout=30)
        server.server_close()


def test_a_local_openai_compatible_server_fills_the_former_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The bundled former inherits `base_url` from the shared OpenAI connection fields, so an
    # OpenAI-compatible local server already satisfies `FormationBackend` with no new adapter and
    # no new dependency. Two servers, because `generation` and `formation` are separate slots over
    # separate clients: the write-path call must land on the former's endpoint only.
    monkeypatch.setenv("OPENAI_API_KEY", "a-local-server-ignores-this")
    with (
        _local_openai_server(_LOCAL_FORMATION_REPLY) as (answer_url, answer_requests),
        _local_openai_server(_LOCAL_FORMATION_REPLY) as (former_url, former_requests),
    ):
        composition = configuration.resolve_memory_config(
            {
                "data_dir": tmp_path,
                "embedding": {"provider": "openai", "base_url": answer_url},
                "generation": {"provider": "openai", "base_url": answer_url},
                "formation": {
                    "provider": "openai",
                    "model": "qwen3-8b-local",
                    "base_url": former_url,
                    "temperature": 0.0,
                    "max_retries": 0,
                },
            }
        )
        try:
            former = composition.plugins.former
            assert former is not None
            assert former.formation_model == "qwen3-8b-local"
            proposals = former.form(
                [
                    FormationInput(
                        memory_id="observation-1",
                        content=ModelInput(text="Ada said she prefers tea, and sounded relieved."),
                        context=ObservationContext(),
                    )
                ]
            )
        finally:
            composition.close()

    assert [request["path"] for request in former_requests] == ["/v1/chat/completions"]
    assert answer_requests == []
    # What the local server must implement, pinned so the documented requirement stays true.
    sent = cast(dict[str, object], former_requests[0]["request"])
    assert sent["model"] == "qwen3-8b-local"
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["temperature"] == 0.0

    assert len(proposals) == 1
    trait, affect = proposals[0]
    assert (trait.kind, trait.subject, trait.predicate, trait.value) == (
        MemoryKind.TRAIT,
        "Ada",
        "prefers",
        "tea",
    )
    assert trait.basis is EvidenceBasis.MODEL_INFERENCE
    assert trait.valid_from == datetime(2026, 9, 1, 8, tzinfo=timezone.utc)
    assert trait.valid_until == datetime(2026, 9, 2, 8, tzinfo=timezone.utc)
    assert (affect.kind, affect.valence, affect.arousal, affect.cue_modality) == (
        MemoryKind.AFFECT,
        0.5,
        0.2,
        Modality.TEXT,
    )


def test_a_local_former_still_needs_a_credential_in_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Declarative configuration deliberately has no credential field, so the official SDK's own
    # environment lookup applies even to an endpoint that authenticates nobody: a local deployment
    # must still export a placeholder key. That cost is documented in docs/configuration.md rather
    # than worked around here, because a credential inside configuration data is the worse trade.
    for name in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(openai.OpenAIError):
        configuration.resolve_memory_config(
            {
                "data_dir": tmp_path,
                "embedding": {"provider": "openai", "base_url": "http://127.0.0.1:1/v1"},
                "formation": {"provider": "openai", "base_url": "http://127.0.0.1:1/v1"},
            }
        )


def test_declarative_error_paths_match_the_input_shape() -> None:
    with pytest.raises(ValidationError) as failure:
        configuration._validated_config({"embedding": {"provider": "openai", "dimension": 0}})

    assert str(failure.value).startswith("config.embedding.dimension:")


def test_builtin_openai_generation_config_maps_friendly_names_to_the_sdk_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    marker = cast(GenerationBackend, object())

    def build(**values: object) -> GenerationBackend:
        captured.update(values)
        return marker

    monkeypatch.setattr(recipes_module, "_owned_openai_models", build)
    spec = configuration.OpenAIGenerationConfig(
        provider="openai",
        model="gpt-5-mini",
        modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        temperature=0.1,
        max_tokens=512,
        base_url="http://localhost:11434/v1",
        timeout=20,
        max_retries=1,
    )

    assert configuration._build_generation(spec) is marker
    assert captured == {
        "base_url": "http://localhost:11434/v1",
        "timeout": 20.0,
        "max_retries": 1,
        "generation_model": "gpt-5-mini",
        "generation_capabilities": frozenset({Modality.TEXT, Modality.IMAGE}),
        "generation_temperature": 0.1,
        "generation_max_tokens": 512,
    }


def test_builtin_openai_embedding_config_maps_friendly_names_to_the_sdk_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    marker = object()

    def build(**values: object) -> object:
        captured.update(values)
        return marker

    monkeypatch.setattr(recipes_module, "_owned_openai_models", build)
    spec = configuration.OpenAIEmbeddingConfig(
        provider="openai",
        model="tencent/WeMM-Embedding-2B",
        dimension=2048,
        modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
        request_format="messages",
        base_url="http://localhost:8000/v1",
        timeout=600,
    )

    assert configuration._build_embedding(spec) is marker
    assert captured == {
        "base_url": "http://localhost:8000/v1",
        "timeout": 600.0,
        "embedding_model": "tencent/WeMM-Embedding-2B",
        "embedding_dimension": 2048,
        "embedding_capabilities": frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
        "embedding_request_format": "messages",
    }


def test_declarative_openai_embedding_defaults_stay_text_and_input_shaped() -> None:
    spec = configuration.OpenAIEmbeddingConfig(provider="openai")

    assert spec.modalities == frozenset({Modality.TEXT})
    assert spec.request_format == "input"

    with pytest.raises(PydanticValidationError, match="request_format"):
        configuration.OpenAIEmbeddingConfig.model_validate(
            {"provider": "openai", "request_format": "chat"}
        )


def test_declarative_openai_embedding_rejects_a_model_that_declares_no_modality() -> None:
    # An empty set builds a Memory whose every write fails with "does not support: text". The
    # value is out of range, so it is refused here rather than at the first add.
    with pytest.raises(PydanticValidationError, match="too_short"):
        MindBridgeConfig.model_validate(
            {
                "embedding": {
                    "provider": "openai",
                    "model": "m",
                    "dimension": 8,
                    "modalities": [],
                }
            }
        )


def test_builtin_openai_formation_config_maps_friendly_names_to_the_sdk_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    marker = cast(FormationBackend, object())

    def build(**values: object) -> FormationBackend:
        captured.update(values)
        return marker

    monkeypatch.setattr(recipes_module, "_owned_openai_models", build)
    spec = configuration.OpenAIFormationConfig(
        provider="openai",
        model="gpt-5-mini",
        modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        temperature=0.0,
        seed=7,
        max_tokens=2048,
        base_url="http://localhost:11434/v1",
    )

    assert configuration._build_formation(spec) is marker
    # The bundled adapter derives its formation contract from the generation controls, and
    # `video_limit` is answer-only, so it must not appear here.
    assert captured == {
        "base_url": "http://localhost:11434/v1",
        "generation_model": "gpt-5-mini",
        "generation_capabilities": frozenset({Modality.TEXT, Modality.IMAGE}),
        "generation_temperature": 0.0,
        "generation_seed": 7,
        "generation_max_tokens": 2048,
    }


class _FormingEmbedder(TinyEmbedder):
    """One bundled-adapter stand-in covering every slot the openai provider can fill."""

    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "tiny-former"
    formation_space = "tiny-former:v1"
    generation_capabilities = frozenset({Modality.TEXT})

    def answer(self, question: ModelInput, hits: Sequence[SearchHit]) -> AnswerResult:
        raise AssertionError("not called")

    def __init__(self) -> None:
        self.formed: list[str] = []

    def form(
        self,
        inputs: Sequence[FormationInput],
    ) -> tuple[tuple[FormationProposal, ...], ...]:
        self.formed.extend(value.memory_id for value in inputs)
        return tuple(
            (
                FormationProposal(
                    kind=MemoryKind.ENTITY,
                    content=f"Berlin is a city mentioned by {value.memory_id}.",
                    subject="Berlin",
                    predicate="is_a",
                    value="city",
                    confidence=0.9,
                ),
            )
            for value in inputs
        )


def _formation_stub(monkeypatch: pytest.MonkeyPatch) -> _FormingEmbedder:
    backend = _FormingEmbedder()

    def build(**_values: object) -> _FormingEmbedder:
        return backend

    monkeypatch.setattr(recipes_module, "_owned_openai_models", build)
    return backend


def test_declarative_formation_slot_reaches_the_write_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`former` had no declarative provider, so every `from_config` deployment and every
    benchmark run produced zero derived memories. Closing the slot is only worth anything if a
    JSON config actually installs the backend and `add` actually calls it."""
    backend = _formation_stub(monkeypatch)

    with Memory.from_config(
        {
            "data_dir": tmp_path,
            "embedding": {"provider": "openai", "model": "tiny-test", "dimension": 4},
            "formation": {
                "provider": "openai",
                "model": "gpt-5-mini",
                "modalities": ["text"],
                "max_tokens": 2048,
            },
        }
    ) as memory:
        source = memory.add("Ada moved to Berlin in March.")
        records = memory.list(limit=10).items

    assert backend.formed == [source.id]
    derived = [record for record in records if record.id != source.id]
    assert [record.content for record in derived] == [f"Berlin is a city mentioned by {source.id}."]
    context = derived[0].context
    assert context is not None
    assert context.kind is MemoryKind.ENTITY
    assert context.basis is EvidenceBasis.MODEL_INFERENCE


def test_declarative_formation_stays_off_unless_it_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Formation is an extra LLM round-trip on the write path, so omitting the slot must keep the
    non-extracting write path rather than opting a deployment in through some other slot."""
    backend = _formation_stub(monkeypatch)

    composition = resolve_memory_config(
        {
            "data_dir": tmp_path,
            "embedding": {"provider": "openai", "model": "tiny-test", "dimension": 4},
            "generation": {"provider": "openai", "model": "gpt-5-mini"},
        }
    )
    try:
        assert composition.plugins.former is None
        with Memory.from_plugins(
            tmp_path, plugins=composition.plugins, config=composition.settings
        ) as memory:
            memory.add("Ada moved to Berlin in March.")
    finally:
        composition.close()

    assert backend.formed == []


def test_composition_closes_the_optional_former(tmp_path: Path) -> None:
    class Former:
        formation_capabilities = frozenset({Modality.TEXT})
        formation_model = "test"
        formation_space = "test:v1"

        def __init__(self) -> None:
            self.closed = False

        def form(
            self,
            inputs: Sequence[FormationInput],
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(() for _value in inputs)

        def close(self) -> None:
            self.closed = True

    former = Former()
    composition = configuration.MemoryComposition(
        tmp_path,
        MemoryPlugins(embedder=TinyEmbedder(), former=former),
        MemoryConfig(),
    )

    composition.close()

    assert former.closed is True
