from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError as PydanticValidationError

import mindbridge.configuration as configuration
import mindbridge.recipes as recipes_module
from mindbridge import MemoryConfig, MemorySettings, MindBridgeConfig
from mindbridge.exceptions import ValidationError
from mindbridge.models.base import GenerationBackend
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
        "generation_temperature": 0.1,
        "generation_max_tokens": 512,
    }
