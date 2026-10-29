"""Public speech configuration, store identity, and the pinned upstream result contract."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from importlib.util import find_spec
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from mindbridge import FunASRTranscriber, Memory, MindBridgeConfig, StorageError
from mindbridge.configuration import FunASRSpeechConfig, _build_speech


def test_funasr_configuration_forwards_execution_settings() -> None:
    config = MindBridgeConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "speech": {
                "provider": "funasr",
                "device": "cuda:1",
                "ncpu": 2,
                "vad_device": "cpu",
                "speaker_device": "cuda:0",
            },
        }
    )
    assert isinstance(config.speech, FunASRSpeechConfig)
    transcriber = _build_speech(config.speech)
    assert isinstance(transcriber, FunASRTranscriber)
    assert transcriber._device == "cuda:1"
    assert transcriber._ncpu == 2
    assert transcriber._submodel_devices == {"vad_kwargs": "cpu", "spk_kwargs": "cuda:0"}
    transcriber.close()
    defaults = FunASRSpeechConfig(provider="funasr")
    assert defaults.ncpu == 4
    assert defaults.vad_device is None
    assert defaults.speaker_device is None


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_funasr_configuration_requires_a_strict_cpu_budget(value: object) -> None:
    with pytest.raises(PydanticValidationError):
        FunASRSpeechConfig.model_validate({"provider": "funasr", "ncpu": value})


def test_funasr_upstream_assigns_the_speaker_with_most_total_overlap() -> None:
    if find_spec("funasr") is None:
        pytest.skip("requires the local extra")
    # FunASR and ModelScope install process-global logging handlers on import.
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-W",
            "error",
            "-c",
            "from funasr.models.campplus.utils import distribute_spk; "
            "sentences = [{'start': 0, 'end': 1000, 'text': 'hello'}]; "
            "distribute_spk(sentences, [(0, 0.4, 0), (0.4, 1.0, 1)]); "
            "assert sentences[0]['spk'] == 1",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )


def test_funasr_old_analysis_space_cannot_be_reopened(tmp_path: Path) -> None:
    from mindbridge import EmbedTask, Modality, ModelInput

    class Embedder:
        embedding_model = "speech-contract"
        embedding_space = "speech-contract:2:v1"
        embedding_dimension = 2
        embedding_capabilities = frozenset({Modality.TEXT})

        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            return tuple((1.0, 0.0) for _ in inputs)

        def close(self) -> None:
            pass

    class OldTranscriber(FunASRTranscriber):
        @property
        def transcription_space(self) -> str:
            return "FunAudioLLM/Fun-ASR-Nano-2512:speech:b47a175e755dea61"

    with Memory(tmp_path, embedder=Embedder(), transcriber=OldTranscriber()):
        pass
    with pytest.raises(StorageError, match=r"transcription\.space_id"):
        Memory(tmp_path, embedder=Embedder(), transcriber=FunASRTranscriber())
