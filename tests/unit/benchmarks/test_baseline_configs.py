from __future__ import annotations

from pathlib import Path

import pytest

from mindbridge.benchmarks.eval import _load_memory_config
from mindbridge.benchmarks.task_catalog import TASKS, expand

_ROOT = Path(__file__).resolve().parents[3]
_BASELINES = _ROOT / "docs" / "examples" / "baselines"


@pytest.mark.parametrize(
    "name",
    ("rtx5090-qwen38-wemm9b-text.yaml", "rtx5090-qwen38-wemm9b-media.yaml"),
)
def test_5090_baseline_has_exact_models_endpoints_and_no_secret(name: str) -> None:
    path = _BASELINES / name
    config, overrides = _load_memory_config(path)

    assert config is not None
    assert config.embedding.provider == "openai"
    assert config.embedding.model == "tencent/WeMM-Embedding-9B"
    assert config.embedding.dimension == 4096
    assert config.embedding.base_url == "https://xyrobot-embed.xyrobot.com/v1"
    assert config.embedding.api_key is None
    assert config.generation is not None
    assert config.generation.model == "Qwen3.8-27B"
    assert config.generation.base_url == "http://xyrobot-vl.xyrobot.com/v1"
    assert config.generation.api_key is None
    assert overrides.judge.api_key is None
    assert overrides.server_metrics.generation_url == "http://xyrobot-vl.xyrobot.com/metrics"
    assert overrides.server_metrics.embedding_url == "https://xyrobot-embed.xyrobot.com/metrics"
    assert overrides.run.repeat_index == 0
    paths = (
        overrides.download.benchmarks_root,
        overrides.download.data_root,
        overrides.download.hf_home,
    )
    assert not any(path.is_absolute() for path in paths if path is not None)


def test_media_baseline_selects_runtime_image_video_and_asr_routes() -> None:
    config, overrides = _load_memory_config(_BASELINES / "rtx5090-qwen38-wemm9b-media.yaml")

    assert config is not None
    assert config.speech is not None and config.speech.provider == "funasr"
    assert config.settings.index_speech is True
    selected = overrides.run.tasks
    assert selected is not None
    tasks = expand(tuple(selected.split(",")))
    assert set(tasks) == {"atm-bench-main", "egolifeqa"}
    patterns: set[str] = set()
    for task in tasks:
        source = TASKS[task].media_source
        if source is not None:
            patterns.update(source.patterns)
    assert "data/raw_memory/image/*" in patterns
    assert "data/raw_memory/video/*" in patterns
    assert "A?_*/DAY*/*.mp4" in patterns
