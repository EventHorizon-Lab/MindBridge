"""The external-world map resource is an explicit, generation-only opt-in."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import cast

import pytest
from pydantic import ValidationError as PydanticValidationError

from mindbridge import MindBridgeConfig, OfflineGazetteer, OpenAIModels, ValidationError


def test_default_config_and_package_import_do_not_enable_or_load_geospatial_dependencies() -> None:
    config = MindBridgeConfig.model_validate(
        {"embedding": {"provider": "openai"}, "generation": {"provider": "openai"}}
    )
    assert config.generation is not None and config.generation.offline_gazetteer is None
    command = "import json,sys; from mindbridge import OfflineGazetteer; print(json.dumps(sorted({'shapely','pyproj','osmium'}.intersection(sys.modules))))"
    result = subprocess.run(
        [sys.executable, "-I", "-c", command],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == []
    with pytest.raises(ValidationError, match="generation_gazetteer"):
        OpenAIModels(generation_gazetteer=cast(OfflineGazetteer, object()))


@pytest.mark.parametrize("slot", ["embedding", "formation", "vision", "consolidation"])
def test_map_configuration_cannot_enable_write_time_world_knowledge(slot: str) -> None:
    with pytest.raises(PydanticValidationError):
        MindBridgeConfig.model_validate(
            {
                "embedding": {"provider": "openai"},
                slot: {
                    "provider": "openai",
                    "offline_gazetteer": {"index_path": "map.sqlite3", "index_sha256": "0" * 64},
                },
            }
        )


@pytest.mark.parametrize(
    "values",
    [
        {"index_sha256": "bad"},
        {"radius_m": 501},
        {"radius_m": float("inf")},
        {"limit": 9},
        {"limit": True},
        {"budget_chars": 0},
    ],
)
def test_map_resource_bounds_fail_before_clients_or_storage(values: dict[str, object]) -> None:
    with pytest.raises(PydanticValidationError):
        MindBridgeConfig.model_validate(
            {
                "embedding": {"provider": "openai"},
                "generation": {
                    "provider": "openai",
                    "offline_gazetteer": {
                        "index_path": "map.sqlite3",
                        "index_sha256": "0" * 64,
                        **values,
                    },
                },
            }
        )
