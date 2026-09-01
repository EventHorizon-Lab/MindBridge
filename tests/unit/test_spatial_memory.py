from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder

from mindbridge import (
    FormationInput,
    FormationProposal,
    Memory,
    MemoryKind,
    ObservationContext,
    RetrievalScope,
    SpatialAnchor,
    SpatialContext,
    ValidationError,
)


class EntityFormer:
    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "entity-test"
    formation_space = "entity-test:v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(
            (
                FormationProposal(
                    kind=MemoryKind.ENTITY,
                    content="The observed object is a red toolbox",
                    subject="red toolbox",
                ),
            )
            for _value in inputs
        )

    def close(self) -> None:
        pass


def test_spatial_search_is_exact_within_one_coordinate_frame(tmp_path: Path) -> None:
    now = datetime(2026, 4, 1, tzinfo=timezone.utc)
    map_pose = SpatialContext(
        frame_id="map",
        anchor=SpatialAnchor.SUBJECT,
        x=1,
        y=2,
        orientation_xyzw=(0, 0, 0, -2),
        position_uncertainty_m=0.1,
    )
    camera_pose = SpatialContext(
        frame_id="camera",
        anchor=SpatialAnchor.SUBJECT,
        x=1,
        y=2,
    )

    with Memory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0) as memory:
        nearby = memory.add(
            "red toolbox",
            occurred_at=now,
            context=ObservationContext(valid_from=now, spatial=map_pose),
        )
        memory.add(
            "red toolbox",
            occurred_at=now,
            context=ObservationContext(valid_from=now, spatial=camera_pose),
        )
        hits = memory.search(
            "toolbox",
            limit=10,
            scope=RetrievalScope(
                near=SpatialContext(
                    frame_id="map",
                    anchor=SpatialAnchor.SUBJECT,
                    x=0,
                    y=2,
                    position_uncertainty_m=0.2,
                ),
                radius_m=0.7,
            ),
        )

        assert [hit.id for hit in hits] == [nearby.id]
        assert hits[0].context is not None
        assert hits[0].context.spatial is not None
        assert hits[0].context.spatial.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_spatial_context_rejects_invalid_physical_values() -> None:
    with pytest.raises(ValidationError):
        SpatialContext(
            frame_id="map",
            anchor=SpatialAnchor.SUBJECT,
            x=math.nan,
            y=0,
        )
    with pytest.raises(ValidationError):
        SpatialContext(
            frame_id="map",
            anchor=SpatialAnchor.SUBJECT,
            x=0,
            y=0,
            position_uncertainty_m=-1,
        )
    with pytest.raises(ValidationError):
        SpatialContext(
            frame_id="map",
            anchor=SpatialAnchor.SUBJECT,
            x=0,
            y=0,
            orientation_xyzw=(0, 0, 0, 0),
        )


def test_spatial_orientation_normalizes_extreme_finite_values() -> None:
    huge = SpatialContext(
        frame_id="map",
        anchor=SpatialAnchor.SUBJECT,
        x=0,
        y=0,
        orientation_xyzw=(1e308, 0, 0, 0),
    )
    tiny = SpatialContext(
        frame_id="map",
        anchor=SpatialAnchor.SUBJECT,
        x=0,
        y=0,
        orientation_xyzw=(1e-308, 0, 0, 0),
    )

    assert huge.orientation_xyzw == (1.0, 0.0, 0.0, 0.0)
    assert tiny.orientation_xyzw == (1.0, 0.0, 0.0, 0.0)


def test_formed_semantics_do_not_merge_across_spatial_frames(tmp_path: Path) -> None:
    map_pose = SpatialContext(
        frame_id="map",
        anchor=SpatialAnchor.SUBJECT,
        x=1,
        y=2,
    )
    camera_pose = SpatialContext(
        frame_id="camera",
        anchor=SpatialAnchor.SUBJECT,
        x=1,
        y=2,
    )
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=EntityFormer(),
        minimum_relevance=0,
    ) as memory:
        memory.add("red toolbox in map", context=ObservationContext(spatial=map_pose))
        memory.add("red toolbox in camera", context=ObservationContext(spatial=camera_pose))

        entities = [
            hit.context
            for hit in memory.search("observed red toolbox", limit=10)
            if hit.context is not None and hit.context.kind is MemoryKind.ENTITY
        ]

        assert len(entities) == 2
        assert {context.spatial.frame_id for context in entities if context.spatial} == {
            "camera",
            "map",
        }
