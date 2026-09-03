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


def test_a_symbolic_place_scopes_retrieval_and_needs_no_former(tmp_path: Path) -> None:
    """ "In the kitchen" is the spatial question a household actually asks.

    The metric axis already existed -- pose, frame, quaternion, radius -- but a robot that cannot
    localise can still label a room, and a person never asks for a radius. This is the second,
    symbolic axis, and it deliberately hangs off the memory record rather than the formed-semantics
    row: that row only exists when a former is configured, so a place scope built on it would have
    silently skipped the default composition entirely.
    """
    with Memory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0) as memory:
        kitchen = memory.add(
            "the kettle boiled dry again",
            context=ObservationContext(place_id="kitchen"),
        )
        garage = memory.add(
            "the kettle is mentioned in the garage log too",
            context=ObservationContext(place_id="garage"),
        )
        unlabelled = memory.add("the kettle with no place recorded")

        scoped = {
            hit.id
            for hit in memory.search("kettle", limit=10, scope=RetrievalScope(place_id="kitchen"))
        }
        assert scoped == {kitchen.id}
        # Read-back: the label is equality-matched, so an application that cannot see what was
        # stored cannot notice it wrote "the kitchen" where "kitchen" was meant.
        assert memory.get(kitchen.id).place_id == "kitchen"
        assert memory.get(unlabelled.id).place_id is None
        assert {record.place_id for record in memory.list(limit=10).items} == {
            "kitchen",
            "garage",
            None,
        }
        assert garage.id not in scoped
        # An unlabelled memory is not "everywhere": a place scope excludes it.
        assert unlabelled.id not in scoped
        # No scope means no place filter, not "memories with no place".
        assert len(memory.search("kettle", limit=10)) == 3

    # The label is equality-matched, so the contract rejects what would silently partition a store.
    for bad in ("", " kitchen", "kitchen "):
        with pytest.raises(ValidationError):
            ObservationContext(place_id=bad)
        with pytest.raises(ValidationError):
            RetrievalScope(place_id=bad)


def test_a_sparse_place_keeps_widening_instead_of_stopping_early(tmp_path: Path) -> None:
    """The candidate loop must count survivors of the place filter, not of the ranking.

    Retrieval widens its candidate window while fewer than `limit` memories survive the scope. If
    the place filter is applied only when hydrating results and not when counting survivors, the
    loop counts memories from other rooms as satisfied, stops at the first window, and returns
    nothing from the room that was asked about -- the under-return this axis exists to avoid.
    """
    with Memory(tmp_path, embedder=TinyEmbedder(), minimum_relevance=0) as memory:
        # Ranked above the target simply by being added first; the embedder is uninformative.
        for index in range(30):
            memory.add(
                f"the hallway cupboard entry number {index}",
                context=ObservationContext(place_id="hallway"),
            )
        first = memory.add(
            "the spare fuses live behind the bread bin",
            context=ObservationContext(place_id="kitchen"),
        )
        second = memory.add(
            "the tea towels are in the drawer under the sink",
            context=ObservationContext(place_id="kitchen"),
        )

        hits = memory.search(
            "entry cupboard fuses towels", limit=3, scope=RetrievalScope(place_id="kitchen")
        )

        assert {hit.id for hit in hits} == {first.id, second.id}
