"""Offline geometry and immutable external-index boundaries."""

from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import IO, Any, cast

import pytest
from _gazetteer_support import index_fixture, multi_index_fixture

from mindbridge import OfflineGazetteer, StorageError, ValidationError
from mindbridge.infrastructure.local import gazetteer as implementation


def test_true_geometry_holes_boundary_road_and_bbox_only_false_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = index_fixture(tmp_path / "region.sqlite3")
    adapter = OfflineGazetteer(path, index_sha256=digest)

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("coordinates must never leave the process")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    hole = adapter.nearby(0, 0)["candidates"]
    assert isinstance(hole, list)
    assert hole[0]["covers_coordinate"] is False
    assert hole[0]["distance_m"] > 100
    inside = adapter.nearby(0, 0.002)["candidates"]
    assert isinstance(inside, list)
    assert inside[0]["spatial_relation"] == "containing_area"
    boundary = adapter.nearby(0.003, 0.000234)["candidates"]
    assert isinstance(boundary, list)
    assert boundary[0]["distance_m"] == 0
    assert boundary[0]["coordinate_on_boundary"] is True
    road = adapter.nearby(0.0201, 0.049)["candidates"]
    assert isinstance(road, list)
    assert road[0]["osm_id"] == 2
    assert road[0]["distance_m"] == pytest.approx(11.1, abs=0.15)
    assert adapter.nearby(0.075, 0.065)["candidates"] == []
    assert adapter.nearby(20, 20)["reason"] == "outside_extract_bbox"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_provenance_explicit_tags_shared_readers_and_cache_are_detached(tmp_path: Path) -> None:
    path, digest = index_fixture(tmp_path / "region.sqlite3")
    first = OfflineGazetteer(path, index_sha256=digest)
    second = OfflineGazetteer(path, index_sha256=digest)
    with ThreadPoolExecutor(max_workers=2) as workers:
        values = list(workers.map(lambda adapter: adapter.nearby(0.022, 0.049), (first, second)))
    assert values[0] == values[1]
    candidates = values[0]["candidates"]
    assert isinstance(candidates, list)
    assert candidates[0]["name:en"] == "Named Museum"
    assert candidates[0]["address_tags"] == {"addr:street": "Explicit Street"}
    assert "Never expose" not in json.dumps(values)
    assert values[0]["photo_identity_confidence"] == "unknown"
    assert values[0]["global_complete"] is False
    candidates.clear()
    assert first.nearby(0.022, 0.049)["candidates"]
    source = first.provenance["source"]
    assert isinstance(source, dict)
    source.clear()
    assert first.provenance["source"]


def test_changed_index_is_rejected_even_for_cached_coordinates(tmp_path: Path) -> None:
    path, digest = index_fixture(tmp_path / "region.sqlite3")
    adapter = OfflineGazetteer(path, index_sha256=digest)
    adapter.nearby(0, 0)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(StorageError, match="changed"):
        adapter.nearby(0, 0)


def test_only_pinned_main_snapshot_is_read_when_unbound_wal_updates_exist(
    tmp_path: Path,
) -> None:
    path, _ = index_fixture(tmp_path / "region.sqlite3")
    with closing(sqlite3.connect(path)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        before = path.read_bytes()
        digest = hashlib.sha256(before).hexdigest()
        signature = path.stat()
        adapter = OfflineGazetteer(path, index_sha256=digest)
        with writer:
            writer.execute(
                "UPDATE features SET tags_json=? WHERE osm_id=3",
                (json.dumps({"name:en": "Unbound WAL Museum"}),),
            )
            writer.execute(
                "UPDATE metadata SET value=? WHERE key='recipe'", (json.dumps("unbound"),)
            )
        assert Path(str(path) + "-wal").stat().st_size > 0
        assert path.read_bytes() == before
        assert path.stat().st_mtime_ns == signature.st_mtime_ns
        with closing(sqlite3.connect(path)) as ordinary_reader:
            assert ordinary_reader.execute(
                "SELECT value FROM metadata WHERE key='recipe'"
            ).fetchone() == (json.dumps("unbound"),)
        # Construction must also validate the pinned main metadata, not WAL metadata.
        after_wal = OfflineGazetteer(path, index_sha256=digest)
        for reader in (adapter, after_wal):
            candidates = reader.nearby(0.022, 0.049)["candidates"]
            assert isinstance(candidates, list)
            assert candidates[0]["name:en"] == "Named Museum"
            assert "Unbound WAL Museum" not in json.dumps(candidates)


def test_grid_free_queries_do_not_change_global_network_policy_or_rehash_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = index_fixture(tmp_path / "region.sqlite3")
    adapter = OfflineGazetteer(path, index_sha256=digest)
    pyproj = pytest.importorskip("pyproj")
    previous = pyproj.network.is_network_enabled()
    original_open = Path.open

    def guarded_open(
        value: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        assert value != path, "a lookup must not reread the entire pinned index"
        return original_open(value, mode, buffering, encoding, errors, newline)

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline lookup must not open a socket")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    try:
        pyproj.network.set_network_enabled(True)
        first = adapter.nearby(0.022, 0.049)
        assert first == adapter.nearby(0.022, 0.049)
        assert pyproj.network.is_network_enabled()
    finally:
        pyproj.network.set_network_enabled(previous)


@pytest.mark.parametrize("change", ["sha", "schema", "recipe", "source", "license"])
def test_unknown_or_unpinned_index_is_not_migrated(tmp_path: Path, change: str) -> None:
    path, digest = index_fixture(tmp_path / "region.sqlite3")
    if change == "sha":
        digest = "0" * 64
    else:
        with closing(sqlite3.connect(path)) as connection, connection:
            if change == "schema":
                connection.execute("ALTER TABLE features ADD COLUMN unknown TEXT")
            else:
                key, value = {
                    "recipe": ("recipe", "another-schema"),
                    "source": ("source", {}),
                    "license": ("license_url", ""),
                }[change]
                connection.execute(
                    "UPDATE metadata SET value=? WHERE key=?", (json.dumps(value), key)
                )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    before = path.read_bytes()
    with pytest.raises(ValidationError, match="index"):
        OfflineGazetteer(path, index_sha256=digest)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "options",
    [
        {"radius_m": 0},
        {"radius_m": 501},
        {"radius_m": float("nan")},
        {"radius_m": True},
        {"limit": 9},
        {"limit": True},
        {"budget_chars": 0},
        {"budget_chars": True},
    ],
)
def test_bounded_options_reject_before_opening_any_index(options: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        OfflineGazetteer("not-opened", index_sha256="0" * 64, **options)


def test_excessive_prefilter_reports_unknown_instead_of_a_false_nearest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = index_fixture(tmp_path / "region.sqlite3")
    monkeypatch.setattr(implementation, "_MAX_PREFILTER_ROWS", 1)
    adapter = OfflineGazetteer(path, index_sha256=digest)
    value = adapter.nearby(0.022, 0.049)
    assert value["candidates"] == []
    assert value["lookup_incomplete"] is True


def test_multiple_sources_filter_real_extracts_and_sort_distances_together(tmp_path: Path) -> None:
    path, digest = multi_index_fixture(tmp_path / "regions.sqlite3")
    adapter = OfflineGazetteer(path, index_sha256=digest)
    value = adapter.nearby(0.022, 0.049)
    assert "source" not in value
    assert isinstance(value["sources"], list) and len(value["sources"]) == 2
    candidates = cast(list[dict[str, Any]], value["candidates"])
    assert [(row["osm_id"], row["source_index"]) for row in candidates] == [(20, 1), (10, 0)]
    assert candidates[0]["distance_m"] == 0 and candidates[1]["distance_m"] > 50
    assert (
        OfflineGazetteer(path, index_sha256=digest, limit=1).nearby(0.022, 0.049)["candidates"]
        == candidates[:1]
    )
    # Source 1 has a closer feature, but this coordinate is outside its true extract bbox.
    west = cast(list[dict[str, Any]], adapter.nearby(0.022, 0.0399)["candidates"])
    assert [(row["osm_id"], row["source_index"]) for row in west] == [(11, 0)]
    distant = cast(list[dict[str, Any]], adapter.nearby(10, 10)["candidates"])
    assert [(row["osm_id"], row["source_index"]) for row in distant] == [(22, 1)]
    assert adapter.nearby(50, 50)["reason"] == "outside_extract_bbox"
    cast(list[dict[str, Any]], value["sources"])[0].clear()
    assert cast(list[dict[str, Any]], adapter.provenance["sources"])[0]["url"]


@pytest.mark.parametrize("value", [None, -1, 2, 0.5, "invalid"])
def test_multiple_source_indices_are_validated_before_queries(
    tmp_path: Path, value: object
) -> None:
    path, _ = multi_index_fixture(tmp_path / "regions.sqlite3")
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE features SET source_index=? WHERE id=10", (value,))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValidationError, match="index"):
        OfflineGazetteer(path, index_sha256=digest)


@pytest.mark.parametrize(
    "change",
    [
        "single_source",
        "mixed_source",
        "missing_column",
        "v1_recipe",
        "bad_provenance",
        "duplicate_osm",
    ],
)
def test_multiple_source_schema_and_provenance_fail_closed(tmp_path: Path, change: str) -> None:
    path, _ = multi_index_fixture(tmp_path / "regions.sqlite3")
    with closing(sqlite3.connect(path)) as db, db:
        sources = json.loads(
            db.execute("SELECT value FROM metadata WHERE key='sources'").fetchone()[0]
        )
        if change == "single_source":
            db.execute(
                "UPDATE metadata SET value=? WHERE key='sources'", (json.dumps(sources[:1]),)
            )
        elif change == "mixed_source":
            db.execute("INSERT INTO metadata VALUES('source',?)", (json.dumps(sources[0]),))
        elif change == "missing_column":
            db.execute("ALTER TABLE features DROP COLUMN source_index")
        elif change == "v1_recipe":
            db.execute(
                "UPDATE metadata SET value=? WHERE key='recipe'",
                (json.dumps(implementation._INDEX_RECIPE),),
            )
        elif change == "bad_provenance":
            sources[1]["sha256"] = "unbound"
            db.execute("UPDATE metadata SET value=? WHERE key='sources'", (json.dumps(sources),))
        else:
            db.execute("ALTER TABLE features RENAME TO original_features")
            db.execute("CREATE TABLE features AS SELECT * FROM original_features")
            db.execute(
                "INSERT INTO features SELECT 999,osm_type,osm_id,kind,tags_json,geometry_wkb,source_index FROM original_features WHERE id=10"
            )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    before = path.read_bytes()
    with pytest.raises(ValidationError, match="index"):
        OfflineGazetteer(path, index_sha256=digest)
    assert path.read_bytes() == before
