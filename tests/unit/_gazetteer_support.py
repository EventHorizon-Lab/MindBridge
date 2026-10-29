"""A small named-geometry v1 external index, with no benchmark inputs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest


def index_fixture(path: Path) -> tuple[Path, str]:
    geometry = pytest.importorskip("shapely.geometry")
    features = (
        (
            "relation",
            1,
            "named_area",
            {"name": "Courtyard College"},
            geometry.Polygon(
                [(-0.003, -0.003), (0.003, -0.003), (0.003, 0.003), (-0.003, 0.003)],
                holes=[[(-0.001, -0.001), (0.001, -0.001), (0.001, 0.001), (-0.001, 0.001)]],
            ),
        ),
        (
            "way",
            2,
            "named_road",
            {"name": "Long Road"},
            geometry.LineString([(-0.1, 0.02), (0.1, 0.02)]),
        ),
        (
            "node",
            3,
            "named_node",
            {
                "name": "原名",
                "name:en": "Named Museum",
                "official_name": "Museum Official Name",
                "addr:street": "Explicit Street",
                "note": "Never expose this freeform note",
            },
            geometry.Point(0.049, 0.022),
        ),
        (
            "way",
            4,
            "named_road",
            {"name": "L-shaped road"},
            geometry.LineString([(0.06, 0.06), (0.08, 0.06), (0.08, 0.08)]),
        ),
    )
    metadata: dict[str, Any] = {
        "recipe": "offline-osm-named-geometries-v1-aeqd-wgs84-step0005",
        "source": {
            "url": "https://example.invalid/public-region.osm.pbf",
            "snapshot_date": "2026-01-01",
            "sha256": "1" * 64,
            "extract_bbox_lonlat": [-1, -1, 1, 1],
            "header": {"osmosis_replication_timestamp": "2026-01-01T00:00:00Z"},
        },
        "attribution": "© OpenStreetMap contributors; ODbL 1.0",
        "license_url": "https://www.openstreetmap.org/copyright",
        "geometry_segment_max_degrees": 0.0005,
        "complete_world_or_region_map": False,
    }
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript(
            "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
            "CREATE TABLE features(id INTEGER PRIMARY KEY,osm_type TEXT,osm_id INTEGER,kind TEXT,tags_json TEXT,geometry_wkb BLOB,UNIQUE(osm_type,osm_id));"
            "CREATE VIRTUAL TABLE bounds USING rtree(id,min_lon,max_lon,min_lat,max_lat);"
        )
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)",
            [(key, json.dumps(value)) for key, value in metadata.items()],
        )
        for index, (osm_type, osm_id, kind, tags, shape) in enumerate(features):
            shape = pytest.importorskip("shapely").segmentize(shape, 0.0005)
            connection.execute(
                "INSERT INTO features VALUES(?,?,?,?,?,?)",
                (index, osm_type, osm_id, kind, json.dumps(tags), shape.wkb),
            )
            west, south, east, north = shape.bounds
            connection.execute(
                "INSERT INTO bounds VALUES(?,?,?,?,?)", (index, west, east, south, north)
            )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def multi_index_fixture(path: Path) -> tuple[Path, str]:
    """Two independent extracts, including overlap and geometry beyond a source bbox."""
    geometry = pytest.importorskip("shapely.geometry")
    index_fixture(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        source = json.loads(
            connection.execute("SELECT value FROM metadata WHERE key='source'").fetchone()[0]
        )
        second = {
            **source,
            "url": "https://example.invalid/second-region.osm.pbf",
            "sha256": "2" * 64,
            "snapshot_date": "2025-02-03",
            "extract_bbox_lonlat": [0.04, 0.02, 11, 11],
            "header": {"osmosis_replication_timestamp": "2025-02-03T04:05:06Z"},
        }
        connection.execute("DELETE FROM metadata WHERE key='source'")
        connection.execute(
            "INSERT INTO metadata VALUES('sources',?)", (json.dumps([source, second]),)
        )
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='recipe'",
            (json.dumps("offline-osm-named-geometries-v2-aeqd-wgs84-step0005"),),
        )
        connection.execute("ALTER TABLE features ADD COLUMN source_index INTEGER")
        connection.execute("DELETE FROM features")
        connection.execute("DELETE FROM bounds")
        values = [
            (10, "First region museum", 0.049, 0.0225, 0),
            (11, "First region west", 0.039, 0.022, 0),
            (12, "First extract geometry beyond its bbox", 10, 10, 0),
            (20, "Second region museum", 0.049, 0.022, 1),
            (21, "Second region west", 0.0401, 0.022, 1),
            (22, "Second region distant", 10.0001, 10, 1),
        ]
        for identifier, name, longitude, latitude, source_index in values:
            shape = geometry.Point(longitude, latitude)
            connection.execute(
                "INSERT INTO features VALUES(?,?,?,?,?,?,?)",
                (
                    identifier,
                    "node",
                    identifier,
                    "named_node",
                    json.dumps({"name": name}),
                    shape.wkb,
                    source_index,
                ),
            )
            connection.execute(
                "INSERT INTO bounds VALUES(?,?,?,?,?)",
                (identifier, longitude, longitude, latitude, latitude),
            )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()
