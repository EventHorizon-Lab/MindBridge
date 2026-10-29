"""Public SDK transport adds map candidates only to integrity-checked reader evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import httpx2 as httpx
import pytest
from _feature_support import TinyEmbedder
from _gazetteer_support import index_fixture, multi_index_fixture
from PIL import Image
from PIL.TiffImagePlugin import IFDRational
from test_openai_sdk import _sdk_client

import mindbridge.configuration as configuration
from mindbridge import (
    AssetRef,
    Memory,
    Modality,
    ModelError,
    ModelInput,
    OfflineGazetteer,
    OpenAIModels,
    SearchHit,
)
from mindbridge.evidence import answer_evidence_payloads
from mindbridge.models import openai_sdk as backend
from mindbridge.models.base import GenerationBackend


def photo(tmp_path: Path) -> AssetRef:
    path = tmp_path / "source.jpg"
    tags = Image.Exif()
    tags[34853] = {1: "N", 2: (0, 1, IFDRational(96, 5)), 3: "E", 4: (0, 2, IFDRational(282, 5))}
    Image.new("RGB", (32, 24), "blue").save(path, exif=tags)
    return AssetRef(
        id="photo",
        modality=Modality.IMAGE,
        media_type="image/jpeg",
        path=path,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def hit(asset: AssetRef, identifier: str = "source") -> SearchHit:
    return SearchHit(
        id=identifier,
        content="A retained caption.",
        score=1,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        assets=(asset,),
        modality=Modality.IMAGE,
        metadata={"source_id": "caller-photo"},
    )


@pytest.mark.parametrize("multiple", [False, True])
def test_public_memory_answer_uses_external_map_without_changing_source(
    tmp_path: Path, multiple: bool
) -> None:
    path, digest = (multi_index_fixture if multiple else index_fixture)(tmp_path / "map.sqlite3")
    answer = "Second region museum" if multiple else "Named Museum"
    gazetteer = OfflineGazetteer(path, index_sha256=digest)
    asset = photo(tmp_path)
    sent: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if sent[-1].get("stream"):
            event = {
                "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 3, "total_tokens": 103},
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"content": answer}, "finish_reason": "stop"}]
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as transport,
        _sdk_client(transport) as client,
    ):
        answerer = OpenAIModels(
            client,
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_gazetteer=gazetteer,
        )
        with Memory(
            tmp_path / "memory",
            embedder=TinyEmbedder(),
            answerer=answerer,
            reinforce_on_answer=False,
            minimum_relevance=0,
        ) as memory:
            assert asset.path is not None
            source = memory.add(asset.path, metadata={"source_id": "caller-photo"})
            before = memory.get(source.id)
            result = memory.ask("Where was this photo captured?", limit=1)
            assert result.answer == answer
            assert memory.get(source.id) == before
            assert not source.content
            assert source.assets[0].sha256 == asset.sha256
    assert len(sent) == 1
    parts = sent[0]["messages"][1]["content"]
    assert sum(part["type"] == "image_url" for part in parts) == 1
    wire = [
        json.loads(part["text"])
        for part in parts
        if part["type"] == "text" and part["text"].startswith("{")
    ]
    evidence = next(part["memory"] for part in wire if "memory" in part)
    assert evidence["metadata"]["source_id"] == "caller-photo"
    metadata = evidence["untrusted_source_metadata"][0]
    assert metadata["gps"]["latitude"] == pytest.approx(0.022)
    enrichment = metadata["offline_osm_candidates"]
    assert enrichment["candidates"][0]["name" if multiple else "name:en"] == answer
    if multiple:
        assert [row["source_index"] for row in enrichment["map_sources"]] == [0, 1]
        assert [row["snapshot_date"] for row in enrichment["map_sources"]] == [
            "2026-01-01",
            "2025-02-03",
        ]
        assert [row["sha256"] for row in enrichment["map_sources"]] == ["1" * 64, "2" * 64]
        assert (
            enrichment["map_sources"][1]["osmosis_replication_timestamp"] == "2025-02-03T04:05:06Z"
        )
        assert "map_snapshot_date" not in enrichment and "map_data_timestamp" not in enrichment
    else:
        assert "map_sources" not in enrichment
    assert enrichment["untrusted"] is True and enrichment["photo_identity_confidence"] == "unknown"
    assert enrichment["global_complete"] is False
    assert "not the photographed" in enrichment["interpretation"]


def test_global_budget_counts_every_duplicate_metadata_occurrence_and_keeps_whole_candidates(
    tmp_path: Path,
) -> None:
    path, digest = index_fixture(tmp_path / "map.sqlite3")
    asset = photo(tmp_path)
    question, hits = (
        ModelInput(text="Which place?", assets=(asset,)),
        (hit(asset), hit(asset, "other")),
    )
    payloads = answer_evidence_payloads(hits)
    original = backend._answer_text_parts(question, hits, payloads)
    roomy = OfflineGazetteer(path, index_sha256=digest, limit=1, budget_chars=100_000)
    full = backend._answer_text_parts(question, hits, payloads, gazetteer=roomy)
    cost = sum(map(len, full)) - sum(map(len, original))
    assert cost > 0
    exact = OfflineGazetteer(path, index_sha256=digest, limit=1, budget_chars=cost)
    assert backend._answer_text_parts(question, hits, payloads, gazetteer=exact) == full
    insufficient = OfflineGazetteer(path, index_sha256=digest, limit=1, budget_chars=cost - 1)
    assert backend._answer_text_parts(question, hits, payloads, gazetteer=insufficient) == original
    for before, after in zip(original, full, strict=True):
        left, right = json.loads(before), json.loads(after)
        left = left.get("memory", left)
        right = right.get("memory", right)
        del right["untrusted_source_metadata"][0]["offline_osm_candidates"]
        assert left == right


def test_omitted_pixels_and_changed_original_never_receive_map_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = index_fixture(tmp_path / "map.sqlite3")
    gazetteer = OfflineGazetteer(path, index_sha256=digest)
    asset = photo(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("no map lookup for omitted or unverified input")

    monkeypatch.setattr(gazetteer, "nearby", forbidden)
    adapter = OpenAIModels(
        generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
        generation_gazetteer=gazetteer,
    )
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_BYTES", 1)
    prepared = adapter._answer_request("What is known?", (hit(asset),))
    assert isinstance(prepared, tuple)
    assert "offline_osm_candidates" not in json.dumps(prepared[0])
    monkeypatch.undo()
    monkeypatch.setattr(gazetteer, "nearby", forbidden)
    cached = backend._source_capture_metadata((asset,))
    assert asset.path is not None
    asset.path.write_bytes(b"changed")
    with pytest.raises(ModelError) as failure:
        backend._answer_text_parts(
            ModelInput(text="Where?"),
            (hit(asset),),
            answer_evidence_payloads((hit(asset),)),
            capture_metadata=cached,
            gazetteer=gazetteer,
        )
    assert failure.value.reason == "asset_changed"


def test_disabled_path_and_write_recipes_remain_unchanged(tmp_path: Path) -> None:
    path, digest = index_fixture(tmp_path / "map.sqlite3")
    gazetteer = OfflineGazetteer(path, index_sha256=digest)
    plain, enabled = OpenAIModels(), OpenAIModels(generation_gazetteer=gazetteer)
    assert plain.embedding_space == enabled.embedding_space
    assert plain.vision_space == enabled.vision_space
    assert plain.formation_space == enabled.formation_space
    asset = photo(tmp_path)
    evidence = (hit(asset),)
    original = backend._answer_text_parts(
        ModelInput(text="Where?"), evidence, answer_evidence_payloads(evidence)
    )
    explicit = backend._answer_text_parts(
        ModelInput(text="Where?"), evidence, answer_evidence_payloads(evidence), gazetteer=None
    )
    assert original == explicit
    assert "offline_osm_candidates" not in "".join(original)


def test_config_factory_injects_only_the_generation_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = index_fixture(tmp_path / "map.sqlite3")
    captured: dict[str, object] = {}
    marker = cast(GenerationBackend, object())

    def build(values: dict[str, object]) -> GenerationBackend:
        captured.update(values)
        return marker

    monkeypatch.setattr(configuration, "_openai_factory", build)
    value = configuration.OpenAIGenerationConfig.model_validate(
        {
            "provider": "openai",
            "offline_gazetteer": {
                "index_path": str(path),
                "index_sha256": digest,
                "budget_chars": 1234,
            },
        }
    )
    assert configuration._build_generation(value) is marker
    adapter = captured["generation_gazetteer"]
    assert isinstance(adapter, OfflineGazetteer) and adapter.budget_chars == 1234
    assert not value.extra_body


def test_multiple_source_headers_share_the_actual_repeated_metadata_budget(tmp_path: Path) -> None:
    path, digest = multi_index_fixture(tmp_path / "map.sqlite3")
    asset = photo(tmp_path)
    question, hits = ModelInput(text="Where?", assets=(asset,)), (hit(asset), hit(asset, "another"))
    payloads = answer_evidence_payloads(hits)
    original = backend._answer_text_parts(question, hits, payloads)
    full = backend._answer_text_parts(
        question,
        hits,
        payloads,
        gazetteer=OfflineGazetteer(path, index_sha256=digest, limit=2, budget_chars=100000),
    )
    cost = sum(map(len, full)) - sum(map(len, original))
    exact = backend._answer_text_parts(
        question,
        hits,
        payloads,
        gazetteer=OfflineGazetteer(path, index_sha256=digest, limit=2, budget_chars=cost),
    )
    assert exact == full
    reduced = backend._answer_text_parts(
        question,
        hits,
        payloads,
        gazetteer=OfflineGazetteer(path, index_sha256=digest, limit=2, budget_chars=cost - 1),
    )
    first_cost = sum(map(len, reduced)) - sum(map(len, original))
    assert 0 < first_cost < cost
    for part in reduced:
        value = json.loads(part)
        value = value.get("memory", value)
        enrichment = value["untrusted_source_metadata"][0]["offline_osm_candidates"]
        assert [candidate["source_index"] for candidate in enrichment["candidates"]] == [1]
        assert [source["source_index"] for source in enrichment["map_sources"]] == [1]
        assert (
            enrichment["map_sources"][0]["url"] == "https://example.invalid/second-region.osm.pbf"
        )
    no_room = backend._answer_text_parts(
        question,
        hits,
        payloads,
        gazetteer=OfflineGazetteer(path, index_sha256=digest, limit=2, budget_chars=first_cost - 1),
    )
    assert no_room == original


def test_v1_reader_header_keeps_its_single_source_serialized_contract(tmp_path: Path) -> None:
    path, digest = index_fixture(tmp_path / "map.sqlite3")
    gazetteer = OfflineGazetteer(path, index_sha256=digest)
    asset = photo(tmp_path)
    evidence = (hit(asset),)
    parts = backend._answer_text_parts(
        ModelInput(text="Where?"), evidence, answer_evidence_payloads(evidence), gazetteer=gazetteer
    )
    actual = json.loads(parts[1])["memory"]["untrusted_source_metadata"][0][
        "offline_osm_candidates"
    ]
    expected = {
        "untrusted": True,
        "map_snapshot_date": "2026-01-01",
        "map_data_timestamp": "2026-01-01T00:00:00Z",
        "index_sha256": digest,
        "attribution": "© OpenStreetMap contributors; ODbL 1.0",
        "license_url": "https://www.openstreetmap.org/copyright",
        "geometry_recipe": "offline-osm-named-geometries-v1-aeqd-wgs84-step0005",
        "radius_m": 500.0,
        "candidate_limit": 8,
        "interpretation": "Nearby or covering map geometry is not the photographed object's identity or a mentioned place. GPS error and map changes since capture remain possible; explicit address tags describe the map feature only.",
        "photo_identity_confidence": "unknown",
        "global_complete": False,
        "candidates": gazetteer.nearby(0.022, 0.049)["candidates"],
    }
    assert backend._json_text(actual) == backend._json_text(expected)
