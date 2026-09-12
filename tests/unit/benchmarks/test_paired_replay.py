"""Tests for the paired-replay research driver."""

from __future__ import annotations

import asyncio
import base64
import fcntl
import importlib.util
import json
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from mindbridge.benchmarks.eval_arms import _BaselineGenerator
from mindbridge.benchmarks.eval_config import _load_memory_config
from mindbridge.benchmarks.model_config import ModelConfig
from mindbridge.types import SearchHit


def _driver() -> ModuleType:
    path = Path(__file__).parents[3] / "benchmarks" / "paired_replay.py"
    spec = importlib.util.spec_from_file_location("paired_replay_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_driver_refuses_to_clone_a_store_with_a_live_owner(tmp_path: Path) -> None:
    driver = _driver()
    store = tmp_path / "unit"
    store.mkdir()
    (store / "state.sqlite3").write_bytes(b"sqlite")
    lock_path = store / ".mindbridge.lock"
    with lock_path.open("w+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="still owned"):
            driver._assert_closed_store(store)


def test_driver_clones_distinct_arm_source_stores(tmp_path: Path) -> None:
    driver = _driver()
    sources = {}
    for arm, content in (("baseline", b"schema16"), ("candidate", b"schema17")):
        store = tmp_path / f"{arm}-source"
        store.mkdir()
        (store / "state.sqlite3").write_bytes(content)
        (store / ".mindbridge.lock").write_bytes(b"")
        sources[arm] = {"unit-a": str(store)}
    plan = {"source_stores": sources, "clone_root": str(tmp_path / "clones")}

    manifest = driver._clone_sources(plan, ("unit-a",))

    arm_sources = manifest["unit-a"]["arm_sources"]
    assert (
        Path(arm_sources["baseline"]["clone"]).joinpath("state.sqlite3").read_bytes() == b"schema16"
    )
    assert (
        Path(arm_sources["candidate"]["clone"]).joinpath("state.sqlite3").read_bytes()
        == b"schema17"
    )
    assert driver._sources_unchanged(manifest)


@pytest.mark.parametrize("pollution", ["sqlite", "query_failures"])
@pytest.mark.parametrize("distinct_arm_sources", [False, True])
def test_driver_rejects_reused_clones_changed_since_creation(
    tmp_path: Path, pollution: str, distinct_arm_sources: bool
) -> None:
    driver = _driver()
    clone_root = tmp_path / "clones"
    if distinct_arm_sources:
        declared = {}
        for arm in ("baseline", "candidate"):
            source = tmp_path / f"{arm}-source"
            source.mkdir()
            (source / "state.sqlite3").write_bytes(arm.encode())
            (source / ".mindbridge.lock").write_bytes(b"")
            declared[arm] = {"unit-a": str(source)}
        plan = {"source_stores": declared, "clone_root": str(clone_root)}
    else:
        source_run = tmp_path / "source-run"
        source = source_run / driver._safe_component("unit-a", "unit")
        source.mkdir(parents=True)
        (source / "state.sqlite3").write_bytes(b"shared")
        (source / ".mindbridge.lock").write_bytes(b"")
        plan = {"source_run": str(source_run), "clone_root": str(clone_root)}

    units = driver._clone_sources(plan, ("unit-a",))
    manifest = tmp_path / "source-store-manifest.json"
    driver._write_json(manifest, {"schema_version": driver.SCHEMA_VERSION, "units": units})
    assert driver._reuse_closed_clones(plan, ("unit-a",), manifest) == units

    clone = clone_root / "baseline" / driver._safe_component("unit-a", "unit")
    if pollution == "sqlite":
        (clone / "state.sqlite3").write_bytes(b"polluted")
    else:
        (clone / "query_failures.jsonl").write_text('{"qid":"q1"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="existing clone content changed for baseline/unit-a"):
        driver._reuse_closed_clones(plan, ("unit-a",), manifest)


def test_blind_replay_mode_is_explicit_and_store_free() -> None:
    driver = _driver()

    assert driver._replay_mode({}) == "compile"
    assert driver._replay_mode({"mode": "blind"}) == "blind"
    with pytest.raises(ValueError, match="compile or blind"):
        driver._replay_mode({"mode": "unknown"})


def test_driver_freezes_question_content_or_text_as_public_compile_input(
    tmp_path: Path,
) -> None:
    driver = _driver()

    class Question:
        def __init__(self) -> None:
            self.content = ("first", tmp_path / "image.png", "second")
            self.source_question = "bare source question"

    question = Question()
    assert driver._compile_query({}, question) == question.content
    assert driver._compile_query({"compile_query": "question_text"}, question) == "first\nsecond"
    assert (
        driver._compile_query({"compile_query": "source_question"}, question)
        == "bare source question"
    )
    with pytest.raises(ValueError, match="question_content, question_text, or source_question"):
        driver._compile_query({"compile_query": "labels"}, question)


def test_driver_builds_and_records_only_the_declared_compile_scope() -> None:
    driver = _driver()
    reference = datetime(2023, 10, 22, 9, 55, tzinfo=timezone.utc)

    assert driver._compile_scope({}, reference) is None
    assert driver._scope_record(None) == {
        "valid_at": None,
        "known_at": None,
        "place_id": None,
        "identity_id": None,
    }
    scope = driver._compile_scope(
        {
            "scope": {
                "valid_at": "reference_at",
                "known_at": "2026-09-08T13:19:27+00:00",
                "place_id": None,
                "identity_id": None,
            }
        },
        reference,
    )
    assert driver._scope_record(scope) == {
        "valid_at": reference.isoformat(),
        "known_at": "2026-09-08T13:19:27+00:00",
        "place_id": None,
        "identity_id": None,
    }
    with pytest.raises(ValueError, match="unsupported fields"):
        driver._compile_scope({"scope": {"account_id": "logical-scope"}}, reference)


def test_driver_verifies_the_frozen_source_store_archive(tmp_path: Path) -> None:
    driver = _driver()
    archive = tmp_path / "closed-store.tar.gz"
    archive.write_bytes(b"immutable closed store")
    digest = driver._sha256_file(archive)

    driver._verify_frozen_store_archive(
        {
            "source_store_archive": str(archive),
            "source_store_archive_sha256": digest,
        }
    )
    with pytest.raises(RuntimeError, match="source store archive digest mismatch"):
        driver._verify_frozen_store_archive(
            {
                "source_store_archive": str(archive),
                "source_store_archive_sha256": "0" * 64,
            }
        )
    with pytest.raises(ValueError, match="must be supplied together"):
        driver._verify_frozen_store_archive({"source_store_archive": str(archive)})


def test_driver_verifies_each_arm_source_store_receipt(tmp_path: Path) -> None:
    driver = _driver()
    receipts = {}
    for arm in ("baseline", "candidate"):
        path = tmp_path / f"{arm}.json"
        path.write_text(f'{{"arm":"{arm}"}}\n', encoding="utf-8")
        receipts[arm] = {"unit-a": {"path": str(path), "sha256": driver._sha256_file(path)}}

    driver._verify_source_store_receipts({"source_store_receipts": receipts})
    receipts["candidate"]["unit-a"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="candidate/unit-a"):
        driver._verify_source_store_receipts({"source_store_receipts": receipts})


def test_driver_records_hit_score_confidence_and_times() -> None:
    driver = _driver()
    occurred_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    hit = SearchHit(
        id="memory-1",
        content="evidence",
        score=0.75,
        created_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        occurred_at=occurred_at,
    )

    assert driver._hit_state(hit) == {
        "memory_id": "memory-1",
        "retrieval_score": 0.75,
        "confidence": 1.0,
        "created_at": "2026-01-03T00:00:00+00:00",
        "occurred_at": "2026-01-02T00:00:00+00:00",
        "occurred_end": None,
        "valid_from": None,
        "valid_until": None,
        "known_at": None,
    }


def test_driver_captures_the_actual_openai_request_body_without_network(tmp_path: Path) -> None:
    driver = _driver()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """generation:
  provider: openai
  base_url: http://xyrobot-vl.xyrobot.com/v1
  api_key: EMPTY
  model: Qwen3.8-27B
  modalities: [text, image, video]
  timeout: 300
  max_retries: 3
  video_limit: 8
  extra_body:
    chat_template_kwargs:
      enable_thinking: false
""",
        encoding="utf-8",
    )
    config, _ = _load_memory_config(config_path)
    assert config is not None and config.generation is not None
    generation = config.generation
    assert generation.timeout is not None
    model = ModelConfig(
        generation_api_key=(
            None if generation.api_key is None else generation.api_key.get_secret_value()
        ),
        generation_base_url=str(generation.base_url),
        generation_model=generation.model,
        timeout_seconds=generation.timeout,
        generation_capabilities=frozenset(generation.modalities),
        generation_min_video_seconds=generation.video_limit,
    )
    generator = _BaselineGenerator(model, seed=42, gen_kwargs="", generation=config)

    captured = asyncio.run(
        driver._capture_generation(generator, "frozen question", "frozen context")
    )
    body = json.loads(base64.b64decode(captured["body_b64"]))

    assert captured["url"] == "http://xyrobot-vl.xyrobot.com/v1/chat/completions"
    assert body["messages"][1]["content"] == (
        "Context:\nfrozen context\n\nQuestion:\nfrozen question"
    )
    assert body["model"] == "Qwen3.8-27B"
    assert body["seed"] == 42
    assert body["temperature"] == 0.0
    # The SDK moves ``extra_body`` members to the real provider payload's top level.
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "extra_body" not in body


def test_embedding_proxy_replays_exact_success_bytes_and_never_caches_errors(
    tmp_path: Path,
) -> None:
    driver = _driver()

    class Upstream(BaseHTTPRequestHandler):
        requests = 0

        def do_POST(self) -> None:
            type(self).requests += 1
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length)
            if self.path.endswith("/fail"):
                status = 503
                response = b'{"error":"temporary"}'
            else:
                status = 200
                response = b'{"data":[{"embedding":[0.25,-0.5]}],"echo":' + body + b"}"
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, fmt: str, *args: object) -> None:
            return None

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.daemon_threads = True
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    try:
        with driver._embedding_proxy(upstream_url, tmp_path / "cache.json") as (proxy, cache):
            body = b'{"input":["same exact query"],"model":"embed"}'
            first = httpx.post(f"{proxy}/embeddings", content=body)
            second = httpx.post(f"{proxy}/embeddings", content=body)
            failed_first = httpx.post(f"{proxy}/fail", content=b"same failure")
            failed_second = httpx.post(f"{proxy}/fail", content=b"same failure")
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join()

    assert first.content == second.content
    assert first.status_code == second.status_code == 200
    assert failed_first.status_code == failed_second.status_code == 503
    assert Upstream.requests == 3
    assert cache.entry_count == 1
    assert [event["cache_hit"] for event in cache.events] == [False, True, False, False]
    assert [event["network_call"] for event in cache.events] == [True, False, True, True]
    assert cache.events[0]["response_sha256"] == cache.events[1]["response_sha256"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", True, "status must be a successful integer"),
        ("status", 503, "status must be a successful integer"),
        ("content_type", None, "content_type must be a non-empty string"),
        ("body_b64", "not base64!", "body_b64 is invalid"),
        ("response_sha256", "0" * 64, "response digest mismatch"),
        ("extra", "unexpected", "invalid schema"),
    ],
)
def test_embedding_cache_rejects_corrupt_entries_on_load(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    driver = _driver()
    body = b'{"data":[]}'
    key = driver._sha256_bytes(b"request")
    entry: dict[str, object] = {
        "status": 200,
        "content_type": "application/json",
        "body_b64": base64.b64encode(body).decode("ascii"),
        "response_sha256": driver._sha256_bytes(body),
    }
    entry[field] = value
    path = tmp_path / "cache.json"
    driver._write_json(path, {"schema_version": driver.SCHEMA_VERSION, "entries": {key: entry}})

    with pytest.raises(ValueError, match=message):
        driver._EmbeddingResponseCache(path)


def test_embedding_cache_revalidates_an_entry_on_hit(tmp_path: Path) -> None:
    driver = _driver()
    body = b'{"data":[]}'
    key = driver._sha256_bytes(b"request")
    cache = driver._EmbeddingResponseCache(tmp_path / "cache.json")
    cache.store(
        key,
        {
            "status": 200,
            "content_type": "application/json",
            "body_b64": base64.b64encode(body).decode("ascii"),
            "response_sha256": driver._sha256_bytes(body),
        },
    )
    cache._entries[key]["body_b64"] = base64.b64encode(b"tampered").decode("ascii")

    with pytest.raises(ValueError, match="response digest mismatch"):
        cache.get(key)


def test_embedding_proxy_can_fail_closed_without_an_upstream_request(tmp_path: Path) -> None:
    driver = _driver()

    class Upstream(BaseHTTPRequestHandler):
        requests = 0

        def do_POST(self) -> None:
            type(self).requests += 1
            self.send_response(500)
            self.send_header("content-length", "0")
            self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            return None

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.daemon_threads = True
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    try:
        with driver._embedding_proxy(
            upstream_url, tmp_path / "cache.json", allow_network=False
        ) as (proxy, cache):
            response = httpx.post(f"{proxy}/embeddings", content=b'"uncached"')
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join()

    assert response.status_code == 424
    assert Upstream.requests == 0
    assert cache.entry_count == 0
    assert cache.events == [
        {
            "sequence": 0,
            "method": "POST",
            "path": "/v1/embeddings",
            "content_type": None,
            "body_sha256": driver._sha256_bytes(b'"uncached"'),
            "request_key": cache.events[0]["request_key"],
            "cache_hit": False,
            "network_call": False,
            "status": 424,
            "cached_success": False,
            "response_sha256": cache.events[0]["response_sha256"],
            "response_bytes": cache.events[0]["response_bytes"],
        }
    ]
