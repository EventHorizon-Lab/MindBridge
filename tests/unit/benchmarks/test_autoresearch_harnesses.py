from __future__ import annotations

import ast
import importlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindbridge.models import openai_sdk
from mindbridge.types import SearchHit

REPOSITORY = Path(__file__).resolve().parents[3]
HARNESS_ROOT = REPOSITORY / "autoresearch/orchestrator-260830-2214"
SOURCE_FACT_ROOT = HARNESS_ROOT / "source-fact-sidecar-v1"


@pytest.fixture(scope="module")
def harnesses() -> SimpleNamespace:
    sys.path.insert(0, str(SOURCE_FACT_ROOT))
    sys.path.insert(0, str(HARNESS_ROOT))
    return SimpleNamespace(
        gallery=importlib.import_module("run_memory_id_gallery_ab"),
        rounded=importlib.import_module("replay_rounded_score"),
        source_fact=importlib.import_module("run"),
        answer_gate=importlib.import_module("answer_gate"),
        sweep=importlib.import_module("sweep_candidate_pool"),
    )


def test_replay_payloads_match_product_and_keep_score_control(harnesses: SimpleNamespace) -> None:
    hit = SearchHit(
        id="memory-1",
        content="source",
        score=0.875,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )
    product = openai_sdk._hit_payload(hit)
    assert harnesses.gallery._hit_payload(hit, include_memory_id=True) == product
    without_id = harnesses.gallery._hit_payload(hit, include_memory_id=False)
    assert without_id == {key: value for key, value in product.items() if key != "memory_id"}
    assert without_id["created_at"] == "2026-08-31T00:00:00+00:00"

    token = harnesses.rounded._ARM.set("score")
    try:
        control = harnesses.rounded._hit_payload(hit)
    finally:
        harnesses.rounded._ARM.reset(token)
    token = harnesses.rounded._ARM.set("compact")
    try:
        compact = harnesses.rounded._hit_payload(hit)
    finally:
        harnesses.rounded._ARM.reset(token)
    assert control == {**product, "score": hit.score}
    assert compact == product


def test_resume_discards_partial_and_duplicate_candidate_groups(
    harnesses: SimpleNamespace, tmp_path: Path
) -> None:
    rows = [
        *(
            {"task": "task", "sample_id": "complete", "candidate_pool": candidate}
            for candidate in (50, 100, 200)
        ),
        {"task": "task", "sample_id": "partial", "candidate_pool": 50},
        *(
            {"task": "task", "sample_id": "duplicate", "candidate_pool": candidate}
            for candidate in (50, 100, 200, 200)
        ),
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    resumed = harnesses.sweep._resume_rows(path)

    assert [row["sample_id"] for row in resumed] == ["complete"] * 3
    assert [json.loads(line) for line in path.read_text().splitlines()] == resumed


def test_answer_cache_binds_queries_stores_and_generation_protocol(
    harnesses: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stores = {}
    for label in ("baseline", "candidate"):
        path = tmp_path / label
        path.mkdir()
        with sqlite3.connect(path / "state.sqlite3") as connection:
            connection.execute("CREATE TABLE evidence (value TEXT NOT NULL)")
            connection.execute("INSERT INTO evidence VALUES (?)", (label,))
        stores[label] = path
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(harnesses.answer_gate, "BASELINE_PATH", stores["baseline"])
    monkeypatch.setattr(harnesses.answer_gate, "CANDIDATE_PATH", stores["candidate"])
    monkeypatch.setattr(harnesses.answer_gate, "CACHE_PATH", cache_path)
    query = SimpleNamespace(question_id="q1", content="Where is it?")
    query_digest = harnesses.answer_gate._query_digest((query,))
    cache = harnesses.answer_gate._load_cache("unit", query_digest)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    changed_query = SimpleNamespace(question_id="q1", content="When was it?")
    with pytest.raises(RuntimeError, match="query_set_sha256"):
        harnesses.answer_gate._load_cache(
            "unit", harnesses.answer_gate._query_digest((changed_query,))
        )

    with sqlite3.connect(stores["candidate"] / "state.sqlite3") as connection:
        connection.execute("UPDATE evidence SET value = 'changed'")
    with pytest.raises(RuntimeError, match="store_sha256"):
        harnesses.answer_gate._load_cache("unit", query_digest)

    with sqlite3.connect(stores["candidate"] / "state.sqlite3") as connection:
        connection.execute("UPDATE evidence SET value = 'candidate'")
    monkeypatch.setattr(
        harnesses.answer_gate.openai_sdk,
        "_GROUNDED_SYSTEM_PROMPT",
        "changed prompt",
    )
    with pytest.raises(RuntimeError, match="generation_protocol_sha256"):
        harnesses.answer_gate._load_cache("unit", query_digest)


def test_fact_resume_rejects_extractor_identity_changes(harnesses: SimpleNamespace) -> None:
    identity = harnesses.source_fact._extractor_identity("locomo", "unit", "source-sha")
    payload = {
        **identity,
        "extractor": dict(identity["extractor"]),
        "entries": {},
    }
    harnesses.source_fact._validate_extractor_identity(payload, identity)
    payload["extractor"]["prompt_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="prompt_sha256"):
        harnesses.source_fact._validate_extractor_identity(payload, identity)


def test_no_retry_claims_use_zero_retry_clients() -> None:
    paths = (
        SOURCE_FACT_ROOT / "answer_gate.py",
        HARNESS_ROOT / "repeat_memory_id_m3_q01_5rep.py",
        HARNESS_ROOT / "run_evidence_index_m3_frozen_ab.py",
        HARNESS_ROOT / "run_memory_id_m3_frozen_ab.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        clients = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"OpenAI", "AsyncOpenAI"}
        ]
        assert clients
        assert all(
            any(
                keyword.arg == "max_retries"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == 0
                for keyword in client.keywords
            )
            for client in clients
        )
