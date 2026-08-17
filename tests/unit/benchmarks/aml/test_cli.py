"""AML CLI wiring, manifest, and resumability tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from mindbridge.api.aml_contracts import derive_tenant_id
from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion
from mindbridge.benchmarks.aml.cli import BENCHMARKS, _Arguments, _run, _write_manifest
from mindbridge.benchmarks.aml.driver import eval_user_id
from mindbridge.benchmarks.artifacts import load_deployment_snapshot, sidecar_manifest_path


def test_every_benchmark_names_a_loader_an_emitter_and_a_pipeline() -> None:
    assert set(BENCHMARKS) == {
        "locomo",
        "longmemeval",
        "beam",
        "clbench",
        "personamem-v1",
        "personamem-v2",
    }
    for name, spec in BENCHMARKS.items():
        assert callable(spec.load), name
        assert callable(spec.emit), name
        assert spec.pipeline.exists(), f"{name} pipeline is not vendored at {spec.pipeline}"


def _case(user_id: str, question_id: str) -> AmlCase:
    return AmlCase(
        user_id=user_id,
        messages=({"role": "user", "content": f"{user_id} said hello"},),
        questions=(AmlQuestion(question_id=question_id, question="What happened?", payload={}),),
    )


def _handler(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/aml/add":
            payload = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "request_id": payload["request_id"],
                    "user_id": payload["user_id"],
                    "session_id": payload["session_id"],
                },
            )
        return httpx.Response(200, json={"data": [{"id": "mem_1", "content": "hi"}]})

    return handle


@pytest.mark.asyncio
async def test_run_writes_one_jsonl_row_per_question(tmp_path: Path) -> None:
    output_path = tmp_path / "rows.jsonl"
    seen: list[httpx.Request] = []
    cases = (_case("locomo:conv-0", "q0"), _case("locomo:conv-1", "q0"))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_handler(seen)), base_url="http://test"
    ) as client:
        tenant_ids = await _run(
            client,
            BENCHMARKS["locomo"],
            cases,
            run_id="run-1",
            benchmark="locomo",
            top_k=10,
            concurrency=2,
            output_path=output_path,
            tenant_prefix="bench_aml",
        )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert {row["id"] for row in rows} == {"locomo:conv-0#q0", "locomo:conv-1#q0"}
    assert set(tenant_ids) == {
        eval_user_id("run-1", "locomo", "locomo:conv-0"),
        eval_user_id("run-1", "locomo", "locomo:conv-1"),
    }
    for user_id, tenant_id in tenant_ids.items():
        assert tenant_id == derive_tenant_id("bench_aml", user_id)


@pytest.mark.asyncio
async def test_run_resumes_without_recontacting_finished_cases_or_duplicating_rows(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "rows.jsonl"
    cases = (_case("locomo:conv-0", "q0"), _case("locomo:conv-1", "q0"))

    # A clean, uninterrupted run of both cases -- the ground truth to match.
    clean_seen: list[httpx.Request] = []
    clean_output = tmp_path / "clean.jsonl"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_handler(clean_seen)), base_url="http://test"
    ) as client:
        await _run(
            client,
            BENCHMARKS["locomo"],
            cases,
            run_id="run-1",
            benchmark="locomo",
            top_k=10,
            concurrency=2,
            output_path=clean_output,
            tenant_prefix="bench_aml",
        )
    expected_rows = [json.loads(line) for line in clean_output.read_text().splitlines()]

    # Simulate a run interrupted after finishing only the first case: only its
    # (real, already-scored) row is on disk already.
    finished_row = next(row for row in expected_rows if row["id"] == "locomo:conv-0#q0")
    output_path.write_text(json.dumps(finished_row) + "\n", encoding="utf-8")

    resumed_seen: list[httpx.Request] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_handler(resumed_seen)), base_url="http://test"
    ) as client:
        await _run(
            client,
            BENCHMARKS["locomo"],
            cases,
            run_id="run-1",
            benchmark="locomo",
            top_k=10,
            concurrency=2,
            output_path=output_path,
            tenant_prefix="bench_aml",
        )

    # The already-finished case must not be re-added or re-searched.
    resumed_user_ids = {request.url.params.get("user_id") for request in resumed_seen}
    resumed_bodies = [json.loads(request.content) for request in resumed_seen if request.content]
    touched_user_ids = {body.get("user_id") for body in resumed_bodies} | resumed_user_ids
    assert eval_user_id("run-1", "locomo", "locomo:conv-0") not in touched_user_ids

    resumed_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert sorted(resumed_rows, key=lambda row: str(row["id"])) == sorted(
        expected_rows, key=lambda row: str(row["id"])
    )
    # No duplicate ids landed in the resumed file.
    assert len(resumed_rows) == len({row["id"] for row in resumed_rows})


def test_manifest_carries_source_deployment_and_tenant_mapping(tmp_path: Path) -> None:
    deployment_path = tmp_path / "deployment.json"
    deployment_path.write_text(
        json.dumps(
            {
                "server_generator": {
                    "plugin": "openai",
                    "distribution": "mindbridge",
                    "version": "0.1.0",
                    "config": {"model_id": "qwen3.8-max"},
                },
                "server_embedder": {
                    "plugin": "openai",
                    "distribution": "mindbridge",
                    "version": "0.1.0",
                    "config": {"space_id": "jina-v5"},
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "rows.jsonl"
    arguments = _Arguments(
        benchmark="locomo",
        dataset_paths=(tmp_path / "locomo10.json",),
        output_path=output_path,
        api_base_url="https://memory.example.test",
        code_revision="mindbridge-commit",
        deployment_config_path=deployment_path,
        run_id="run-1",
        tenant_prefix="bench_aml",
        top_k=10,
        concurrency=2,
    )
    deployment = load_deployment_snapshot(deployment_path)
    tenant_ids = {
        "eval:run-1:locomo:locomo:conv-0": derive_tenant_id(
            "bench_aml", "eval:run-1:locomo:locomo:conv-0"
        )
    }

    _write_manifest(arguments, deployment, tenant_ids)

    manifest_path = sidecar_manifest_path(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_repository"] == "AML-memory/agent-memory-leaderboard"
    assert manifest["source_revision"] == "5761ed58502d24153115cbdc010e44957cb18c3a"
    assert len(manifest["source_sha256"]) == 64
    assert manifest["code_revision"] == "mindbridge-commit"
    assert manifest["deployment"]["server_generator"]["plugin"] == "openai"
    assert manifest["run_id"] == "run-1"
    assert manifest["tenant_prefix"] == "bench_aml"
    assert manifest["recall_limit"] == 10
    assert manifest["request_concurrency"] == 2
    assert manifest["tenant_ids"] == tenant_ids
