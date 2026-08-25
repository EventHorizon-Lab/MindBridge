"""AML CLI wiring, manifest, and resumability tests."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from mindbridge.api.aml_contracts import derive_tenant_id
from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion
from mindbridge.benchmarks.aml.cli import (
    BENCHMARKS,
    _Arguments,
    _require_compatible_existing_manifest,
    _require_nonempty_cases,
    _run,
    _write_manifest,
)
from mindbridge.benchmarks.aml.driver import eval_user_id
from mindbridge.benchmarks.artifacts import load_deployment_snapshot, sidecar_manifest_path


def test_every_benchmark_names_a_loader_an_emitter_and_a_pipeline() -> None:
    assert set(BENCHMARKS) == {
        "locomo-refined",
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


def _case(user_id: str, question_id: str, payload: dict[str, object] | None = None) -> AmlCase:
    return AmlCase(
        user_id=user_id,
        messages=({"role": "user", "content": f"{user_id} said hello"},),
        questions=(
            AmlQuestion(question_id=question_id, question="What happened?", payload=payload or {}),
        ),
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
    cases = (_case("locomo-refined:conv-0", "q0"), _case("locomo-refined:conv-1", "q0"))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_handler(seen)), base_url="http://test"
    ) as client:
        tenant_ids = await _run(
            client,
            BENCHMARKS["locomo-refined"],
            cases,
            run_id="run-1",
            benchmark="locomo-refined",
            top_k=10,
            concurrency=2,
            output_path=output_path,
            tenant_prefix="bench_aml",
        )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert {row["id"] for row in rows} == {"locomo-refined:conv-0#q0", "locomo-refined:conv-1#q0"}
    assert set(tenant_ids) == {
        eval_user_id("run-1", "locomo-refined", "locomo-refined:conv-0"),
        eval_user_id("run-1", "locomo-refined", "locomo-refined:conv-1"),
    }
    for user_id, tenant_id in tenant_ids.items():
        assert tenant_id == derive_tenant_id("bench_aml", user_id)


@pytest.mark.asyncio
async def test_run_checkpoints_early_cases_instead_of_finishing_them_all_at_once(
    tmp_path: Path,
) -> None:
    """A kill mid-run has to leave finished cases behind, which needs a bound on cases."""
    # run_case gathers every add before issuing a search, so with only a request bound each
    # case's adds queue ahead of every case's searches and nothing completes until nearly the
    # whole add phase is done. Recording when the first search is issued measures exactly
    # that: with a case bound, early cases have already finished and been written by then.
    output_path = tmp_path / "rows.jsonl"
    seen: list[httpx.Request] = []
    adds_before_first_search: list[int] = []

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
        if not adds_before_first_search:
            adds_before_first_search.append(sum(1 for item in seen if item.url.path == "/aml/add"))
        return httpx.Response(200, json={"data": [{"id": "mem_1", "content": "hi"}]})

    cases = tuple(_case(f"locomo-refined:conv-{index}", "q0") for index in range(8))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as client:
        await _run(
            client,
            BENCHMARKS["locomo-refined"],
            cases,
            run_id="run-1",
            benchmark="locomo-refined",
            top_k=10,
            concurrency=2,
            output_path=output_path,
            tenant_prefix="bench_aml",
        )

    # Two cases in flight means at most two adds land before the first search. Unbounded, all
    # eight would, and no row could be written until then.
    assert adds_before_first_search == [2]
    assert len(output_path.read_text().splitlines()) == 8


@pytest.mark.parametrize(
    ("benchmark", "payload"),
    [
        # The empty payload keeps the driver's own
        # `{case.user_id}#{question.question_id}` id, which is what a loader that
        # sets no "id" leaves behind. Every real loader now sets one (see
        # loaders/*.py), and it overwrites the driver's id when `run_case` builds a
        # row (`row.update(question.payload)`) -- Blocking 2 (final review,
        # 2026-08-17): the CLI's resume check computed pending work from the
        # driver's id format alone, which never matches what those loaders actually
        # write, so a resumed run treated every one of their cases as unfinished and
        # silently doubled every memory. Both id sources stay covered here.
        ("locomo-refined", {}),
        ("longmemeval", {"id": "own-id"}),
        ("beam", {"id": "own-id"}),
        ("clbench", {"id": "own-id"}),
        ("personamem-v1", {"id": "own-id"}),
        ("personamem-v2", {"id": "own-id"}),
    ],
)
@pytest.mark.asyncio
async def test_run_resumes_without_recontacting_finished_cases_or_duplicating_rows(
    tmp_path: Path, benchmark: str, payload: dict[str, object]
) -> None:
    output_path = tmp_path / "rows.jsonl"
    cases = (
        _case(f"{benchmark}:conv-0", "q0", payload),
        _case(f"{benchmark}:conv-1", "q0", payload),
    )
    spec = BENCHMARKS[benchmark]

    # A clean, uninterrupted run of both cases -- the ground truth to match.
    clean_seen: list[httpx.Request] = []
    clean_output = tmp_path / "clean.jsonl"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_handler(clean_seen)), base_url="http://test"
    ) as client:
        await _run(
            client,
            spec,
            cases,
            run_id="run-1",
            benchmark=benchmark,
            top_k=10,
            concurrency=2,
            output_path=clean_output,
            tenant_prefix="bench_aml",
        )
    expected_rows = [json.loads(line) for line in clean_output.read_text().splitlines()]

    # Simulate a run interrupted after finishing only the first case: only its
    # (real, already-scored) row is on disk already. The row's actual id is
    # whatever `run_case` wrote -- payload["id"] when the loader supplies one,
    # else the driver's own `{case.user_id}#{question.question_id}` -- not
    # necessarily the latter, which is the whole point of this test.
    first_case, first_question = cases[0], cases[0].questions[0]
    finished_id = str(payload.get("id") or f"{first_case.user_id}#{first_question.question_id}")
    finished_row = next(row for row in expected_rows if row["id"] == finished_id)
    output_path.write_text(json.dumps(finished_row) + "\n", encoding="utf-8")

    resumed_seen: list[httpx.Request] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_handler(resumed_seen)), base_url="http://test"
    ) as client:
        await _run(
            client,
            spec,
            cases,
            run_id="run-1",
            benchmark=benchmark,
            top_k=10,
            concurrency=2,
            output_path=output_path,
            tenant_prefix="bench_aml",
        )

    # The already-finished case must not be re-added or re-searched.
    resumed_user_ids = {request.url.params.get("user_id") for request in resumed_seen}
    resumed_bodies = [json.loads(request.content) for request in resumed_seen if request.content]
    touched_user_ids = {body.get("user_id") for body in resumed_bodies} | resumed_user_ids
    assert eval_user_id("run-1", benchmark, first_case.user_id) not in touched_user_ids

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
    # One row on disk -- `question_count` must be positive for a manifest to
    # validate at all (Important 2, task-13 review), so this run "produced"
    # exactly one question.
    output_path.write_text(
        json.dumps({"id": "locomo-refined:conv-0#q0", "question": "q0"}) + "\n",
        encoding="utf-8",
    )
    arguments = _Arguments(
        benchmark="locomo-refined",
        dataset_paths=(tmp_path / "locomo_refined.json",),
        output_path=output_path,
        api_base_url="https://memory.example.test",
        deployment_config_path=deployment_path,
        run_id="run-1",
        tenant_prefix="bench_aml",
        top_k=10,
        concurrency=2,
        request_timeout_seconds=600.0,
        quiet=True,
    )
    deployment = load_deployment_snapshot(deployment_path)
    tenant_ids = {
        "eval:run-1:locomo-refined:locomo-refined:conv-0": derive_tenant_id(
            "bench_aml", "eval:run-1:locomo-refined:locomo-refined:conv-0"
        )
    }

    _write_manifest(arguments, deployment, tenant_ids)

    manifest_path = sidecar_manifest_path(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_repository"] == "AML-memory/agent-memory-leaderboard"
    assert len(manifest["source_sha256"]) == 64
    assert manifest["deployment"]["server_generator"]["plugin"] == "openai"
    assert manifest["run_id"] == "run-1"
    assert manifest["tenant_prefix"] == "bench_aml"
    assert manifest["recall_limit"] == 10
    assert manifest["request_concurrency"] == 2
    assert manifest["client_derived_tenant_ids"] == tenant_ids
    assert manifest["question_count"] == 1
    assert manifest["oversized_unsliced_question_count"] == 0
    assert manifest["deployment_sha256"] == deployment.sha256
    assert manifest["request_timeout_seconds"] == 600.0
    assert len(manifest["predictions_sha256"]) == 64


def test_write_manifest_counts_oversized_unsliced_questions_from_output_rows(
    tmp_path: Path,
) -> None:
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
    # Three rows already on disk -- as if this run resumed from a prior,
    # partially-completed invocation -- two of which carry CL-Bench's
    # `question_unsliced` marker.
    output_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "clbench:task-1#task-1", "question": "q1", "question_unsliced": False},
                {"id": "clbench:task-2#task-2", "question": "q2", "question_unsliced": True},
                {"id": "clbench:task-3#task-3", "question": "q3", "question_unsliced": True},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    arguments = _Arguments(
        benchmark="clbench",
        dataset_paths=(tmp_path / "CL-bench.jsonl",),
        output_path=output_path,
        api_base_url="https://memory.example.test",
        deployment_config_path=deployment_path,
        run_id="run-1",
        tenant_prefix="bench_aml",
        top_k=10,
        concurrency=2,
        request_timeout_seconds=600.0,
        quiet=True,
    )
    deployment = load_deployment_snapshot(deployment_path)

    _write_manifest(arguments, deployment, {})

    manifest = json.loads(sidecar_manifest_path(output_path).read_text(encoding="utf-8"))
    assert manifest["question_count"] == 3
    assert manifest["oversized_unsliced_question_count"] == 2


def _deployment_path(tmp_path: Path) -> Path:
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
    return deployment_path


def _arguments(
    tmp_path: Path, output_path: Path, deployment_path: Path, **overrides: object
) -> _Arguments:
    fields: dict[str, object] = {
        "benchmark": "locomo-refined",
        "dataset_paths": (tmp_path / "locomo_refined.json",),
        "output_path": output_path,
        "api_base_url": "https://memory.example.test",
        "deployment_config_path": deployment_path,
        "run_id": "run-1",
        "tenant_prefix": "bench_aml",
        "top_k": 10,
        "concurrency": 2,
        "request_timeout_seconds": 600.0,
        "quiet": True,
    }
    fields.update(overrides)
    return _Arguments(**fields)  # type: ignore[arg-type]


def test_require_compatible_existing_manifest_allows_a_first_run(tmp_path: Path) -> None:
    """No sidecar manifest yet -- nothing to disagree with."""
    deployment = load_deployment_snapshot(_deployment_path(tmp_path))

    _require_compatible_existing_manifest(
        tmp_path / "rows.jsonl",
        benchmark="locomo-refined",
        run_id="run-1",
        deployment=deployment.snapshot,
        recall_limit=10,
    )


def test_require_compatible_existing_manifest_allows_a_matching_resume(tmp_path: Path) -> None:
    """Same benchmark/run_id/deployment/recall_limit as the existing sidecar
    -- this is a legitimate resume of an interrupted run and must proceed."""
    deployment_path = _deployment_path(tmp_path)
    deployment = load_deployment_snapshot(deployment_path)
    output_path = tmp_path / "rows.jsonl"
    output_path.write_text(json.dumps({"id": "locomo-refined:conv-0#q0", "question": "q0"}) + "\n")
    arguments = _arguments(tmp_path, output_path, deployment_path)
    _write_manifest(arguments, deployment, {})

    _require_compatible_existing_manifest(
        output_path,
        benchmark="locomo-refined",
        run_id="run-1",
        deployment=deployment.snapshot,
        recall_limit=10,
    )


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("run_id", {"run_id": "run-2"}),
        ("benchmark", {"benchmark": "longmemeval"}),
        ("recall_limit", {"top_k": 20}),
    ],
)
def test_require_compatible_existing_manifest_refuses_a_mismatched_resume(
    tmp_path: Path, field: str, overrides: dict[str, object]
) -> None:
    """The exact bug this guard exists for: point a second invocation with a
    different run_id/benchmark/top_k at the same --output. Without the
    guard, `_run` would resume (skip every case as already-done, since row
    ids carry no run identity) and `_write_manifest` would silently restamp
    the sidecar with the new run's provenance -- attributing rows to a run
    that never produced them. As of this test being written, the current
    `main()` has no such guard: this test fails until one exists."""
    deployment_path = _deployment_path(tmp_path)
    deployment = load_deployment_snapshot(deployment_path)
    output_path = tmp_path / "rows.jsonl"
    output_path.write_text(json.dumps({"id": "locomo-refined:conv-0#q0", "question": "q0"}) + "\n")
    first_run = _arguments(tmp_path, output_path, deployment_path)
    _write_manifest(first_run, deployment, {})

    second_run = _arguments(tmp_path, output_path, deployment_path, **overrides)
    with pytest.raises(ValueError, match=field):
        _require_compatible_existing_manifest(
            output_path,
            benchmark=second_run.benchmark,
            run_id=second_run.run_id,
            deployment=deployment.snapshot,
            recall_limit=second_run.top_k,
        )


def test_require_compatible_existing_manifest_refuses_a_different_deployment(
    tmp_path: Path,
) -> None:
    deployment_path = _deployment_path(tmp_path)
    deployment = load_deployment_snapshot(deployment_path)
    output_path = tmp_path / "rows.jsonl"
    output_path.write_text(json.dumps({"id": "locomo-refined:conv-0#q0", "question": "q0"}) + "\n")
    arguments = _arguments(tmp_path, output_path, deployment_path)
    _write_manifest(arguments, deployment, {})

    other_deployment_path = tmp_path / "other-deployment.json"
    other_deployment_path.write_text(
        json.dumps(
            {
                "server_generator": {
                    "plugin": "openai",
                    "distribution": "mindbridge",
                    "version": "0.1.0",
                    "config": {"model_id": "a-different-model"},
                },
                "server_embedder": {
                    "plugin": "openai",
                    "distribution": "mindbridge",
                    "version": "0.1.0",
                    "config": {"space_id": "jina-v5"},
                },
            }
        )
    )
    other_deployment = load_deployment_snapshot(other_deployment_path)

    with pytest.raises(ValueError, match="deployment"):
        _require_compatible_existing_manifest(
            output_path,
            benchmark="locomo-refined",
            run_id="run-1",
            deployment=other_deployment.snapshot,
            recall_limit=10,
        )


def test_require_nonempty_cases_allows_at_least_one_case() -> None:
    cases = (_case("locomo-refined:conv-0", "q0"),)
    _require_nonempty_cases(
        cases, benchmark="locomo-refined", dataset_paths=(Path("locomo_refined.json"),)
    )


def test_require_nonempty_cases_refuses_zero_cases_and_names_the_dataset_path(
    tmp_path: Path,
) -> None:
    """The realistic trigger: a `--dataset` path pointed one directory too
    high or too low (e.g. BEAM's `chats_root.glob("*/*/chat.json")` silently
    yields nothing). Without this guard the CLI goes on to write a
    complete-looking manifest for zero questions. As of this test being
    written, no such guard exists: this test fails until one does."""
    dataset_path = tmp_path / "chats"
    with pytest.raises(ValueError, match=re.escape(str(dataset_path.resolve()))):
        _require_nonempty_cases((), benchmark="beam", dataset_paths=(dataset_path,))


# U+2028 LINE SEPARATOR is legal inside a JSON string and is not a JSON line
# delimiter, but `str.splitlines()` breaks on it -- and it is one of only
# three such characters `json.dumps(ensure_ascii=False)` emits raw (U+2029
# and U+0085 are the others; the rest are control characters below 0x20,
# which JSON requires escaping regardless). The official CL-Bench release
# carries 343 of them, so retrieved context really does contain them.
_SHARD_SEPARATOR = "\u2028"


def _separator_handler(
    seen: list[httpx.Request],
) -> Callable[[httpx.Request], httpx.Response]:
    """Answer recalls with content carrying a raw Unicode line separator."""

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
        return httpx.Response(
            200,
            json={"data": [{"id": "mem_1", "content": f"before{_SHARD_SEPARATOR}after"}]},
        )

    return handle


@pytest.mark.asyncio
async def test_run_writes_shards_that_a_splitlines_reader_cannot_shred(tmp_path: Path) -> None:
    """One row per line however the reader splits, with the separator preserved.

    `docs/benchmarking.md` hands these shards to the vendored scorers, which
    are standalone upstream scripts rather than callers of anything here, and
    third-party tooling reads them too. So the guarantee belongs on the write
    side and cannot be stated as "our readers cope": a shard must survive a
    reader that splits on Unicode line boundaries, which is what `splitlines()`
    does. Escaping costs ~1% on these corpora -- all seven are 0.00% CJK.
    """
    output_path = tmp_path / "rows.jsonl"
    seen: list[httpx.Request] = []
    cases = (_case("locomo-refined:conv-0", "q0"), _case("locomo-refined:conv-1", "q0"))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_separator_handler(seen)), base_url="http://test"
    ) as client:
        await _run(
            client,
            BENCHMARKS["locomo-refined"],
            cases,
            run_id="run-1",
            benchmark="locomo-refined",
            top_k=10,
            concurrency=2,
            output_path=output_path,
            tenant_prefix="bench_aml",
        )

    raw = output_path.read_text(encoding="utf-8")
    newline_rows = [line for line in raw.split("\n") if line]
    assert len(newline_rows) == 2
    assert len(raw.splitlines()) == len(newline_rows), (
        "a splitlines() reader sees more lines than there are records, so the shard "
        "carries a raw separator that shreds a record mid-string"
    )
    # Escaping must not lose the character: it has to survive the round trip.
    decoded = [json.loads(line) for line in newline_rows]
    assert all(_SHARD_SEPARATOR in row["retrieved_context"] for row in decoded), decoded[0]
