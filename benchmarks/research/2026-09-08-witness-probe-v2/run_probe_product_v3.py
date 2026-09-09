"""Run the frozen witness probe with one isolated store per case and source arm."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = Path(__file__).resolve().parent
MANIFEST = ARTIFACT / "manifest.json"
AMENDMENT = ARTIFACT / "amendment.json"
OUTPUT = ROOT / ".benchmarks/results/e2e-witness-probe-v2"
DATA_ROOT = ROOT / ".benchmarks/data/e2e-witness-probe-v2"
REFERENCE_AT = datetime.fromisoformat("2026-09-08T00:00:00+00:00")
EXPECTED_MANIFEST_SHA256 = "0f784891d55bc20b47a3830a8cd79955a050474d8cb154f1727e3ac88db6c8fa"
EXPECTED_AMENDMENT_SHA256 = "c5453d3d12d368698e89fda68af77c2fc66ed42969b36d1e78bac16da92d8409"
SOURCE_ARMS = {
    "pre_witness": {
        "root": ROOT / ".benchmarks/research/2026-09-08-pre-witness-product-snapshot/extracted/src",
        "archive": ROOT
        / ".benchmarks/research/2026-09-08-pre-witness-product-snapshot/working-tree.tar.gz",
        "archive_sha256": "fd70d57c821d00807b85de4640507235a72212950e36fcb4221ca4f304957d84",
    },
    "witness": {
        "root": ROOT
        / ".benchmarks/research/2026-09-08-witness-schema17-product-snapshot-v2/extracted/src",
        "archive": ROOT
        / ".benchmarks/research/2026-09-08-witness-schema17-product-snapshot-v2/working-tree.tar.gz",
        "archive_sha256": "eaf593a5493dbe67d79d6b8dcb27f070e2f71608783f5ab31caf5c8cb75e9810",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_frozen_inputs() -> dict[str, object]:
    actual_manifest = _sha256(MANIFEST)
    actual_amendment = _sha256(AMENDMENT)
    if actual_manifest != EXPECTED_MANIFEST_SHA256:
        raise SystemExit(f"manifest changed: {actual_manifest}")
    if actual_amendment != EXPECTED_AMENDMENT_SHA256:
        raise SystemExit(f"amendment changed: {actual_amendment}")
    for arm, values in SOURCE_ARMS.items():
        archive = values["archive"]
        expected = values["archive_sha256"]
        assert isinstance(archive, Path) and isinstance(expected, str)
        actual = _sha256(archive)
        if actual != expected:
            raise SystemExit(f"{arm} archive changed: {actual}")
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if document["protocol"]["arms"] != ["pre_witness", "witness"]:
        raise SystemExit("unexpected arms")
    cases = document["cases"]
    if len(cases) != 8 or len({case["id"] for case in cases}) != 8:
        raise SystemExit("expected eight unique cases")
    return document


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _rows(database: sqlite3.Connection, table: str) -> list[dict[str, object]]:
    cursor = database.execute(f"SELECT * FROM {table}")
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, (_json_safe(value) for value in row), strict=True)) for row in cursor]


def _stored_state(data_dir: Path) -> dict[str, object]:
    database = sqlite3.connect(f"file:{data_dir / 'state.sqlite3'}?mode=ro", uri=True)
    try:
        tables = {
            str(row[0])
            for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        wanted = (
            "memory_records",
            "memory_semantics",
            "memory_versions",
            "memory_evidence",
            "memory_evidence_clauses",
            "memory_evidence_clause_members",
            "memory_evidence_clause_versions",
            "formation_runs",
        )
        return {
            "schema_version": int(database.execute("PRAGMA user_version").fetchone()[0]),
            "integrity_check": str(database.execute("PRAGMA integrity_check").fetchone()[0]),
            "tables": {table: _rows(database, table) for table in wanted if table in tables},
        }
    finally:
        database.close()


def _hit(hit: object) -> dict[str, object]:
    context = getattr(hit, "context", None)
    return {
        "id": hit.id,
        "content": hit.content,
        "score": hit.score,
        "memory_type": hit.memory_type.value,
        "occurred_at": (None if hit.occurred_at is None else hit.occurred_at.isoformat()),
        "kind": None if context is None else context.kind.value,
        "basis": None if context is None else context.basis.value,
        "evidence_ids": [] if context is None else list(context.evidence_ids),
        "confidence": None if context is None else context.confidence,
    }


def _answer(result: object) -> dict[str, object]:
    reason = result.abstention_reason
    return {
        "answer": result.answer,
        "abstained": result.abstained,
        "abstention_reason": None if reason is None else reason.value,
        "cited_hits": [_hit(hit) for hit in result.hits],
    }


async def _worker(arm: str, case_id: str) -> None:
    import httpx
    from openai import OpenAI

    import mindbridge
    from mindbridge import (
        AsyncMemory,
        ContextBudget,
        MemoryConfig,
        MemoryPlugins,
        MemoryType,
        Modality,
        OpenAIModels,
    )

    document = _validate_frozen_inputs()
    case = next((value for value in document["cases"] if value["id"] == case_id), None)
    if case is None:
        raise SystemExit(f"unknown case: {case_id}")
    expected_root = Path(SOURCE_ARMS[arm]["root"]).resolve()  # noqa: ASYNC240
    imported = Path(mindbridge.__file__).resolve()  # noqa: ASYNC240
    if expected_root not in imported.parents:
        raise SystemExit(f"wrong import root: {imported}; expected {expected_root}")

    data_dir = DATA_ROOT / arm / case_id
    result_path = OUTPUT / "cases" / arm / f"{case_id}.json"
    transport_path = OUTPUT / "transport" / arm / f"{case_id}.jsonl"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return
        raise SystemExit(f"existing incomplete result requires inspection: {result_path}")
    if data_dir.exists():
        raise SystemExit(f"data directory already exists without a complete result: {data_dir}")

    class CaptureTransport(httpx.BaseTransport):
        def __init__(self, label: str) -> None:
            self.label = label
            self.inner = httpx.HTTPTransport()
            self.ordinal = 0

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            self.ordinal += 1
            request_body = request.read()
            response = self.inner.handle_request(request)
            response_body = response.read()
            captured = {
                "at": datetime.now(timezone.utc).isoformat(),
                "client": self.label,
                "ordinal": self.ordinal,
                "method": request.method,
                "url": str(request.url),
                "request_sha256": hashlib.sha256(request_body).hexdigest(),
                "request_body": request_body.decode("utf-8"),
                "status_code": response.status_code,
                "response_sha256": hashlib.sha256(response_body).hexdigest(),
                "response_body": response_body.decode("utf-8", errors="replace"),
            }
            _append_jsonl(transport_path, captured)
            headers = response.headers
            extensions = response.extensions
            response.close()
            return httpx.Response(
                status_code=captured["status_code"],
                headers=headers,
                content=response_body,
                request=request,
                extensions=extensions,
            )

        def close(self) -> None:
            self.inner.close()

    embedding_http = httpx.Client(transport=CaptureTransport("embedding"), timeout=300)
    generation_http = httpx.Client(transport=CaptureTransport("generation"), timeout=300)
    embedding_client = OpenAI(
        api_key="EMPTY",
        base_url="https://xyrobot-embed.xyrobot.com/v1",
        max_retries=3,
        timeout=300,
        http_client=embedding_http,
    )
    generation_client = OpenAI(
        api_key="EMPTY",
        base_url="http://xyrobot-vl.xyrobot.com/v1",
        max_retries=3,
        timeout=300,
        http_client=generation_http,
    )
    models = OpenAIModels(
        embedding_client=embedding_client,
        generation_client=generation_client,
        embedding_model="tencent/WeMM-Embedding-2B",
        embedding_dimension=2048,
        embedding_request_format="messages",
        generation_model="Qwen3.8-27B",
        embedding_capabilities=frozenset({Modality.TEXT}),
        generation_capabilities=frozenset({Modality.TEXT}),
        generation_seed=42,
        generation_temperature=0,
        generation_max_tokens=8192,
        generation_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        generation_stream=True,
    )
    plugins = MemoryPlugins(embedder=models, answerer=models, former=models)
    config = MemoryConfig(reinforce_on_answer=False)
    observations = tuple(str(value) for value in case["observations"])
    started = datetime.now(timezone.utc)
    try:
        async with AsyncMemory.from_plugins(data_dir, plugins=plugins, config=config) as memory:
            records = await memory.add_many(
                observations,
                occurred_at=[REFERENCE_AT] * len(observations),
                metadata=[{"probe_observation": index} for index in range(len(observations))],
                memory_type=MemoryType.EPISODIC,
            )
            raw_answer = await memory.ask(
                str(case["question"]),
                limit=12,
                reference_at=REFERENCE_AT,
                link_identities=False,
            )
            bundle = await memory.compile(
                str(case["question"]),
                budget=ContextBudget(max_items=12, max_chars=8000),
                reference_at=REFERENCE_AT,
            )
            compiled_answer = await asyncio.to_thread(
                models.answer,
                f"{case['question']}\n[Reference time: {REFERENCE_AT.isoformat()}]",
                bundle.hits,
            )
            runtime = {
                "status": "complete",
                "started_at": started.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "arm": arm,
                "case_id": case_id,
                "question": case["question"],
                "observations": observations,
                "target_dependencies": case["target_dependencies"],
                "source_record_ids": [record.id for record in records],
                "raw_ask": _answer(raw_answer),
                "compile": {
                    "answer": _answer(compiled_answer),
                    "bundle_hits": [_hit(hit) for hit in bundle.hits],
                    "bundle_rendered": bundle.render(),
                    "chars": bundle.chars,
                    "omitted": bundle.omitted,
                    "unknowns": [
                        {"kind": item.kind.value, "detail": item.detail} for item in bundle.unknowns
                    ],
                },
                "data_dir": str(data_dir.resolve()),
                "source_root": str(expected_root),
                "source_archive_sha256": SOURCE_ARMS[arm]["archive_sha256"],
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "amendment_sha256": EXPECTED_AMENDMENT_SHA256,
                "runner_sha256": _sha256(Path(__file__)),
                "import_path": str(imported),
                "transport_journal": str(transport_path.resolve()),
            }
        runtime["store"] = _stored_state(data_dir)
        runtime["transport_journal_sha256"] = _sha256(transport_path)
        _write_json(result_path, runtime)
    except BaseException as error:
        _write_json(
            result_path.with_name(f"{case_id}.failure.json"),
            {
                "status": "failed",
                "at": datetime.now(timezone.utc).isoformat(),
                "arm": arm,
                "case_id": case_id,
                "error_type": type(error).__name__,
                "error": " ".join(str(error).split())[:2000],
                "transport_journal": str(transport_path.resolve()),
            },
        )
        raise


def _supervise() -> None:
    document = _validate_frozen_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not DATA_ROOT.exists():
        DATA_ROOT.mkdir(parents=True)
    identities = [
        (arm, str(case["id"])) for case in document["cases"] for arm in document["protocol"]["arms"]
    ]
    _write_json(
        OUTPUT / "runtime-manifest.json",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "manifest": str(MANIFEST.resolve()),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "amendment": str(AMENDMENT.resolve()),
            "amendment_sha256": EXPECTED_AMENDMENT_SHA256,
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__)),
            "python": sys.executable,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "source_arms": {
                arm: {key: str(value) for key, value in values.items()}
                for arm, values in SOURCE_ARMS.items()
            },
            "planned_identities": [f"{arm}:{case_id}" for arm, case_id in identities],
        },
    )
    for arm, case_id in identities:
        result_path = OUTPUT / "cases" / arm / f"{case_id}.json"
        if (
            result_path.exists()
            and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete"
        ):
            continue
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE_ARMS[arm]["root"])
        environment["PYTHONHASHSEED"] = "42"
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "worker", arm, case_id],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        _append_jsonl(
            OUTPUT / "supervisor.jsonl",
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "arm": arm,
                "case_id": case_id,
                "returncode": process.returncode,
            },
        )
        if process.returncode != 0:
            raise SystemExit(process.returncode)
    results = [
        json.loads((OUTPUT / "cases" / arm / f"{case_id}.json").read_text(encoding="utf-8"))
        for arm, case_id in identities
    ]
    _write_json(
        OUTPUT / "summary.json",
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "planned": len(identities),
            "completed": len(results),
            "by_arm": {arm: sum(result["arm"] == arm for result in results) for arm in SOURCE_ARMS},
            "result_sha256": {
                f"{arm}:{case_id}": _sha256(OUTPUT / "cases" / arm / f"{case_id}.json")
                for arm, case_id in identities
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("run")
    worker = subparsers.add_parser("worker")
    worker.add_argument("arm", choices=tuple(SOURCE_ARMS))
    worker.add_argument("case_id")
    arguments = parser.parse_args()
    if arguments.command == "validate":
        print(json.dumps(_validate_frozen_inputs(), ensure_ascii=False, sort_keys=True))
    elif arguments.command == "run":
        _supervise()
    else:
        asyncio.run(_worker(arguments.arm, arguments.case_id))


if __name__ == "__main__":
    main()
