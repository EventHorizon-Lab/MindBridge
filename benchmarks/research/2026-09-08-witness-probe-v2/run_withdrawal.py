"""Delete observation zero from resolved probe cases and rerun compiled QA."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / ".benchmarks/results/e2e-witness-probe-v2"
OUTPUT = ROOT / ".benchmarks/results/e2e-witness-probe-withdrawal-v1"
SOURCE_DATA = ROOT / ".benchmarks/data/e2e-witness-probe-v2"
CLONES = ROOT / ".benchmarks/data/e2e-witness-probe-withdrawal-v1"
CACHE = OUTPUT / "shared-embedding-cache.json"
REFERENCE_AT = datetime.fromisoformat("2026-09-08T00:00:00+00:00")
CASES = ("identity", "pet")
SOURCE_ARMS = {
    "pre_witness": ROOT
    / ".benchmarks/research/2026-09-08-pre-witness-product-snapshot/extracted/src",
    "witness": ROOT
    / ".benchmarks/research/2026-09-08-witness-schema17-product-snapshot-v3/extracted/src",
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


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _tree_manifest(root: Path) -> dict[str, object]:
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": len(entries),
        "bytes": sum(int(item["bytes"]) for item in entries),
    }


@contextmanager
def _closed(path: Path) -> Iterator[None]:
    with (path / ".mindbridge.lock").open("rb") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"store is active: {path}") from None
        yield


def _state(data_dir: Path) -> dict[str, object]:
    database = sqlite3.connect(f"file:{data_dir / 'state.sqlite3'}?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    try:
        tables = {
            str(row[0])
            for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        result: dict[str, object] = {
            "schema_version": int(database.execute("PRAGMA user_version").fetchone()[0]),
            "integrity_check": str(database.execute("PRAGMA integrity_check").fetchone()[0]),
        }
        for table in (
            "memory_records",
            "memory_semantics",
            "memory_versions",
            "memory_evidence",
            "memory_evidence_clauses",
            "memory_evidence_clause_members",
            "memory_evidence_clause_versions",
        ):
            if table in tables:
                result[table] = [dict(row) for row in database.execute(f"SELECT * FROM {table}")]
        return result
    finally:
        database.close()


def _hit(hit: object) -> dict[str, object]:
    context = hit.context
    return {
        "id": hit.id,
        "content": hit.content,
        "score": hit.score,
        "kind": None if context is None else context.kind.value,
        "evidence_ids": [] if context is None else list(context.evidence_ids),
    }


def _answer(result: object) -> dict[str, object]:
    return {
        "answer": result.answer,
        "abstained": result.abstained,
        "abstention_reason": (
            None if result.abstention_reason is None else result.abstention_reason.value
        ),
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
        Modality,
        OpenAIModels,
    )

    data_dir = CLONES / arm / case_id
    probe = json.loads((PROBE / "cases" / arm / f"{case_id}.json").read_text(encoding="utf-8"))
    result_path = OUTPUT / "cases" / arm / f"{case_id}.json"
    journal = OUTPUT / "transport" / arm / f"{case_id}.jsonl"
    if (
        result_path.exists()
        and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete"
    ):
        return
    expected = SOURCE_ARMS[arm].resolve()
    imported = Path(mindbridge.__file__).resolve()  # noqa: ASYNC240
    if expected not in imported.parents:
        raise RuntimeError(f"wrong import: {imported}, expected {expected}")

    class Transport(httpx.BaseTransport):
        def __init__(self, client: str) -> None:
            self.client = client
            self.inner = httpx.HTTPTransport()

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            body = request.read()
            key = hashlib.sha256(body).hexdigest()
            cached = None
            if self.client == "embedding" and CACHE.exists():
                cached = json.loads(CACHE.read_text(encoding="utf-8")).get(key)
            if cached is None:
                response = self.inner.handle_request(request)
                response_body = response.read()
                status = response.status_code
                headers = dict(response.headers)
                extensions = response.extensions
                response.close()
                if self.client == "embedding" and 200 <= status < 300:
                    values = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
                    values[key] = {
                        "status": status,
                        "headers": headers,
                        "body": response_body.decode("utf-8"),
                        "response_sha256": hashlib.sha256(response_body).hexdigest(),
                    }
                    _write_json(CACHE, values)
            else:
                status = int(cached["status"])
                headers = cached["headers"]
                response_body = str(cached["body"]).encode()
                extensions = {}
            _append_jsonl(
                journal,
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "client": self.client,
                    "cache_hit": cached is not None,
                    "request_sha256": key,
                    "request_body": body.decode("utf-8"),
                    "status": status,
                    "response_sha256": hashlib.sha256(response_body).hexdigest(),
                    "response_body": response_body.decode("utf-8", errors="replace"),
                },
            )
            return httpx.Response(
                status,
                headers=headers,
                content=response_body,
                request=request,
                extensions=extensions,
            )

        def close(self) -> None:
            self.inner.close()

    embedding_http = httpx.Client(transport=Transport("embedding"), timeout=300)
    generation_http = httpx.Client(transport=Transport("generation"), timeout=300)
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
    before = _state(data_dir)
    async with AsyncMemory.from_plugins(
        data_dir,
        plugins=MemoryPlugins(embedder=models, answerer=models),
        config=MemoryConfig(reinforce_on_answer=False),
    ) as memory:
        deleted_id = probe["source_record_ids"][0]
        deleted = await memory.delete(deleted_id)
        bundle = await memory.compile(
            probe["question"],
            budget=ContextBudget(max_items=12, max_chars=8000),
            reference_at=REFERENCE_AT,
        )
        answer = await asyncio.to_thread(
            models.answer,
            f"{probe['question']}\n[Reference time: {REFERENCE_AT.isoformat()}]",
            bundle.hits,
        )
    after = _state(data_dir)
    _write_json(
        result_path,
        {
            "status": "complete",
            "arm": arm,
            "case_id": case_id,
            "question": probe["question"],
            "deleted_observation_index": 0,
            "deleted_memory_id": deleted_id,
            "delete_returned": deleted,
            "source_record_ids": probe["source_record_ids"],
            "bundle_hits": [_hit(hit) for hit in bundle.hits],
            "bundle_rendered": bundle.render(),
            "answer": _answer(answer),
            "store_before": before,
            "store_after": after,
            "source_root": str(expected),
            "import_path": str(imported),
            "runner_sha256": _sha256(Path(__file__)),
            "probe_result_sha256": _sha256(PROBE / "cases" / arm / f"{case_id}.json"),
            "transport_sha256": _sha256(journal),
        },
    )


def _prepare() -> None:
    if OUTPUT.exists() or CLONES.exists():
        raise SystemExit("withdrawal output or clone root already exists")
    OUTPUT.mkdir(parents=True)
    CLONES.mkdir(parents=True)
    source_manifest: dict[str, object] = {}
    for arm in SOURCE_ARMS:
        for case_id in CASES:
            source = SOURCE_DATA / arm / case_id
            destination = CLONES / arm / case_id
            with _closed(source):
                before = _tree_manifest(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination, copy_function=shutil.copy2)
                after = _tree_manifest(source)
            copied = _tree_manifest(destination)
            if before != after or copied != before:
                raise RuntimeError(f"non-isolated clone: {arm}/{case_id}")
            source_manifest[f"{arm}:{case_id}"] = {
                "source": str(source),
                "destination": str(destination),
                "content": before,
            }
    _write_json(OUTPUT / "source-store-manifest.json", source_manifest)
    _write_json(CACHE, {})


def _supervise() -> None:
    _prepare()
    identities = [(arm, case_id) for case_id in CASES for arm in SOURCE_ARMS]
    _write_json(
        OUTPUT / "runtime-manifest.json",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": _sha256(Path(__file__)),
            "cases": list(CASES),
            "arms": list(SOURCE_ARMS),
            "reference_at": REFERENCE_AT.isoformat(),
            "budget": {"max_items": 12, "max_chars": 8000},
            "rule": "all probe cases that actually produced a resolved cross-input target claim",
        },
    )
    for arm, case_id in identities:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE_ARMS[arm])
        environment["PYTHONHASHSEED"] = "42"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "worker", arm, case_id],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        _append_jsonl(
            OUTPUT / "supervisor.jsonl",
            {"arm": arm, "case_id": case_id, "returncode": process.returncode},
        )
        if process.returncode:
            raise SystemExit(process.returncode)
    results = [
        json.loads((OUTPUT / "cases" / arm / f"{case_id}.json").read_text(encoding="utf-8"))
        for arm, case_id in identities
    ]
    _write_json(
        OUTPUT / "summary.json",
        {
            "status": "complete",
            "completed": len(results),
            "embedding_cache_entries": len(json.loads(CACHE.read_text(encoding="utf-8"))),
            "answers": {f"{value['arm']}:{value['case_id']}": value["answer"] for value in results},
            "result_sha256": {
                f"{arm}:{case_id}": _sha256(OUTPUT / "cases" / arm / f"{case_id}.json")
                for arm, case_id in identities
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("arm", choices=tuple(SOURCE_ARMS), nargs="?")
    parser.add_argument("case_id", choices=CASES, nargs="?")
    arguments = parser.parse_args()
    if arguments.command == "run":
        _supervise()
    else:
        if arguments.arm is None or arguments.case_id is None:
            parser.error("worker requires arm and case_id")
        asyncio.run(_worker(arguments.arm, arguments.case_id))


if __name__ == "__main__":
    main()
