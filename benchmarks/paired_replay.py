#!/usr/bin/env python3
"""Run an exact, no-generation replay against two frozen MindBridge source trees.

The controller verifies that source stores are closed, copies each unit into a distinct physical
directory per arm, and launches a fresh interpreter with that arm's source tree first on
``PYTHONPATH``.  The worker compiles through the public ``AsyncMemory`` API and sends the final
generation call through the real OpenAI SDK into an in-process capture transport.  No generation
request reaches the network.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import fcntl
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
ARM_NAMES = ("baseline", "candidate")
EMBEDDING_PROXY_ENV = "MINDBRIDGE_REPLAY_EMBEDDING_PROXY"


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, object]:
    """Describe a store by content, independent of its clone path and file timestamps."""
    root = root.resolve()
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    payload = _json_bytes(entries)
    return {
        "sha256": _sha256_bytes(payload),
        "files": len(entries),
        "bytes": sum(int(x["bytes"]) for x in entries),
    }


def _safe_component(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty and trimmed")
    encoded = base64.b32encode(value.encode("utf-8")).decode("ascii").rstrip("=").lower()
    return f"{label}-{encoded}"


@contextmanager
def _closed_store_lock(path: Path) -> Iterator[None]:
    if not path.is_dir() or not (path / "state.sqlite3").is_file():
        raise FileNotFoundError(f"source store is missing state.sqlite3: {path}")
    lock_path = path / ".mindbridge.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(f"source store is missing its owner lock: {lock_path}")
    with lock_path.open("rb") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"source store is still owned: {path}") from None
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _assert_closed_store(path: Path) -> None:
    with _closed_store_lock(path):
        pass


def _load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"replay plan must use schema_version {SCHEMA_VERSION}")
    for key in (
        "name",
        "task",
        "benchmarks_root",
        "config",
        "selection",
        "clone_root",
        "arms",
    ):
        if key not in value:
            raise ValueError(f"replay plan is missing {key}")
    if "source_run" not in value and "source_stores" not in value:
        raise ValueError("replay plan is missing source_run or source_stores")
    if set(value["arms"]) != set(ARM_NAMES):
        raise ValueError("replay plan must contain exactly baseline and candidate arms")
    _replay_mode(value)
    if str(value.get("compile_query", "question_content")) not in {
        "question_content",
        "question_text",
        "source_question",
    }:
        raise ValueError(
            "compile_query must be question_content, question_text, or source_question"
        )
    if str(value.get("embedding_cache_network", "allow")) not in {"allow", "forbid"}:
        raise ValueError("embedding_cache_network must be allow or forbid")
    return value


def _replay_mode(plan: Mapping[str, Any]) -> str:
    mode = str(plan.get("mode", "compile"))
    if mode not in {"compile", "blind"}:
        raise ValueError("replay plan mode must be compile or blind")
    return mode


def _compile_query(plan: Mapping[str, Any], question: Any) -> Any:  # noqa: ANN401
    """Return the frozen public ``compile`` input for this replay."""
    mode = str(plan.get("compile_query", "question_content"))
    content = question.content
    if mode == "question_content":
        return content[0] if len(content) == 1 else content
    if mode == "question_text":
        text = "\n".join(str(part) for part in content if not isinstance(part, Path))
        if not text:
            raise ValueError("question_text compile input requires at least one text part")
        return text
    if mode == "source_question":
        source_question = question.source_question
        if not isinstance(source_question, str) or not source_question.strip():
            raise ValueError("source_question compile input requires a frozen source question")
        return source_question
    raise ValueError("compile_query must be question_content, question_text, or source_question")


def _compile_scope(plan: Mapping[str, Any], reference_at: datetime) -> Any:  # noqa: ANN401
    """Build the explicitly frozen public retrieval scope, or preserve the SDK default."""
    declared = plan.get("scope")
    if declared is None:
        return None
    if not isinstance(declared, Mapping):
        raise ValueError("replay plan scope must be an object")
    supported = {"valid_at", "known_at", "place_id", "identity_id"}
    unknown = set(declared) - supported
    if unknown:
        raise ValueError(f"replay plan scope has unsupported fields: {', '.join(sorted(unknown))}")

    from mindbridge import RetrievalScope

    def timestamp(name: str) -> datetime | None:
        value = declared.get(name)
        if value is None:
            return None
        if value == "reference_at":
            return reference_at
        if not isinstance(value, str):
            raise ValueError(f"replay plan scope {name} must be an ISO timestamp")
        return datetime.fromisoformat(value)

    return RetrievalScope(
        valid_at=timestamp("valid_at"),
        known_at=timestamp("known_at"),
        place_id=declared.get("place_id"),
        identity_id=declared.get("identity_id"),
    )


def _scope_record(scope: object | None) -> dict[str, object]:
    """Serialize exactly the scope supplied to ``AsyncMemory.compile``."""
    typed_scope = cast(Any, scope)
    return {
        "valid_at": (
            None
            if typed_scope is None or typed_scope.valid_at is None
            else typed_scope.valid_at.isoformat()
        ),
        "known_at": (
            None
            if typed_scope is None or typed_scope.known_at is None
            else typed_scope.known_at.isoformat()
        ),
        "place_id": None if typed_scope is None else typed_scope.place_id,
        "identity_id": None if typed_scope is None else typed_scope.identity_id,
    }


def _load_task(plan: Mapping[str, Any]) -> Any:  # noqa: ANN401
    from mindbridge.benchmarks.eval_adapters import load_task
    from mindbridge.benchmarks.task_catalog import TASKS

    selection = plan["selection"]
    task_name = str(plan["task"])
    return load_task(
        TASKS[task_name],
        root=Path(plan["benchmarks_root"]),
        limit=selection["limit"],
        offset=selection["offset"],
    )


def _selected_units(plan: Mapping[str, Any]) -> tuple[str, ...]:
    task = _load_task(plan)
    units = tuple(unit.unit_id for unit in task.units)
    questions = sum(len(unit.questions) for unit in task.units)
    selection = plan["selection"]
    if len(units) != selection["expected_units"] or questions != selection["expected_questions"]:
        raise RuntimeError(
            f"frozen selection mismatch: got {len(units)} units/{questions} questions, "
            f"expected {selection['expected_units']}/{selection['expected_questions']}"
        )
    roster = [
        {
            "unit_id": unit.unit_id,
            "question_ids": [question.question_id for question in unit.questions],
        }
        for unit in task.units
    ]
    roster_sha256 = _sha256_bytes(_json_bytes(roster))
    if roster_sha256 != selection["roster_sha256"]:
        raise RuntimeError(
            f"frozen selection roster mismatch: got {roster_sha256}, "
            f"expected {selection['roster_sha256']}"
        )
    if expected_source_roster := selection.get("source_roster_sha256"):
        source_rosters = [
            _sha256_bytes(
                json.dumps(
                    [
                        {
                            "source_id": item.source_id,
                            "content": [str(part) for part in item.content],
                            "start_seconds": item.start_seconds,
                            "end_seconds": item.end_seconds,
                            "occurred_at": (
                                None if item.occurred_at is None else item.occurred_at.isoformat()
                            ),
                            "occurred_end": (
                                None if item.occurred_end is None else item.occurred_end.isoformat()
                            ),
                        }
                        for item in unit.memories
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            for unit in task.units
        ]
        actual_source_roster: object = (
            source_rosters[0] if len(source_rosters) == 1 else source_rosters
        )
        if actual_source_roster != expected_source_roster:
            raise RuntimeError(
                "frozen selection source roster mismatch: "
                f"got {actual_source_roster}, expected {expected_source_roster}"
            )
    for field in ("dataset_sha256", "evaluation_sha256"):
        actual = getattr(task, field)
        if actual != selection[field]:
            raise RuntimeError(
                f"frozen selection {field} mismatch: got {actual}, expected {selection[field]}"
            )
    return units


def _clone_sources(  # noqa: C901 - common and arm-specific stores share isolation checks
    plan: Mapping[str, Any], units: Sequence[str]
) -> dict[str, object]:
    clone_root = Path(plan["clone_root"]).resolve()
    if clone_root.exists():
        raise FileExistsError(f"clone root already exists: {clone_root}")
    clone_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    clone_root.mkdir(mode=0o700)
    sources: dict[str, object] = {}
    declared_stores = plan.get("source_stores")
    if declared_stores is not None and (
        not isinstance(declared_stores, dict) or set(declared_stores) != set(ARM_NAMES)
    ):
        raise ValueError("source_stores must declare exactly baseline and candidate")
    for unit_id in units:
        if declared_stores is None:
            source_run = Path(plan["source_run"]).resolve()
            source = source_run / _safe_component(unit_id, "unit")
            with _closed_store_lock(source):
                source_manifest = tree_manifest(source)
                destinations = {}
                for arm in ARM_NAMES:
                    destination = clone_root / arm / _safe_component(unit_id, "unit")
                    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    shutil.copytree(source, destination, copy_function=shutil.copy2)
                    clone_manifest = tree_manifest(destination)
                    if clone_manifest != source_manifest:
                        raise RuntimeError(f"clone content mismatch for {arm}/{unit_id}")
                    destinations[arm] = str(destination)
                if tree_manifest(source) != source_manifest:
                    raise RuntimeError(f"source store changed while cloning: {source}")
            sources[unit_id] = {
                "source": str(source),
                "content": source_manifest,
                "clones": destinations,
                "clone_content": dict.fromkeys(ARM_NAMES, source_manifest),
            }
            continue
        arm_sources = {}
        for arm in ARM_NAMES:
            arm_mapping = declared_stores[arm]
            if not isinstance(arm_mapping, dict) or unit_id not in arm_mapping:
                raise ValueError(f"source_stores.{arm} is missing unit {unit_id}")
            source = Path(arm_mapping[unit_id]).resolve()
            destination = clone_root / arm / _safe_component(unit_id, "unit")
            with _closed_store_lock(source):
                source_manifest = tree_manifest(source)
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                shutil.copytree(source, destination, copy_function=shutil.copy2)
                clone_manifest = tree_manifest(destination)
                if clone_manifest != source_manifest:
                    raise RuntimeError(f"clone content mismatch for {arm}/{unit_id}")
                if tree_manifest(source) != source_manifest:
                    raise RuntimeError(f"source store changed while cloning: {source}")
            arm_sources[arm] = {
                "source": str(source),
                "content": source_manifest,
                "clone": str(destination),
                "clone_content": clone_manifest,
            }
        sources[unit_id] = {"arm_sources": arm_sources}
    return sources


def _run_worker(
    plan_path: Path,
    output: Path,
    arm: str,
    *,
    embedding_proxy: str | None = None,
) -> subprocess.CompletedProcess[str]:
    plan = _load_plan(plan_path)
    source_root = Path(plan["arms"][arm]["source_root"]).resolve()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    environment["PYTHONHASHSEED"] = str(plan["seed"])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if embedding_proxy is not None:
        environment[EMBEDDING_PROXY_ENV] = embedding_proxy
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--plan",
        str(plan_path.resolve()),
        "--arm",
        arm,
        "--output",
        str(output.resolve()),
    ]
    return subprocess.run(command, env=environment, text=True, capture_output=True, check=False)


def _verify_frozen_arms(plan: Mapping[str, Any]) -> None:
    for arm in ARM_NAMES:
        frozen = plan["arms"][arm]
        source_root = Path(frozen["source_root"]).resolve()
        archive = Path(frozen["source_archive"]).resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"{arm} source root does not exist: {source_root}")
        actual = _sha256_file(archive)
        if actual != frozen["source_archive_sha256"]:
            raise RuntimeError(
                f"{arm} source archive digest mismatch: got {actual}, "
                f"expected {frozen['source_archive_sha256']}"
            )


def _verify_frozen_store_archive(plan: Mapping[str, Any]) -> None:
    """Verify an optional immutable source-store archive before making evaluation clones."""
    archive_value = plan.get("source_store_archive")
    digest_value = plan.get("source_store_archive_sha256")
    if archive_value is None and digest_value is None:
        return
    if not isinstance(archive_value, str) or not isinstance(digest_value, str):
        raise ValueError(
            "source_store_archive and source_store_archive_sha256 must be supplied together"
        )
    archive = Path(archive_value).resolve()
    actual = _sha256_file(archive)
    if actual != digest_value:
        raise RuntimeError(
            f"source store archive digest mismatch: got {actual}, expected {digest_value}"
        )


def _verify_source_store_receipts(plan: Mapping[str, Any]) -> None:
    receipts = plan.get("source_store_receipts")
    if receipts is None:
        return
    if not isinstance(receipts, dict) or set(receipts) != set(ARM_NAMES):
        raise ValueError("source_store_receipts must declare exactly baseline and candidate")
    for arm, units in receipts.items():
        if not isinstance(units, dict):
            raise ValueError(f"source_store_receipts.{arm} must map unit IDs to receipts")
        for unit_id, receipt in units.items():
            if not isinstance(receipt, dict) or set(receipt) != {"path", "sha256"}:
                raise ValueError(f"invalid source store receipt for {arm}/{unit_id}")
            actual = _sha256_file(Path(receipt["path"]).resolve())
            if actual != receipt["sha256"]:
                raise RuntimeError(
                    f"source store receipt digest mismatch for {arm}/{unit_id}: "
                    f"got {actual}, expected {receipt['sha256']}"
                )


class _EmbeddingResponseCache:
    """Durably map exact serialized embedding requests to successful raw responses."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._lock = threading.Lock()
        self.events: list[dict[str, object]] = []
        self._entries: dict[str, dict[str, object]] = {}
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"embedding cache has an unsupported schema: {self.path}")
            entries = value.get("entries")
            if not isinstance(entries, dict):
                raise ValueError(f"embedding cache entries are invalid: {self.path}")
            self._entries = {
                key: self._validated_entry(key, entry) for key, entry in entries.items()
            }

    @staticmethod
    def _validated_entry(key: object, entry: object) -> dict[str, object]:
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(character not in "0123456789abcdef" for character in key)
        ):
            raise ValueError("embedding cache entry key must be a lowercase SHA-256 digest")
        required = {"status", "content_type", "body_b64", "response_sha256"}
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError(f"embedding cache entry {key} has an invalid schema")
        status = entry["status"]
        if type(status) is not int or not 200 <= status < 300:
            raise ValueError(f"embedding cache entry {key} status must be a successful integer")
        content_type = entry["content_type"]
        if not isinstance(content_type, str) or not content_type.strip():
            raise ValueError(f"embedding cache entry {key} content_type must be a non-empty string")
        body_b64 = entry["body_b64"]
        if not isinstance(body_b64, str):
            raise ValueError(f"embedding cache entry {key} body_b64 must be a string")
        try:
            body = base64.b64decode(body_b64, validate=True)
        except (binascii.Error, UnicodeEncodeError, ValueError):
            raise ValueError(f"embedding cache entry {key} body_b64 is invalid") from None
        response_sha256 = entry["response_sha256"]
        if (
            not isinstance(response_sha256, str)
            or len(response_sha256) != 64
            or any(character not in "0123456789abcdef" for character in response_sha256)
        ):
            raise ValueError(f"embedding cache entry {key} response_sha256 is invalid")
        if _sha256_bytes(body) != response_sha256:
            raise ValueError(f"embedding cache entry {key} response digest mismatch")
        return dict(entry)

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, key: str) -> dict[str, object] | None:
        with self._lock:
            found = self._entries.get(key)
            return None if found is None else self._validated_entry(key, found)

    def store(self, key: str, response: dict[str, object]) -> None:
        with self._lock:
            self._entries[key] = self._validated_entry(key, response)
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            _write_json(
                temporary,
                {"schema_version": SCHEMA_VERSION, "entries": self._entries},
            )
            os.replace(temporary, self.path)

    def record(self, event: dict[str, object]) -> None:
        with self._lock:
            self.events.append({"sequence": len(self.events), **event})


def _embedding_upstream(plan: Mapping[str, Any]) -> str:
    import yaml

    value = yaml.safe_load(Path(plan["config"]).read_text(encoding="utf-8"))
    try:
        base_url = value["embedding"]["base_url"]
    except (KeyError, TypeError):
        raise ValueError("replay config must declare embedding.base_url") from None
    parsed = urlsplit(str(base_url))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("replay embedding.base_url must be an absolute HTTP URL")
    return str(base_url).rstrip("/")


@contextmanager
def _embedding_proxy(
    upstream_base_url: str, cache_path: Path, *, allow_network: bool = True
) -> Iterator[tuple[str, _EmbeddingResponseCache]]:
    import httpx

    cache = _EmbeddingResponseCache(cache_path)
    upstream = urlsplit(upstream_base_url)
    upstream_origin = f"{upstream.scheme}://{upstream.netloc}"
    client = httpx.Client(timeout=300.0)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length)
            request_identity = {
                "method": "POST",
                "path": self.path,
                "content_type": self.headers.get("content-type"),
                "body_sha256": _sha256_bytes(body),
            }
            request_key = _sha256_bytes(_json_bytes(request_identity))
            cached = cache.get(request_key)
            if cached is not None:
                response_body = base64.b64decode(cast(str, cached["body_b64"]), validate=True)
                cached_status = cached["status"]
                if not isinstance(cached_status, int):
                    raise RuntimeError("cached embedding status must be an integer")
                status = cached_status
                content_type = str(cached["content_type"])
                cache.record(
                    {
                        **request_identity,
                        "request_key": request_key,
                        "cache_hit": True,
                        "network_call": False,
                        "status": status,
                        "response_sha256": _sha256_bytes(response_body),
                        "response_bytes": len(response_body),
                    }
                )
                self._respond(status, content_type, response_body)
                return
            if not allow_network:
                response_body = _json_bytes(
                    {"error": {"message": "exact embedding cache miss; network is forbidden"}}
                )
                cache.record(
                    {
                        **request_identity,
                        "request_key": request_key,
                        "cache_hit": False,
                        "network_call": False,
                        "status": 424,
                        "cached_success": False,
                        "response_sha256": _sha256_bytes(response_body),
                        "response_bytes": len(response_body),
                    }
                )
                self._respond(424, "application/json", response_body)
                return
            headers = {
                name: value
                for name in ("authorization", "content-type", "accept", "user-agent")
                if (value := self.headers.get(name)) is not None
            }
            try:
                response = client.post(upstream_origin + self.path, content=body, headers=headers)
                response_body = response.content
                status = response.status_code
                content_type = response.headers.get("content-type", "application/json")
            except httpx.HTTPError as error:
                response_body = _json_bytes({"error": {"message": str(error)}})
                status = 502
                content_type = "application/json"
            if 200 <= status < 300:
                cache.store(
                    request_key,
                    {
                        "status": status,
                        "content_type": content_type,
                        "body_b64": base64.b64encode(response_body).decode("ascii"),
                        "response_sha256": _sha256_bytes(response_body),
                    },
                )
            cache.record(
                {
                    **request_identity,
                    "request_key": request_key,
                    "cache_hit": False,
                    "network_call": True,
                    "status": status,
                    "response_sha256": _sha256_bytes(response_body),
                    "response_bytes": len(response_body),
                    "cached_success": 200 <= status < 300,
                }
            )
            self._respond(status, content_type, response_body)

        def _respond(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    proxy_base_url = f"http://127.0.0.1:{server.server_port}{upstream.path.rstrip('/')}"
    try:
        yield proxy_base_url, cache
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        client.close()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, values: Iterable[object]) -> None:
    path.write_bytes(b"".join(_json_bytes(value) + b"\n" for value in values))


def _compare(
    plan: Mapping[str, Any],
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(baseline) != len(candidate):
        raise RuntimeError("paired arms emitted different row counts")
    rows = []
    for left, right in zip(baseline, candidate, strict=True):
        identity_fields = (
            "task",
            "unit_id",
            "question_id",
            "question_sha256",
            "reference_at",
            "provider_request_url",
        )
        mismatched = [field for field in identity_fields if left[field] != right[field]]
        if mismatched:
            raise RuntimeError(f"pair identity mismatch for {left['question_id']}: {mismatched}")
        equal = left["provider_request_body_b64"] == right["provider_request_body_b64"]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task": left["task"],
                "unit_id": left["unit_id"],
                "question_id": left["question_id"],
                "baseline_request_sha256": left["provider_request_sha256"],
                "candidate_request_sha256": right["provider_request_sha256"],
                "exact_provider_request_bytes_equal": equal,
                "future_paired_answer_reuse_permitted": equal,
                "existing_baseline_score_transfer_eligible": False,
                "generation_performed": False,
                "baseline_context_sha256": left["context_sha256"],
                "candidate_context_sha256": right["context_sha256"],
                "exact_hit_state_equal": left.get("hits") == right.get("hits"),
                "baseline_compile_latency_ms": left.get("compile_latency_ms"),
                "candidate_compile_latency_ms": right.get("compile_latency_ms"),
            }
        )
    expected = int(plan["selection"]["expected_questions"])
    if len(rows) != expected:
        raise RuntimeError(f"comparison emitted {len(rows)} rows, expected {expected}")
    equal_count = sum(bool(row["exact_provider_request_bytes_equal"]) for row in rows)
    hit_state_equal = sum(bool(row["exact_hit_state_equal"]) for row in rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "name": plan["name"],
        "task": plan["task"],
        "planned_questions": expected,
        "compared_questions": len(rows),
        "exact_request_equal": equal_count,
        "exact_request_changed": len(rows) - equal_count,
        "exact_hit_state_equal": hit_state_equal,
        "exact_hit_state_changed": len(rows) - hit_state_equal,
        "generation_requests_sent": 0,
        "future_paired_answers_reusable": equal_count,
        "existing_baseline_scores_transferable": 0,
        "candidate_generation_required_later": len(rows) - equal_count,
        "all_pairs_accounted_for": len(rows) == expected,
    }
    return rows, summary


def _reuse_closed_clones(  # noqa: C901 - validates both supported source manifest layouts
    plan: Mapping[str, Any], units: Sequence[str], manifest_path: Path
) -> dict[str, object]:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = value.get("units") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(sources, dict)
        or set(sources) != set(units)
    ):
        raise RuntimeError("existing clone manifest does not match the frozen unit roster")
    clone_root = Path(plan["clone_root"]).resolve()
    for unit_id in units:
        unit = sources[unit_id]
        if "arm_sources" in unit:
            for arm in ARM_NAMES:
                arm_source = unit["arm_sources"][arm]
                source = Path(arm_source["source"])
                with _closed_store_lock(source):
                    if tree_manifest(source) != arm_source["content"]:
                        raise RuntimeError(f"original source store changed since cloning: {source}")
                clone = Path(arm_source["clone"]).resolve()
                expected = clone_root / arm / _safe_component(unit_id, "unit")
                if clone != expected:
                    raise RuntimeError(
                        f"unexpected existing clone path for {arm}/{unit_id}: {clone}"
                    )
                clone_content = arm_source.get("clone_content")
                if not isinstance(clone_content, dict):
                    raise RuntimeError(
                        f"existing clone manifest is missing content for {arm}/{unit_id}"
                    )
                with _closed_store_lock(clone):
                    if tree_manifest(clone) != clone_content:
                        raise RuntimeError(f"existing clone content changed for {arm}/{unit_id}")
            continue
        source = Path(unit["source"])
        with _closed_store_lock(source):
            if tree_manifest(source) != unit["content"]:
                raise RuntimeError(f"original source store changed since cloning: {source}")
        clone_contents = unit.get("clone_content")
        if not isinstance(clone_contents, dict) or set(clone_contents) != set(ARM_NAMES):
            raise RuntimeError(f"existing clone manifest is missing content for {unit_id}")
        for arm in ARM_NAMES:
            clone = Path(unit["clones"][arm]).resolve()
            expected = clone_root / arm / _safe_component(unit_id, "unit")
            if clone != expected:
                raise RuntimeError(f"unexpected existing clone path for {arm}/{unit_id}: {clone}")
            with _closed_store_lock(clone):
                if tree_manifest(clone) != clone_contents[arm]:
                    raise RuntimeError(f"existing clone content changed for {arm}/{unit_id}")
    return sources


def _sources_unchanged(sources: Mapping[str, Any]) -> bool:
    for unit in sources.values():
        if "arm_sources" in unit:
            if any(
                tree_manifest(Path(value["source"])) != value["content"]
                for value in unit["arm_sources"].values()
            ):
                return False
        elif tree_manifest(Path(unit["source"])) != unit["content"]:
            return False
    return True


def controller(  # noqa: C901 - compile and store-free control share provenance checks
    plan_path: Path,
    output: Path,
    *,
    reuse_clones_from: Path | None = None,
    embedding_cache: Path | None = None,
) -> int:
    plan_path = plan_path.resolve()
    plan = _load_plan(plan_path)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(mode=0o700, parents=True)
    plan_sha256 = _sha256_file(plan_path)
    _verify_frozen_arms(plan)
    _verify_frozen_store_archive(plan)
    _verify_source_store_receipts(plan)
    units = _selected_units(plan)
    mode = _replay_mode(plan)
    if mode == "blind":
        if reuse_clones_from is not None or embedding_cache is not None:
            raise ValueError("blind replay accepts neither stores nor an embedding cache")
        sources: dict[str, object] = {}
    else:
        sources = (
            _clone_sources(plan, units)
            if reuse_clones_from is None
            else _reuse_closed_clones(plan, units, reuse_clones_from.resolve())
        )
    _write_json(
        output / "source-store-manifest.json", {"schema_version": SCHEMA_VERSION, "units": sources}
    )
    logs = {}
    proxy_context = (
        _embedding_proxy(
            _embedding_upstream(plan),
            embedding_cache,
            allow_network=str(plan.get("embedding_cache_network", "allow")) == "allow",
        )
        if mode == "compile" and embedding_cache is not None
        else nullcontext((None, None))
    )
    with proxy_context as (embedding_proxy, cache):
        for arm in ARM_NAMES:
            event_start = 0 if cache is None else len(cache.events)
            result = _run_worker(
                plan_path,
                output / f"{arm}.jsonl",
                arm,
                embedding_proxy=embedding_proxy,
            )
            logs[arm] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            if cache is not None:
                for event in cache.events[event_start:]:
                    event["arm"] = arm
            _write_json(output / f"{arm}-process.json", logs[arm])
            if result.returncode:
                raise RuntimeError(
                    f"{arm} worker failed with exit code {result.returncode}: "
                    f"{result.stderr[-2000:]}"
                )
    sources_unchanged = mode == "blind" or _sources_unchanged(sources)
    if not sources_unchanged:
        raise RuntimeError("one or more original source stores changed during replay")
    baseline = _read_jsonl(output / "baseline.jsonl")
    candidate = _read_jsonl(output / "candidate.jsonl")
    comparisons, summary = _compare(plan, baseline, candidate)
    _write_jsonl(output / "comparison.jsonl", comparisons)
    embedding_control = None
    if cache is not None:
        _write_jsonl(output / "embedding-requests.jsonl", cache.events)
        snapshot = output / "embedding-cache-snapshot.json"
        shutil.copy2(cache.path, snapshot)
        by_arm = {
            arm: {
                "requests": sum(event["arm"] == arm for event in cache.events),
                "hits": sum(
                    event["arm"] == arm and bool(event["cache_hit"]) for event in cache.events
                ),
                "network_calls": sum(
                    event["arm"] == arm and bool(event["network_call"]) for event in cache.events
                ),
            }
            for arm in ARM_NAMES
        }
        embedding_control = {
            "cache_path": str(cache.path),
            "network_policy": str(plan.get("embedding_cache_network", "allow")),
            "cache_snapshot": str(snapshot.resolve()),
            "cache_snapshot_sha256": _sha256_file(snapshot),
            "entry_count": cache.entry_count,
            "requests": len(cache.events),
            "hits": sum(bool(event["cache_hit"]) for event in cache.events),
            "misses": sum(not bool(event["cache_hit"]) for event in cache.events),
            "network_calls": sum(bool(event["network_call"]) for event in cache.events),
            "errors_cached": 0,
            "by_arm": by_arm,
        }
    elif mode == "blind":
        embedding_control = {
            "entry_count": 0,
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "network_calls": 0,
            "errors_cached": 0,
            "by_arm": {arm: {"requests": 0, "hits": 0, "network_calls": 0} for arm in ARM_NAMES},
        }
    summary.update(
        {
            "plan_sha256": plan_sha256,
            "config_sha256": _sha256_file(Path(plan["config"])),
            "selection_sha256": _sha256_file(Path(plan["selection_manifest"])),
            "arms": {name: plan["arms"][name] for name in ARM_NAMES},
            "source_store_manifest_sha256": _sha256_file(output / "source-store-manifest.json"),
            "source_stores_unchanged_after_replay": sources_unchanged,
            "source_stores_accessed": mode == "compile",
            "replay_mode": mode,
            "driver_sha256": _sha256_file(Path(__file__).resolve()),
            "reused_closed_clones_from": (
                None if reuse_clones_from is None else str(reuse_clones_from.resolve())
            ),
            "embedding_control": embedding_control,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


class _CaptureTransport:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def __call__(self, request: Any) -> Any:  # noqa: ANN401
        import httpx

        assert isinstance(request, httpx.Request)
        body = bytes(request.content)
        self.requests.append(
            {
                "method": request.method,
                "url": str(request.url),
                "content_type": request.headers.get("content-type"),
                "body": body,
            }
        )
        response = {
            "id": "paired-replay-capture",
            "object": "chat.completion",
            "created": 0,
            "model": "paired-replay-capture",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "CAPTURE_ONLY"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return httpx.Response(200, json=response, request=request)


def _canonical_atom(atom: str | Path) -> dict[str, object]:
    if isinstance(atom, Path):
        return {
            "kind": "media",
            "sha256": _sha256_file(atom),
            "bytes": atom.stat().st_size,
            "suffix": atom.suffix.casefold(),
        }
    return {"kind": "text", "value": atom}


def _question_record(question: object) -> dict[str, object]:
    typed_question = cast(Any, question)
    record = {
        "question_id": typed_question.question_id,
        "content": [_canonical_atom(atom) for atom in typed_question.content],
        "reference_at": typed_question.reference_at.isoformat(),
        "source_question": typed_question.source_question,
        "metadata": typed_question.metadata,
        "score_kind": typed_question.score_kind,
        "expected_choice": typed_question.expected_choice,
        "cutoff_seconds": typed_question.cutoff_seconds,
        "refusal": typed_question.refusal,
        "references": typed_question.references,
    }
    return record


def _hit_state(hit: object) -> dict[str, object]:
    typed_hit = cast(Any, hit)
    context = typed_hit.context
    return {
        "memory_id": typed_hit.id,
        "retrieval_score": typed_hit.score,
        "confidence": 1.0 if context is None else context.confidence,
        "created_at": typed_hit.created_at.isoformat(),
        "occurred_at": (
            None if typed_hit.occurred_at is None else typed_hit.occurred_at.isoformat()
        ),
        "occurred_end": (
            None if typed_hit.occurred_end is None else typed_hit.occurred_end.isoformat()
        ),
        "valid_from": None
        if context is None or context.valid_from is None
        else context.valid_from.isoformat(),
        "valid_until": None
        if context is None or context.valid_until is None
        else context.valid_until.isoformat(),
        "known_at": None if context is None else context.recorded_at.isoformat(),
    }


def _import_identity(source_root: Path) -> dict[str, object]:
    import mindbridge

    imported = Path(mindbridge.__file__).resolve()
    if not imported.is_relative_to(source_root.resolve()):
        raise RuntimeError(f"imported MindBridge outside frozen source root: {imported}")
    files = []
    for relative in (
        "__init__.py",
        "memory.py",
        "context.py",
        "infrastructure/local/store.py",
        "benchmarks/eval.py",
    ):
        path = source_root / "mindbridge" / relative
        files.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "module_file": str(imported),
        "files": files,
        "import_clock_sha256": _sha256_bytes(_json_bytes(files)),
    }


async def _capture_generation(
    generator: object,
    question: str,
    context: str | None,
    *,
    question_assets: Sequence[Path] = (),
    evidence_hits: Sequence[Any] = (),
) -> dict[str, object]:
    import httpx
    from openai import AsyncOpenAI

    typed_generator = cast(Any, generator)
    original = typed_generator._client
    await original.close()
    capture = _CaptureTransport()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
    typed_generator._client = AsyncOpenAI(
        api_key=typed_generator._client.api_key,
        base_url=typed_generator._client.base_url,
        timeout=typed_generator._client.timeout,
        http_client=cast(Any, http_client),
    )
    try:
        if "question_assets" in inspect.signature(typed_generator.answer).parameters:
            await typed_generator.answer(
                question,
                context,
                question_assets=question_assets,
                evidence_hits=evidence_hits,
            )
        else:
            await typed_generator.answer(question, context)
    finally:
        await typed_generator.close()
    if len(capture.requests) != 1:
        raise RuntimeError(f"capture transport saw {len(capture.requests)} requests")
    request = capture.requests[0]
    body = request.pop("body")
    assert isinstance(body, bytes)
    return {
        **request,
        "body_b64": base64.b64encode(body).decode("ascii"),
        "sha256": _sha256_bytes(body),
        "bytes": len(body),
    }


async def worker(  # noqa: C901 - compile and store-free control share capture serialization
    plan_path: Path, output: Path, arm: str
) -> int:
    from mindbridge import ContextBudget
    from mindbridge.benchmarks.eval import _BackendPool, _BaselineGenerator, _load_memory_config
    from mindbridge.benchmarks.model_config import ModelConfig

    plan = _load_plan(plan_path)
    source_root = Path(plan["arms"][arm]["source_root"]).resolve()  # noqa: ASYNC240
    identity = _import_identity(source_root)
    _selected_units(plan)
    task = _load_task(plan)
    config, _ = _load_memory_config(Path(plan["config"]))
    if config is None or config.generation is None:
        raise RuntimeError("replay config must declare generation")
    proxy = os.environ.get(EMBEDDING_PROXY_ENV)
    if proxy is not None:
        if config.embedding is None:
            raise RuntimeError("embedding proxy requires a configured embedding backend")
        config = config.model_copy(
            update={"embedding": config.embedding.model_copy(update={"base_url": proxy})}
        )
    generation = config.generation
    assert generation is not None
    if generation.timeout is None:
        raise RuntimeError("replay config must pin generation.timeout")
    model = ModelConfig(
        generation_api_key=(
            None if generation.api_key is None else generation.api_key.get_secret_value()
        ),
        generation_base_url=str(generation.base_url),
        generation_model=generation.model,
        timeout_seconds=generation.timeout,
        generation_capabilities=frozenset(generation.modalities),
        generation_min_video_seconds=generation.min_video_seconds,
    )
    if _replay_mode(plan) == "blind":
        rows = []
        for unit in task.units:
            for question in unit.questions:
                question_record = _question_record(question)
                question_assets = tuple(part for part in question.content if isinstance(part, Path))
                generator = _BaselineGenerator(
                    model,
                    seed=int(plan["seed"]),
                    gen_kwargs=str(plan.get("gen_kwargs", "")),
                    generation=config,
                )
                request = await _capture_generation(
                    generator,
                    "\n".join(str(part) for part in question.content if not isinstance(part, Path)),
                    None,
                    question_assets=question_assets,
                )
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "arm": arm,
                        "task": plan["task"],
                        "unit_id": unit.unit_id,
                        "question_id": question.question_id,
                        "question": question_record,
                        "question_sha256": _sha256_bytes(_json_bytes(question_record)),
                        "reference_at": question.reference_at.isoformat(),
                        "budget": {"max_items": 0, "max_chars": 0},
                        "memory_ids": [],
                        "hits": [],
                        "scope": {
                            "valid_at": question.reference_at.isoformat(),
                            "known_at": question.reference_at.isoformat(),
                        },
                        "context": "",
                        "context_sha256": _sha256_bytes(b""),
                        "context_chars": 0,
                        "context_items": 0,
                        "compile_latency_ms": None,
                        "media": [],
                        "query_media": [_canonical_atom(path) for path in question_assets],
                        "provider_request_method": request["method"],
                        "provider_request_url": request["url"],
                        "provider_request_content_type": request["content_type"],
                        "provider_request_body_b64": request["body_b64"],
                        "provider_request_sha256": request["sha256"],
                        "provider_request_bytes": request["bytes"],
                        "generation_performed": False,
                        "worker_process": {
                            "pid": os.getpid(),
                            "executable": sys.executable,
                            "python": sys.version,
                        },
                        "import_identity": identity,
                        "source_archive_sha256": plan["arms"][arm]["source_archive_sha256"],
                        "config_sha256": _sha256_file(Path(plan["config"])),
                        "selection_sha256": _sha256_file(Path(plan["selection_manifest"])),
                        "shared_embedding_cache": False,
                        "replay_mode": "blind",
                    }
                )
        expected = int(plan["selection"]["expected_questions"])
        if len(rows) != expected:
            raise RuntimeError(f"worker emitted {len(rows)} rows, expected {expected}")
        _write_jsonl(output, rows)
        return 0
    pool = _BackendPool(
        model,
        device=None,
        batch_size=8,
        needs_speech=False,
        seed=int(plan["seed"]),
        gen_kwargs=str(plan.get("gen_kwargs", "")),
        memory_config=config,
    )
    rows = []
    try:
        for unit in task.units:
            clone = Path(plan["clone_root"]) / arm / _safe_component(unit.unit_id, "unit")
            async with pool.memory(clone) as memory:
                for question in unit.questions:
                    question_record = _question_record(question)
                    content = _compile_query(plan, question)
                    compile_scope = _compile_scope(plan, question.reference_at)
                    compile_started = time.perf_counter()
                    bundle = await memory.compile(
                        content,
                        budget=ContextBudget(
                            max_items=int(plan["budget"]["max_items"]),
                            max_chars=int(plan["budget"]["max_chars"]),
                        ),
                        reference_at=question.reference_at,
                        scope=compile_scope,
                    )
                    compile_latency_ms = (time.perf_counter() - compile_started) * 1_000
                    rendered = bundle.render()
                    generator = _BaselineGenerator(
                        model,
                        seed=int(plan["seed"]),
                        gen_kwargs=str(plan.get("gen_kwargs", "")),
                        generation=config,
                    )
                    question_text = "\n".join(
                        str(part) for part in question.content if not isinstance(part, Path)
                    )
                    request = await _capture_generation(
                        generator,
                        question_text,
                        rendered,
                        question_assets=tuple(
                            part for part in question.content if isinstance(part, Path)
                        ),
                        evidence_hits=bundle.hits,
                    )
                    media = [
                        {
                            "memory_id": hit.id,
                            "asset_id": asset.id,
                            "sha256": asset.sha256,
                            "media_type": asset.media_type,
                        }
                        for hit in bundle.hits
                        for asset in hit.assets
                    ]
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "arm": arm,
                            "task": plan["task"],
                            "unit_id": unit.unit_id,
                            "question_id": question.question_id,
                            "question": question_record,
                            "question_sha256": _sha256_bytes(_json_bytes(question_record)),
                            "reference_at": question.reference_at.isoformat(),
                            "budget": plan["budget"],
                            "memory_ids": [hit.id for hit in bundle.hits],
                            "hits": [_hit_state(hit) for hit in bundle.hits],
                            "scope": _scope_record(compile_scope),
                            "context": rendered,
                            "context_sha256": _sha256_bytes(rendered.encode("utf-8")),
                            "context_chars": bundle.chars,
                            "context_items": len(bundle.hits),
                            "compile_latency_ms": compile_latency_ms,
                            "media": media,
                            "provider_request_method": request["method"],
                            "provider_request_url": request["url"],
                            "provider_request_content_type": request["content_type"],
                            "provider_request_body_b64": request["body_b64"],
                            "provider_request_sha256": request["sha256"],
                            "provider_request_bytes": request["bytes"],
                            "generation_performed": False,
                            "worker_process": {
                                "pid": os.getpid(),
                                "executable": sys.executable,
                                "python": sys.version,
                            },
                            "import_identity": identity,
                            "source_archive_sha256": plan["arms"][arm]["source_archive_sha256"],
                            "config_sha256": _sha256_file(Path(plan["config"])),
                            "selection_sha256": _sha256_file(Path(plan["selection_manifest"])),
                            "shared_embedding_cache": proxy is not None,
                            "compile_query": str(plan.get("compile_query", "question_content")),
                        }
                    )
    finally:
        pool.close()
    expected = int(plan["selection"]["expected_questions"])
    if len(rows) != expected:
        raise RuntimeError(f"worker emitted {len(rows)} rows, expected {expected}")
    _write_jsonl(output, rows)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=ARM_NAMES)
    parser.add_argument("--reuse-clones-from", type=Path)
    parser.add_argument("--embedding-cache", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.worker:
        if arguments.arm is None:
            parser.error("--worker requires --arm")
        return asyncio.run(worker(arguments.plan, arguments.output, arguments.arm))
    if arguments.arm is not None:
        parser.error("--arm is only valid with --worker")
    return controller(
        arguments.plan,
        arguments.output,
        reuse_clones_from=arguments.reuse_clones_from,
        embedding_cache=arguments.embedding_cache,
    )


if __name__ == "__main__":
    raise SystemExit(main())
