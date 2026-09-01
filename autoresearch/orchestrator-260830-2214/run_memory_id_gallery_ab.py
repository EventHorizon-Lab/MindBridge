#!/usr/bin/env python3
"""Run a SHA-fixed frozen-media Mem-Gallery A/B toggling only internal memory IDs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import os
import random
import re
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import run_memory_id_closed_ab as base
import run_memory_id_locomo_ab as text_ab

from mindbridge.benchmarks.official_scorers import (
    combine_judge_scores,
    finalize_scores,
    judge_plan,
    local_scores,
    parse_judge_response,
    sample_primary_metric,
    scorer_protocol,
)
from mindbridge.models import openai_sdk as sdk
from mindbridge.models.base import ModelInput
from mindbridge.types import AssetRef, MemoryType, Modality, SearchHit

if TYPE_CHECKING:
    from openai import AsyncOpenAI

TASK = "mem-gallery"
SELECTION_SIZE = 32
BENCHMARK_MARKER = "/.benchmarks/"
MEMORY_ID_OUTER_BYTES = 83


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--stores-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="https://inner-prism.cece.com/api/v1")
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--selection-size", type=int, default=SELECTION_SIZE)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _datetime(value: object) -> datetime | None:
    rendered = base._isoformat(value)
    return None if rendered is None else datetime.fromisoformat(rendered)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_ref(
    row: sqlite3.Row,
    store_path: Path,
    *,
    unit_id: str,
    asset_manifest: dict[str, dict[str, Any]],
) -> AssetRef:
    asset_id = row["asset_id"]
    if asset_id != row["sha256"]:
        raise ValueError("frozen asset ID does not equal its SHA-256")
    path = store_path.parent / row["relative_path"]
    info = path.stat()
    if info.st_size != row["size_bytes"]:
        raise ValueError(f"frozen asset size changed: {asset_id}")
    file_sha256 = _file_sha256(path)
    if file_sha256 != row["sha256"]:
        raise ValueError(f"frozen asset digest changed: {asset_id}")
    key = f"{unit_id}:{asset_id}"
    asset_manifest[key] = {
        "unit_id": unit_id,
        "asset_id": asset_id,
        "modality": row["asset_modality"],
        "mime_type": row["mime_type"],
        "size_bytes": row["size_bytes"],
        "sha256": file_sha256,
        "relative_path": row["relative_path"],
    }
    return AssetRef(
        id=asset_id,
        modality=Modality(row["asset_modality"]),
        media_type=row["mime_type"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        name=row["name"],
        path=path,
    )


def _store_records(
    store_path: Path,
    wanted: Sequence[str],
    *,
    unit_id: str,
    asset_manifest: dict[str, dict[str, Any]],
    allowed_asset_modalities: frozenset[Modality] = frozenset({Modality.IMAGE}),
) -> dict[str, SearchHit]:
    connection = sqlite3.connect(f"file:{store_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        raw_records: dict[str, sqlite3.Row] = {}
        assets_by_memory: defaultdict[str, list[tuple[int, AssetRef]]] = defaultdict(list)
        asset_cache: dict[str, AssetRef] = {}
        for batch in base._chunks(wanted):
            placeholders = ",".join("?" for _value in batch)
            for row in connection.execute(
                f"""
                SELECT memory_id, content, modality, memory_type, metadata_json,
                       occurred_at, occurred_end, created_at
                FROM memory_records
                WHERE memory_id IN ({placeholders})
                """,
                tuple(batch),
            ).fetchall():
                raw_records[row["memory_id"]] = row
            for row in connection.execute(
                f"""
                SELECT ma.memory_id, ma.position, a.asset_id,
                       a.modality AS asset_modality, a.mime_type, a.size_bytes,
                       a.sha256, a.relative_path, a.name
                FROM memory_assets AS ma
                JOIN media_assets AS a ON a.asset_id = ma.asset_id
                WHERE ma.memory_id IN ({placeholders})
                ORDER BY ma.memory_id, ma.position
                """,
                tuple(batch),
            ).fetchall():
                asset = asset_cache.get(row["asset_id"])
                if asset is None:
                    asset = _asset_ref(
                        row,
                        store_path,
                        unit_id=unit_id,
                        asset_manifest=asset_manifest,
                    )
                    asset_cache[row["asset_id"]] = asset
                assets_by_memory[row["memory_id"]].append((row["position"], asset))
    finally:
        connection.close()
    missing = sorted(set(wanted) - raw_records.keys())
    if missing:
        raise ValueError(f"frozen store is missing {len(missing)} requested memories")
    records: dict[str, SearchHit] = {}
    for memory_id, row in raw_records.items():
        positioned = assets_by_memory[memory_id]
        if [position for position, _asset in positioned] != list(range(len(positioned))):
            raise ValueError(f"frozen asset positions are not contiguous: {memory_id}")
        assets = tuple(asset for _position, asset in positioned)
        if any(asset.modality not in allowed_asset_modalities for asset in assets):
            raise ValueError("selected routed asset has an unexpected modality")
        metadata = json.loads(row["metadata_json"])
        if not isinstance(metadata, dict):
            raise ValueError("stored metadata is not an object")
        records[memory_id] = SearchHit(
            id=memory_id,
            content=row["content"],
            score=0.0,
            created_at=_datetime(row["created_at"]),
            occurred_at=_datetime(row["occurred_at"]),
            occurred_end=_datetime(row["occurred_end"]),
            metadata=metadata,
            assets=assets,
            modality=Modality(row["modality"]),
            memory_type=MemoryType(row["memory_type"]),
        )
    return records


def _media_type(path: Path) -> str:
    value, _encoding = mimetypes.guess_type(path.name)
    if value not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"unsupported frozen query image: {path}")
    return value


def _question_input(
    sample: Mapping[str, Any],
    benchmark_root: Path,
) -> tuple[ModelInput, dict[str, Any] | None]:
    prompt = sample["prompt"]
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(not isinstance(part, str) for part in prompt)
    ):
        raise ValueError("Mem-Gallery prompt is not a non-empty text list")
    archived_paths = [part for part in prompt if BENCHMARK_MARKER in part]
    if len(archived_paths) > 1 or (archived_paths and prompt[-1] != archived_paths[0]):
        raise ValueError("Mem-Gallery query image envelope is ambiguous")
    text_parts = prompt[:-1] if archived_paths else prompt
    assets: tuple[AssetRef, ...] = ()
    provenance = None
    if archived_paths:
        archived_path = archived_paths[0]
        suffix = archived_path.split(BENCHMARK_MARKER, 1)[1]
        resolved = benchmark_root / suffix
        size_bytes = resolved.stat().st_size
        sha256 = _file_sha256(resolved)
        media_type = _media_type(resolved)
        asset = AssetRef(
            id=sha256,
            modality=Modality.IMAGE,
            media_type=media_type,
            size_bytes=size_bytes,
            sha256=sha256,
            name=resolved.name,
            path=resolved,
        )
        assets = (asset,)
        provenance = {
            "sample_id": sample["sample_id"],
            "archived_path": archived_path,
            "resolved_path": str(resolved),
            "asset_id": sha256,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "mime_type": media_type,
        }
    return ModelInput(text="\n\n".join(text_parts), assets=assets), provenance


def _load_inputs(  # noqa: C901 - provenance validation is deliberately one closed audit path
    samples_path: Path,
    results_path: Path,
    stores_root: Path,
    benchmark_root: Path,
    selection_size: int,
) -> tuple[
    list[tuple[int, dict[str, Any], str]],
    dict[str, dict[str, SearchHit]],
    dict[str, ModelInput],
    dict[str, Any],
]:
    archived = json.loads(results_path.read_text(encoding="utf-8"))
    matching = [summary for summary in archived["tasks"] if summary["task"] == TASK]
    if archived["status"] != "completed" or len(matching) != 1 or matching[0]["error_count"]:
        raise ValueError("source benchmark result is not a clean completed Mem-Gallery run")
    if base._sha256(samples_path) != archived["samples_sha256"]:
        raise ValueError("source samples digest does not match results.json")
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    if len(samples) != matching[0]["question_count"]:
        raise ValueError("source question count does not match results.json")
    if any(sample["task"] != TASK for sample in samples):
        raise ValueError("source samples are not exclusively Mem-Gallery")
    if not 0 < selection_size <= len(samples):
        raise ValueError("selection size is outside the source sample range")
    ranked = sorted(
        enumerate(samples),
        key=lambda item: (text_ab._selection_key(item[1]), item[1]["sample_id"]),
    )
    selected = [
        (source_index, sample, text_ab._selection_key(sample))
        for source_index, sample in ranked[:selection_size]
    ]
    stores = text_ab._locate_stores(stores_root)
    wanted_by_unit: defaultdict[str, list[str]] = defaultdict(list)
    for _source_index, sample, _key in selected:
        wanted_by_unit[sample["unit_id"]].extend(sample["memory_ids"])
    records_by_unit: dict[str, dict[str, SearchHit]] = {}
    asset_manifest: dict[str, dict[str, Any]] = {}
    store_digests: dict[str, str] = {}
    store_paths: dict[str, str] = {}
    for unit_id, values in wanted_by_unit.items():
        component = text_ab._safe_unit_component(unit_id)
        if component not in stores:
            raise ValueError(f"no frozen store found for unit {unit_id}")
        store_path = stores[component]
        wanted = tuple(dict.fromkeys(values))
        records_by_unit[unit_id] = _store_records(
            store_path,
            wanted,
            unit_id=unit_id,
            asset_manifest=asset_manifest,
        )
        store_digests[unit_id] = base._sha256(store_path)
        store_paths[unit_id] = str(store_path)
    questions: dict[str, ModelInput] = {}
    query_assets = []
    for _source_index, sample, _key in selected:
        question, query_provenance = _question_input(sample, benchmark_root)
        questions[sample["sample_id"]] = question
        if query_provenance is not None:
            query_assets.append(query_provenance)
        hits = tuple(records_by_unit[sample["unit_id"]][value] for value in sample["memory_ids"])
        grounded = sdk._fit_grounding_media(question, hits)
        if tuple(hit.id for hit in grounded) != tuple(sample["memory_ids"]):
            raise ValueError("frozen IDs cannot reproduce Mem-Gallery grounding fit")
        if any(
            tuple(asset.id for asset in before.assets) != tuple(asset.id for asset in after.assets)
            for before, after in zip(hits, grounded, strict=True)
        ):
            raise ValueError("frozen Mem-Gallery hits require unavailable pre-fit media state")
    selection_manifest = [
        {"sample_id": sample["sample_id"], "sha256": key, "source_index": source_index}
        for source_index, sample, key in selected
    ]
    selection_bytes = json.dumps(
        selection_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    store_bytes = json.dumps(store_digests, sort_keys=True, separators=(",", ":")).encode()
    asset_bytes = json.dumps(asset_manifest, sort_keys=True, separators=(",", ":")).encode()
    query_bytes = json.dumps(query_assets, sort_keys=True, separators=(",", ":")).encode()
    provenance = {
        "source_run_id": archived["run_id"],
        "samples_sha256": archived["samples_sha256"],
        "results_sha256": base._sha256(results_path),
        "source_question_count": len(samples),
        "source_media_roots": archived.get("media_roots"),
        "benchmark_root": str(benchmark_root),
        "selection_algorithm": "lowest sha256(UTF-8 sample_id), tie-break sample_id",
        "selection_size": selection_size,
        "selection_manifest_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "selection_manifest": selection_manifest,
        "used_store_count": len(store_digests),
        "store_manifest_sha256": hashlib.sha256(store_bytes).hexdigest(),
        "store_sha256_by_unit": store_digests,
        "store_path_by_unit": store_paths,
        "unique_memory_count": sum(len(records) for records in records_by_unit.values()),
        "routed_asset_count": len(asset_manifest),
        "routed_asset_bytes": sum(asset["size_bytes"] for asset in asset_manifest.values()),
        "asset_manifest_sha256": hashlib.sha256(asset_bytes).hexdigest(),
        "query_asset_count": len(query_assets),
        "query_asset_manifest_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "query_assets": query_assets,
    }
    return selected, records_by_unit, questions, provenance


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _hit_payload(hit: SearchHit, *, include_memory_id: bool) -> dict[str, object]:
    payload = sdk._hit_payload(hit)
    if not include_memory_id:
        del payload["memory_id"]
    return payload


def _request_content(
    question: ModelInput,
    hits: Sequence[SearchHit],
    *,
    include_memory_id: bool,
    cache: dict[str, str],
) -> list[dict[str, object]]:
    texts = (
        _json_text(
            {
                "question": question.text,
                "assets": [asset.id for asset in question.assets],
            }
        ),
        *(
            _json_text(
                {
                    "memory": {
                        **_hit_payload(hit, include_memory_id=include_memory_id),
                        "assets": [asset.id for asset in hit.assets],
                    }
                }
            )
            for hit in hits
        ),
    )
    parts: list[dict[str, object]] = [{"type": "text", "text": texts[0]}]
    seen_assets: set[str] = set()
    for asset in question.assets:
        if asset.id not in seen_assets:
            parts.append(sdk._generation_asset_part(asset, cache))
            seen_assets.add(asset.id)
    for hit, text in zip(hits, texts[1:], strict=True):
        parts.append({"type": "text", "text": text})
        for asset in hit.assets:
            if asset.id not in seen_assets:
                parts.append(sdk._generation_asset_part(asset, cache))
                seen_assets.add(asset.id)
    return parts


def _content_bytes(content: Sequence[Mapping[str, object]]) -> bytes:
    return json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()


def _normalized_content(content: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    normalized = []
    for part in content:
        item = dict(part)
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            payload = json.loads(item["text"])
            memory = payload.get("memory")
            if isinstance(memory, dict):
                memory.pop("memory_id", None)
            item["text"] = _json_text(payload)
        normalized.append(item)
    return normalized


def _answer_diagnostics(
    hits: Sequence[SearchHit],
    prediction: str,
) -> dict[str, Any]:
    source_ids = tuple(
        dict.fromkeys(
            str(hit.metadata["source_id"])
            for hit in hits
            if isinstance(hit.metadata.get("source_id"), str)
        )
    )
    return {
        "unknown": prediction.strip() == base.UNKNOWN_ANSWER,
        "retrieved_memory_ids_emitted": [hit.id for hit in hits if hit.id in prediction],
        "opaque_hex_ids_emitted": sorted(set(base.OPAQUE_ID.findall(prediction))),
        "metadata_source_ids_emitted": [
            value for value in source_ids if value.casefold() in prediction.casefold()
        ],
    }


def _local_score(sample: Mapping[str, Any], prediction: str) -> dict[str, float]:
    return dict(
        local_scores(
            TASK,
            score_kind="free_text",
            prediction=prediction,
            parsed_choice=None,
            expected_choice=None,
            references=sample["references"],
            question=sample["prompt"][0],
            metadata=sample["metadata"],
            evidence_source_ids=tuple(item["source_id"] for item in sample["evidence"]),
        )
    )


async def _judge(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    sample: Mapping[str, Any],
    prediction: str,
) -> dict[str, Any] | None:
    plan = judge_plan(
        TASK,
        question=sample["prompt"][0],
        references=sample["references"],
        prediction=prediction,
        metadata=sample["metadata"],
    )
    if plan is None:
        return None
    responses: list[str] = []
    parsed: list[Mapping[str, float]] = []
    usages: list[dict[str, int]] = []
    for messages in plan.calls:
        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "temperature": 0.0,
        }
        if plan.max_tokens is not None:
            request["max_tokens"] = plan.max_tokens
        extra_body = dict(plan.extra_body or {})
        if "qwen3" in re.sub(r"[^a-z0-9]+", "", model.casefold()):
            extra_body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        if extra_body:
            request["extra_body"] = extra_body
        async with semaphore:
            response = await client.chat.completions.create(**request)
        choices = getattr(response, "choices", None)
        text = None if not choices else getattr(choices[0].message, "content", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("judge returned an invalid answer")
        text = text.strip()
        responses.append(text)
        parsed.append(parse_judge_response(plan, text))
        usages.append(base._usage(getattr(response, "usage", None)))
    return {
        "responses": responses,
        "scores": combine_judge_scores(plan, parsed),
        "usage": {
            name: sum(usage[name] for usage in usages)
            for name in ("input_tokens", "output_tokens", "total_tokens")
        },
    }


async def _run_pair(
    selection_index: int,
    source_index: int,
    sample: dict[str, Any],
    selection_sha256: str,
    records: Mapping[str, SearchHit],
    question: ModelInput,
    arm_order: tuple[str, str],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
) -> dict[str, Any]:
    hits = tuple(records[memory_id] for memory_id in sample["memory_ids"])
    grounded = sdk._fit_grounding_media(question, hits)
    cache: dict[str, str] = {}
    contents = {
        "with_memory_id": _request_content(
            question,
            grounded,
            include_memory_id=True,
            cache=cache,
        ),
        "without_memory_id": _request_content(
            question,
            grounded,
            include_memory_id=False,
            cache=cache,
        ),
    }
    if _normalized_content(contents["with_memory_id"]) != contents["without_memory_id"]:
        raise ValueError("A/B request bodies differ by more than memory_id")
    content_bytes = {arm: _content_bytes(value) for arm, value in contents.items()}
    if len(content_bytes["with_memory_id"]) - len(content_bytes["without_memory_id"]) != (
        len(grounded) * MEMORY_ID_OUTER_BYTES
    ):
        raise ValueError("A/B payload delta is not one compact memory_id field per hit")
    arms: dict[str, dict[str, Any]] = {}
    for arm in arm_order:
        generated = await base._generate(
            client,
            semaphore,
            model=model,
            request_text=contents[arm],
        )
        generated["payload_utf8_bytes"] = len(content_bytes[arm])
        generated["payload_sha256"] = hashlib.sha256(content_bytes[arm]).hexdigest()
        generated.update(_answer_diagnostics(grounded, generated["prediction"]))
        generated["scores"] = _local_score(sample, generated["prediction"])
        arms[arm] = generated
    if arms["with_memory_id"]["prediction"] == arms["without_memory_id"]["prediction"]:
        judged = await _judge(
            client,
            semaphore,
            model=model,
            sample=sample,
            prediction=arms["with_memory_id"]["prediction"],
        )
        if judged is not None:
            for arm in arms:
                arms[arm]["judge"] = {**judged, "shared": True}
                arms[arm]["scores"] = finalize_scores(
                    TASK, {**arms[arm]["scores"], **judged["scores"]}
                )
    else:
        for arm in arm_order:
            judged = await _judge(
                client,
                semaphore,
                model=model,
                sample=sample,
                prediction=arms[arm]["prediction"],
            )
            if judged is not None:
                arms[arm]["judge"] = {**judged, "shared": False}
                arms[arm]["scores"] = finalize_scores(
                    TASK, {**arms[arm]["scores"], **judged["scores"]}
                )
    primary = sample_primary_metric(TASK)
    if any(primary not in arm["scores"] for arm in arms.values()):
        raise ValueError(f"official scorer did not produce {primary}")
    return {
        "selection_index": selection_index,
        "source_index": source_index,
        "selection_sha256": selection_sha256,
        "sample_id": sample["sample_id"],
        "unit_id": sample["unit_id"],
        "question_id": sample["question_id"],
        "point": sample["metadata"].get("point"),
        "request_arm_order": arm_order,
        "hit_count": len(grounded),
        "question_asset_count": len(question.assets),
        "grounding_asset_count": len({asset.id for hit in grounded for asset in hit.assets}),
        "source_archived_prediction": sample["prediction"],
        "source_archived_score": sample["score"],
        "arms": arms,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arm_names = ("with_memory_id", "without_memory_id")
    metric_names = set.intersection(
        *(set(row["arms"][arm]["scores"]) for row in rows for arm in arm_names)
    )
    aggregate: dict[str, Any] = {}
    for arm in arm_names:
        aggregate[arm] = {
            "metrics": {
                metric: sum(row["arms"][arm]["scores"][metric] for row in rows) / len(rows)
                for metric in sorted(metric_names)
            },
            "generation_usage": base._sum_usage(rows, arm),
            "unknown_count": sum(row["arms"][arm]["unknown"] for row in rows),
            "opaque_hex_emission_count": sum(
                bool(row["arms"][arm]["opaque_hex_ids_emitted"]) for row in rows
            ),
            "metadata_source_id_emission_count": sum(
                bool(row["arms"][arm]["metadata_source_ids_emitted"]) for row in rows
            ),
            "payload_utf8_bytes": sum(row["arms"][arm]["payload_utf8_bytes"] for row in rows),
        }
    primary = sample_primary_metric(TASK)
    deltas = [
        row["arms"]["without_memory_id"]["scores"][primary]
        - row["arms"]["with_memory_id"]["scores"][primary]
        for row in rows
    ]
    return {
        "primary_metric": primary,
        "arms": aggregate,
        "without_minus_with_primary": (
            aggregate["without_memory_id"]["metrics"][primary]
            - aggregate["with_memory_id"]["metrics"][primary]
        ),
        "wins": sum(delta > 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "identical_prediction_count": sum(
            row["arms"]["with_memory_id"]["prediction"]
            == row["arms"]["without_memory_id"]["prediction"]
            for row in rows
        ),
    }


async def _main() -> None:
    arguments = _arguments()
    if arguments.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    selected, records_by_unit, questions, provenance = _load_inputs(
        arguments.samples,
        arguments.results,
        arguments.stores_root,
        arguments.benchmark_root,
        arguments.selection_size,
    )
    old_bytes = 0
    new_bytes = 0
    for _source_index, sample, _key in selected:
        hits = tuple(records_by_unit[sample["unit_id"]][value] for value in sample["memory_ids"])
        question = questions[sample["sample_id"]]
        cache: dict[str, str] = {}
        old_bytes += len(
            _content_bytes(_request_content(question, hits, include_memory_id=True, cache=cache))
        )
        new_bytes += len(
            _content_bytes(_request_content(question, hits, include_memory_id=False, cache=cache))
        )
    expected_delta = (
        sum(len(sample["memory_ids"]) for _, sample, _ in selected) * MEMORY_ID_OUTER_BYTES
    )
    if old_bytes - new_bytes != expected_delta:
        raise ValueError("validation payload delta is not exactly one memory_id field per hit")
    if arguments.validate_only:
        print(
            json.dumps(
                {
                    **provenance,
                    "with_memory_id_bytes": old_bytes,
                    "without_memory_id_bytes": new_bytes,
                    "saved_bytes": old_bytes - new_bytes,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    api_key = os.environ.get("MINDBRIDGE_GENERATION_API_KEY")
    if not api_key:
        raise RuntimeError("MINDBRIDGE_GENERATION_API_KEY is required")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=arguments.base_url, timeout=3600)
    semaphore = asyncio.Semaphore(arguments.concurrency)
    rng = random.Random(base.SEED)
    orders = []
    for _item in selected:
        order = ["with_memory_id", "without_memory_id"]
        rng.shuffle(order)
        orders.append(tuple(order))
    started = perf_counter()
    completed = 0

    async def run(selection_index: int) -> dict[str, Any]:
        nonlocal completed
        source_index, sample, selection_sha256 = selected[selection_index]
        row = await _run_pair(
            selection_index,
            source_index,
            sample,
            selection_sha256,
            records_by_unit[sample["unit_id"]],
            questions[sample["sample_id"]],
            orders[selection_index],
            client,
            semaphore,
            arguments.model,
        )
        completed += 1
        if completed % 4 == 0 or completed == len(selected):
            print(f"completed {completed}/{len(selected)}", flush=True)
        return row

    try:
        rows = list(await asyncio.gather(*(run(index) for index in range(len(selected)))))
    finally:
        await client.close()
    rows.sort(key=lambda row: row["selection_index"])
    output = {
        "schema_version": 1,
        "experiment": "gallery_sha32_closed_store_memory_id_field_ab",
        "protocol": {
            "task": TASK,
            "scorer_protocol": scorer_protocol(TASK),
            "primary_metric": sample_primary_metric(TASK),
            "primary_metric_is_official": True,
            "judge_model": arguments.model,
            "judge_is_official": False,
            "model": arguments.model,
            "base_url": arguments.base_url,
            "seed": base.SEED,
            "temperature": 0,
            "enable_thinking": False,
            "max_tokens": None,
            "stream": True,
            "concurrency": arguments.concurrency,
            "system_prompt": base.SYSTEM_PROMPT,
            "system_prompt_sha256": hashlib.sha256(base.SYSTEM_PROMPT.encode()).hexdigest(),
            "only_intended_request_difference": "presence of each hit's memory_id field",
            "gold_used_in_generation": False,
        },
        "provenance": provenance,
        "duration_seconds": perf_counter() - started,
        "summary": _summary(rows),
        "samples": rows,
    }
    arguments.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    asyncio.run(_main())
