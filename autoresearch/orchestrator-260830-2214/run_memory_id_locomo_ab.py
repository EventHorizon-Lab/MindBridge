#!/usr/bin/env python3
"""Run a SHA-fixed frozen-hit LoCoMo A/B that only toggles internal memory IDs."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import random
import re
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import run_memory_id_closed_ab as base

from mindbridge.benchmarks.official_scorers import (
    combine_judge_scores,
    finalize_scores,
    judge_plan,
    local_scores,
    parse_judge_response,
    sample_primary_metric,
    scorer_protocol,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

TASK = "locomo-refined"
SELECTION_SIZE = 32
QUESTION_PREFIX = "Answer concisely using only the memories.\nQuestion: "
QUESTION_SUFFIX = "\nAnswer:"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--stores-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="https://inner-prism.cece.com/api/v1")
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--selection-size", type=int, default=SELECTION_SIZE)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _safe_unit_component(unit_id: str) -> str:
    encoded = base64.b32encode(unit_id.encode()).decode("ascii").rstrip("=").lower()
    return f"unit-{encoded}"


def _selection_key(sample: Mapping[str, Any]) -> str:
    return hashlib.sha256(sample["sample_id"].encode()).hexdigest()


def _source_question(sample: Mapping[str, Any]) -> str:
    prompt = sample["prompt"]
    if not isinstance(prompt, list) or len(prompt) != 1 or not isinstance(prompt[0], str):
        raise ValueError("LoCoMo frozen sample does not have one text prompt")
    value = prompt[0]
    if not value.startswith(QUESTION_PREFIX) or not value.endswith(QUESTION_SUFFIX):
        raise ValueError("LoCoMo frozen prompt does not match the source-question envelope")
    return value[len(QUESTION_PREFIX) : -len(QUESTION_SUFFIX)]


def _locate_stores(root: Path) -> dict[str, Path]:
    located: dict[str, Path] = {}
    for path in root.rglob("state.sqlite3"):
        component = path.parent.name
        if not component.startswith("unit-"):
            continue
        if component in located:
            raise ValueError(f"duplicate frozen store for {component}")
        located[component] = path
    if not located:
        raise ValueError("no frozen unit stores found")
    return located


def _store_records(
    store_path: Path,
    wanted: Sequence[str],
) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(f"file:{store_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        routed_media = 0
        records: dict[str, dict[str, Any]] = {}
        for batch in base._chunks(wanted):
            placeholders = ",".join("?" for _value in batch)
            routed_media += connection.execute(
                f"SELECT count(*) FROM memory_assets WHERE memory_id IN ({placeholders})",
                tuple(batch),
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT memory_id, content, memory_type, metadata_json,
                       occurred_at, occurred_end, created_at
                FROM memory_records
                WHERE memory_id IN ({placeholders})
                """,
                tuple(batch),
            ).fetchall()
            for row in rows:
                metadata = json.loads(row["metadata_json"])
                if not isinstance(metadata, dict):
                    raise ValueError("stored metadata is not an object")
                record = {
                    "memory_id": row["memory_id"],
                    "content": row["content"],
                    "memory_type": row["memory_type"],
                    "occurred_at": base._isoformat(row["occurred_at"]),
                    "occurred_end": base._isoformat(row["occurred_end"]),
                    "created_at": base._isoformat(row["created_at"]),
                    "metadata": metadata,
                }
                records[record["memory_id"]] = record
    finally:
        connection.close()
    if routed_media:
        raise ValueError("LoCoMo frozen hits unexpectedly contain routed media")
    missing = sorted(set(wanted) - records.keys())
    if missing:
        raise ValueError(f"frozen store is missing {len(missing)} requested memories")
    return records


def _load_inputs(
    samples_path: Path,
    results_path: Path,
    stores_root: Path,
    selection_size: int,
) -> tuple[
    list[tuple[int, dict[str, Any], str]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
]:
    archived = json.loads(results_path.read_text(encoding="utf-8"))
    matching = [summary for summary in archived["tasks"] if summary["task"] == TASK]
    if archived["status"] != "completed" or len(matching) != 1 or matching[0]["error_count"]:
        raise ValueError("source benchmark result is not a clean completed LoCoMo run")
    if base._sha256(samples_path) != archived["samples_sha256"]:
        raise ValueError("source samples digest does not match results.json")
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    if len(samples) != matching[0]["question_count"]:
        raise ValueError("source question count does not match results.json")
    if any(sample["task"] != TASK for sample in samples):
        raise ValueError("source samples are not exclusively LoCoMo")
    if not 0 < selection_size <= len(samples):
        raise ValueError("selection size is outside the source sample range")
    ranked = sorted(
        enumerate(samples),
        key=lambda item: (_selection_key(item[1]), item[1]["sample_id"]),
    )
    selected = [
        (source_index, sample, _selection_key(sample))
        for source_index, sample in ranked[:selection_size]
    ]
    for _source_index, sample, _key in selected:
        _source_question(sample)

    stores = _locate_stores(stores_root)
    wanted_by_unit: defaultdict[str, list[str]] = defaultdict(list)
    for _source_index, sample, _key in selected:
        wanted_by_unit[sample["unit_id"]].extend(sample["memory_ids"])
    records_by_unit: dict[str, dict[str, dict[str, Any]]] = {}
    store_digests: dict[str, str] = {}
    store_paths: dict[str, str] = {}
    for unit_id, values in wanted_by_unit.items():
        component = _safe_unit_component(unit_id)
        if component not in stores:
            raise ValueError(f"no frozen store found for unit {unit_id}")
        store_path = stores[component]
        wanted = tuple(dict.fromkeys(values))
        records_by_unit[unit_id] = _store_records(store_path, wanted)
        store_digests[unit_id] = base._sha256(store_path)
        store_paths[unit_id] = str(store_path)
    selection_manifest = [
        {"sample_id": sample["sample_id"], "sha256": key, "source_index": source_index}
        for source_index, sample, key in selected
    ]
    manifest_bytes = json.dumps(
        selection_manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    store_manifest_bytes = json.dumps(
        store_digests,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    provenance = {
        "source_run_id": archived["run_id"],
        "samples_sha256": archived["samples_sha256"],
        "results_sha256": base._sha256(results_path),
        "source_question_count": len(samples),
        "selection_algorithm": "lowest sha256(UTF-8 sample_id), tie-break sample_id",
        "selection_size": selection_size,
        "selection_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selection_manifest": selection_manifest,
        "used_store_count": len(store_digests),
        "store_manifest_sha256": hashlib.sha256(store_manifest_bytes).hexdigest(),
        "store_sha256_by_unit": store_digests,
        "store_path_by_unit": store_paths,
        "unique_memory_count": sum(len(records) for records in records_by_unit.values()),
    }
    return selected, records_by_unit, provenance


def _local_score(sample: Mapping[str, Any], prediction: str) -> dict[str, float]:
    return dict(
        local_scores(
            TASK,
            score_kind="free_text",
            prediction=prediction,
            parsed_choice=None,
            expected_choice=None,
            references=sample["references"],
            question=_source_question(sample),
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
        question=_source_question(sample),
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
    records: Mapping[str, Mapping[str, Any]],
    arm_order: tuple[str, str],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
) -> dict[str, Any]:
    texts = {
        "with_memory_id": base._request_text(sample, records, include_memory_id=True),
        "without_memory_id": base._request_text(sample, records, include_memory_id=False),
    }
    old_hits = json.loads(texts["with_memory_id"])["hits"]
    new_hits = json.loads(texts["without_memory_id"])["hits"]
    if not all(
        ({key: value for key, value in old.items() if key != "memory_id"} == new)
        for old, new in zip(old_hits, new_hits, strict=True)
    ):
        raise ValueError("A/B request bodies differ by more than memory_id")
    arms: dict[str, dict[str, Any]] = {}
    for arm in arm_order:
        generated = await base._generate(
            client,
            semaphore,
            model=model,
            request_text=texts[arm],
        )
        generated["payload_utf8_bytes"] = len(texts[arm].encode())
        generated["payload_sha256"] = hashlib.sha256(texts[arm].encode()).hexdigest()
        generated.update(base._answer_diagnostics(sample, records, generated["prediction"]))
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
        "category": sample["metadata"].get("category"),
        "request_arm_order": arm_order,
        "hit_count": len(sample["memory_ids"]),
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
    selected, records_by_unit, provenance = _load_inputs(
        arguments.samples,
        arguments.results,
        arguments.stores_root,
        arguments.selection_size,
    )
    old_bytes = sum(
        len(
            base._request_text(
                sample,
                records_by_unit[sample["unit_id"]],
                include_memory_id=True,
            ).encode()
        )
        for _source_index, sample, _key in selected
    )
    new_bytes = sum(
        len(
            base._request_text(
                sample,
                records_by_unit[sample["unit_id"]],
                include_memory_id=False,
            ).encode()
        )
        for _source_index, sample, _key in selected
    )
    if old_bytes - new_bytes != sum(len(sample["memory_ids"]) for _, sample, _ in selected) * 79:
        raise ValueError("payload byte delta is not exactly one compact memory_id field per hit")
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
        "experiment": "locomo_sha32_closed_store_memory_id_field_ab",
        "protocol": {
            "task": TASK,
            "scorer_protocol": scorer_protocol(TASK),
            "primary_metric": sample_primary_metric(TASK),
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
