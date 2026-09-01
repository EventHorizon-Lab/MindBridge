#!/usr/bin/env python3
"""Run the frozen M3 bedroom n15 A/B toggling only internal memory IDs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import run_memory_id_closed_ab as base
import run_memory_id_gallery_ab as gallery
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
from mindbridge.types import Modality, SearchHit

if TYPE_CHECKING:
    from openai import AsyncOpenAI

TASK = "m3-bench-robot"
UNIT_ID = "bedroom_01"
QUESTION_COUNT = 15
GENERATION_SEED = 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--stores-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="https://inner-prism.cece.com/api/v1")
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _load_inputs(  # noqa: C901 - frozen provenance checks intentionally stay in one path
    samples_path: Path,
    results_path: Path,
    stores_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, SearchHit], dict[str, Any]]:
    archived = json.loads(results_path.read_text(encoding="utf-8"))
    matching = [summary for summary in archived["tasks"] if summary["task"] == TASK]
    if archived["status"] != "completed" or len(matching) != 1 or matching[0]["error_count"]:
        raise ValueError("source result is not a clean completed M3 run")
    if base._sha256(samples_path) != archived["samples_sha256"]:
        raise ValueError("source samples digest does not match results.json")
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    if len(samples) != QUESTION_COUNT or matching[0]["question_count"] != QUESTION_COUNT:
        raise ValueError("source run is not the fixed M3 n15 unit")
    if any(sample["task"] != TASK or sample["unit_id"] != UNIT_ID for sample in samples):
        raise ValueError("source samples are not exclusively M3 bedroom_01")
    if any(len(sample["prompt"]) != 1 for sample in samples):
        raise ValueError("M3 frozen questions are not text-only prompts")
    stores = text_ab._locate_stores(stores_root)
    component = text_ab._safe_unit_component(UNIT_ID)
    if component not in stores:
        raise ValueError("frozen M3 bedroom store is missing")
    store_path = stores[component]
    wanted = tuple(
        dict.fromkeys(memory_id for sample in samples for memory_id in sample["memory_ids"])
    )
    asset_manifest: dict[str, dict[str, Any]] = {}
    records = gallery._store_records(
        store_path,
        wanted,
        unit_id=UNIT_ID,
        asset_manifest=asset_manifest,
        allowed_asset_modalities=frozenset({Modality.IMAGE, Modality.VIDEO, Modality.AUDIO}),
    )
    fit_manifest = []
    for index, sample in enumerate(samples):
        question = ModelInput(text=sample["prompt"][0])
        hits = tuple(records[memory_id] for memory_id in sample["memory_ids"])
        grounded = sdk._fit_grounding_media(question, hits)
        if tuple(hit.id for hit in grounded) != tuple(sample["memory_ids"]):
            raise ValueError("frozen IDs cannot reproduce M3 grounding fit")
        fit_manifest.append(
            {
                "index": index,
                "sample_id": sample["sample_id"],
                "hit_count": len(grounded),
                "routed_asset_ids": [asset.id for hit in grounded for asset in hit.assets],
            }
        )
    asset_bytes = json.dumps(asset_manifest, sort_keys=True, separators=(",", ":")).encode()
    fit_bytes = json.dumps(fit_manifest, sort_keys=True, separators=(",", ":")).encode()
    snapshot_paths = [store_path]
    wal_path = Path(f"{store_path}-wal")
    if wal_path.exists():
        snapshot_paths.append(wal_path)
    store_snapshot = {path.name: base._sha256(path) for path in snapshot_paths}
    snapshot_bytes = json.dumps(store_snapshot, sort_keys=True, separators=(",", ":")).encode()
    provenance = {
        "source_run_id": archived["run_id"],
        "samples_sha256": archived["samples_sha256"],
        "results_sha256": base._sha256(results_path),
        "question_count": len(samples),
        "unit_id": UNIT_ID,
        "store_path": str(store_path),
        "store_snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "store_snapshot_files": store_snapshot,
        "unique_memory_count": len(records),
        "asset_count": len(asset_manifest),
        "asset_bytes": sum(asset["size_bytes"] for asset in asset_manifest.values()),
        "asset_manifest_sha256": hashlib.sha256(asset_bytes).hexdigest(),
        "grounding_fit_manifest_sha256": hashlib.sha256(fit_bytes).hexdigest(),
        "grounding_fit_manifest": fit_manifest,
        "archived_generation_seed": archived["model"]["generation_seed"],
        "archived_generation_min_video_seconds": archived["model"]["generation_min_video_seconds"],
    }
    if provenance["archived_generation_seed"] != GENERATION_SEED:
        raise ValueError("source M3 generation seed is not the fixed seed")
    if provenance["archived_generation_min_video_seconds"] is not None:
        raise ValueError("source M3 run did not route native videos")
    return samples, records, provenance


def _text_request(
    question: ModelInput,
    hits: Sequence[SearchHit],
    *,
    include_memory_id: bool,
) -> str:
    return gallery._json_text(
        {
            "question": question.text,
            "hits": [
                gallery._hit_payload(hit, include_memory_id=include_memory_id) for hit in hits
            ],
        }
    )


def _request_pair(
    question: ModelInput,
    hits: Sequence[SearchHit],
) -> tuple[dict[str, str | list[dict[str, object]]], dict[str, bytes]]:
    cache: dict[str, str] = {}
    if any(asset for hit in hits for asset in hit.assets):
        contents: dict[str, str | list[dict[str, object]]] = {
            "with_memory_id": gallery._request_content(
                question,
                hits,
                include_memory_id=True,
                cache=cache,
            ),
            "without_memory_id": gallery._request_content(
                question,
                hits,
                include_memory_id=False,
                cache=cache,
            ),
        }
        if gallery._normalized_content(contents["with_memory_id"]) != contents["without_memory_id"]:
            raise ValueError("M3 multimodal A/B differs by more than memory_id")
        payloads = {
            arm: gallery._content_bytes(value)
            for arm, value in contents.items()
            if isinstance(value, list)
        }
        if set(payloads) != set(contents):
            raise ValueError("M3 multimodal content serialization is incomplete")
        expected_delta = len(hits) * gallery.MEMORY_ID_OUTER_BYTES
    else:
        contents = {
            "with_memory_id": _text_request(question, hits, include_memory_id=True),
            "without_memory_id": _text_request(question, hits, include_memory_id=False),
        }
        old = json.loads(contents["with_memory_id"])
        new = json.loads(contents["without_memory_id"])
        for hit in old["hits"]:
            hit.pop("memory_id")
        if old != new:
            raise ValueError("M3 text A/B differs by more than memory_id")
        payloads = {arm: value.encode() for arm, value in contents.items()}
        expected_delta = len(hits) * 79
    if len(payloads["with_memory_id"]) - len(payloads["without_memory_id"]) != expected_delta:
        raise ValueError("M3 payload byte delta does not match the sole intended field")
    return contents, payloads


async def _generate(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    content: str | list[dict[str, object]],
) -> dict[str, Any]:
    started = perf_counter()
    first_chunk_seconds = None
    first_token_seconds = None
    usage: object | None = None
    finish_reason = None
    parts: list[str] = []
    async with semaphore:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": base.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            seed=GENERATION_SEED,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            elapsed = perf_counter() - started
            if first_chunk_seconds is None:
                first_chunk_seconds = elapsed
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            value = getattr(choices[0], "finish_reason", None)
            if isinstance(value, str):
                finish_reason = value
            delta = getattr(getattr(choices[0], "delta", None), "content", None)
            if isinstance(delta, str) and delta:
                if first_token_seconds is None:
                    first_token_seconds = elapsed
                parts.append(delta)
    prediction = "".join(parts).strip()
    if not prediction or finish_reason in {"length", "content_filter"}:
        raise ValueError(f"invalid generation response: finish_reason={finish_reason!r}")
    return {
        "prediction": prediction,
        "usage": base._usage(usage),
        "duration_seconds": perf_counter() - started,
        "first_chunk_seconds": first_chunk_seconds,
        "first_token_seconds": first_token_seconds,
        "finish_reason": finish_reason,
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
            question=text_ab._source_question(sample),
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
) -> dict[str, Any]:
    plan = judge_plan(
        TASK,
        question=text_ab._source_question(sample),
        references=sample["references"],
        prediction=prediction,
        metadata=sample["metadata"],
    )
    if plan is None:
        raise ValueError("M3 prediction did not produce a judge plan")
    responses = []
    parsed = []
    usages = []
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
            raise ValueError("M3 judge returned an invalid answer")
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
    index: int,
    sample: dict[str, Any],
    records: Mapping[str, SearchHit],
    arm_order: tuple[str, str],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
) -> dict[str, Any]:
    question = ModelInput(text=sample["prompt"][0])
    retrieved = tuple(records[memory_id] for memory_id in sample["memory_ids"])
    hits = sdk._fit_grounding_media(question, retrieved)
    contents, payloads = _request_pair(question, hits)
    arms: dict[str, dict[str, Any]] = {}
    for arm in arm_order:
        generated = await _generate(
            client,
            semaphore,
            model=model,
            content=contents[arm],
        )
        generated["payload_utf8_bytes"] = len(payloads[arm])
        generated["payload_sha256"] = hashlib.sha256(payloads[arm]).hexdigest()
        generated.update(gallery._answer_diagnostics(hits, generated["prediction"]))
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
        for arm in arms:
            arms[arm]["judge"] = {**judged, "shared": True}
            arms[arm]["scores"] = finalize_scores(TASK, {**arms[arm]["scores"], **judged["scores"]})
    else:
        for arm in arm_order:
            judged = await _judge(
                client,
                semaphore,
                model=model,
                sample=sample,
                prediction=arms[arm]["prediction"],
            )
            arms[arm]["judge"] = {**judged, "shared": False}
            arms[arm]["scores"] = finalize_scores(TASK, {**arms[arm]["scores"], **judged["scores"]})
    if any("accuracy" not in arm["scores"] for arm in arms.values()):
        raise ValueError("M3 judge did not produce accuracy")
    return {
        "index": index,
        "sample_id": sample["sample_id"],
        "question_id": sample["question_id"],
        "question_types": sample["metadata"].get("question_types"),
        "request_arm_order": arm_order,
        "hit_count": len(hits),
        "grounding_asset_count": len({asset.id for hit in hits for asset in hit.assets}),
        "source_archived_prediction": sample["prediction"],
        "source_archived_score": sample["score"],
        "arms": arms,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for arm in ("with_memory_id", "without_memory_id"):
        aggregate[arm] = {
            "accuracy": sum(row["arms"][arm]["scores"]["accuracy"] for row in rows) / len(rows),
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
    deltas = [
        row["arms"]["without_memory_id"]["scores"]["accuracy"]
        - row["arms"]["with_memory_id"]["scores"]["accuracy"]
        for row in rows
    ]
    return {
        "arms": aggregate,
        "without_minus_with_accuracy": (
            aggregate["without_memory_id"]["accuracy"] - aggregate["with_memory_id"]["accuracy"]
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
    samples, records, provenance = _load_inputs(
        arguments.samples,
        arguments.results,
        arguments.stores_root,
    )
    old_bytes = 0
    new_bytes = 0
    for sample in samples:
        question = ModelInput(text=sample["prompt"][0])
        hits = sdk._fit_grounding_media(
            question,
            tuple(records[memory_id] for memory_id in sample["memory_ids"]),
        )
        _contents, payloads = _request_pair(question, hits)
        old_bytes += len(payloads["with_memory_id"])
        new_bytes += len(payloads["without_memory_id"])
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

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=arguments.base_url,
        timeout=3600,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(arguments.concurrency)
    rng = random.Random(base.SEED)
    orders = []
    for _sample in samples:
        order = ["with_memory_id", "without_memory_id"]
        rng.shuffle(order)
        orders.append(tuple(order))
    started = perf_counter()
    completed = 0

    async def run(index: int) -> dict[str, Any]:
        nonlocal completed
        row = await _run_pair(
            index,
            samples[index],
            records,
            orders[index],
            client,
            semaphore,
            arguments.model,
        )
        completed += 1
        if completed % 3 == 0 or completed == len(samples):
            print(f"completed {completed}/{len(samples)}", flush=True)
        return row

    try:
        rows = list(await asyncio.gather(*(run(index) for index in range(len(samples)))))
    finally:
        await client.close()
    rows.sort(key=lambda row: row["index"])
    output = {
        "schema_version": 1,
        "experiment": "m3_bedroom_n15_closed_store_memory_id_field_ab",
        "protocol": {
            "task": TASK,
            "scorer_protocol": scorer_protocol(TASK),
            "primary_metric": sample_primary_metric(TASK),
            "judge_model": arguments.model,
            "judge_is_official": False,
            "model": arguments.model,
            "base_url": arguments.base_url,
            "seed": GENERATION_SEED,
            "temperature": 0,
            "enable_thinking": False,
            "max_tokens": None,
            "provider_retries": 0,
            "stream": True,
            "concurrency": arguments.concurrency,
            "minimum_video_seconds": None,
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
