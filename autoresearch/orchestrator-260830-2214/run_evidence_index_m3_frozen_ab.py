#!/usr/bin/env python3
"""Compare frozen M3 memory IDs with short ordered evidence indices."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import run_memory_id_closed_ab as base
import run_memory_id_gallery_ab as gallery
import run_memory_id_m3_frozen_ab as m3

from mindbridge.models import openai_sdk as sdk
from mindbridge.models.base import ModelInput
from mindbridge.types import SearchHit

if TYPE_CHECKING:
    from openai import AsyncOpenAI

BASELINE = "memory_id"
CANDIDATE = "evidence_index"


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


def _hit_payload(hit: SearchHit, index: int, *, arm: str) -> dict[str, object]:
    identifying: dict[str, object]
    if arm == BASELINE:
        identifying = {"memory_id": hit.id}
    elif arm == CANDIDATE:
        identifying = {"evidence_index": index}
    else:
        raise ValueError(f"unknown arm: {arm}")
    return {
        **identifying,
        **gallery._hit_payload(hit, include_memory_id=False),
    }


def _text_request(question: ModelInput, hits: Sequence[SearchHit], *, arm: str) -> str:
    return gallery._json_text(
        {
            "question": question.text,
            "hits": [_hit_payload(hit, index, arm=arm) for index, hit in enumerate(hits, start=1)],
        }
    )


def _media_request(
    question: ModelInput,
    hits: Sequence[SearchHit],
    *,
    arm: str,
    cache: dict[str, str],
) -> list[dict[str, object]]:
    texts = (
        gallery._json_text(
            {
                "question": question.text,
                "assets": [asset.id for asset in question.assets],
            }
        ),
        *(
            gallery._json_text(
                {
                    "memory": {
                        **_hit_payload(hit, index, arm=arm),
                        "assets": [asset.id for asset in hit.assets],
                    }
                }
            )
            for index, hit in enumerate(hits, start=1)
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


def _normalized(value: str | Sequence[Mapping[str, object]]) -> object:
    if isinstance(value, str):
        payload = json.loads(value)
        for hit in payload["hits"]:
            hit.pop("memory_id", None)
            hit.pop("evidence_index", None)
        return payload
    normalized = []
    for part in value:
        item = dict(part)
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            payload = json.loads(item["text"])
            memory = payload.get("memory")
            if isinstance(memory, dict):
                memory.pop("memory_id", None)
                memory.pop("evidence_index", None)
            item["text"] = gallery._json_text(payload)
        normalized.append(item)
    return normalized


def _request_pair(
    question: ModelInput,
    hits: Sequence[SearchHit],
) -> tuple[
    dict[str, str | list[dict[str, object]]],
    dict[str, bytes],
]:
    cache: dict[str, str] = {}
    if any(asset for hit in hits for asset in hit.assets):
        contents: dict[str, str | list[dict[str, object]]] = {
            arm: _media_request(question, hits, arm=arm, cache=cache)
            for arm in (BASELINE, CANDIDATE)
        }
        payloads = {
            arm: gallery._content_bytes(value)
            for arm, value in contents.items()
            if isinstance(value, list)
        }
    else:
        contents = {arm: _text_request(question, hits, arm=arm) for arm in (BASELINE, CANDIDATE)}
        payloads = {arm: value.encode() for arm, value in contents.items()}
    if set(payloads) != set(contents):
        raise ValueError("payload serialization did not cover both arms")
    if _normalized(contents[BASELINE]) != _normalized(contents[CANDIDATE]):
        raise ValueError("A/B differs by more than memory_id versus evidence_index")
    if len(payloads[CANDIDATE]) >= len(payloads[BASELINE]):
        raise ValueError("evidence_index payload is not smaller than the memory_id payload")
    return contents, payloads


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
        generated = await m3._generate(
            client,
            semaphore,
            model=model,
            content=contents[arm],
        )
        generated["payload_utf8_bytes"] = len(payloads[arm])
        generated["payload_sha256"] = hashlib.sha256(payloads[arm]).hexdigest()
        generated.update(gallery._answer_diagnostics(hits, generated["prediction"]))
        generated["scores"] = m3._local_score(sample, generated["prediction"])
        arms[arm] = generated
    if arms[BASELINE]["prediction"] == arms[CANDIDATE]["prediction"]:
        judged = await m3._judge(
            client,
            semaphore,
            model=model,
            sample=sample,
            prediction=arms[BASELINE]["prediction"],
        )
        for arm in arms:
            arms[arm]["judge"] = {**judged, "shared": True}
            arms[arm]["scores"].update(judged["scores"])
    else:
        for arm in arm_order:
            judged = await m3._judge(
                client,
                semaphore,
                model=model,
                sample=sample,
                prediction=arms[arm]["prediction"],
            )
            arms[arm]["judge"] = {**judged, "shared": False}
            arms[arm]["scores"].update(judged["scores"])
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


def _sum_usage(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, int]:
    return {
        name: sum(row["arms"][arm]["usage"][name] for row in rows)
        for name in ("input_tokens", "output_tokens", "total_tokens")
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for arm in (BASELINE, CANDIDATE):
        aggregate[arm] = {
            "accuracy": sum(row["arms"][arm]["scores"]["accuracy"] for row in rows) / len(rows),
            "generation_usage": _sum_usage(rows, arm),
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
        row["arms"][CANDIDATE]["scores"]["accuracy"] - row["arms"][BASELINE]["scores"]["accuracy"]
        for row in rows
    ]
    errors = 0
    acceptance = {
        "candidate_accuracy_gte_baseline": (
            aggregate[CANDIDATE]["accuracy"] >= aggregate[BASELINE]["accuracy"]
        ),
        "errors_zero": errors == 0,
        "candidate_opaque_hash_emission_zero": (
            aggregate[CANDIDATE]["opaque_hex_emission_count"] == 0
        ),
        "candidate_generation_tokens_lower": (
            aggregate[CANDIDATE]["generation_usage"]["total_tokens"]
            < aggregate[BASELINE]["generation_usage"]["total_tokens"]
        ),
    }
    return {
        "arms": aggregate,
        "candidate_minus_baseline_accuracy": (
            aggregate[CANDIDATE]["accuracy"] - aggregate[BASELINE]["accuracy"]
        ),
        "wins": sum(delta > 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "identical_prediction_count": sum(
            row["arms"][BASELINE]["prediction"] == row["arms"][CANDIDATE]["prediction"]
            for row in rows
        ),
        "error_count": errors,
        "acceptance": {**acceptance, "passed": all(acceptance.values())},
    }


async def _main() -> None:
    arguments = _arguments()
    if arguments.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    samples, records, provenance = m3._load_inputs(
        arguments.samples,
        arguments.results,
        arguments.stores_root,
    )
    payload_totals = {BASELINE: 0, CANDIDATE: 0}
    for sample in samples:
        question = ModelInput(text=sample["prompt"][0])
        hits = sdk._fit_grounding_media(
            question,
            tuple(records[memory_id] for memory_id in sample["memory_ids"]),
        )
        _contents, payloads = _request_pair(question, hits)
        for arm in payload_totals:
            payload_totals[arm] += len(payloads[arm])
    if arguments.validate_only:
        print(json.dumps({**provenance, "payload_utf8_bytes": payload_totals}, sort_keys=True))
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
        order = [BASELINE, CANDIDATE]
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
        "experiment": "m3_bedroom_n15_memory_id_vs_evidence_index_ab",
        "protocol": {
            "task": m3.TASK,
            "model": arguments.model,
            "base_url": arguments.base_url,
            "generation_seed": m3.GENERATION_SEED,
            "temperature": 0,
            "enable_thinking": False,
            "max_tokens": None,
            "minimum_video_seconds": None,
            "concurrency": arguments.concurrency,
            "baseline": "original 64-hex memory_id field",
            "candidate": "memory_id removed; 1-based integer evidence_index in ordered hit position",
            "system_prompt": base.SYSTEM_PROMPT,
            "system_prompt_sha256": hashlib.sha256(base.SYSTEM_PROMPT.encode()).hexdigest(),
            "system_prompt_change": False,
            "gold_used_in_generation": False,
            "no_retry_or_best_of": True,
            "acceptance_gate": (
                "candidate accuracy >= baseline; errors=0; candidate opaque hash emission=0; "
                "candidate generation tokens < baseline"
            ),
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
