#!/usr/bin/env python3
"""Run the preregistered five paired repetitions for frozen M3 bedroom Q01."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import run_memory_id_closed_ab as base
import run_memory_id_m3_frozen_ab as m3

ORDERS = (
    ("with_memory_id", "without_memory_id"),
    ("without_memory_id", "with_memory_id"),
    ("with_memory_id", "without_memory_id"),
    ("without_memory_id", "with_memory_id"),
    ("with_memory_id", "without_memory_id"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--stores-root", type=Path, required=True)
    parser.add_argument("--source-ab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="https://inner-prism.cece.com/api/v1")
    parser.add_argument("--model", default="qwen3.8-flash")
    return parser.parse_args()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    with_scores = [row["arms"]["with_memory_id"]["scores"]["accuracy"] for row in rows]
    without_scores = [row["arms"]["without_memory_id"]["scores"]["accuracy"] for row in rows]
    deltas = [new - old for old, new in zip(with_scores, without_scores, strict=True)]
    usage = {}
    for arm in ("with_memory_id", "without_memory_id"):
        usage[arm] = {
            name: sum(row["arms"][arm]["usage"][name] for row in rows)
            for name in ("input_tokens", "output_tokens", "total_tokens")
        }
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    with_total = sum(with_scores)
    without_total = sum(without_scores)
    return {
        "with_memory_id_scores": with_scores,
        "without_memory_id_scores": without_scores,
        "with_memory_id_score_sum": with_total,
        "without_memory_id_score_sum": without_total,
        "wins": wins,
        "losses": losses,
        "ties": sum(delta == 0 for delta in deltas),
        "generation_usage": usage,
        "opaque_hex_emission_count": {
            arm: sum(bool(row["arms"][arm]["opaque_hex_ids_emitted"]) for row in rows)
            for arm in ("with_memory_id", "without_memory_id")
        },
        "unknown_count": {
            arm: sum(row["arms"][arm]["unknown"] for row in rows)
            for arm in ("with_memory_id", "without_memory_id")
        },
        "acceptance_condition": {
            "without_score_sum_gte_with": without_total >= with_total,
            "paired_losses_lte_wins": losses <= wins,
            "passed": without_total >= with_total and losses <= wins,
        },
    }


async def _main() -> None:
    arguments = _arguments()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    source_ab = json.loads(arguments.source_ab.read_text(encoding="utf-8"))
    if source_ab["experiment"] != "m3_bedroom_n15_closed_store_memory_id_field_ab":
        raise ValueError("source A/B is not the fixed M3 n15 run")
    q14 = next(row for row in source_ab["samples"] if row["question_id"] == "bedroom_01_Q14")
    samples, records, provenance = m3._load_inputs(
        arguments.samples,
        arguments.results,
        arguments.stores_root,
    )
    sample = next(row for row in samples if row["question_id"] == "bedroom_01_Q01")
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
    semaphore = asyncio.Semaphore(1)
    started = perf_counter()
    rows = []
    try:
        for repetition, order in enumerate(ORDERS, start=1):
            row = await m3._run_pair(
                0,
                sample,
                records,
                order,
                client,
                semaphore,
                arguments.model,
            )
            row["repetition"] = repetition
            row["request_arm_order"] = order
            rows.append(row)
            print(f"completed {repetition}/{len(ORDERS)}", flush=True)
    finally:
        await client.close()
    output = {
        "schema_version": 1,
        "experiment": "m3_bedroom_q01_memory_id_field_5rep",
        "protocol": {
            "task": m3.TASK,
            "question_id": "bedroom_01_Q01",
            "model": arguments.model,
            "base_url": arguments.base_url,
            "generation_seed": m3.GENERATION_SEED,
            "temperature": 0,
            "enable_thinking": False,
            "max_tokens": None,
            "minimum_video_seconds": None,
            "orders": ORDERS,
            "repetition_count_preregistered": len(ORDERS),
            "no_retry_or_best_of": True,
            "system_prompt": base.SYSTEM_PROMPT,
            "system_prompt_sha256": hashlib.sha256(base.SYSTEM_PROMPT.encode()).hexdigest(),
            "only_intended_request_difference": "presence of each hit's memory_id field",
        },
        "provenance": {
            **provenance,
            "source_ab_sha256": base._sha256(arguments.source_ab),
            "source_ab_q01": next(
                row for row in source_ab["samples"] if row["question_id"] == "bedroom_01_Q01"
            ),
        },
        "q14_source_ab_hash_diagnostic": {
            arm: {
                "prediction": q14["arms"][arm]["prediction"],
                "opaque_hex_ids_emitted": q14["arms"][arm]["opaque_hex_ids_emitted"],
                "retrieved_memory_ids_emitted": q14["arms"][arm]["retrieved_memory_ids_emitted"],
            }
            for arm in ("with_memory_id", "without_memory_id")
        },
        "duration_seconds": perf_counter() - started,
        "summary": _summary(rows),
        "repetitions": rows,
    }
    arguments.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    asyncio.run(_main())
