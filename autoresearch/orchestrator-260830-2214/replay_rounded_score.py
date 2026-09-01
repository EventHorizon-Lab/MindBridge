#!/usr/bin/env python3
"""Replay fixed EgoLife evidence with and without retrieval scores."""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import mindbridge.models.openai_sdk as openai_sdk
from mindbridge import Memory
from mindbridge.benchmarks.eval import _BackendPool, _parsed_choice
from mindbridge.benchmarks.model_config import ModelConfig
from mindbridge.types import Modality, SearchHit

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "final/baseline-ba4bcced90b9-locked-v2/egolife-q50-99/samples.jsonl"
CURRENT = ROOT / "final/current-e806cf3b1dd4-locked-v3/egolife-q50-99/samples.jsonl"
STORE = Path(
    "/dev/shm/mindbridge-ar-locked-b2d7f2918105/"
    "current-e806cf3b1dd4-locked-v3/egolife-q50-99/"
    "benchmark-mvtw63djmzsxcyi/"
    "run-mn2xe4tfnz2c2zjyga3ggzrtmiywizbufvwg6y3lmvsc25rtfvswo33mnftgkllrguyc2ojz/"
    "unit-ieyv6ssbjncq"
)

_ARM = contextvars.ContextVar("rounded_score_arm", default="score")
_ORIGINAL_HIT_PAYLOAD = openai_sdk._hit_payload


def _hit_payload(hit: SearchHit) -> dict[str, object]:
    payload = _ORIGINAL_HIT_PAYLOAD(hit)
    if _ARM.get() == "score":
        payload["score"] = hit.score
    return payload


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["sample_id"]: row for line in path.read_text().splitlines() if (row := json.loads(line))
    }


def _selected(stable_count: int) -> tuple[list[str], list[str]]:
    baseline, current = _rows(BASELINE), _rows(CURRENT)
    flips = sorted(
        sample_id
        for sample_id in current
        if current[sample_id]["score"] != baseline[sample_id]["score"]
    )
    stable = sorted(
        (
            sample_id
            for sample_id in current
            if current[sample_id]["score"] == baseline[sample_id]["score"]
        ),
        key=lambda value: hashlib.sha256(value.encode()).digest(),
    )[:stable_count]
    return flips, stable


def _answer(
    pool: _BackendPool,
    arm: str,
    sample_id: str,
    repetition: int,
    question: object,
    hits: tuple[SearchHit, ...],
    choices: tuple[str, ...],
    expected: str | None,
) -> dict[str, object]:
    token = _ARM.set(arm)
    try:
        answer = pool.models.answer(question, hits).answer  # type: ignore[arg-type]
    finally:
        _ARM.reset(token)
    parsed = _parsed_choice("egolifeqa", answer, choices)
    return {
        "sample_id": sample_id,
        "repetition": repetition,
        "arm": arm,
        "prediction": answer,
        "parsed_choice": parsed,
        "correct": None if expected is None else parsed == expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--stable-count", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repetitions, args.workers) < 1 or args.stable_count < 0:
        parser.error("counts must be positive and stable-count must be non-negative")

    baseline, current = _rows(BASELINE), _rows(CURRENT)
    expected = {
        sample_id: next(
            (
                row["parsed_choice"]
                for row in (baseline[sample_id], current[sample_id])
                if row["score"] == 1.0
            ),
            None,
        )
        for sample_id in current
    }
    flips, stable = _selected(args.stable_count)
    sample_ids = (*flips, *stable)
    config = replace(
        ModelConfig.from_environment(),
        generation_min_video_seconds=2,
        generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
    )
    openai_sdk._hit_payload = _hit_payload
    pool = _BackendPool(
        config,
        device="cuda",
        batch_size=1,
        needs_speech=True,
        seed=20260830,
        gen_kwargs="enable_thinking=false",
    )
    try:
        routed: dict[str, tuple[object, tuple[SearchHit, ...], tuple[str, ...]]] = {}
        evidence_replay = {}
        with Memory(
            STORE,
            embedder=pool._embedder,
            answerer=pool._answerer,
            transcriber=pool._transcriber,
            index_speech=True,
        ) as memory:
            for sample_id in sample_ids:
                row = current[sample_id]
                prompt = row["prompt"][0]
                hits = memory.search(prompt, limit=20)
                retrieved = tuple(hit.id for hit in hits)
                recorded = tuple(row["memory_ids"])
                evidence_replay[sample_id] = {
                    "order_exact": retrieved == recorded,
                    "overlap": len(set(retrieved) & set(recorded)) / len(recorded),
                    "retrieved": retrieved,
                    "recorded": recorded,
                }
                with memory._operation() as operation:
                    prepared = memory._prepare_content(prompt, operation)
                    question = memory._route_generation(prepared, operation)
                    routed_hits = memory._route_generation_hits(hits, operation)
                routed[sample_id] = (
                    question,
                    routed_hits,
                    tuple(row["metadata"]["choices"]),
                )

        jobs = []
        for repetition in range(args.repetitions):
            for sample_id in sample_ids:
                question, hits, choices = routed[sample_id]
                for arm in ("compact", "score"):
                    jobs.append((arm, sample_id, repetition, question, hits, choices))
        jobs.sort(key=lambda job: hashlib.sha256(f"{job[1]}:{job[2]}:{job[0]}".encode()).digest())
        outcomes = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    _answer,
                    pool,
                    arm,
                    sample_id,
                    repetition,
                    question,
                    hits,
                    choices,
                    expected[sample_id],
                )
                for arm, sample_id, repetition, question, hits, choices in jobs
            ]
            for future in as_completed(futures):
                outcomes.append(future.result())
    finally:
        openai_sdk._hit_payload = _ORIGINAL_HIT_PAYLOAD
        pool.close()

    outcomes.sort(key=lambda row: (row["sample_id"], row["repetition"], row["arm"]))
    known_accuracy = {
        arm: sum(
            row["correct"] is True
            for row in outcomes
            if row["arm"] == arm and row["correct"] is not None
        )
        / sum(row["arm"] == arm and row["correct"] is not None for row in outcomes)
        for arm in ("compact", "score")
    }
    parsed_rate = {
        arm: sum(row["parsed_choice"] is not None for row in outcomes if row["arm"] == arm)
        / sum(row["arm"] == arm for row in outcomes)
        for arm in ("compact", "score")
    }
    payload = {
        "schema": "mindbridge-score-replay-v2",
        "selection": {
            "flips": flips,
            "stable_controls": stable,
            "baseline_accuracy": sum(baseline[sample_id]["score"] for sample_id in sample_ids)
            / len(sample_ids),
            "current_accuracy": sum(current[sample_id]["score"] for sample_id in sample_ids)
            / len(sample_ids),
        },
        "repetitions": args.repetitions,
        "evidence_replay": evidence_replay,
        "known_label_accuracy": known_accuracy,
        "parsed_choice_rate": parsed_rate,
        "outcomes": outcomes,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "known_label_accuracy": known_accuracy,
                "parsed_choice_rate": parsed_rate,
                "samples": len(sample_ids),
            }
        )
    )


if __name__ == "__main__":
    main()
