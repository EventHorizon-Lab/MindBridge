"""Rejudge saved ICM predictions without ingesting data or generating answers."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from mindbridge.benchmarks.eval_config import _JudgeConfig
from mindbridge.benchmarks.eval_judge import _judge_call
from mindbridge.benchmarks.eval_results import SampleResult
from mindbridge.benchmarks.icm_bench import load_icm_bench
from mindbridge.benchmarks.official_scorers import (
    SCORER_VERSION,
    judge_model_is_official,
    judge_plan,
    scorer_protocol,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def _text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"saved prediction requires {key}")
    return value


def _references(row: Mapping[str, object]) -> tuple[str, ...]:
    value = row.get("references")
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError("saved references must be a nonempty list of text")
    return tuple(value)


def load_predictions(path: Path) -> tuple[dict[str, object], ...]:
    """Validate all rows before spending any judge requests."""
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("predictions must be JSONL objects")
        for key in ("sample_id", "task", "unit_id", "question_id", "dataset_sha256"):
            _text(row, key)
        if row["task"] != "icm-bench":
            raise ValueError("rejudge input must contain only icm-bench samples")
        if not isinstance(row.get("prediction"), str):
            raise ValueError("saved prediction must be text")
        if row["sample_id"] in seen:
            raise ValueError("duplicate saved sample_id")
        seen.add(row["sample_id"])
        rows.append(row)
    if not rows:
        raise ValueError("predictions must not be empty")
    return tuple(rows)


def attach_questions(
    rows: Sequence[Mapping[str, object]], annotations: Path
) -> tuple[dict[str, object], ...]:
    """Join original questions by ID and verify saved references and available question text.

    Old generation prompts may include task templates; never reverse-engineer them into the
    original judge question. The annotation digest is recorded separately from the evaluation
    corpus digest, whose upstream definition may also include media.
    """
    questions = {q.question_id: q for q in load_icm_bench(annotations)}
    matched = []
    for row in rows:
        question = questions.get(_text(row, "question_id"))
        if question is None:
            raise ValueError("saved question_id is absent from ICM annotations")
        if _references(row) != (question.reference_answer,):
            raise ValueError("saved references disagree with ICM annotations or were not logged")
        saved_question = row.get("source_question")
        if saved_question is not None and saved_question != question.question:
            raise ValueError("saved source_question disagrees with ICM annotations")
        matched.append(
            {
                **row,
                "source_question": question.question,
                "question_binding": "id_reference_and_text"
                if saved_question is not None
                else "legacy_id_and_reference",
            }
        )
    return tuple(matched)


async def rejudge(
    rows: Sequence[Mapping[str, object]], *, client: AsyncOpenAI, config: _JudgeConfig
) -> tuple[dict[str, object], ...]:
    """Keep old scores and predictions; report judge failures as unscored, never correct."""
    semaphore = asyncio.Semaphore(config.concurrency)
    output: list[dict[str, object]] = []
    for row in rows:
        result: dict[str, object] = {
            "sample_id": row["sample_id"],
            "prediction": row["prediction"],
            "prediction_sha256": hashlib.sha256(str(row["prediction"]).encode("utf-8")).hexdigest(),
            "dataset_sha256": row["dataset_sha256"],
            "old_score": row.get("score"),
            "old_scorer_protocol": row.get("scorer_protocol"),
            "old_judge_model": row.get("judge_model"),
            "scorer_protocol": scorer_protocol("icm-bench"),
            "scorer_version": SCORER_VERSION,
            "judge_model": config.model,
            "official_judge_model": judge_model_is_official("icm-bench", config.model),
            "question_binding": row.get("question_binding"),
            "score": None,
        }
        if row.get("error_code") not in (None, "JudgeError") or row.get("ingest_failure_count"):
            result["status"] = "skipped_failed_sample"
        elif not row["prediction"]:
            result.update(status="empty_prediction", score=0.0)
        else:
            result.update(await _judge_row(row, client=client, config=config, semaphore=semaphore))
        output.append(result)
    return tuple(output)


async def _judge_row(
    row: Mapping[str, object],
    *,
    client: AsyncOpenAI,
    config: _JudgeConfig,
    semaphore: asyncio.Semaphore,
) -> dict[str, object]:
    plan = judge_plan(
        "icm-bench",
        question=_text(row, "source_question"),
        references=_references(row),
        prediction=str(row["prediction"]),
        metadata={},
    )
    assert plan is not None
    sample = SampleResult(
        task="icm-bench",
        benchmark="icm-bench",
        dataset_sha256=str(row["dataset_sha256"]),
        evaluation_sha256=str(row.get("evaluation_sha256", "")),
        unit_id=str(row["unit_id"]),
        question_id=str(row["question_id"]),
        prediction=str(row["prediction"]),
        parsed_choice=None,
        score=None,
        exact_match=None,
        latency_ms=0,
        confidence=0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=None,
        metadata={},
        arm=str(row.get("arm", "mindbridge")),
    )
    try:
        scores, response, _cached = await _judge_call(
            client,
            plan.calls[0],
            sample=sample,
            plan=plan,
            call_index=0,
            cache=None,
            semaphore=semaphore,
            config=config,
        )
        return {"status": "judged", "score": scores["accuracy"], "judge_response": response}
    except Exception as error:
        # A transport exception may contain credentials. Persist its class, not its message.
        return {"status": "judge_failed", "error_type": type(error).__name__}


async def _main(arguments: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    rows = attach_questions(load_predictions(arguments.predictions), arguments.annotations)
    if arguments.output.exists():
        raise ValueError("rejudge output already exists; choose a new path")
    config = _JudgeConfig(
        model=arguments.model,
        base_url=arguments.base_url,
        api_key=os.environ.get(arguments.api_key_env),
        concurrency=1,
    )
    document: dict[str, object] = {
        "schema_version": 1,
        "measurement": "icm_saved_prediction_rejudge",
        "predictions_sha256": hashlib.sha256(arguments.predictions.read_bytes()).hexdigest(),
        "annotations_sha256": hashlib.sha256(arguments.annotations.read_bytes()).hexdigest(),
    }
    # Reserve the destination before requests; another process cannot overwrite the results.
    with arguments.output.open("x", encoding="utf-8") as output:
        async with AsyncOpenAI(
            api_key=config.api_key, base_url=config.base_url, timeout=config.timeout_seconds
        ) as client:
            document["samples"] = await rejudge(rows, client=client, config=config)
        json.dump(document, output, ensure_ascii=False, allow_nan=False, indent=2)
        output.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
