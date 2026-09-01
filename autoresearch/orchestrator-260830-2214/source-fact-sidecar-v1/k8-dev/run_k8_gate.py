"""Isolated conv-26 answer gate: source-only K20 versus source-bound K8."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from openai import OpenAI

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parent
sys.path.insert(0, str(SOURCE_ROOT))

import answer_gate as gate  # noqa: E402

from mindbridge import Memory as ProductMemory  # noqa: E402
from mindbridge import ModelInput, SearchHit  # noqa: E402
from mindbridge.models.openai_sdk import OpenAIModels  # noqa: E402

SEED = 1234
LIMITS = {"baseline": 20, "candidate": 8}
CACHE_PATH = HERE / "cache.json"
RESULT_PATH = HERE / "result.json"
REPORT_PATH = HERE / "report.md"


def _source_memory_ids(path: Path) -> dict[str, str]:
    with sqlite3.connect(f"file:{path / 'state.sqlite3'}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT memory_id, metadata_json FROM memory_records ORDER BY memory_id"
        ).fetchall()
    result: dict[str, str] = {}
    for memory_id, metadata_json in rows:
        source_id = str(json.loads(metadata_json)["source_id"])
        if source_id in result:
            raise RuntimeError(f"duplicate source_id in store: {source_id}")
        result[source_id] = str(memory_id)
    return result


SOURCE_MEMORY_IDS = _source_memory_ids(HERE / "baseline")
if _source_memory_ids(HERE / "candidate") != SOURCE_MEMORY_IDS:
    raise RuntimeError("isolated stores do not contain identical source records")


class _StrictOpenAI(OpenAI):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("max_retries", 0)
        super().__init__(*args, **cast(Any, kwargs))


class _SeededModels(OpenAIModels):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["generation_seed"] = SEED
        super().__init__(*args, **cast(Any, kwargs))


class _StrictGuard(gate._GenerationGuard):
    def stream_answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
    ) -> object:
        for hit in hits:
            source_id = str(hit.metadata.get("source_id", ""))
            if hit.id != SOURCE_MEMORY_IDS.get(source_id):
                raise gate.GenerationProvenanceError(
                    "generation received a derived or non-source memory id"
                )
        return super().stream_answer(question, hits)


class _LimitMemory:
    def __init__(self, data_dir: Path, **kwargs: object) -> None:
        self._label = data_dir.name
        self._memory = ProductMemory(data_dir, **cast(Any, kwargs))

    def ask(self, question: ModelInput, *, limit: int = 20) -> object:
        del limit
        return self._memory.ask(question, limit=LIMITS[self._label])

    def close(self) -> None:
        self._memory.close()


def _load_cache(unit_id: str, query_digest: str) -> dict[str, Any]:
    identity = gate._cache_identity(
        unit_id,
        query_digest,
        limits=LIMITS,
        generation_seed=SEED,
    )
    identity["best_of"] = 1
    if not CACHE_PATH.exists():
        return {**identity, "answers": {}, "judgments": {}}
    payload = cast(dict[str, Any], json.loads(CACHE_PATH.read_text(encoding="utf-8")))
    for key, value in identity.items():
        if payload.get(key) != value:
            raise RuntimeError(f"cache identity mismatch for {key}")
    return payload


def _judge_request(
    client: OpenAI,
    messages: Sequence[object],
    *,
    extra_body: Mapping[str, object] | None,
) -> tuple[str, dict[str, int | None]]:
    request_extra = dict(extra_body or {})
    request_extra.setdefault("chat_template_kwargs", {"enable_thinking": False})
    response = client.chat.completions.create(
        model=gate.MODEL,
        messages=cast(Any, messages),
        temperature=0.0,
        seed=SEED,
        extra_body=request_extra,
    )
    usage = response.usage
    text = str(response.choices[0].message.content or "").strip()
    return text, {
        "requests": 1,
        "input_tokens": None if usage is None else usage.prompt_tokens,
        "output_tokens": None if usage is None else usage.completion_tokens,
        "total_tokens": None if usage is None else usage.total_tokens,
    }


def _lower(candidate: object, baseline: object) -> bool:
    return (
        isinstance(candidate, int | float)
        and isinstance(baseline, int | float)
        and candidate < baseline
    )


def _finalize(payload: dict[str, Any]) -> dict[str, Any]:
    for label in LIMITS:
        arm = payload[label]
        arm["retrieved_source_coverage@limit"] = arm.pop("retrieved_source_coverage@20")
        arm["retrieved_source_hit@limit"] = arm.pop("retrieved_source_hit@20")
    for row in payload["rows"]:
        for label in LIMITS:
            arm = row[label]
            arm["retrieved_source_coverage@limit"] = arm.pop("retrieved_source_coverage@20")
            arm["retrieved_source_hit@limit"] = arm.pop("retrieved_source_hit@20")

    baseline = payload["baseline"]
    candidate = payload["candidate"]
    checks = {
        "primary_not_lower": (
            candidate["metrics"]["llm_judge"] >= baseline["metrics"]["llm_judge"]
        ),
        "errors_zero": baseline["errors"] == candidate["errors"] == 0,
        "generation_tokens_lower": _lower(
            candidate["generation_tokens"]["total_tokens"],
            baseline["generation_tokens"]["total_tokens"],
        ),
        "ask_p50_lower": _lower(
            candidate["ask_latency_ms"]["p50"], baseline["ask_latency_ms"]["p50"]
        ),
        "ttft_p50_lower": _lower(
            candidate["generation_ttft_ms"]["p50"],
            baseline["generation_ttft_ms"]["p50"],
        ),
    }
    accepted = all(checks.values())
    payload["status"] = "accepted" if accepted else "rejected"
    payload["limits"] = LIMITS
    payload.pop("limit", None)
    payload["generation"].update(
        {
            "seed": SEED,
            "provider_retries": 0,
            "best_of": 1,
            "generation_received_derived_fact_text_or_id": False,
        }
    )
    payload["scorer"].update({"seed": SEED, "provider_retries": 0})
    payload["acceptance_gate"] = {
        "rule": (
            "llm_judge not lower; both arms errors=0; candidate generation tokens, "
            "ask p50, and TTFT p50 each lower"
        ),
        "checks": checks,
        "accepted": accepted,
    }
    gate._write_json(RESULT_PATH, payload)
    _write_report(payload)
    return payload


def _write_report(payload: Mapping[str, Any]) -> None:
    baseline = payload["baseline"]
    candidate = payload["candidate"]
    lines = [
        "# Conv-26 source-bound K8 answer gate",
        "",
        f"Verdict: **{str(payload['status']).upper()}**",
        "",
        "| Metric | source-only K20 | source-bound K8 |",
        "| --- | ---: | ---: |",
    ]
    for key, label in (
        ("llm_judge", "LLM judge"),
        ("token_f1", "Token F1"),
        ("bleu_1", "BLEU-1"),
    ):
        lines.append(
            f"| {label} | {baseline['metrics'][key]:.6f} | {candidate['metrics'][key]:.6f} |"
        )
    for key, label in (
        ("retrieved_source_coverage@limit", "Evidence coverage@arm limit"),
        ("retrieved_source_hit@limit", "Evidence Hit@arm limit"),
    ):
        lines.append(f"| {label} | {baseline[key]:.6f} | {candidate[key]:.6f} |")
    for section, key, label in (
        ("generation_tokens", "total_tokens", "Generation tokens"),
        ("ask_latency_ms", "p50", "Ask p50 ms"),
        ("ask_latency_ms", "p95", "Ask p95 ms"),
        ("generation_ttft_ms", "p50", "TTFT p50 ms"),
        ("generation_ttft_ms", "p95", "TTFT p95 ms"),
    ):
        lines.append(f"| {label} | {baseline[section][key]} | {candidate[section][key]} |")
    lines.append(f"| Errors | {baseline['errors']} | {candidate['errors']} |")
    lines.extend(
        [
            "",
            "Generation was fixed at qwen3.8-flash, temperature 0, seed 1234, no-think, "
            "no retries, and best-of 1. Question order is the frozen lowest-SHA 32; arm order "
            "alternates by question-ID SHA parity. A fail-closed guard verified that every "
            "generation hit used an original source record ID and exact original source content.",
            "",
            "Acceptance checks: "
            + ", ".join(
                f"{key}={'pass' if value else 'fail'}"
                for key, value in payload["acceptance_gate"]["checks"].items()
            ),
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _patch_gate() -> None:
    gate.BASELINE_PATH = HERE / "baseline"
    gate.CANDIDATE_PATH = HERE / "candidate"
    gate.CACHE_PATH = CACHE_PATH
    gate.RESULT_PATH = RESULT_PATH
    gate.OpenAI = _StrictOpenAI
    gate.OpenAIModels = _SeededModels
    gate.Memory = _LimitMemory
    gate._GenerationGuard = _StrictGuard
    gate._load_cache = _load_cache
    gate._judge_request = _judge_request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _patch_gate()
    states = gate._validate_store_pair()
    if args.check:
        print(json.dumps({"status": "ok", "store_state": states}, sort_keys=True))
        return 0
    payload = _finalize(gate.run_gate(device=args.device, phase="all"))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "result": str(RESULT_PATH),
                "report": str(REPORT_PATH),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
