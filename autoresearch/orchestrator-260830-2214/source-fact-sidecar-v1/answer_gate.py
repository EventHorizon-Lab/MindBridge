"""Paired answer-quality gate for source-bound retrieval keys.

The generator sees only hydrated source memories.  Gold answers are loaded only by
the scoring phase after both paired answers have been produced.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sqlite3
import statistics
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from run import HERE, Query, _digest, _queries, _selected_locomo, _sources, _write_json

from mindbridge import EmbeddingBackend, Memory, Modality, ModelInput, SearchHit
from mindbridge.benchmarks.eval import _BorrowedBackend
from mindbridge.benchmarks.eval_telemetry import (
    BENCHMARK_TASK,
    BENCHMARK_TASK_SPAN,
    EvaluationTelemetry,
)
from mindbridge.benchmarks.official_scorers import (
    SCORER_VERSION,
    combine_judge_scores,
    judge_plan,
    local_scores,
    parse_judge_response,
    scorer_protocol,
)
from mindbridge.models import JinaOmniEmbedder, openai_sdk
from mindbridge.models.openai_sdk import OpenAIModels

TASK = "locomo-refined"
MODEL = "qwen3.8-flash"
RUN_ROOT = HERE / "runs-bound" / "locomo"
BASELINE_PATH = RUN_ROOT / "baseline"
CANDIDATE_PATH = RUN_ROOT / "candidate"
CACHE_PATH = HERE / "answer-quality-cache.json"
RESULT_PATH = HERE / "answer-quality-result.json"
EXPECTED_RECIPE = (
    "zvec-0.7:hnsw-cosine-m50-efc500:fts-dual-language:"
    "grouped-range:context-keys-v9:quantization-none"
)


class GenerationProvenanceError(RuntimeError):
    """The paired gate attempted to send non-source content to generation."""


class _GenerationGuard:
    """Forward one shared answerer while proving its inputs are original sources."""

    def __init__(self, backend: OpenAIModels, source_content: Mapping[str, str]) -> None:
        self._backend = backend
        self._source_content = source_content
        self.label = ""
        self.question_id = ""
        self.calls: dict[str, dict[str, object]] = {}

    def __getattr__(self, name: str) -> object:
        return getattr(self._backend, name)

    def stream_answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
    ) -> Iterator[str]:
        source_ids = []
        for hit in hits:
            source_id = str(hit.metadata.get("source_id", ""))
            expected = self._source_content.get(source_id)
            if expected is None or hit.content != expected:
                raise GenerationProvenanceError(
                    "generation received a non-source or mutated source record"
                )
            if source_id.startswith("derived::") or "[Derived statement]" in hit.content:
                raise GenerationProvenanceError("generation received derived fact text")
            source_ids.append(source_id)
        key = f"{self.label}:{self.question_id}"
        if key in self.calls:
            raise RuntimeError(f"duplicate guarded generation call: {key}")
        self.calls[key] = {
            "source_ids": source_ids,
            "source_count": len(source_ids),
            "all_original_sources": True,
        }
        return cast(Iterator[str], self._backend.stream_answer(question, hits))

    def close(self) -> None:
        return None


def _store_state(path: Path) -> dict[str, object]:
    database = path / "state.sqlite3"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        metadata = dict(
            connection.execute(
                "SELECT key, value FROM store_metadata "
                "WHERE key IN ('embedding.space_id', 'index.recipe')"
            )
        )
        memories = int(connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0])
        embeddings = int(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
    return {
        "memories": memories,
        "embeddings": embeddings,
        "embedding_space_id": metadata.get("embedding.space_id"),
        "index_recipe": metadata.get("index.recipe"),
    }


def _validate_store_pair() -> dict[str, dict[str, object]]:
    states = {
        "baseline": _store_state(BASELINE_PATH),
        "candidate": _store_state(CANDIDATE_PATH),
    }
    for label, state in states.items():
        if state["memories"] != 419:
            raise RuntimeError(f"{label} memory count changed: {state['memories']}")
        if state["index_recipe"] != EXPECTED_RECIPE:
            raise RuntimeError(f"{label} recipe is not safe to reopen: {state['index_recipe']}")
    if states["baseline"]["embeddings"] != 419:
        raise RuntimeError("baseline embedding count changed")
    if states["candidate"]["embeddings"] != 1140:
        raise RuntimeError("candidate source-bound keys are missing")
    if states["baseline"]["embedding_space_id"] != states["candidate"]["embedding_space_id"]:
        raise RuntimeError("paired stores use different embedding spaces")
    return states


def _store_digest(path: Path) -> str:
    digest = hashlib.sha256()
    database = path / "state.sqlite3"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        for statement in connection.iterdump():
            digest.update(statement.encode())
            digest.update(b"\n")
    assets = path / "assets"
    if assets.is_dir():
        for asset in sorted(item for item in assets.rglob("*") if item.is_file()):
            digest.update(asset.relative_to(path).as_posix().encode())
            digest.update(b"\0")
            digest.update(asset.read_bytes())
    return digest.hexdigest()


def _query_digest(queries: Sequence[Query]) -> str:
    return _digest(
        json.dumps(
            [
                {
                    "question_id": query.question_id,
                    "content": query.content,
                }
                for query in queries
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _cache_identity(
    unit_id: str,
    query_digest: str,
    *,
    limits: Mapping[str, int],
    generation_seed: int | None,
) -> dict[str, object]:
    store_sha256 = {
        "baseline": _store_digest(BASELINE_PATH),
        "candidate": _store_digest(CANDIDATE_PATH),
    }
    generation_protocol_sha256 = _digest(
        json.dumps(
            {
                "system_prompt": openai_sdk._GROUNDED_SYSTEM_PROMPT,
                "answer_text_parts": inspect.getsource(openai_sdk._answer_text_parts),
                "hit_payload": inspect.getsource(openai_sdk._hit_payload),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    request_inputs = {
        "query_set_sha256": query_digest,
        "store_sha256": store_sha256,
        "generation_protocol_sha256": generation_protocol_sha256,
        "limits": dict(limits),
        "generation_seed": generation_seed,
    }
    return {
        "schema_version": 2,
        "task": TASK,
        "unit_id": unit_id,
        **request_inputs,
        "request_inputs_sha256": _digest(
            json.dumps(request_inputs, sort_keys=True, separators=(",", ":"))
        ),
        "model": MODEL,
        "generation_base_url": os.environ.get(
            "MINDBRIDGE_GENERATION_BASE_URL", "https://inner-prism.cece.com/api/v1"
        ),
        "generation_temperature": 0.0,
        "generation_max_tokens": None,
        "no_think": True,
        "provider_retries": 0,
        "scorer_version": SCORER_VERSION,
        "scorer_protocol": scorer_protocol(TASK),
    }


def _load_cache(unit_id: str, query_digest: str) -> dict[str, Any]:
    identity = _cache_identity(
        unit_id,
        query_digest,
        limits={"baseline": 20, "candidate": 20},
        generation_seed=None,
    )
    if not CACHE_PATH.exists():
        return {**identity, "answers": {}, "judgments": {}}
    payload = cast(dict[str, Any], json.loads(CACHE_PATH.read_text(encoding="utf-8")))
    for key, value in identity.items():
        if payload.get(key) != value:
            raise RuntimeError(f"answer cache identity mismatch for {key}")
    return payload


def _node_metric(telemetry: Mapping[str, object], node: str, *keys: str) -> float | None:
    value: object = telemetry
    for key in ("nodes", node, *keys):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, int | float) else None


def _generation_tokens(telemetry: Mapping[str, object]) -> dict[str, object]:
    usage = telemetry.get("token_usage")
    if not isinstance(usage, Mapping):
        return {"complete": False, "total_tokens": None}
    by_module = usage.get("by_module")
    generation = by_module.get("generation") if isinstance(by_module, Mapping) else None
    if not isinstance(generation, Mapping):
        return {"complete": False, "total_tokens": None}
    return {
        key: generation.get(key)
        for key in (
            "complete",
            "request_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        )
    }


def _answer_order(question_id: str) -> tuple[str, str]:
    return (
        ("candidate", "baseline")
        if int(_digest(question_id), 16) % 2
        else ("baseline", "candidate")
    )


def _answer_phase(  # noqa: C901 - paired instrumentation and fail-closed provenance are coupled
    cache: dict[str, Any],
    *,
    device: str,
    sources: Mapping[str, str],
    unit_id: str,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    api_key = os.environ.get("MINDBRIDGE_GENERATION_API_KEY")
    if not api_key:
        raise RuntimeError("MINDBRIDGE_GENERATION_API_KEY is required")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get(
            "MINDBRIDGE_GENERATION_BASE_URL", "https://inner-prism.cece.com/api/v1"
        ),
        timeout=120,
        max_retries=0,
    )
    models = OpenAIModels(
        generation_client=client,
        generation_model=MODEL,
        generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
        generation_temperature=0.0,
        generation_max_tokens=None,
        generation_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    embedder_model = JinaOmniEmbedder(device=device, batch_size=32)
    borrowed_embedder = cast(EmbeddingBackend, _BorrowedBackend(embedder_model))
    guard = _GenerationGuard(models, sources)
    telemetry = EvaluationTelemetry()
    memories: dict[str, Memory] = {}
    before = _validate_store_pair()
    try:
        memories = {
            "baseline": Memory(
                BASELINE_PATH,
                embedder=borrowed_embedder,
                answerer=cast(Any, guard),
                tracer=telemetry.tracer,
            ),
            "candidate": Memory(
                CANDIDATE_PATH,
                embedder=borrowed_embedder,
                answerer=cast(Any, guard),
                tracer=telemetry.tracer,
            ),
        }
        opened = _validate_store_pair()
        if opened != before:
            raise RuntimeError("reopening the paired stores changed authoritative state")
        queries = _queries("locomo", unit_id)
        for query in queries:
            for label in _answer_order(query.question_id):
                cache_key = f"{label}:{query.question_id}"
                if cache_key in cache["answers"]:
                    continue
                guard.label = label
                guard.question_id = query.question_id
                task_key = f"answer-gate:{cache_key}"
                started = time.perf_counter()
                with telemetry.tracer.start_as_current_span(
                    BENCHMARK_TASK_SPAN,
                    attributes={BENCHMARK_TASK: task_key},
                ):
                    try:
                        result = memories[label].ask(query.content, limit=20)
                        error: str | None = None
                    except GenerationProvenanceError:
                        raise
                    except Exception as caught:
                        result = None
                        error = f"{type(caught).__name__}: {' '.join(str(caught).split())[:500]}"
                latency_ms = (time.perf_counter() - started) * 1_000
                measurement = telemetry.result(task_key, question_count=1)
                guarded = guard.calls.get(cache_key)
                if guarded is None and error is None:
                    raise RuntimeError("generation guard did not observe the public ask call")
                guarded = guarded or {
                    "source_ids": [],
                    "source_count": 0,
                    "all_original_sources": True,
                }
                returned_ids = (
                    []
                    if result is None
                    else [str(hit.metadata["source_id"]) for hit in result.hits]
                )
                if any(source_id not in sources for source_id in returned_ids):
                    raise RuntimeError("public ask returned a non-source record")
                cache["answers"][cache_key] = {
                    "answer": "" if result is None else result.answer,
                    "retrieved_source_ids": guarded["source_ids"],
                    "returned_source_ids": returned_ids,
                    "generation_input_all_original_sources": True,
                    "ask_latency_ms": latency_ms,
                    "generation_ttft_ms": _node_metric(
                        measurement,
                        "mindbridge.model.generation",
                        "ttft_ms",
                        "average",
                    ),
                    "generation_ttfc_ms": _node_metric(
                        measurement,
                        "mindbridge.model.generation",
                        "time_to_first_chunk_ms",
                        "average",
                    ),
                    "generation_tokens": _generation_tokens(measurement),
                    "error": error,
                }
                _write_json(CACHE_PATH, cache)
    finally:
        for memory in memories.values():
            memory.close()
        telemetry.close()
        embedder_model.close()
        client.close()
    after = _validate_store_pair()
    if after != before:
        raise RuntimeError("answer phase changed authoritative store state")
    return before, after


def _judge_request(
    client: OpenAI,
    messages: Sequence[object],
    *,
    extra_body: Mapping[str, object] | None,
) -> tuple[str, dict[str, int | None]]:
    request_extra = dict(extra_body or {})
    request_extra.setdefault("chat_template_kwargs", {"enable_thinking": False})
    response = client.chat.completions.create(
        model=MODEL,
        messages=cast(Any, messages),
        temperature=0.0,
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


def _judge_phase(cache: dict[str, Any], *, unit_id: str) -> None:
    if all(
        f"{label}:{query.question_id}" in cache["judgments"]
        for query in _queries("locomo", unit_id)
        for label in ("baseline", "candidate")
    ):
        return
    api_key = os.environ.get("MINDBRIDGE_GENERATION_API_KEY")
    if not api_key:
        raise RuntimeError("MINDBRIDGE_GENERATION_API_KEY is required")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get(
            "MINDBRIDGE_GENERATION_BASE_URL", "https://inner-prism.cece.com/api/v1"
        ),
        timeout=120,
        max_retries=0,
    )
    # Gold is deliberately loaded only after the paired generation phase is complete.
    questions = {item.question_id: item for item in _selected_locomo().questions}
    try:
        for query in _queries("locomo", unit_id):
            official = questions[query.question_id]
            for label in _answer_order(query.question_id):
                cache_key = f"{label}:{query.question_id}"
                if cache_key in cache["judgments"]:
                    continue
                answer = str(cache["answers"][cache_key]["answer"])
                answer_error = cache["answers"][cache_key]["error"]
                retrieved_ids = tuple(cache["answers"][cache_key]["retrieved_source_ids"])
                metrics = local_scores(
                    TASK,
                    score_kind="answer",
                    prediction=answer,
                    parsed_choice=None,
                    expected_choice=None,
                    references=official.reference_answers,
                    question=official.question,
                    metadata={},
                    evidence_source_ids=retrieved_ids,
                )
                plan = judge_plan(
                    TASK,
                    question=official.question,
                    references=official.reference_answers,
                    prediction=answer,
                    metadata={},
                )
                responses: list[str] = []
                usage_rows: list[dict[str, int | None]] = []
                judge_error: str | None = None
                if plan is not None and answer_error is None:
                    try:
                        scores = []
                        for call in plan.calls:
                            response, usage = _judge_request(
                                client,
                                [
                                    {"role": message.role, "content": message.content}
                                    for message in call
                                ],
                                extra_body=plan.extra_body,
                            )
                            responses.append(response)
                            usage_rows.append(usage)
                            scores.append(parse_judge_response(plan, response))
                        metrics.update(combine_judge_scores(plan, scores))
                    except Exception as caught:
                        judge_error = (
                            f"{type(caught).__name__}: {' '.join(str(caught).split())[:500]}"
                        )
                metrics.setdefault("llm_judge", 0.0)
                evidence = set(official.evidence_dialog_ids)
                selected = set(retrieved_ids[:20])
                coverage = len(evidence & selected) / len(evidence) if evidence else 0.0
                cache["judgments"][cache_key] = {
                    "metrics": metrics,
                    "retrieved_source_coverage@20": coverage,
                    "retrieved_source_hit@20": float(bool(evidence & selected)),
                    "judge_responses_sha256": [_digest(value) for value in responses],
                    "judge_usage": usage_rows,
                    "error": answer_error or judge_error,
                }
                _write_json(CACHE_PATH, cache)
    finally:
        client.close()


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * probability + 0.999) - 1))]


def _sum_known(rows: Sequence[Mapping[str, object]], key: str) -> int | None:
    values = [row.get(key) for row in rows]
    return (
        sum(cast(int, value) for value in values)
        if values and all(isinstance(value, int) for value in values)
        else None
    )


def _aggregate(cache: Mapping[str, Any], label: str, question_ids: Sequence[str]) -> dict[str, Any]:
    answers = [cache["answers"][f"{label}:{question_id}"] for question_id in question_ids]
    judgments = [cache["judgments"][f"{label}:{question_id}"] for question_id in question_ids]
    metrics = {
        metric: statistics.fmean(float(row["metrics"][metric]) for row in judgments)
        for metric in ("llm_judge", "token_f1", "bleu_1")
    }
    latencies = [float(row["ask_latency_ms"]) for row in answers]
    ttfts = [
        float(row["generation_ttft_ms"])
        for row in answers
        if row.get("generation_ttft_ms") is not None
    ]
    generation_usage = [cast(Mapping[str, object], row["generation_tokens"]) for row in answers]
    judge_usage = [
        usage
        for row in judgments
        for usage in cast(Sequence[Mapping[str, object]], row["judge_usage"])
    ]
    return {
        "metrics": metrics,
        "retrieved_source_coverage@20": statistics.fmean(
            float(row["retrieved_source_coverage@20"]) for row in judgments
        ),
        "retrieved_source_hit@20": statistics.fmean(
            float(row["retrieved_source_hit@20"]) for row in judgments
        ),
        "ask_latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": statistics.median(latencies),
            "p95": _percentile(latencies, 0.95),
        },
        "generation_ttft_ms": {
            "count": len(ttfts),
            "mean": statistics.fmean(ttfts) if ttfts else None,
            "p50": statistics.median(ttfts) if ttfts else None,
            "p95": _percentile(ttfts, 0.95) if ttfts else None,
        },
        "generation_tokens": {
            "complete": all(row.get("complete") is True for row in generation_usage),
            "input_tokens": _sum_known(generation_usage, "input_tokens"),
            "output_tokens": _sum_known(generation_usage, "output_tokens"),
            "total_tokens": _sum_known(generation_usage, "total_tokens"),
        },
        "judge_tokens": {
            "input_tokens": _sum_known(judge_usage, "input_tokens"),
            "output_tokens": _sum_known(judge_usage, "output_tokens"),
            "total_tokens": _sum_known(judge_usage, "total_tokens"),
        },
        "errors": sum(
            answer.get("error") is not None or judgment.get("error") is not None
            for answer, judgment in zip(answers, judgments, strict=True)
        ),
    }


def run_gate(*, device: str, phase: str) -> dict[str, Any]:
    unit_id, source_rows = _sources("locomo")
    queries = _queries("locomo", unit_id)
    query_digest = _query_digest(queries)
    cache = _load_cache(unit_id, query_digest)
    sources = {source.source_id: source.content for source in source_rows}
    store_before = _validate_store_pair()
    store_after = store_before
    if phase in {"all", "answer"}:
        store_before, store_after = _answer_phase(
            cache,
            device=device,
            sources=sources,
            unit_id=unit_id,
        )
    if phase in {"all", "judge"}:
        missing_answers = [
            f"{label}:{query.question_id}"
            for query in queries
            for label in ("baseline", "candidate")
            if f"{label}:{query.question_id}" not in cache["answers"]
        ]
        if missing_answers:
            raise RuntimeError(f"judge phase has {len(missing_answers)} missing paired answers")
        _judge_phase(cache, unit_id=unit_id)
    question_ids = [query.question_id for query in queries]
    complete = all(
        f"{label}:{question_id}" in cache["judgments"]
        for question_id in question_ids
        for label in ("baseline", "candidate")
    )
    if not complete:
        return {
            "status": "partial",
            "answer_count": len(cache["answers"]),
            "judgment_count": len(cache["judgments"]),
        }
    baseline = _aggregate(cache, "baseline", question_ids)
    candidate = _aggregate(cache, "candidate", question_ids)
    primary_delta = candidate["metrics"]["llm_judge"] - baseline["metrics"]["llm_judge"]
    accepted = (
        primary_delta >= 0
        and candidate["errors"] <= baseline["errors"]
        and candidate["errors"] == 0
    )
    rows = []
    for question_id in question_ids:
        row: dict[str, object] = {"question_id": question_id}
        for label in ("baseline", "candidate"):
            answer = cache["answers"][f"{label}:{question_id}"]
            judgment = cache["judgments"][f"{label}:{question_id}"]
            row[label] = {
                "answer": answer["answer"],
                "retrieved_source_ids": answer["retrieved_source_ids"],
                "returned_source_ids": answer["returned_source_ids"],
                "generation_input_all_original_sources": answer[
                    "generation_input_all_original_sources"
                ],
                "ask_latency_ms": answer["ask_latency_ms"],
                "generation_ttft_ms": answer["generation_ttft_ms"],
                "generation_tokens": answer["generation_tokens"],
                "metrics": judgment["metrics"],
                "retrieved_source_coverage@20": judgment["retrieved_source_coverage@20"],
                "retrieved_source_hit@20": judgment["retrieved_source_hit@20"],
                "judge_usage": judgment["judge_usage"],
                "error": answer["error"] or judgment["error"],
            }
        rows.append(row)
    payload = {
        "schema_version": 1,
        "status": "accepted" if accepted else "rejected",
        "acceptance_gate": {
            "rule": "candidate llm_judge >= baseline and candidate errors <= baseline with zero candidate errors",
            "primary_metric": "llm_judge",
            "primary_delta": primary_delta,
            "accepted": accepted,
        },
        "task": TASK,
        "unit_id": unit_id,
        "query_count": len(question_ids),
        "query_set_sha256": query_digest,
        "paired_order": "SHA-256 question-id parity; fixed baseline/candidate alternation",
        "concurrency": 1,
        "limit": 20,
        "generation": {
            "model": MODEL,
            "temperature": 0.0,
            "max_tokens": None,
            "no_think": True,
            "provider_retries": 0,
            "best_of": 1,
            "generation_received_derived_fact_text": False,
        },
        "scorer": {
            "model": MODEL,
            "version": SCORER_VERSION,
            "protocol": scorer_protocol(TASK),
            "official_model_match": False,
            "same_model_prompt_and_parser_for_both_arms": True,
            "provider_retries": 0,
        },
        "store_state_before": store_before,
        "store_state_after": store_after,
        "baseline": baseline,
        "candidate": candidate,
        "rows": rows,
    }
    _write_json(RESULT_PATH, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "answer", "judge", "check"), default="all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.phase == "check":
        states = _validate_store_pair()
        payload = {"status": "ok", "store_state": states}
    else:
        payload = run_gate(device=args.device, phase=args.phase)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
