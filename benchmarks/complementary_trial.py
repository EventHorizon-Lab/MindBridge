"""Frozen paired research comparison using Memory.ask and an injectable backend."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx2 as httpx
from openai import OpenAI

from mindbridge import (
    AnswerPolicy,
    AnswerResult,
    Memory,
    Modality,
    ModelInput,
    OpenAIModels,
    RetrievalMode,
    SearchHit,
)
from mindbridge.benchmarks.complementary import (
    SELECTOR_PROMPT,
    parse_plan,
    select_evidence,
    selector_payload,
)
from mindbridge.benchmarks.icm_bench import load_icm_bench
from mindbridge.benchmarks.official_scorers import judge_plan, parse_judge_response

ROOT = Path.home() / ".local/share/openresearch/benchmark-cache/mindbridge"
BASELINE = ROOT / "icm-four-baseline-20260917"
REFERENCE = datetime(2026, 9, 17, tzinfo=timezone.utc)
ARMS = ("rank", "relevance", "coverage")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def redact_media(value: object) -> object:
    if isinstance(value, str) and value.startswith("data:"):
        return {"sha256": hashlib.sha256(value.encode()).hexdigest(), "encoded_chars": len(value)}
    if isinstance(value, dict):
        return {k: redact_media(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_media(v) for v in value]
    return value


class Audit:
    """Actual HTTP payloads, without credentials or raw encoded media."""

    def __init__(self, output: Path) -> None:
        self.output, self.phase, self.case, self.rows = output, "initialization", "", []

    def response(self, response: httpx.Response) -> None:
        response.read()
        request = json.loads(response.request.content)
        media, text_chars = Counter(), 0
        for message in request.get("messages", []):
            content = message.get("content", "")
            if isinstance(content, str):
                text_chars += len(content)
            else:
                for part in content:
                    if part.get("type") == "text":
                        text_chars += len(part["text"])
                    elif part.get("type") in {"image_url", "video_url", "input_audio"}:
                        media[part["type"]] += 1
        try:
            body = response.json()
        except ValueError:
            body = {}
        row = {
            "case": self.case,
            "phase": self.phase,
            "http_status": response.status_code,
            "text_chars": text_chars,
            "media_parts": dict(media),
            "request": redact_media(request),
            "usage": body.get("usage"),
            "response_model": body.get("model"),
            "fingerprint": body.get("system_fingerprint"),
            "choices": body.get("choices"),
        }
        self.rows.append({k: v for k, v in row.items() if k not in {"request", "choices"}})
        write_row(self.output, row)


class SelectorBackend:
    """All arms receive identical routed candidates through the public SDK."""

    def __init__(
        self,
        backend: OpenAIModels,
        client: OpenAI,
        model: str,
        audit: Audit,
        recipe: dict[str, Any],
        output: Path,
    ) -> None:
        self.backend, self.client, self.model = backend, client, model
        self.audit, self.recipe, self.output = audit, recipe, output
        self.generation_capabilities = backend.generation_capabilities
        self.arm, self.plan, self.payload = "rank", None, None
        self.selector_seconds = 0.0
        self.selected, self.pool = (), ()
        self.row_limit = recipe["reader_rows"]

    def reset(self) -> None:
        self.plan = self.payload = None
        self.selector_seconds = 0.0

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        pool = tuple(hits[: self.recipe["candidate_rows"]])
        payload = selector_payload(question.text, pool)
        if self.payload is None:
            self.payload = payload
            self.audit.phase = "shared_selector"
            started = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                seed=0,
                max_tokens=6000,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                messages=[
                    {"role": "system", "content": SELECTOR_PROMPT},
                    {"role": "user", "content": payload},
                ],
            )
            raw = response.choices[0].message.content or ""
            self.selector_seconds = time.perf_counter() - started
            error_type = None
            try:
                self.plan = parse_plan(raw, len(pool))
            except ValueError as error:
                error_type = type(error).__name__
            write_row(
                self.output / "plans.jsonl",
                {
                    "case": self.audit.case,
                    "input_sha256": digest(payload),
                    "input": json.loads(payload),
                    "raw": raw,
                    "valid": self.plan is not None,
                    "error": error_type,
                },
            )
        elif payload != self.payload:
            raise RuntimeError("paired arms observed different candidate content/order")
        self.pool = pool
        self.selected = select_evidence(
            pool,
            self.plan,
            arm=self.arm if self.plan is not None else "rank",
            rows=self.row_limit,
            chars=self.recipe["reader_chars"],
        )
        self.audit.phase = "answer/" + self.arm
        return self.backend.answer(
            question, self.selected, answer_policy=answer_policy, exhaustive=False
        )

    def close(self) -> None:
        self.backend.close()


def open_memory(
    path: Path,
    embedder: OpenAIModels,
    reader: SelectorBackend,
    *,
    vision: OpenAIModels | None = None,
) -> Memory:
    return Memory(
        path,
        embedder=embedder,
        answerer=reader,
        vision_describer=vision,
        retrieval_mode=RetrievalMode.HYBRID,
        recall_planning=False,
        evidence_expansion=False,
        reinforce_on_answer=False,
        minimum_relevance=0,
        ambiguity_margin=0,
        evidence_budget_chars=None,
    )


def vision_backend(client: OpenAI, model: str) -> OpenAIModels:
    # The describer contract accepts visual modalities only, unlike the answerer.
    return OpenAIModels(
        generation_client=client,
        generation_model=model,
        generation_capabilities=frozenset({Modality.IMAGE, Modality.VIDEO}),
        generation_temperature=0,
        generation_seed=0,
        generation_video_limit=8,
        generation_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def score(client: OpenAI, model: str, question: str, reference: str, prediction: str) -> float:
    plan = judge_plan(
        "icm-bench", question=question, references=(reference,), prediction=prediction, metadata={}
    )
    if plan is None:
        return 0.0
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=8192,
        messages=[{"role": message.role, "content": message.content} for message in plan.calls[0]],
    )
    return parse_judge_response(plan, response.choices[0].message.content or "")["accuracy"]


def source(hit: SearchHit) -> str:
    return str(hit.metadata.get("source_id", hit.id))


def evaluate(
    memory: Memory, reader: SelectorBackend, audit: Audit, case: dict[str, Any], output: Path
) -> list[dict[str, Any]]:
    reader.reset()
    audit.case = case["id"]
    order = list(ARMS)
    random.Random(case["id"]).shuffle(order)
    predictions = []
    # Every prediction is complete before the reference enters any judge request.
    for arm in order:
        reader.arm, audit.phase = arm, "sdk/" + arm
        start = time.perf_counter()
        selector_before = reader.selector_seconds
        answer = memory.ask(
            case["question"],
            limit=reader.recipe["candidate_rows"],
            reference_at=REFERENCE,
            answer_policy="best_effort" if case["dataset"] == "icm-profile-dev" else "strict",
            link_identities=False,
        )
        predictions.append(
            {
                "case": case["id"],
                "dataset": case["dataset"],
                "family": case.get("family"),
                "arm": arm,
                "prediction": answer.answer,
                "abstained": answer.abstained,
                "seconds": time.perf_counter() - start,
                "shared_selector_seconds": reader.selector_seconds - selector_before,
                "plan_valid": reader.plan is not None,
                "candidate_ids": [source(h) for h in reader.pool],
                "selected_ids": [source(h) for h in reader.selected],
                "returned_ids": [source(h) for h in answer.hits],
                "selected_chars": sum(len(h.content) for h in reader.selected),
                "selected_media": [a.id for h in reader.selected for a in h.assets],
                "returned_media": [a.id for h in answer.hits for a in h.assets],
            }
        )
        write_row(output / "predictions.jsonl", predictions[-1])
    for row in predictions:
        audit.phase = "judge/" + row["arm"]
        row["score"] = score(
            reader.client, reader.model, case["question"], case["reference"], row["prediction"]
        )
        gold = set(case.get("gold", []))
        row["gold_source_recall"] = (
            len(gold & set(row["selected_ids"])) / len(gold) if gold else None
        )
        row["complete_gold_sources"] = gold <= set(row["selected_ids"]) if gold else None
        write_row(output / "samples.jsonl", row)
    print(
        "CASE_RESULT "
        + json.dumps(
            {
                "id": case["id"],
                "dataset": case["dataset"],
                "plan_valid": reader.plan is not None,
                "scores": {r["arm"]: r["score"] for r in predictions},
                "selected_counts": {r["arm"]: len(r["selected_ids"]) for r in predictions},
            }
        ),
        flush=True,
    )
    return predictions


def run_cases(
    recipe: dict[str, Any],
    output: Path,
    embedder: OpenAIModels,
    reader: SelectorBackend,
    audit: Audit,
    vision: OpenAIModels,
) -> list[dict[str, Any]]:
    from paired_replay import _closed_store_lock

    fixture_bytes = Path(__file__).with_name("complementary_scenarios.json").read_bytes()
    if hashlib.sha256(fixture_bytes).hexdigest() != recipe["fixture_sha256"]:
        raise RuntimeError("diagnostic fixture changed since protocol freeze")
    fixtures = json.loads(fixture_bytes)
    results = []
    for case in fixtures:
        reader.row_limit = recipe["diagnostic_rows"]
        with open_memory(output / "stores" / case["id"], embedder, reader) as memory:
            audit.case, audit.phase = case["id"], "ingest"
            ids = {}
            for observation in case["observations"]:
                record = memory.add(observation["text"], metadata={"source_id": observation["id"]})
                ids[observation["id"]] = record.id
            for deleted in case.get("delete", []):
                assert memory.delete(ids[deleted])
            results.extend(evaluate(memory, reader, audit, case, output))
    stores = list((BASELINE / "stores/icm-bench").rglob("state.sqlite3"))
    if len(stores) != 1:
        raise RuntimeError("expected exactly one frozen ICM baseline store")
    copied = output / "stores/icm-profile"
    with _closed_store_lock(stores[0].parent):
        source_hash = hashlib.sha256(stores[0].read_bytes()).hexdigest()
        if source_hash != recipe["source_sqlite_sha256"]:
            raise RuntimeError("baseline snapshot changed since protocol freeze")
        shutil.copytree(stores[0].parent, copied)
    questions = {
        q.question_id: q
        for q in load_icm_bench(BASELINE / "datasets/icm-bench/annotations/qa_test.jsonl")
    }
    print(
        "STORE_SNAPSHOT "
        + json.dumps(
            {
                "parent_run": "a5f37edc-3a12-4f05-b294-a2da79de09fa",
                "sqlite_sha256": source_hash,
            }
        ),
        flush=True,
    )
    reader.row_limit = recipe["reader_rows"]
    with open_memory(copied, embedder, reader, vision=vision) as memory:
        for identifier in recipe["question_ids"]:
            q = questions[identifier]
            if q.before_clip is not None:
                raise RuntimeError("full-store trial only supports full-timeline Profile")
            case = {
                "id": identifier,
                "dataset": "icm-profile-dev",
                "question": q.question,
                "reference": q.reference_answer,
                "gold": list(q.evidence_video_ids),
            }
            results.extend(evaluate(memory, reader, audit, case, output))
    if hashlib.sha256(stores[0].read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("original baseline changed during the experiment")
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = {}
    for dataset in sorted({row["dataset"] for row in results}):
        summaries[dataset] = {}
        for arm in ARMS:
            rows = [r for r in results if r["dataset"] == dataset and r["arm"] == arm]
            summaries[dataset][arm] = {
                "n": len(rows),
                "accuracy": statistics.mean(r["score"] for r in rows),
                "valid_plans": sum(r["plan_valid"] for r in rows),
                "mean_selected_chars": statistics.mean(r["selected_chars"] for r in rows),
                "source_complete": sum(r["complete_gold_sources"] is True for r in rows),
            }
        control = {
            r["case"]: r for r in results if r["dataset"] == dataset and r["arm"] == "relevance"
        }
        treatment = [r for r in results if r["dataset"] == dataset and r["arm"] == "coverage"]
        summaries[dataset]["coverage_vs_relevance"] = {
            "wins": sum(r["score"] > control[r["case"]]["score"] for r in treatment),
            "losses": sum(r["score"] < control[r["case"]]["score"] for r in treatment),
        }
    return summaries


def main() -> int:
    from support_coverage import _env

    recipe = json.loads(Path(__file__).with_suffix(".json").read_text())
    settings = _env()
    output = ROOT / recipe["suite_id"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.jsonl").exists():
        raise RuntimeError("refusing to overwrite a trial; preserve partial evidence")
    print("TRIAL_RECIPE " + json.dumps(recipe), flush=True)
    print(
        "TRIAL_MODELS "
        + json.dumps(
            {k: settings[k] for k in ["MINDBRIDGE_EMBEDDING_MODEL", "MINDBRIDGE_GENERATION_MODEL"]}
        ),
        flush=True,
    )
    audit = Audit(output / "http-audit.jsonl")
    client = OpenAI(
        base_url=settings["MINDBRIDGE_GENERATION_BASE_URL"],
        api_key=settings["MINDBRIDGE_GENERATION_API_KEY"],
        timeout=180,
        max_retries=2,
        http_client=httpx.Client(timeout=180, event_hooks={"response": [audit.response]}),
    )
    embedding_client = OpenAI(
        base_url=settings["MINDBRIDGE_EMBEDDING_BASE_URL"],
        api_key=settings.get("MINDBRIDGE_EMBEDDING_API_KEY", "EMPTY"),
        timeout=180,
        max_retries=2,
    )
    modalities = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})
    embedder = OpenAIModels(
        embedding_client=embedding_client,
        embedding_model=settings["MINDBRIDGE_EMBEDDING_MODEL"],
        embedding_dimension=2048,
        embedding_request_format="messages",
        embedding_capabilities=modalities,
    )
    backend = OpenAIModels(
        generation_client=client,
        generation_model=settings["MINDBRIDGE_GENERATION_MODEL"],
        generation_capabilities=modalities,
        generation_temperature=0,
        generation_seed=0,
        generation_video_limit=8,
        generation_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    reader = SelectorBackend(
        backend, client, settings["MINDBRIDGE_GENERATION_MODEL"], audit, recipe, output
    )
    vision = vision_backend(client, settings["MINDBRIDGE_GENERATION_MODEL"])
    try:
        results = run_cases(recipe, output, embedder, reader, audit, vision)
    finally:
        client.close()
        embedding_client.close()
    report = {
        "recipe": recipe,
        "results": summarize(results),
        "http_requests": audit.rows,
        "limitations": [
            "ICM development subset, already inspected",
            "synthetic text diagnostics are not multimodal generalization",
            "shared selector costs allocated to both reranking arms",
            "reader caps equal, actual token costs may differ",
            "not directly comparable to full SDK baseline with recall planning",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "TRIAL_SUMMARY "
        + json.dumps(
            {
                "results": report["results"],
                "output": str(output),
                "limitations": report["limitations"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
