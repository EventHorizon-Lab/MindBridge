"""Frozen paired research comparison using Memory.ask and an injectable backend."""

from __future__ import annotations

import fcntl
import hashlib
import json
import random
import shutil
import statistics
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

import httpx2 as httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

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
    Arm,
    EvidencePlan,
    parse_plan,
    select_evidence,
    selector_payload,
)
from mindbridge.benchmarks.eval_adapters import dataset_digest
from mindbridge.benchmarks.icm_bench import ICMQuestion, load_icm_bench
from mindbridge.benchmarks.official_scorers import judge_plan, parse_judge_response
from mindbridge.benchmarks.task_catalog import TASKS

ROOT = Path.home() / ".local/share/openresearch/benchmark-cache/mindbridge"
BASELINE = ROOT / "icm-four-baseline-20260917"
REFERENCE = datetime(2026, 9, 17, tzinfo=timezone.utc)
ARMS: tuple[Arm, ...] = ("rank", "relevance", "coverage")

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonemptyText = Annotated[str, Field(min_length=1, pattern=r"\S")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    suite_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]
    candidate_rows: PositiveInt = Field(le=100)
    reader_rows: PositiveInt
    reader_chars: PositiveInt
    diagnostic_rows: PositiveInt
    question_ids: list[NonemptyText] = Field(min_length=1)
    selection_rule: NonemptyText
    selector_version: NonemptyText
    video_limit: PositiveInt
    reuse: NonemptyText
    source_sqlite_sha256: Sha256
    source_tree_sha256: Sha256
    fixture_sha256: Sha256
    protocol_sha256: Sha256

    @model_validator(mode="after")
    def valid_selection(self) -> Recipe:
        if max(self.reader_rows, self.diagnostic_rows) > self.candidate_rows:
            raise ValueError("reader rows must fit the candidate pool")
        if len(set(self.question_ids)) != len(self.question_ids):
            raise ValueError("question IDs must be unique")
        return self


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: NonemptyText
    text: NonemptyText


class TrialCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: NonemptyText
    dataset: NonemptyText
    question: NonemptyText
    reference: NonemptyText
    family: NonemptyText | None = None
    pair: NonemptyText | None = None
    variant: int | None = None
    gold: list[NonemptyText] = Field(default_factory=list)


class FixtureCase(TrialCase):
    id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
    observations: list[Observation] = Field(min_length=1)
    delete: list[NonemptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_sources(self) -> FixtureCase:
        if self.id == "icm-profile":
            raise ValueError("fixture ID icm-profile is reserved for the baseline store")
        sources = {observation.id for observation in self.observations}
        if len(sources) != len(self.observations):
            raise ValueError("observation IDs must be unique")
        if len(set(self.delete)) != len(self.delete) or not set(self.delete) <= sources:
            raise ValueError("deletions must reference unique observation IDs")
        if not set(self.gold) <= sources - set(self.delete):
            raise ValueError("gold sources must reference live observation IDs")
        return self


class Prediction(TypedDict):
    case: str
    dataset: str
    family: str | None
    arm: Arm
    prediction: str
    abstained: bool
    seconds: float
    shared_selector_seconds: float
    plan_valid: bool
    candidate_ids: list[str]
    selected_ids: list[str]
    returned_ids: list[str]
    selected_chars: int
    selected_media: list[str]
    returned_media: list[str]


class TrialResult(Prediction):
    score: float
    gold_source_recall: float | None
    complete_gold_sources: bool | None


class TreeManifest(TypedDict):
    sha256: str
    files: int
    bytes: int


class TreeEntry(TypedDict):
    path: str
    bytes: int
    sha256: str


def load_trial_inputs() -> tuple[Recipe, tuple[FixtureCase, ...]]:
    recipe = Recipe.model_validate_json(Path(__file__).with_suffix(".json").read_bytes())
    fixture_bytes = Path(__file__).with_name("complementary_scenarios.json").read_bytes()
    if hashlib.sha256(fixture_bytes).hexdigest() != recipe.fixture_sha256:
        raise RuntimeError("diagnostic fixture changed since protocol freeze")
    fixtures = TypeAdapter(list[FixtureCase]).validate_json(fixture_bytes)
    if not fixtures or len({case.id for case in fixtures}) != len(fixtures):
        raise ValueError("fixture case IDs must be non-empty and unique")
    return recipe, tuple(fixtures)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_row(path: Path, row: object) -> None:
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
        self.output, self.phase, self.case = output, "initialization", ""
        self.rows: list[dict[str, object]] = []

    def response(self, response: httpx.Response) -> None:
        response.read()
        request = json.loads(response.request.content)
        media: Counter[str] = Counter()
        text_chars = 0
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
        recipe: Recipe,
        output: Path,
    ) -> None:
        self.backend, self.client, self.model = backend, client, model
        self.audit, self.recipe, self.output = audit, recipe, output
        self.generation_capabilities = backend.generation_capabilities
        self.arm: Arm = "rank"
        self.plan: EvidencePlan | None = None
        self.payload: str | None = None
        self.selector_seconds = 0.0
        self.selected: tuple[SearchHit, ...] = ()
        self.pool: tuple[SearchHit, ...] = ()
        self.row_limit = recipe.reader_rows

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
        pool = tuple(hits[: self.recipe.candidate_rows])
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
            self.selector_seconds = time.perf_counter() - started
        elif payload != self.payload:
            raise RuntimeError("paired arms observed different candidate content/order")
        self.pool = pool
        self.selected = select_evidence(
            pool,
            self.plan,
            arm=self.arm if self.plan is not None else "rank",
            rows=self.row_limit,
            chars=self.recipe.reader_chars,
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
    messages: list[ChatCompletionMessageParam] = []
    for message in plan.calls[0]:
        if message.role == "system":
            messages.append({"role": "system", "content": message.content})
        else:
            messages.append({"role": "user", "content": message.content})
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=8192,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=messages,
    )
    return parse_judge_response(plan, response.choices[0].message.content or "")["accuracy"]


def source(hit: SearchHit) -> str:
    return str(hit.metadata.get("source_id", hit.id))


def evaluate(
    memory: Memory, reader: SelectorBackend, audit: Audit, case: TrialCase, output: Path
) -> list[TrialResult]:
    reader.reset()
    audit.case = case.id
    order = list(ARMS)
    random.Random(case.id).shuffle(order)
    predictions: list[Prediction] = []
    # Every prediction is complete before the reference enters any judge request.
    for arm in order:
        reader.arm, audit.phase = arm, "sdk/" + arm
        start = time.perf_counter()
        selector_before = reader.selector_seconds
        answer = memory.ask(
            case.question,
            limit=reader.recipe.candidate_rows,
            reference_at=REFERENCE,
            answer_policy="best_effort" if case.dataset == "icm-profile-dev" else "strict",
            link_identities=False,
        )
        selector_elapsed = reader.selector_seconds - selector_before
        predictions.append(
            {
                "case": case.id,
                "dataset": case.dataset,
                "family": case.family,
                "arm": arm,
                "prediction": answer.answer,
                "abstained": answer.abstained,
                "seconds": time.perf_counter() - start - selector_elapsed,
                "shared_selector_seconds": 0.0 if arm == "rank" else reader.selector_seconds,
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
    results: list[TrialResult] = []
    for prediction in predictions:
        audit.phase = "judge/" + prediction["arm"]
        accuracy = score(
            reader.client, reader.model, case.question, case.reference, prediction["prediction"]
        )
        gold, selected = set(case.gold), set(prediction["selected_ids"])
        row: TrialResult = {
            **prediction,
            "score": accuracy,
            "gold_source_recall": len(gold & selected) / len(gold) if gold else None,
            "complete_gold_sources": gold <= selected if gold else None,
        }
        write_row(output / "samples.jsonl", row)
        results.append(row)
    print(
        "CASE_RESULT "
        + json.dumps(
            {
                "id": case.id,
                "dataset": case.dataset,
                "plan_valid": reader.plan is not None,
                "scores": {r["arm"]: r["score"] for r in results},
                "selected_counts": {r["arm"]: len(r["selected_ids"]) for r in results},
            }
        ),
        flush=True,
    )
    return results


def frozen_questions(path: Path) -> dict[str, ICMQuestion]:
    if dataset_digest(path) != TASKS["icm-bench"].digest:
        raise RuntimeError("ICM question dataset changed since protocol freeze")
    return {q.question_id: q for q in load_icm_bench(path)}


def load_profile_cases(recipe: Recipe) -> tuple[TrialCase, ...]:
    questions = frozen_questions(BASELINE / "datasets/icm-bench/annotations/qa_test.jsonl")
    profiles = []
    for identifier in recipe.question_ids:
        if identifier not in questions:
            raise ValueError(f"selected question is missing from the frozen dataset: {identifier}")
        question = questions[identifier]
        if question.before_clip is not None:
            raise RuntimeError("full-store trial only supports full-timeline Profile")
        profiles.append(
            TrialCase(
                id=identifier,
                dataset="icm-profile-dev",
                question=question.question,
                reference=question.reference_answer,
                gold=list(question.evidence_video_ids),
            )
        )
    return tuple(profiles)


def tree_manifest(root: Path) -> TreeManifest:
    """Describe a store by content, independent of its clone path and file timestamps."""
    root = root.resolve()
    entries: list[TreeEntry] = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": dataset_digest(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    payload = json.dumps(
        entries, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
    }


@contextmanager
def _closed_store_lock(path: Path) -> Iterator[None]:
    if not path.is_dir() or not (path / "state.sqlite3").is_file():
        raise FileNotFoundError(f"source store is missing state.sqlite3: {path}")
    lock_path = path / ".mindbridge.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(f"source store is missing its owner lock: {lock_path}")
    with lock_path.open("rb") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"source store is still owned: {path}") from None
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def copy_baseline(source: Path, destination: Path, recipe: Recipe) -> TreeManifest:
    with _closed_store_lock(source):
        if dataset_digest(source / "state.sqlite3") != recipe.source_sqlite_sha256:
            raise RuntimeError("baseline snapshot changed since protocol freeze")
        manifest = tree_manifest(source)
        if manifest["sha256"] != recipe.source_tree_sha256:
            raise RuntimeError("baseline store tree changed since protocol freeze")
        shutil.copytree(source, destination)
        if tree_manifest(destination) != manifest:
            raise RuntimeError("baseline store changed while copying")
    return manifest


def run_cases(
    recipe: Recipe,
    fixtures: Sequence[FixtureCase],
    profiles: Sequence[TrialCase],
    output: Path,
    embedder: OpenAIModels,
    reader: SelectorBackend,
    audit: Audit,
    vision: OpenAIModels,
) -> list[TrialResult]:
    results: list[TrialResult] = []
    for case in fixtures:
        reader.row_limit = recipe.diagnostic_rows
        with open_memory(output / "stores" / case.id, embedder, reader) as memory:
            audit.case, audit.phase = case.id, "ingest"
            ids = {}
            for observation in case.observations:
                record = memory.add(observation.text, metadata={"source_id": observation.id})
                ids[observation.id] = record.id
            for deleted in case.delete:
                assert memory.delete(ids[deleted])
            results.extend(evaluate(memory, reader, audit, case, output))
    stores = list((BASELINE / "stores/icm-bench").rglob("state.sqlite3"))
    if len(stores) != 1:
        raise RuntimeError("expected exactly one frozen ICM baseline store")
    copied = output / "stores/icm-profile"
    manifest = copy_baseline(stores[0].parent, copied, recipe)
    print(
        "STORE_SNAPSHOT "
        + json.dumps(
            {
                "parent_run": "a5f37edc-3a12-4f05-b294-a2da79de09fa",
                "sqlite_sha256": recipe.source_sqlite_sha256,
                "store_tree": manifest,
            }
        ),
        flush=True,
    )
    reader.row_limit = recipe.reader_rows
    with open_memory(copied, embedder, reader, vision=vision) as memory:
        for profile in profiles:
            results.extend(evaluate(memory, reader, audit, profile, output))
    with _closed_store_lock(stores[0].parent):
        if tree_manifest(stores[0].parent) != manifest:
            raise RuntimeError("original baseline changed during the experiment")
    return results


def summarize(results: Sequence[TrialResult]) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
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

    recipe, fixtures = load_trial_inputs()
    output = ROOT / recipe.suite_id
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing to overwrite a trial; preserve partial evidence")
    profiles = load_profile_cases(recipe)
    settings = _env()
    output.mkdir(parents=True, exist_ok=True)
    print("TRIAL_RECIPE " + recipe.model_dump_json(), flush=True)
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
        results = run_cases(recipe, fixtures, profiles, output, embedder, reader, audit, vision)
    finally:
        client.close()
        embedding_client.close()
    report = {
        "recipe": recipe.model_dump(mode="json"),
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
