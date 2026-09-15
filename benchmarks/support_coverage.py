#!/usr/bin/env python3
"""Measure where a question's gold support sits, and whether the reader can use it.

Three arms share one ingest per question, so they differ only in how the grounding window was
filled and never in the store or its embeddings:

* ``window`` -- ``limit`` rows, no character budget. What ``ask`` grounds with nothing set.
* ``window+linked`` -- the same window plus the records it is structurally linked to. This is the
  only configuration in which after-tail expansion fires at all: a 100-candidate rerank pool runs
  to roughly 89 000 characters on this corpus, so any practical ``evidence_budget_chars`` is
  saturated by the ranked tail and leaves expansion nothing to spend.
* ``budget`` -- ``limit`` rows plus ranked tail up to ``--evidence-budget-chars``. The project's
  adopted default, and the size-matched control for the arm above: it spends a comparable prompt
  on more *ranked* rows instead of on linked ones.

Each arm is scored twice. On the **window**, by where the release's turn-level ``has_answer``
labels sit relative to it -- the decomposition the 2026-09-12 loss analysis is stated in. And on
the **answer**, by the official LongMemEval answer-check prompt and parser. The judge is a proxy:
upstream's is ``gpt-4o-2024-08-06`` and this runs the same model that answers, exactly as the
project's own baseline did, so no number here is comparable to a published row.

Generation and embedding endpoints come from ``~/.config/mindbridge-eval.env`` (override with
``MINDBRIDGE_EVAL_ENV``), because a run executes a committed snapshot and a credential must not
live in the tree that snapshot is taken from.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from openai import APIError, OpenAI

from mindbridge import (
    AnswerPolicy,
    AnswerResult,
    GenerationBackend,
    Memory,
    MemoryType,
    Modality,
    ModelInput,
    ObservationContext,
    SearchHit,
)
from mindbridge.benchmarks._official.longmemeval_prompts import (
    build_answer_check_prompt,
    parse_answer_check,
)
from mindbridge.benchmarks.longmemeval import load_longmemeval
from mindbridge.exceptions import ModelError
from mindbridge.models.openai_sdk import OpenAIModels

RECALL_LIMIT = 12
EMBEDDING_DIMENSION = 2048
INGEST_BATCH = 64
# How many rows expansion put in the window, read from the note the kernel writes into the prompt.
# Inferring it from the row count instead is wrong in the direction that hides the effect: a linked
# row is long and the ranked rows it displaces are short, so an expansion that fired can leave the
# window *smaller*. That proxy is what made a confounded first run look like a clean loss.
_EXPANSION_NOTE = re.compile(r"(\d+) further records linked to the evidence above")
_ENV_PATH = Path(os.environ.get("MINDBRIDGE_EVAL_ENV", "~/.config/mindbridge-eval.env"))


@dataclass(frozen=True, slots=True)
class Arm:
    """One way of filling the grounding window, named so the report can be read."""

    name: str
    budget: int | None
    expansion: bool


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one arm gave one question, on the window and on the answer."""

    question_id: str
    arm: str
    support: str
    gold: int
    found: int
    window_rows: int
    window_chars: int
    expansion_rows: int
    prompt_chars: int
    correct: float
    abstained: bool
    # The window's own IDs, so a run can prove two arms saw one index rather than assume it.
    window_ids: tuple[str, ...] = field(default=(), repr=False)


class _Recorder:
    """Delegates to the real reader and keeps the window and prompt it was handed.

    ``AnswerResult.hits`` is the evidence the backend reports having used, which is a subset of
    what it was grounded on. The decomposition is a claim about the window, so the window has to
    be captured on the way in rather than reconstructed from the way out.
    """

    def __init__(self, backend: GenerationBackend) -> None:
        self._backend = backend
        self.generation_capabilities = backend.generation_capabilities
        self.generation_model = backend.generation_model
        self.grounded: tuple[SearchHit, ...] = ()
        self.prompt_chars = 0
        self.expansion_rows = 0

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        self.grounded = tuple(hits)
        text = question.text or ""
        self.prompt_chars = len(text) + sum(len(hit.content) for hit in self.grounded)
        note = _EXPANSION_NOTE.search(text)
        self.expansion_rows = 0 if note is None else int(note.group(1))
        return self._backend.answer(
            question,
            hits,
            answer_policy=answer_policy,
            exhaustive=exhaustive,
        )

    def close(self) -> None:
        self._backend.close()


def _env() -> Mapping[str, str]:
    """Endpoint settings from the out-of-tree env file, with the process environment winning."""
    values: dict[str, str] = {}
    path = _ENV_PATH.expanduser()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            values[name.strip()] = value.strip()
    values.update({k: v for k, v in os.environ.items() if k.startswith("MINDBRIDGE_")})
    return values


def _support_class(gold: frozenset[str], window: frozenset[str], pool: frozenset[str]) -> str:
    inside = len(gold & window)
    if inside == len(gold):
        return "complete_in_window"
    if inside:
        return "partial_in_window"
    return "beyond_window" if gold & pool else "outside_candidates"


def _embedder(settings: Mapping[str, str]) -> OpenAIModels:
    return OpenAIModels(
        embedding_client=OpenAI(
            base_url=settings["MINDBRIDGE_EMBEDDING_BASE_URL"],
            api_key=settings.get("MINDBRIDGE_EMBEDDING_API_KEY", "local"),
            max_retries=3,
            timeout=600.0,
        ),
        embedding_model=settings["MINDBRIDGE_EMBEDDING_MODEL"],
        embedding_dimension=EMBEDDING_DIMENSION,
        embedding_capabilities=frozenset({Modality.TEXT}),
    )


def _generation_client(settings: Mapping[str, str]) -> OpenAI:
    return OpenAI(
        base_url=settings["MINDBRIDGE_GENERATION_BASE_URL"],
        api_key=settings["MINDBRIDGE_GENERATION_API_KEY"],
        max_retries=4,
        timeout=600.0,
    )


def _reader(settings: Mapping[str, str], client: OpenAI) -> OpenAIModels:
    return OpenAIModels(
        generation_client=client,
        generation_model=settings["MINDBRIDGE_GENERATION_MODEL"],
        generation_capabilities=frozenset({Modality.TEXT}),
        generation_temperature=0.0,
        generation_seed=42,
    )


def _judge(
    client: OpenAI,
    model: str,
    question: object,
    prediction: str,
) -> float:
    """Score one answer with the official answer-check prompt and parser."""
    prompt = build_answer_check_prompt(
        question.question_type,  # type: ignore[attr-defined]
        question.question,  # type: ignore[attr-defined]
        question.reference_answer,  # type: ignore[attr-defined]
        prediction,
        abstention=bool(question.abstention),  # type: ignore[attr-defined]
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=64,
    )
    return float(parse_answer_check(response.choices[0].message.content or ""))


def _open(
    data_dir: Path,
    embedder: OpenAIModels,
    answerer: _Recorder | None,
    arm: Arm,
) -> Memory:
    return Memory(
        data_dir,
        embedder=embedder,
        answerer=answerer,
        minimum_relevance=0.0,
        ambiguity_margin=0.0,
        reinforce_on_answer=False,
        recall_planning=False,
        evidence_budget_chars=arm.budget,
        evidence_expansion=arm.expansion,
    )


def _ingest(memory: Memory, question: object, *, capture_context: bool) -> frozenset[str]:
    """Write one question's haystack and return the turn IDs its answer needs.

    One record per turn, which is the granularity the release labels. The capture is the session,
    which is the only grouping a host of this corpus actually knows -- and the edge under test.
    """
    gold: list[str] = []
    contents: list[str] = []
    occurred: list[object] = []
    metadata: list[dict[str, object]] = []
    contexts: list[ObservationContext | None] = []
    for session in question.sessions:  # type: ignore[attr-defined]
        context = ObservationContext(source_id=session.session_id) if capture_context else None
        for turn in session.turns:
            if not turn.content.strip():
                continue
            contents.append(f"[{session.occurred_at.isoformat()}] {turn.role}: {turn.content}")
            occurred.append(session.occurred_at)
            metadata.append({"source_id": turn.turn_id})
            contexts.append(context)
            if turn.has_answer:
                gold.append(turn.turn_id)
    # One model batch per chunk rather than one call per turn: a unit is ~550 turns, and the
    # embedder is over HTTP.
    for offset in range(0, len(contents), INGEST_BATCH):
        window = slice(offset, offset + INGEST_BATCH)
        memory.add_many(
            contents[window],
            occurred_at=occurred[window],  # type: ignore[arg-type]
            metadata=metadata[window],
            memory_type=MemoryType.EPISODIC,
            context=contexts[window],
        )
    return frozenset(gold)


def _turn_ids(hits: Sequence[SearchHit]) -> frozenset[str]:
    return frozenset(str(hit.metadata["source_id"]) for hit in hits if "source_id" in hit.metadata)


# A shared gateway returns the occasional 502, and one of them threw away 53 minutes of a 120
# question run at question 78. The retry unit is the whole question, not the request: a failure
# part way through leaves some arms scored and some not, and a question is cheap to redo while a
# run is not.
_RETRY_ATTEMPTS = 4
_RETRY_BACKOFF_SECONDS = 5.0


def _question_outcomes(
    question: object,
    *,
    root: Path,
    arms: Sequence[Arm],
    settings: Mapping[str, str],
    client: OpenAI,
    capture_context: bool,
) -> tuple[Outcome, ...]:
    """Ingest one question's store once, then read it once per arm."""
    data_dir = root / str(question.question_id)  # type: ignore[attr-defined]
    embedder = _embedder(settings)
    outcomes: list[Outcome] = []
    try:
        # Ingest closes before any arm reads. A session that wrote merges its Zvec segments at
        # close, and segment layout moves what an approximate search returns, so an arm reading
        # the store it just wrote reads a different index from one that opens it afterwards.
        # Measured that way, five of seven apparent expansion losses were questions where
        # expansion had added no row at all: index state, not the setting under test.
        with _open(data_dir, embedder, None, arms[0]) as memory:
            gold = _ingest(memory, question, capture_context=capture_context)
        for arm in arms:
            recorder = _Recorder(_reader(settings, client))
            with _open(data_dir, embedder, recorder, arm) as memory:
                result = memory.ask(
                    "Answer concisely using only the memories.\n"
                    f"Question: {question.question}\nAnswer:",  # type: ignore[attr-defined]
                    limit=RECALL_LIMIT,
                )
                pool = _turn_ids(memory.search(question.question, limit=100))  # type: ignore[attr-defined]
            window = _turn_ids(recorder.grounded)
            outcomes.append(
                Outcome(
                    question_id=str(question.question_id),  # type: ignore[attr-defined]
                    arm=arm.name,
                    support=_support_class(gold, window, pool | window),
                    gold=len(gold),
                    found=len(gold & window),
                    window_rows=len(recorder.grounded),
                    window_chars=sum(len(hit.content) for hit in recorder.grounded),
                    expansion_rows=recorder.expansion_rows,
                    prompt_chars=recorder.prompt_chars,
                    correct=_judge(
                        client,
                        settings["MINDBRIDGE_GENERATION_MODEL"],
                        question,
                        result.answer,
                    ),
                    abstained=result.abstained,
                    window_ids=tuple(hit.id for hit in recorder.grounded),
                )
            )
    finally:
        embedder.close()
    return tuple(outcomes)


def _summarize(outcomes: Sequence[Outcome]) -> dict[str, object]:
    counts = Counter(outcome.support for outcome in outcomes)
    total = len(outcomes) or 1
    return {
        "questions": len(outcomes),
        "accuracy": round(statistics.fmean(outcome.correct for outcome in outcomes), 4),
        "abstention_rate": round(
            statistics.fmean(float(outcome.abstained) for outcome in outcomes), 4
        ),
        "complete_support": round(counts["complete_in_window"] / total, 4),
        "support_counts": dict(sorted(counts.items())),
        "mean_group_recall": round(
            statistics.fmean(outcome.found / max(outcome.gold, 1) for outcome in outcomes), 4
        ),
        "mean_window_rows": round(statistics.fmean(outcome.window_rows for outcome in outcomes), 2),
        "mean_window_chars": round(
            statistics.fmean(outcome.window_chars for outcome in outcomes), 1
        ),
        "mean_prompt_chars": round(
            statistics.fmean(outcome.prompt_chars for outcome in outcomes), 1
        ),
        "mean_expansion_rows": round(
            statistics.fmean(outcome.expansion_rows for outcome in outcomes), 2
        ),
    }


def _accuracy_by_support(outcomes: Sequence[Outcome]) -> dict[str, object]:
    """The instrument's own premise: does complete support in the window predict the answer."""
    grouped: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.support, []).append(outcome)
    return {
        name: {
            "windows": len(rows),
            "accuracy": round(statistics.fmean(row.correct for row in rows), 4),
            "abstention_rate": round(statistics.fmean(float(row.abstained) for row in rows), 4),
        }
        for name, rows in sorted(grouped.items())
    }


def _paired(
    base: Sequence[Outcome],
    arm: Sequence[Outcome],
    *,
    same_budget: bool,
) -> dict[str, object]:
    """Win, loss and tie against one control, question by question on the same store."""
    by_id = {outcome.question_id: outcome for outcome in base}
    support = Counter[str]()
    answer = Counter[str]()
    unexplained = 0
    for outcome in arm:
        before = by_id[outcome.question_id]
        # The self-check that says the pairing held: an arm on the control's own budget that
        # added no linked row must have grounded the identical window, because nothing else
        # differs. A disagreement is index state leaking into the comparison. An arm on a
        # different budget is expected to differ, so the check does not apply to it.
        if same_budget and outcome.expansion_rows == 0 and outcome.window_ids != before.window_ids:
            unexplained += 1
        for label, was, now in (
            (
                "support",
                before.support == "complete_in_window",
                outcome.support == "complete_in_window",
            ),
            ("answer", before.correct > 0.5, outcome.correct > 0.5),
        ):
            counter = support if label == "support" else answer
            counter["won" if now and not was else "lost" if was and not now else "tied"] += 1
    return {
        "complete_support": dict(support),
        "answer": dict(answer),
        "unexplained_window_changes": unexplained if same_budget else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--evidence-budget-chars", type=int, default=24_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--no-capture-context",
        action="store_true",
        help="ingest without an ObservationContext, which is what the harness does today",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = _env()
    missing = [
        name
        for name in (
            "MINDBRIDGE_EMBEDDING_BASE_URL",
            "MINDBRIDGE_EMBEDDING_MODEL",
            "MINDBRIDGE_GENERATION_BASE_URL",
            "MINDBRIDGE_GENERATION_API_KEY",
            "MINDBRIDGE_GENERATION_MODEL",
        )
        if not settings.get(name)
    ]
    if missing:
        parser.error(f"{_ENV_PATH} is missing {', '.join(missing)}")

    arms = (
        Arm("window", budget=None, expansion=False),
        Arm("window+linked", budget=None, expansion=True),
        Arm("budget", budget=args.evidence_budget_chars, expansion=False),
    )
    questions = load_longmemeval(args.dataset)[args.offset : args.offset + args.limit]
    client = _generation_client(settings)
    done = threading.Lock()
    finished = 0
    abandoned: list[tuple[str, str]] = []
    started = time.perf_counter()
    collected: list[tuple[Outcome, ...]] = []
    with tempfile.TemporaryDirectory(prefix="support-coverage-") as root:

        def run(question: object) -> tuple[Outcome, ...]:
            nonlocal finished, abandoned
            question_id = str(question.question_id)  # type: ignore[attr-defined]
            outcomes: tuple[Outcome, ...] = ()
            reason = ""
            for attempt in range(_RETRY_ATTEMPTS):
                try:
                    outcomes = _question_outcomes(
                        question,
                        root=Path(root),
                        arms=arms,
                        settings=settings,
                        client=client,
                        capture_context=not args.no_capture_context,
                    )
                    break
                except (APIError, ModelError) as error:
                    reason = f"{type(error).__name__}: {error}"
                    if attempt + 1 == _RETRY_ATTEMPTS:
                        break
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
            with done:
                finished += 1
                if outcomes:
                    shown = " ".join(
                        f"{o.arm}={o.support[:8]}/{'y' if o.correct > 0.5 else 'n'}"
                        for o in outcomes
                    )
                else:
                    abandoned.append((question_id, reason))
                    shown = f"ABANDONED {reason[:120]}"
                print(
                    f"[{finished}/{len(questions)}] {question_id} {shown}",
                    file=sys.stderr,
                    flush=True,
                )
            return outcomes

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            collected = [rows for rows in pool.map(run, questions) if rows]

    by_arm = {
        arm.name: [row for rows in collected for row in rows if row.arm == arm.name] for arm in arms
    }
    control = by_arm[arms[0].name]
    report = {
        "dataset": str(args.dataset),
        "questions": len(questions),
        "scored_questions": len(collected),
        # Named, not silently dropped: a report whose denominator moved has to say so.
        "abandoned": [{"question_id": qid, "reason": why} for qid, why in abandoned],
        "offset": args.offset,
        "recall_limit": RECALL_LIMIT,
        "evidence_budget_chars": args.evidence_budget_chars,
        "capture_context": not args.no_capture_context,
        "answer_model": settings["MINDBRIDGE_GENERATION_MODEL"],
        "judge": {
            "model": settings["MINDBRIDGE_GENERATION_MODEL"],
            "protocol": "longmemeval_official_answer_check",
            "note": (
                "proxy judge: upstream's is gpt-4o-2024-08-06 and this is the model that also "
                "answered, so no number here is comparable to a published row"
            ),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "arms": {name: _summarize(rows) for name, rows in by_arm.items()},
        "accuracy_by_support": _accuracy_by_support(
            [row for rows in collected for row in rows],
        ),
        "paired_against_window": {
            arm.name: _paired(
                control,
                by_arm[arm.name],
                same_budget=arm.budget == arms[0].budget,
            )
            for arm in arms[1:]
        },
        "samples": [
            {row.arm: {k: v for k, v in asdict(row).items() if k != "window_ids"} for row in rows}
            for rows in collected
        ],
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
