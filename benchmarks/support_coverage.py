#!/usr/bin/env python3
"""Measure where a question's gold support sits, with and without evidence expansion.

No generation model is involved. The answerer is a stub that reports the evidence it was handed,
so every arm runs the real `ask()` path -- ranking, grounding window, budget, expansion -- and is
scored on the *window*, not on an answer. That is the quantity the loss decomposition in
`docs/research/2026-09-12-baseline-loss-decomposition.md` showed predicts the answer: complete
support in the window scored 0.83, partial support 0.40.

Both arms read one store per unit. The store is ingested once, closed, and reopened with the arm's
setting, so the arms differ in exactly one boolean and share their embeddings byte for byte.

    uv run --locked python benchmarks/support_coverage.py \\
        --dataset ~/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval/snapshots/<rev>/longmemeval_s \\
        --limit 60 --output .benchmarks/support-coverage.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

from mindbridge import (
    AnswerPolicy,
    AnswerResult,
    Memory,
    MemoryType,
    Modality,
    ModelInput,
    ObservationContext,
    SearchHit,
)
from mindbridge.benchmarks.longmemeval import load_longmemeval
from mindbridge.models.openai_sdk import OpenAIModels

# The window the reader is grounded on, and the depth the ranking is allowed to reach. Both are
# the baseline run's own settings, so a decomposition here is comparable to the one that motivated
# this measurement rather than to a differently-tuned retrieval.
RECALL_LIMIT = 12
EMBEDDING_DIMENSION = 2048
INGEST_BATCH = 64
# How many rows expansion put in the window, read from the note the kernel writes into the prompt.
# Inferring it from the row count instead is wrong in the direction that hides the effect: a linked
# row is long and the ranked rows it displaces are short, so an expansion that fired can leave the
# window *smaller*. That proxy is what made a confounded first run look like a clean loss.
_EXPANSION_NOTE = re.compile(r"(\d+) further records linked to the evidence above")


class _StubAnswerer:
    """Reports the grounded window instead of answering from it.

    A real reader would decide the score; this measures what the reader would have been given,
    which is the half of the loss this change acts on and the half that needs no generation model.
    """

    generation_capabilities = frozenset({Modality.TEXT})
    generation_model = "stub-window-recorder"

    def __init__(self) -> None:
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
        del answer_policy, exhaustive
        self.grounded = tuple(hits)
        text = question.text or ""
        self.prompt_chars = len(text)
        note = _EXPANSION_NOTE.search(text)
        self.expansion_rows = 0 if note is None else int(note.group(1))
        return AnswerResult(answer="recorded", hits=self.grounded)

    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one question's window held, in the terms the decomposition is stated in."""

    question_id: str
    support: str
    gold: int
    found: int
    window_rows: int
    window_chars: int
    expansion_rows: int
    # The window's own IDs, so a run can prove the two arms saw one index rather than assume it.
    window_ids: tuple[str, ...]


def _support_class(gold: frozenset[str], window: frozenset[str], pool: frozenset[str]) -> str:
    inside = len(gold & window)
    if inside == len(gold):
        return "complete_in_window"
    if inside:
        return "partial_in_window"
    return "beyond_window" if gold & pool else "outside_candidates"


def _models(base_url: str, api_key: str, model: str) -> OpenAIModels:
    return OpenAIModels(
        embedding_client=OpenAI(base_url=base_url, api_key=api_key, max_retries=3),
        embedding_model=model,
        embedding_dimension=EMBEDDING_DIMENSION,
        embedding_capabilities=frozenset({Modality.TEXT}),
    )


def _open(
    data_dir: Path,
    embedder: OpenAIModels,
    answerer: _StubAnswerer,
    *,
    expansion: bool,
    budget: int | None,
) -> Memory:
    return Memory(
        data_dir,
        embedder=embedder,
        answerer=answerer,
        minimum_relevance=0.0,
        ambiguity_margin=0.0,
        reinforce_on_answer=False,
        recall_planning=False,
        evidence_budget_chars=budget,
        evidence_expansion=expansion,
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


def _run_arm(
    memory: Memory,
    answerer: _StubAnswerer,
    question: object,
    gold: frozenset[str],
) -> Outcome:
    memory.ask(question.question, limit=RECALL_LIMIT)  # type: ignore[attr-defined]
    window = _turn_ids(answerer.grounded)
    pool = _turn_ids(memory.search(question.question, limit=100))  # type: ignore[attr-defined]
    rows = len(answerer.grounded)
    return Outcome(
        question_id=question.question_id,  # type: ignore[attr-defined]
        support=_support_class(gold, window, pool | window),
        gold=len(gold),
        found=len(gold & window),
        window_rows=rows,
        window_chars=sum(len(hit.content) for hit in answerer.grounded),
        expansion_rows=answerer.expansion_rows,
        window_ids=tuple(hit.id for hit in answerer.grounded),
    )


def _summarize(outcomes: Sequence[Outcome]) -> dict[str, object]:
    counts = Counter(outcome.support for outcome in outcomes)
    total = len(outcomes) or 1
    return {
        "questions": len(outcomes),
        "counts": dict(sorted(counts.items())),
        "shares": {name: round(count / total, 4) for name, count in sorted(counts.items())},
        "complete_support": round(counts["complete_in_window"] / total, 4),
        "mean_group_recall": round(
            statistics.fmean(outcome.found / max(outcome.gold, 1) for outcome in outcomes), 4
        ),
        "mean_window_rows": round(statistics.fmean(outcome.window_rows for outcome in outcomes), 2),
        "mean_window_chars": round(
            statistics.fmean(outcome.window_chars for outcome in outcomes), 1
        ),
        "mean_expansion_rows": round(
            statistics.fmean(outcome.expansion_rows for outcome in outcomes), 2
        ),
    }


def _paired(base: Sequence[Outcome], arm: Sequence[Outcome]) -> dict[str, object]:
    """Win, loss and tie on complete support, question by question on the same store."""
    by_id = {outcome.question_id: outcome for outcome in base}
    wins = losses = ties = 0
    # The self-check that says the pairing held: with no row added, the two arms must have
    # grounded the identical window. Any disagreement here is index state leaking into the
    # comparison, and the arms are not measuring the setting.
    unexplained = 0
    for outcome in arm:
        before = by_id[outcome.question_id]
        if outcome.expansion_rows == 0 and outcome.window_ids != before.window_ids:
            unexplained += 1
        was = before.support == "complete_in_window"
        now = outcome.support == "complete_in_window"
        if now and not was:
            wins += 1
        elif was and not now:
            losses += 1
        else:
            ties += 1
    return {
        "won": wins,
        "lost": losses,
        "tied": ties,
        "unexplained_window_changes": unexplained,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", default="tencent/WeMM-Embedding-2B")
    parser.add_argument("--evidence-budget-chars", type=int, default=None)
    parser.add_argument(
        "--no-capture-context",
        action="store_true",
        help="ingest without an ObservationContext, which is what the harness does today",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    questions = load_longmemeval(args.dataset)[args.offset : args.offset + args.limit]
    embedder = _models(args.base_url, args.api_key, args.model)
    base: list[Outcome] = []
    expanded: list[Outcome] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="support-coverage-") as root:
        for index, question in enumerate(questions, start=1):
            data_dir = Path(root) / question.question_id
            answerer = _StubAnswerer()
            # Three opens, not two. A session that wrote merges its Zvec segments at close, and
            # segment layout moves what an approximate search returns -- so an arm that reads the
            # store it just wrote is reading a different index from the arm that opens it after.
            # Measured that way, five of seven apparent expansion losses were questions where
            # expansion added no row at all: index state, not the setting under test. Ingest
            # therefore closes before any arm reads, and both arms then open one settled index.
            with _open(
                data_dir,
                embedder,
                answerer,
                expansion=False,
                budget=args.evidence_budget_chars,
            ) as memory:
                gold = _ingest(memory, question, capture_context=not args.no_capture_context)
            for expansion, outcomes in ((False, base), (True, expanded)):
                with _open(
                    data_dir,
                    embedder,
                    answerer,
                    expansion=expansion,
                    budget=args.evidence_budget_chars,
                ) as memory:
                    outcomes.append(_run_arm(memory, answerer, question, gold))
            print(
                f"[{index}/{len(questions)}] {question.question_id} "
                f"base={base[-1].support} expand={expanded[-1].support}",
                file=sys.stderr,
                flush=True,
            )
    embedder.close()

    report = {
        "dataset": str(args.dataset),
        "questions": len(questions),
        "offset": args.offset,
        "recall_limit": RECALL_LIMIT,
        "evidence_budget_chars": args.evidence_budget_chars,
        "capture_context": not args.no_capture_context,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "arms": {"baseline": _summarize(base), "expansion": _summarize(expanded)},
        "paired_complete_support": _paired(base, expanded),
        "samples": [
            {
                "baseline": {k: v for k, v in asdict(before).items() if k != "window_ids"},
                "expansion": {k: v for k, v in asdict(after).items() if k != "window_ids"},
            }
            for before, after in zip(base, expanded, strict=True)
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
