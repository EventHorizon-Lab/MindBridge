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
import importlib
import json
import os
import random
import re
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import cast

from corpora import Unit, gallery_units, write_unit
from openai import APIError, OpenAI

# The installed OpenAI SDK is built on `httpx2`, not `httpx`, and only the former is in the
# dependency closure. Taking the module the client itself uses keeps the hook attachable whichever
# one a future pin brings, instead of importing a name that is not installed.
from openai._base_client import httpx2 as httpx

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
from mindbridge.benchmarks.atm_bench import (
    atm_capture_time,
    atm_email_block,
    load_atm_bench,
    load_atm_emails,
)
from mindbridge.benchmarks.longmemeval import load_longmemeval
from mindbridge.benchmarks.official_scorers import judge_plan, local_scores, parse_judge_response
from mindbridge.benchmarks.prompts import ATM_BENCH_QUERY_PROMPT, atm_format_constraint
from mindbridge.context import evidence_cost
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
# Which corpus this commit measures. The run command is fixed across the tree, so the corpus is a
# committed property of the branch rather than a flag: `--dataset` is read only by the
# `longmemeval` corpus and is stated as unused by the other.
CORPUS: str = "mem-gallery"
# ATM-Bench is one store every question reads, so its questions are answered concurrently inside
# one open rather than each getting a store of their own.
ATM_QUESTIONS = 120
# `evidence_cost` charges an image 2,000 characters, so the text run's 24,000 buys a ranked tail of
# twelve images -- which is the `window` arm over again. The budget arm is scaled to the media the
# linked arm reaches, so the contest stays composition at one media budget rather than volume.
ATM_BUDGET_FACTOR = 3
# Questions per persona, and how many of them are drawn from the only population an edge can move.
# All 1,527 questions across three arms is thirteen hours for a set that is two thirds unaffected
# by construction; 15 per topic with 8 reserved for incomplete windows keeps the edge population
# large and the control population honest. Seeded, so the subset survives a re-run.
GALLERY_QUESTIONS_PER_TOPIC = 15
GALLERY_INCOMPLETE_PER_TOPIC = 8
GALLERY_SAMPLE_SEED = 20260916
# A Mem-Gallery round is a few hundred characters of dialogue and a quarter of them carry an image,
# so a twelve-row window costs far less than ATM's. The budget arm is sized to reach about the same
# row count the linked arm does, which is what keeps the contest composition rather than volume.
GALLERY_BUDGET_CHARS = 45_000
# Set by the run once it knows whether this environment could load the OpenCV face recipe, so the
# report can say whether the identity edge was measured or merely absent.
_FACES_RAN = False
# Whether Mem-Gallery's deterministic primary could be computed at all in this environment.
_F1_AVAILABLE = True


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
    # What the product itself charges this window: characters plus a flat price per media part.
    # `prompt_chars` is text only, and on a photo corpus the media is most of the window, so a
    # comparison stated in characters alone overstated one arm's saving by 1.6x.
    evidence_cost: int
    # What the provider said it was billed for the answer call. The only unmodelled cost here.
    prompt_tokens: int
    media_rows: int
    correct: float
    # Mem-Gallery's judged metric beside its deterministic primary; zero on the corpora that have
    # only one official score.
    judged: float = 0.0
    # Whether the ranking left this question's window incomplete, decided before any arm ran. The
    # edge population and the control population have to be separable in the analysis, because a
    # question with sixteen clue rounds cannot fit a twelve-row window and widening to any depth
    # helps it -- which is the release's shape, not the edge's effect.
    incomplete_stratum: bool = False
    abstained: bool = False
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
        # One `Memory` holds one answerer, and the ATM corpus is one store every question reads,
        # so several threads pass through this object at once. The capture is per ask, so it
        # belongs to the asking thread and not to the recorder.
        self._local = threading.local()

    @property
    def grounded(self) -> tuple[SearchHit, ...]:
        return cast(tuple[SearchHit, ...], getattr(self._local, "grounded", ()))

    @property
    def prompt_chars(self) -> int:
        return cast(int, getattr(self._local, "prompt_chars", 0))

    @property
    def expansion_rows(self) -> int:
        return cast(int, getattr(self._local, "expansion_rows", 0))

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        grounded = tuple(hits)
        text = question.text or ""
        note = _EXPANSION_NOTE.search(text)
        self._local.grounded = grounded
        self._local.prompt_chars = len(text) + sum(len(hit.content) for hit in grounded)
        self._local.expansion_rows = 0 if note is None else int(note.group(1))
        return self._backend.answer(
            question,
            hits,
            answer_policy=answer_policy,
            exhaustive=exhaustive,
        )

    def close(self) -> None:
        self._backend.close()


_BILLED = threading.local()


def _billing_hook(response: httpx.Response) -> None:
    """Accumulate the prompt tokens the provider reports, for the asking thread.

    Cost is the whole claim this measurement makes, and it has been got wrong twice by modelling
    it -- once by counting characters and ignoring media entirely, once by pricing a media part at
    the flat charge `evidence_cost` uses rather than what a model is billed for. The provider says
    what it charged; read that instead. Best effort by construction: a body this cannot parse
    leaves the modelled figure standing rather than failing an answer.
    """
    try:
        response.read()
        usage = response.json().get("usage") or {}
        tokens = int(usage.get("prompt_tokens", 0))
    except Exception:
        return
    _BILLED.tokens = getattr(_BILLED, "tokens", 0) + tokens


def _billed_tokens() -> int:
    return cast(int, getattr(_BILLED, "tokens", 0))


def _reset_billing() -> None:
    _BILLED.tokens = 0


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
    """The local embedder, declared for the modalities this corpus actually holds.

    A media corpus needs the `messages` request format, because that is the only shape this
    endpoint accepts an image or a clip in. It also forces one record per request: the endpoint
    returns a single vector for a list of messages, and the adapter refuses the mismatch rather
    than mis-aligning a batch, which is the safe half of a slower ingest.
    """
    media = CORPUS != "longmemeval"
    return OpenAIModels(
        embedding_client=OpenAI(
            base_url=settings["MINDBRIDGE_EMBEDDING_BASE_URL"],
            api_key=settings.get("MINDBRIDGE_EMBEDDING_API_KEY", "local"),
            max_retries=3,
            # The embedder is local and answers in well under a second.
            timeout=120.0,
        ),
        embedding_model=settings["MINDBRIDGE_EMBEDDING_MODEL"],
        embedding_dimension=EMBEDDING_DIMENSION,
        embedding_request_format="messages" if media else "input",
        embedding_capabilities=(
            frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})
            if media
            else frozenset({Modality.TEXT})
        ),
    )


def _generation_client(settings: Mapping[str, str]) -> OpenAI:
    """The reader and judge endpoint, with a deadline a hung request cannot outlive.

    A 600-second timeout with four client retries lets one hung request hold a worker for over
    three hours, which is what happened at question 120 of an ATM arm: no answer, no failure, and
    a run that would have spent longer waiting on it than on the other 119 together. An answer
    over a 72,000-character window with images returns in tens of seconds, so the deadline is
    generous at 180 and the question-level retry above is what recovers a genuine blip.
    """
    return OpenAI(
        base_url=settings["MINDBRIDGE_GENERATION_BASE_URL"],
        api_key=settings["MINDBRIDGE_GENERATION_API_KEY"],
        max_retries=2,
        timeout=180.0,
        http_client=httpx.Client(
            timeout=180.0,
            event_hooks={"response": [_billing_hook]},
        ),
    )


def _reader(settings: Mapping[str, str], client: OpenAI) -> OpenAIModels:
    return OpenAIModels(
        generation_client=client,
        generation_model=settings["MINDBRIDGE_GENERATION_MODEL"],
        generation_capabilities=(
            frozenset({Modality.TEXT})
            if CORPUS == "longmemeval"
            else frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})
        ),
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
                _reset_billing()
                result = memory.ask(
                    "Answer concisely using only the memories.\n"
                    f"Question: {question.question}\nAnswer:",  # type: ignore[attr-defined]
                    limit=RECALL_LIMIT,
                )
                billed = _billed_tokens()
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
                    evidence_cost=sum(evidence_cost(hit) for hit in recorder.grounded),
                    prompt_tokens=billed,
                    media_rows=sum(1 for hit in recorder.grounded if hit.assets),
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


def _atm_score(question: object, prediction: str, client: OpenAI, model: str) -> float:
    """Score one ATM answer the way ATM scores it.

    Two protocols, not one: `number` and `list_recall` are deterministic upstream and a judge
    never sees them, while `open_end` goes through the pinned judge prompt. Sending the countable
    types to a judge would be a different benchmark that happens to share a name.
    """
    metadata = {"qtype": question.qtype, "evidence_ids": question.evidence_ids}  # type: ignore[attr-defined]
    deterministic = local_scores(
        "atm-bench-main",
        score_kind="accuracy",
        prediction=prediction,
        parsed_choice=None,
        expected_choice=None,
        references=(question.reference_answer,),  # type: ignore[attr-defined]
        question=question.question,  # type: ignore[attr-defined]
        metadata=metadata,
        evidence_source_ids=(),
    )
    if "accuracy" in deterministic:
        return float(deterministic["accuracy"])
    plan = judge_plan(
        "atm-bench-main",
        question=question.question,  # type: ignore[attr-defined]
        references=(question.reference_answer,),  # type: ignore[attr-defined]
        prediction=prediction,
        metadata=metadata,
    )
    if plan is None:
        return 0.0
    messages = [{"role": message.role, "content": message.content} for message in plan.calls[0]]
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.0,
        max_tokens=plan.max_tokens or 600,
    )
    return float(
        parse_judge_response(plan, response.choices[0].message.content or "").get("accuracy", 0.0)
    )


# The provider adapter refuses an inline media item over 20 MiB base64-encoded, which is about
# 15 MiB of bytes, and it refuses it before the request rather than after -- so the video-sampling
# retry that rescues an over-long clip never runs for an over-large one. Twenty-one of ATM's 533
# clips are over it. They are skipped and counted: three of the benchmark's 1,013 questions cite
# one, none of them in the slice this node measures, and a silently missing record would make the
# support decomposition a lie about what the store holds.
_INLINE_MEDIA_BYTES = 20 * 1024 * 1024 * 3 // 4


def _atm_ingest(memory: Memory, data: Path) -> tuple[int, tuple[str, ...]]:
    """Write the whole ATM corpus: emails as text, photographs and clips as their own bytes.

    The capture is the day, which is what a photo corpus actually has: the other pictures from one
    afternoon are evidence a query has no way to name, and they are the edge this node is about.
    Batching is one record per call because the endpoint'"'"'s `messages` embedding format collapses a
    batch into a single vector, and a silently mis-aligned batch is worse than a slower ingest.
    """
    added = 0
    for email in load_atm_emails(data / "raw_memory" / "email" / "emails.json"):
        added += 1
        memory.add(
            atm_email_block(email),
            occurred_at=email.occurred_at,
            metadata={"source_id": email.email_id},
            memory_type=MemoryType.EPISODIC,
            context=ObservationContext(source_id=email.occurred_at.date().isoformat()),
        )
    skipped: list[str] = []
    for folder, pattern in (("image", "*.jpg"), ("video", "*.mp4")):
        for path in sorted((data / "raw_memory" / folder).glob(pattern)):
            if path.stat().st_size > _INLINE_MEDIA_BYTES:
                skipped.append(path.stem)
                continue
            added += 1
            memory.add(
                path,
                occurred_at=atm_capture_time(path.stem),
                metadata={"source_id": path.stem},
                memory_type=MemoryType.EPISODIC,
                context=ObservationContext(source_id=path.stem[:8]),
            )
    return added, tuple(skipped)


def _atm_outcomes(
    *,
    root: Path,
    arms: Sequence[Arm],
    settings: Mapping[str, str],
    client: OpenAI,
    limit: int,
    offset: int,
    workers: int,
    report: Callable[[str], None],
    abandoned: list[tuple[str, str]],
) -> tuple[tuple[Outcome, ...], ...]:
    """Ingest the ATM corpus once, then read it once per arm with its questions concurrent."""
    data = Path(settings["MINDBRIDGE_BENCH_ROOT"]) / "atm-bench" / "data"
    questions = load_atm_bench(data / "atm-bench" / "atm-bench.json")[offset : offset + limit]
    embedder = _embedder(settings)
    store = root / "atm"
    model = settings["MINDBRIDGE_GENERATION_MODEL"]
    by_question: dict[str, list[Outcome]] = {}
    try:
        started = time.perf_counter()
        with _open(store, embedder, None, arms[0]) as memory:
            records, skipped = _atm_ingest(memory, data)
        report(
            f"ingested {records} records in {(time.perf_counter() - started) / 60:.1f} min; "
            f"skipped {len(skipped)} media over the inline limit"
        )
        for arm in arms:
            recorder = _Recorder(_reader(settings, client))
            with _open(store, embedder, recorder, arm) as memory:

                def one(
                    question: object,
                    arm: Arm = arm,
                    memory: Memory = memory,
                    recorder: _Recorder = recorder,
                ) -> Outcome:
                    gold = frozenset(question.evidence_ids)  # type: ignore[attr-defined]
                    prompt = ATM_BENCH_QUERY_PROMPT.text.format(
                        question=question.question,  # type: ignore[attr-defined]
                        format_constraint=atm_format_constraint(question.qtype),  # type: ignore[attr-defined]
                    )
                    _reset_billing()
                    result = memory.ask(prompt, limit=RECALL_LIMIT)
                    billed = _billed_tokens()
                    window = _turn_ids(recorder.grounded)
                    pool = _turn_ids(memory.search(question.question, limit=100))  # type: ignore[attr-defined]
                    outcome = Outcome(
                        question_id=str(question.question_id),  # type: ignore[attr-defined]
                        arm=arm.name,
                        support=_support_class(gold, window, pool | window),
                        gold=len(gold),
                        found=len(gold & window),
                        window_rows=len(recorder.grounded),
                        window_chars=sum(len(hit.content) for hit in recorder.grounded),
                        expansion_rows=recorder.expansion_rows,
                        prompt_chars=recorder.prompt_chars,
                        evidence_cost=sum(evidence_cost(hit) for hit in recorder.grounded),
                        prompt_tokens=billed,
                        media_rows=sum(1 for hit in recorder.grounded if hit.assets),
                        correct=_atm_score(question, result.answer, client, model),
                        abstained=result.abstained,
                        window_ids=tuple(hit.id for hit in recorder.grounded),
                    )
                    report(
                        f"{arm.name} {outcome.question_id} {outcome.support} "
                        f"{'y' if outcome.correct > 0.5 else 'n'}"
                    )
                    return outcome

                with ThreadPoolExecutor(max_workers=max(1, workers)) as pool_executor:
                    for outcome in pool_executor.map(_retrying(one), questions):
                        if outcome is not None:
                            by_question.setdefault(outcome.question_id, []).append(outcome)
    finally:
        embedder.close()
    # Only questions every arm answered, so the arms share one denominator -- and whatever fell
    # short is named rather than quietly dropped, because a report whose denominator moved has to
    # say which questions moved it. An earlier run reported `abandoned: []` beside 116 of 120.
    for question in questions:
        rows = by_question.get(str(question.question_id))  # type: ignore[attr-defined]
        if rows is None:
            abandoned.append((str(question.question_id), "no arm answered it"))  # type: ignore[attr-defined]
        elif len(rows) != len(arms):
            abandoned.append(
                (str(question.question_id), f"answered by {len(rows)} of {len(arms)} arms")  # type: ignore[attr-defined]
            )
    return tuple(tuple(rows) for rows in by_question.values() if len(rows) == len(arms))


def _retrying(work: Callable[[object], Outcome]) -> Callable[[object], Outcome | None]:
    """Retry one unit of work over a flaky gateway, and give up on it rather than on the run."""

    def attempt(item: object) -> Outcome | None:
        for index in range(_RETRY_ATTEMPTS):
            try:
                return work(item)
            except (APIError, ModelError):
                if index + 1 == _RETRY_ATTEMPTS:
                    return None
                time.sleep(_RETRY_BACKOFF_SECONDS * (index + 1))
        return None

    return attempt


def _gallery_score(
    question: str, reference: str, prediction: str, client: OpenAI, model: str
) -> tuple[float, float]:
    """Score one Mem-Gallery answer with the release's own two official metrics.

    `f1` is upstream's primary and is deterministic; `llm_judge` is its judged metric, scored
    0, 0.5 or 1. Both are returned because the primary is the one to report and the judged one is
    the more sensitive paired signal.
    """
    metadata = {"point": "AR"}
    # `f1` is upstream's primary but its scorer needs the `benchmarks` extra, which the tree's
    # fixed run command does not install. `llm_judge` is the other official metric for this family
    # and needs only the pinned prompt, so it carries the comparison when f1 is unavailable -- and
    # it is the more sensitive paired signal anyway, being three-valued rather than a token overlap.
    global _F1_AVAILABLE
    try:
        deterministic = local_scores(
            "mem-gallery",
            score_kind="accuracy",
            prediction=prediction,
            parsed_choice=None,
            expected_choice=None,
            references=(reference,),
            question=question,
            metadata=metadata,
            evidence_source_ids=(),
        )
    except (RuntimeError, ImportError):
        _F1_AVAILABLE = False
        deterministic = {}
    plan = judge_plan(
        "mem-gallery",
        question=question,
        references=(reference,),
        prediction=prediction,
        metadata=metadata,
    )
    judged = 0.0
    if plan is not None:
        messages = [{"role": m.role, "content": m.content} for m in plan.calls[0]]
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.0,
            max_tokens=plan.max_tokens or 600,
        )
        judged = float(
            parse_judge_response(plan, response.choices[0].message.content or "").get(
                "llm_judge", 0.0
            )
        )
    return float(deterministic.get("f1", 0.0)), judged


def _gallery_sample(
    memory: Memory,
    unit: Unit,
) -> tuple[tuple[int, bool], ...]:
    """Choose which of a persona's questions to answer, and say which had incomplete windows.

    The window is read once here, before any arm runs, so the stratum a question belongs to is a
    property of the ranking rather than of whichever arm happened to see it first.
    """
    incomplete: list[int] = []
    complete: list[int] = []
    for index, (question, gold, _reference) in enumerate(unit.questions):
        window = {
            str(hit.metadata.get("source_id"))
            for hit in memory.search(question, limit=RECALL_LIMIT)
        }
        (incomplete if gold - window else complete).append(index)
    rng = random.Random(f"{GALLERY_SAMPLE_SEED}:{unit.unit_id}")
    rng.shuffle(incomplete)
    rng.shuffle(complete)
    picked = incomplete[:GALLERY_INCOMPLETE_PER_TOPIC]
    picked_complete = complete[: GALLERY_QUESTIONS_PER_TOPIC - len(picked)]
    chosen = [(index, True) for index in picked] + [(index, False) for index in picked_complete]
    chosen.sort()
    return tuple(chosen)


def _face_analyzer(settings: Mapping[str, str]) -> object | None:
    """The OpenCV face recipe, or None when this environment cannot run it.

    A run executes the tree's fixed command, which installs the `openai` extra and not `face`, so
    OpenCV is absent from a run's snapshot however present it is in a worktree. Degrading is the
    right answer rather than failing: on Mem-Gallery the capture edge supplies 125 of the 132
    questions any edge could complete and identity supplies 29, so a run without faces measures
    almost all of the headroom and loses only the identity attribution. What it must not do is
    stay quiet about it, so the report says which of the two it was.
    """
    try:
        # The adapter loads OpenCV lazily at first `analyze`, not at construction, so importing
        # it and building it both succeed in an environment that cannot run it -- which is how a
        # run reached its first photograph before failing. The import that decides this is the
        # one the adapter itself will eventually make.
        importlib.import_module("cv2")
        from mindbridge.models.opencv_face import OpenCVFaceAnalyzer

        return OpenCVFaceAnalyzer(
            detector_model=settings["MINDBRIDGE_FACE_DETECTOR"],
            recognizer_model=settings["MINDBRIDGE_FACE_RECOGNIZER"],
            score_threshold=0.9,
        )
    except (ImportError, KeyError, ModelError, OSError):
        return None


def _note_faces(ran: bool) -> None:
    """Record once whether any persona's store had a working face recipe behind it."""
    global _FACES_RAN
    _FACES_RAN = _FACES_RAN or ran


def _gallery_outcome(
    item: object,
    *,
    unit: Unit,
    arm: Arm,
    memory: Memory,
    recorder: _Recorder,
    client: OpenAI,
    model: str,
    report: Callable[[str], None],
) -> Outcome:
    """Answer one sampled question under one arm and score it Mem-Gallery's own two ways."""
    index, incomplete = cast(tuple[int, bool], item)
    question, gold, reference = unit.questions[index]
    _reset_billing()
    result = memory.ask(question, limit=RECALL_LIMIT)
    billed = _billed_tokens()
    window = _turn_ids(recorder.grounded)
    pool = _turn_ids(memory.search(question, limit=100))
    f1, judged = _gallery_score(question, reference, result.answer, client, model)
    outcome = Outcome(
        question_id=f"{unit.unit_id}:{index}",
        arm=arm.name,
        support=_support_class(gold, window, pool | window),
        gold=len(gold),
        found=len(gold & window),
        window_rows=len(recorder.grounded),
        window_chars=sum(len(hit.content) for hit in recorder.grounded),
        expansion_rows=recorder.expansion_rows,
        prompt_chars=recorder.prompt_chars,
        evidence_cost=sum(evidence_cost(hit) for hit in recorder.grounded),
        prompt_tokens=billed,
        media_rows=sum(1 for hit in recorder.grounded if hit.assets),
        correct=f1,
        judged=judged,
        incomplete_stratum=incomplete,
        abstained=result.abstained,
        window_ids=tuple(hit.id for hit in recorder.grounded),
    )
    report(f"{unit.unit_id} {arm.name} q{index} {outcome.support} f1={f1:.2f} judge={judged:.1f}")
    return outcome


def _gallery_outcomes(
    *,
    root: Path,
    arms: Sequence[Arm],
    settings: Mapping[str, str],
    client: OpenAI,
    workers: int,
    report: Callable[[str], None],
    abandoned: list[tuple[str, str]],
) -> tuple[tuple[Outcome, ...], ...]:
    """One store per persona: ingest, classify, then read the sampled questions once per arm."""
    data = Path(settings["MINDBRIDGE_BENCH_ROOT"]) / "mem-gallery" / "data"
    model = settings["MINDBRIDGE_GENERATION_MODEL"]
    by_question: dict[str, list[Outcome]] = {}
    units = gallery_units(data, 10_000)
    for unit in units:
        store = root / unit.unit_id
        embedder = _embedder(settings)
        face = _face_analyzer(settings)
        _note_faces(face is not None)
        try:
            # Ingest and classification close before any arm reads, for the reason every run in
            # this line of work now closes first: a session that wrote merges its Zvec segments at
            # close, and segment layout moves what an approximate search returns.
            with Memory(
                store,
                embedder=embedder,
                face_analyzer=face,  # type: ignore[arg-type]
                minimum_relevance=0.0,
                ambiguity_margin=0.0,
                reinforce_on_answer=False,
            ) as memory:
                media = write_unit(memory, unit)
                if face is not None:
                    for memory_id in media:
                        memory.faces(memory_id)
                sample = _gallery_sample(memory, unit)
            for arm in arms:
                recorder = _Recorder(_reader(settings, client))
                with _open(store, _embedder(settings), recorder, arm) as memory:
                    answer = partial(
                        _gallery_outcome,
                        unit=unit,
                        arm=arm,
                        memory=memory,
                        recorder=recorder,
                        client=client,
                        model=model,
                        report=report,
                    )
                    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool_executor:
                        for outcome in pool_executor.map(_retrying(answer), sample):
                            if outcome is not None:
                                by_question.setdefault(outcome.question_id, []).append(outcome)
        finally:
            embedder.close()
            if face is not None:
                face.close()  # type: ignore[attr-defined]
    for question_id, rows in by_question.items():
        if len(rows) != len(arms):
            abandoned.append((question_id, f"answered by {len(rows)} of {len(arms)} arms"))
    return tuple(tuple(rows) for rows in by_question.values() if len(rows) == len(arms))


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
        "mean_media_rows": round(statistics.fmean(outcome.media_rows for outcome in outcomes), 2),
        "judged": round(statistics.fmean(outcome.judged for outcome in outcomes), 4),
        "mean_evidence_cost": round(
            statistics.fmean(outcome.evidence_cost for outcome in outcomes), 1
        ),
        "mean_prompt_tokens": round(
            statistics.fmean(outcome.prompt_tokens for outcome in outcomes), 1
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


def _longmemeval_outcomes(
    args: argparse.Namespace,
    *,
    arms: Sequence[Arm],
    settings: Mapping[str, str],
    client: OpenAI,
    report: Callable[[str], None],
    abandoned: list[tuple[str, str]],
    lock: threading.Lock,
) -> tuple[list[tuple[Outcome, ...]], int]:
    """One store per question, because each LongMemEval unit is its own haystack."""
    questions = load_longmemeval(args.dataset)[args.offset : args.offset + args.limit]
    with tempfile.TemporaryDirectory(prefix="support-coverage-") as root:

        def run(question: object) -> tuple[Outcome, ...]:
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
            if outcomes:
                shown = " ".join(
                    f"{o.arm}={o.support[:8]}/{'y' if o.correct > 0.5 else 'n'}" for o in outcomes
                )
            else:
                with lock:
                    abandoned.append((question_id, reason))
                shown = f"ABANDONED {reason[:120]}"
            report(f"{question_id} {shown}")
            return outcomes

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            collected = [rows for rows in pool.map(run, questions) if rows]
    return collected, len(questions)


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

    if CORPUS != "longmemeval" and not settings.get("MINDBRIDGE_BENCH_ROOT"):
        parser.error(f"{_ENV_PATH} is missing MINDBRIDGE_BENCH_ROOT")
    if CORPUS == "mem-gallery":
        budget = GALLERY_BUDGET_CHARS
    else:
        budget = args.evidence_budget_chars * (ATM_BUDGET_FACTOR if CORPUS == "atm" else 1)
    arms = (
        Arm("window", budget=None, expansion=False),
        Arm("window+linked", budget=None, expansion=True),
        Arm("budget", budget=budget, expansion=False),
    )
    client = _generation_client(settings)
    done = threading.Lock()
    finished = 0
    abandoned: list[tuple[str, str]] = []
    started = time.perf_counter()
    collected: list[tuple[Outcome, ...]] = []

    def progress() -> Callable[[str], None]:
        """One numbered line per completed unit of work, whichever corpus produced it."""

        def note(line: str) -> None:
            nonlocal finished
            with done:
                finished += 1
                print(f"[{finished}] {line}", file=sys.stderr, flush=True)

        return note

    if CORPUS in {"atm", "mem-gallery"}:
        print(
            f"corpus={CORPUS}; --dataset {args.dataset} is not read by this commit",
            file=sys.stderr,
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="support-coverage-") as root:
            shared = {
                "root": Path(root),
                "arms": arms,
                "settings": settings,
                "client": client,
                "workers": args.workers,
                "report": progress(),
                "abandoned": abandoned,
            }
            if CORPUS == "atm":
                collected = list(
                    _atm_outcomes(limit=ATM_QUESTIONS, offset=args.offset, **shared)  # type: ignore[arg-type]
                )
                counted = ATM_QUESTIONS
                protocol = "atm_bench_official_deterministic_and_judge"
            else:
                collected = list(_gallery_outcomes(**shared))  # type: ignore[arg-type]
                counted = len(collected) + len(abandoned)
                protocol = "mem_gallery_official_f1_and_judge"
        return _emit(
            args,
            settings,
            arms,
            collected,
            abandoned=abandoned,
            question_count=counted,
            judge_protocol=protocol,
            budget=budget,
            started=started,
        )

    collected, counted = _longmemeval_outcomes(
        args,
        arms=arms,
        settings=settings,
        client=client,
        report=progress(),
        abandoned=abandoned,
        lock=done,
    )
    return _emit(
        args,
        settings,
        arms,
        collected,
        abandoned=abandoned,
        question_count=counted,
        judge_protocol="longmemeval_official_answer_check",
        budget=budget,
        started=started,
    )


def _emit(
    args: argparse.Namespace,
    settings: Mapping[str, str],
    arms: Sequence[Arm],
    collected: Sequence[tuple[Outcome, ...]],
    *,
    abandoned: Sequence[tuple[str, str]],
    question_count: int,
    judge_protocol: str,
    budget: int,
    started: float,
) -> int:
    """Write one report, whichever corpus produced it."""
    by_arm = {
        arm.name: [row for rows in collected for row in rows if row.arm == arm.name] for arm in arms
    }
    control = by_arm[arms[0].name]
    report = {
        "corpus": CORPUS,
        "dataset": str(args.dataset) if CORPUS == "longmemeval" else None,
        "questions": question_count,
        "scored_questions": len(collected),
        # Named, not silently dropped: a report whose denominator moved has to say so.
        "abandoned": [{"question_id": qid, "reason": why} for qid, why in abandoned],
        "offset": args.offset,
        "recall_limit": RECALL_LIMIT,
        "evidence_budget_chars": budget,
        "capture_context": not args.no_capture_context,
        "face_recognition_ran": _FACES_RAN,
        "f1_available": _F1_AVAILABLE,
        "answer_model": settings["MINDBRIDGE_GENERATION_MODEL"],
        "judge": {
            "model": settings["MINDBRIDGE_GENERATION_MODEL"],
            "protocol": judge_protocol,
            "note": (
                "proxy judge where a judge is used at all: upstream's is a frontier model and "
                "this is the model that also answered, so no number here is comparable to a "
                "published row"
            ),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "arms": {name: _summarize(rows) for name, rows in by_arm.items()},
        "accuracy_by_support": _accuracy_by_support(
            [row for rows in collected for row in rows],
        ),
        # The edge population and the control population, separated. An arm that helps only where
        # the window was already complete is not doing what this mechanism claims to do.
        "by_stratum": {
            name: {
                arm.name: _summarize(
                    [
                        row
                        for rows in collected
                        for row in rows
                        if row.arm == arm.name and row.incomplete_stratum is incomplete
                    ]
                )
                for arm in arms
            }
            for name, incomplete in (("incomplete_window", True), ("complete_window", False))
            if any(row.incomplete_stratum is incomplete for rows in collected for row in rows)
        },
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
