"""The answer plane: grounding a question and generating from what was grounded.

Retrieval ranks; this module decides how much of the ranking to ground, whether a recall program
adds exhaustive reads beside it, how the question and its evidence are routed into the
generation backend, and how a streamed answer is validated.
"""

from __future__ import annotations

import builtins
from collections import Counter
from collections.abc import Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, suppress
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from time import perf_counter
from typing import Any, cast

from opentelemetry import trace
from opentelemetry.trace import Tracer

from mindbridge._telemetry import (
    ASYNC_QUEUE_TIME,
    MODEL_TTFT,
    OPERATION_TTFT,
    RECALL_COMPLETE,
    RECALL_EXHAUSTIVE_ROWS,
    RECALL_FALLBACK,
    RECALL_NON_SELECTIVE_STEPS,
    RECALL_OPS,
    RECALL_REPLAN,
    RECALL_SHAPE,
    _record_retrieval_results,
    current_model_request_count,
    mark_model_requests,
)
from mindbridge.context import evidence_cost
from mindbridge.exceptions import MindBridgeError, ModelError, StorageError
from mindbridge.infrastructure.local.store import LocalStore, RecallRead, datetime_text
from mindbridge.kernel.content import PreparedContent, memory_modality
from mindbridge.kernel.contracts import Backends, fallback_unsupported, require_audio_transcription
from mindbridge.kernel.derived import derived_text, has_stream_description, has_stream_transcript
from mindbridge.kernel.hydration import Hydrator
from mindbridge.kernel.lifecycle import Lifecycle, OperationAssets
from mindbridge.kernel.materialization import Materializer
from mindbridge.kernel.retrieval import RERANK_CANDIDATES, Retrieval
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.settings import Settings
from mindbridge.kernel.speech import Speech
from mindbridge.kernel.temporal import query_temporal_range, resolved_reference_at, temporal_context
from mindbridge.kernel.tracing import Traced
from mindbridge.kernel.validation import (
    optional_memory_type,
    pushed_memory_types,
    validate_answer_policy,
    validate_limit,
    validated_retrieval_scope,
)
from mindbridge.kernel.vision import Vision
from mindbridge.models.base import ModelInput, RecallPlanningBackend, StreamingGenerationBackend
from mindbridge.recall import (
    RecallPlan,
    RecallResult,
    RecallRows,
    execute,
    fallback_plan,
    parse_recall_plan,
    recall_note,
)
from mindbridge.types import (
    AbstentionReason,
    AnswerChunk,
    AnswerPolicy,
    AnswerResult,
    ContentInput,
    MemoryType,
    Modality,
    RetrievalScope,
    SearchHit,
)

# How many media rows an exhaustive plan may ground on, as a multiple of the window the caller
# asked for. Media is the one kind of evidence whose grounding costs writes -- face and speech
# recognition run on each row before the answer call, and again on a replan -- so a set read that
# matched every clip in the corpus is bounded here rather than by the character budget alone.
_RECALL_MEDIA_FACTOR = 2


ASYNC_QUEUE_TIME_MS: ContextVar[float | None] = ContextVar(
    "mindbridge_async_queue_time_ms",
    default=None,
)


@dataclass(frozen=True, slots=True)
class _RecallContext:
    """What one answering round needs, so a second round repeats nothing but the reading."""

    prepared: PreparedContent
    routed: ModelInput
    assets: OperationAssets
    # The ranked hits this question already earned. A plan's `similar` step is this list, cut to
    # its own depth: the ranking is over the question as asked, media and temporal window
    # included, and re-running it against a rewritten query would be query rewriting -- a
    # separate mechanism, measured separately, and a measured null here.
    ranked: tuple[SearchHit, ...]
    limit: int
    reference: datetime
    scope: RetrievalScope | None
    memory_type: MemoryType | None
    link_identities: bool
    answer_policy: AnswerPolicy


class _RecallReads:
    """Run a recall plan's reads against one store, inside the caller's open operation.

    The exhaustive ops read SQLite through the store's own primitives and are hydrated the way
    every other hit is. They carry no relevance score, because they earned none: they are in the
    evidence set because they match the question's predicate, not because they ranked.
    """

    def __init__(
        self,
        store: LocalStore,
        hydrator: Hydrator,
        context: _RecallContext,
    ) -> None:
        self._store = store
        self._hydrator = hydrator
        self._context = context

    def similar(self, query: str, *, k: int) -> RecallRows:
        del query
        rows = self._context.ranked[:k]
        return RecallRows(rows, len(rows))

    def match(
        self,
        terms: Sequence[str],
        *,
        any_of: bool,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        modality: Modality | None,
        memory_type: MemoryType | None,
        max_rows: int,
        identity_id: str | None = None,
    ) -> RecallRows:
        scope = self._context.scope or RetrievalScope()
        return self._hits(
            self._store.recall.match_memories(
                tuple(terms),
                any_of=any_of,
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                modality=None if modality is None else modality.value,
                memory_type=self.validated_memory_type(memory_type),
                max_rows=max_rows,
                valid_at=scope.valid_at,
                known_at=scope.known_at,
                near=scope.near,
                radius_m=scope.radius_m,
                place_id=scope.place_id,
                identity_id=identity_id or scope.identity_id,
            )
        )

    def window(
        self,
        *,
        occurred_from: datetime,
        occurred_until: datetime,
        modality: Modality | None,
        memory_type: MemoryType | None,
        max_rows: int,
    ) -> RecallRows:
        scope = self._context.scope or RetrievalScope()
        return self._hits(
            self._store.recall.memories_in_window(
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                modality=None if modality is None else modality.value,
                memory_type=self.validated_memory_type(memory_type),
                max_rows=max_rows,
                valid_at=scope.valid_at,
                known_at=scope.known_at,
                near=scope.near,
                radius_m=scope.radius_m,
                place_id=scope.place_id,
                identity_id=scope.identity_id,
            )
        )

    def neighbors(
        self,
        memory_ids: Sequence[str],
        *,
        before: int,
        after: int,
        max_rows: int,
    ) -> RecallRows:
        scope = self._context.scope or RetrievalScope()
        return self._hits(
            self._store.recall.neighbor_memories(
                tuple(memory_ids),
                before=before,
                after=after,
                max_rows=max_rows,
                valid_at=scope.valid_at,
                known_at=scope.known_at,
                near=scope.near,
                radius_m=scope.radius_m,
                place_id=scope.place_id,
                identity_id=scope.identity_id,
            )
        )

    def entity(self, name: str, *, max_rows: int) -> RecallRows:
        """Read what is known about one person, by identity when the name resolves to one.

        A name no identity carries is not an error: the store may hold the person only as words
        in other people's memories, so the read degrades to matching the name as text.
        """
        identity_id = self._store.recall.identity_id_for_name(name)
        return self.match(
            () if identity_id is not None else (name,),
            any_of=True,
            occurred_from=None,
            occurred_until=None,
            modality=None,
            memory_type=None,
            max_rows=max_rows,
            identity_id=identity_id,
        )

    def validated_memory_type(self, requested: MemoryType | None) -> str | None:
        """The caller's own `memory_type` wins: a plan may not widen the question's scope."""
        chosen = self._context.memory_type or requested
        return None if chosen is None else chosen.value

    def _hits(self, read: RecallRead) -> RecallRows:
        """Hydrate one primitive's rows, keeping the count its predicate selected.

        The rows can be fewer than the selection: the bound is applied in SQL and the caller's
        bitemporal, spatial and metric scope is applied by the one hydrating read behind it. Only
        the selection count says whether the bound truncated anything.
        """
        return RecallRows(
            tuple(self._hydrator.search_hit(memory, 0.0) for memory in read),
            read.selected,
        )


def _budgeted_recall(
    program: RecallResult,
    budget_chars: int,
    *,
    media_limit: int,
    max_rows: int,
    required: Sequence[SearchHit],
) -> tuple[tuple[SearchHit, ...], int]:
    """Union the program's set with the window the ranking earned, inside one budget.

    Exhaustive rows come first because they are the answer's shape; ranked rows follow by rank,
    deduplicated by ID. `required` is the window the unplanned path would have grounded, and it
    is admitted whatever either bound says: it is evidence the question already had, so the
    budget may only trim exhaustive rows beyond it and a plan can never ground less than no plan.

    The returned count is how many of the *exhaustive* rows were left out, which is what turns a
    complete set into an incomplete one for the reader -- a ranking cut short is still a ranking,
    and counting one of its rows as a missing "matched record" told a reader holding every record
    its predicate matched that it did not have them.

    Media rows are capped separately. An exhaustive read can name every clip in a corpus, and
    each media row reaching the answer call carries face and speech recognition -- writes, paid
    again on every replan round -- so a set's media rows stop at `media_limit` while its text
    rows do not. The cap bounds what the predicate added, never the ranked window, whose size is
    `limit` and which the unplanned path hands over regardless.

    `max_rows` caps the matched rows the same way for the same reason on the other axis: a
    corpus of short records fits hundreds of rows inside the character budget, and a reader
    handed hundreds answers worse than one handed a window. The rows are already chronological
    when they arrive, so the cap keeps the earliest ones -- the read's own order, not a second
    ranking -- and the rows past it are counted in the returned shortfall, which is what tells
    the reader the set is not complete. Like the other bounds, it may not drop a `required` row.
    """
    exhaustive_ids = {hit.id for hit in program.exhaustive}
    ranked = {hit.id: hit for hit in (*program.ranked, *required) if hit.id not in exhaustive_ids}
    required_ids = {hit.id for hit in required}
    selected: builtins.list[SearchHit] = []
    spent = 0
    media = 0
    omitted = 0
    for index, hit in enumerate((*program.exhaustive, *ranked.values())):
        exhaustive_row = index < len(program.exhaustive)
        cost = evidence_cost(hit)
        if hit.id not in required_ids and (
            (selected and spent + cost > budget_chars)
            or (exhaustive_row and hit.assets and media >= media_limit)
            or (exhaustive_row and index >= max_rows)
        ):
            omitted += 1 if exhaustive_row else 0
            continue
        selected.append(hit)
        spent += cost
        media += 1 if exhaustive_row and hit.assets and hit.id not in required_ids else 0
    return tuple(selected), omitted


def _with_recall_note(question: ModelInput, note: str | None) -> ModelInput:
    """Append what the evidence set is, before the reference clock's own final line."""
    if note is None or not question.text:
        return question
    return replace(question, text=f"{question.text}\n\n{note}")


def _attempt_note(hits: Sequence[SearchHit], result: AnswerResult) -> str:
    """Describe what one round read and what it could not answer, for the next plan.

    How much was read, over what dates, and why it was not enough. No evidence text: the planner
    decides which reads to make next, and handing it records would make it summarize them
    instead. No labels either -- the answer prompt numbers only the qualified subset of the
    evidence, so an `E3` here would name a different record than the one the reader saw, or none.
    """
    dates = sorted(
        hit.occurred_at.date().isoformat() for hit in hits if hit.occurred_at is not None
    )
    span = ""
    if dates:
        span = f" dated {dates[0]}" + ("" if dates[0] == dates[-1] else f" to {dates[-1]}")
    reason = (
        "no usable evidence"
        if result.abstention_reason is None
        else (result.abstention_reason.value)
    )
    return (
        f"read {len(hits)} records{span} and reported {reason}; "
        "the answer was a low-confidence guess"
    )


def _with_reference_time(question: ModelInput, reference_at: datetime) -> ModelInput:
    """Append the answering clock as the final line of what the reader is handed."""
    note = f"Reference time for relative dates: {reference_at.isoformat(timespec='seconds')}"
    return replace(question, text=f"{question.text}\n\n{note}" if question.text else note)


def grounding_hits(
    hits: Sequence[SearchHit],
    limit: int,
    *,
    budget_chars: int | None = None,
) -> tuple[SearchHit, ...]:
    """Ground on the ranking's own order, guaranteeing one hit per modality it contains.

    This used to pop one hit per modality in turn, which capped every modality at
    `ceil(limit / m)` however the scores fell, so the grounded window was the top of the
    ranking only on a single-modality corpus. Measured on the round's mem-gallery library
    (image 231 / text 962, `limit` 20): the rotation rebuilt 6.20 of the 20 slots -- 31 % --
    out of lower-ranked hits, raised the window's media share from 17.7 % to 44.5 %, and gave
    up 1.93 pp of the gold recall the ranking had already found (R@20 0.8320 against a window
    at 0.8127). The intent it served -- one modality must not shut the others out -- is a floor,
    not a rotation, so a modality that would otherwise be absent is promoted into the last
    slots instead of displacing a third of the window.

    A promotion takes the seat of the lowest-ranked hit whose modality survives without it, so
    the top hit is never evicted and no promotion costs the window a modality it already had.
    When more modalities exist than `limit` has slots, the ranking's own order decides which of
    them the window carries.
    """
    best_by_modality: dict[Modality, SearchHit] = {}
    for hit in hits:
        best_by_modality.setdefault(hit.modality, hit)
    selected = list(hits[:limit])
    represented = {hit.modality for hit in selected}
    promoted = [hit for modality, hit in best_by_modality.items() if modality not in represented]
    for hit in promoted[: max(limit - 1, 0)]:
        if len(selected) >= limit:
            counts = Counter(chosen.modality for chosen in selected)
            spare = next(
                (
                    index
                    for index in range(len(selected) - 1, 0, -1)
                    if counts[selected[index].modality] > 1
                ),
                None,
            )
            if spare is None:
                break
            del selected[spare]
        selected.append(hit)
    if budget_chars is None:
        return tuple(selected)
    return (*selected, *_budgeted_hits(hits, selected, budget_chars))


def _budgeted_hits(
    hits: Sequence[SearchHit],
    selected: Sequence[SearchHit],
    budget_chars: int,
) -> tuple[SearchHit, ...]:
    """Extend a grounding set down the ranking while the evidence fits one budget.

    The guaranteed hits are never dropped, so this only ever widens what the answer sees. Media
    is charged per modality because an image or video part costs the model far more than its
    record's text; the provider adapter still enforces its own byte ceiling.
    """
    taken = {hit.id for hit in selected}
    used = sum(evidence_cost(hit) for hit in selected)
    # No near-duplicate suppression here, and that is a measured decision rather than an
    # omission. Character-trigram Jaccard -- the measure VoiceMem uses for this at 0.30 -- cannot
    # separate a restatement from a log entry, because in this domain the bands overlap outright:
    # rephrasings of one fact score 0.635-0.836, while "dad took his medication at 8am" against
    # "...at 9am" scores 0.806, "the bicycle was in the shed on monday" against "...tuesday"
    # 0.737, and a temperature reading against the next day's 0.860. Five such pairs were tested
    # and all five fell inside the restatement band, so no threshold exists that keeps them.
    # Collapsing a medication time or a temperature series is information loss a companion memory
    # cannot afford, and it is strictly worse than the wasted budget it would save. A working
    # version needs a measure that reads the *differing* span rather than the shared template.
    extra: list[SearchHit] = []
    for hit in hits:
        if hit.id in taken:
            continue
        cost = evidence_cost(hit)
        if used + cost > budget_chars:
            # The selected prefix remains mandatory, but this supplemental scan is a packing
            # pass. An oversized lower-ranked hit must not prevent a later, smaller record from
            # using the remaining character budget.
            continue
        used += cost
        extra.append(hit)
    return tuple(extra)


class Answering(Traced):
    """Ground a question, optionally run its recall program, and generate the answer."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        backends: Backends,
        settings: Settings,
        lifecycle: Lifecycle,
        hydrator: Hydrator,
        materializer: Materializer,
        speech: Speech,
        vision: Vision,
        retrieval: Retrieval,
    ) -> None:
        super().__init__(tracer)
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._backends = backends
        self._settings = settings
        self._lifecycle = lifecycle
        self._hydrator = hydrator
        self._materializer = materializer
        self._speech = speech
        self._vision = vision
        self._retrieval = retrieval

    def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        link_identities: bool = True,
        answer_policy: AnswerPolicy = "strict",
    ) -> AnswerResult:
        # Draining `ask_stream()` keeps one implementation of the whole path, so a buffered and
        # a streamed answer cannot drift apart in grounding, abstention, or reinforcement.
        stream = self.ask_stream(
            question,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            link_identities=link_identities,
            answer_policy=answer_policy,
        )
        while True:
            try:
                next(stream)
            except StopIteration as complete:
                return cast(AnswerResult, complete.value)

    def ask_stream(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        link_identities: bool = True,
        answer_policy: AnswerPolicy = "strict",
    ) -> Generator[AnswerChunk, None, AnswerResult]:
        validate_limit(limit, maximum=100)
        validate_answer_policy(answer_policy)
        self._require_answerer()
        return self._ask_chunks(
            question,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            link_identities=link_identities,
            answer_policy=answer_policy,
        )

    def _require_answerer(self) -> None:
        if self._backends.answerer is None:
            raise ModelError(
                "answer backend is not configured",
                reason="backend_not_configured",
                stage="generate",
            )

    def _ask_chunks(
        self,
        question: ContentInput,
        *,
        limit: int,
        memory_type: MemoryType | None,
        reference_at: datetime | None,
        scope: RetrievalScope | None,
        link_identities: bool,
        answer_policy: AnswerPolicy,
    ) -> Generator[AnswerChunk, None, AnswerResult]:
        answer = yield from self._ask_operation(
            question,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            link_identities=link_identities,
            answer_policy=answer_policy,
        )
        # The operation and its span close before the result is handed over. A generator stays
        # suspended at whichever yield the caller stops on, so a terminal chunk yielded inside
        # the operation would pin it open for every caller that reads the result and stops --
        # the ordinary way to consume this -- and `close()` waits on open operations.
        yield AnswerChunk(result=answer)
        return answer

    def _ask_operation(
        self,
        question: ContentInput,
        *,
        limit: int,
        memory_type: MemoryType | None,
        reference_at: datetime | None,
        scope: RetrievalScope | None,
        link_identities: bool,
        answer_policy: AnswerPolicy,
    ) -> Generator[AnswerChunk, None, AnswerResult]:
        started = perf_counter()
        operation_ttft_recorded = False
        with (
            self._trace("mindbridge.ask", kind="operation") as span,
            self._lifecycle.operation() as assets,
        ):
            queue_time_ms = ASYNC_QUEUE_TIME_MS.get()
            if queue_time_ms is not None:
                span.set_attribute(ASYNC_QUEUE_TIME, queue_time_ms)
            validate_limit(limit, maximum=100)
            validate_answer_policy(answer_policy)
            self._require_answerer()
            # `search()` opens a `mindbridge.search` operation span; `ask` reaches the same
            # retrieval plane directly. Keep its stage around every prerequisite through ranked
            # hits so the reported leg matches what answering actually waits for.
            with self._trace("mindbridge.retrieve", kind="stage"):
                with self._trace("mindbridge.content.prepare", kind="stage"):
                    prepared = self._materializer.prepare(question, assets)
                explicit_reference = resolved_reference_at(reference_at)
                reference, temporal_text = temporal_context(
                    prepared.text,
                    explicit_reference or datetime.now(timezone.utc),
                    infer_reference=explicit_reference is None,
                )
                temporal_range = query_temporal_range(temporal_text, reference)
                scope = validated_retrieval_scope(scope)
                prepared_search = partial(
                    self._retrieval.search_prepared,
                    prepared,
                    # Without a budget the answer grounds on `limit` hits, so ranking three times
                    # that is enough headroom for the modality round robin. With one, the budget is
                    # what decides depth, so the whole rerank pool has to be ranked for it to spend.
                    limit=(
                        RERANK_CANDIDATES
                        if self._settings.evidence_budget is not None
                        else min(100, limit * 3)
                    ),
                    operation=assets,
                    memory_types=pushed_memory_types(optional_memory_type(memory_type)),
                    reference_at=reference,
                    temporal_range=temporal_range,
                    occurred_from=None,
                    occurred_until=None,
                    scope=scope,
                    require_unambiguous=limit == 1,
                    capture_trace=False,
                )
                speech_assets = self._speech.answer_speech_assets(prepared.assets)
                if speech_assets and all(
                    Modality(asset.modality) in self._backends.embedding_capabilities
                    for asset in speech_assets
                ):
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        context = copy_context()
                        identities = executor.submit(
                            context.run,
                            self._speech.recognize,
                            speech_assets,
                            assets,
                        )
                        hits = prepared_search().hits
                        _record_retrieval_results(hits)
                        identities.result()
                else:
                    if speech_assets:
                        self._speech.recognize(speech_assets, assets)
                    hits = prepared_search().hits
                    _record_retrieval_results(hits)
            round_context = _RecallContext(
                prepared=prepared,
                # Every question the answerer reads as text gets the reference time, not just the
                # ones whose phrasing a parser recognized. "How long ago did grandpa visit?"
                # narrows no retrieval window and names no date, yet it is exactly the question
                # the reader cannot answer without knowing when it is being asked. Applied after
                # routing, because that is what decides whether the reader gets text at all -- a
                # spoken question becomes text there -- and because routing appends speech
                # identities and transcripts, which would otherwise land after the line
                # documented as final.
                routed=self._route_generation(prepared, assets),
                assets=assets,
                ranked=hits,
                limit=limit,
                reference=reference,
                scope=scope,
                memory_type=optional_memory_type(memory_type),
                link_identities=link_identities,
                answer_policy=answer_policy,
            )
            # `closing` rather than plain iteration: an abandoned stream throws `GeneratorExit`
            # at the yield below, and without this the inner generator would only be closed when
            # the cleared frame drops its last reference. That defers closing the provider's
            # response and ends `mindbridge.model.generation` after its own parent, which loses
            # the call's model usage and emits a malformed trace.
            with closing(self._answer_rounds(round_context)) as deltas:
                while True:
                    try:
                        delta = next(deltas)
                    except StopIteration as complete:
                        result, hits = complete.value
                        break
                    if delta.strip() and not operation_ttft_recorded:
                        operation_ttft_ms = (perf_counter() - started) * 1_000.0
                        if queue_time_ms is not None:
                            operation_ttft_ms += queue_time_ms
                        span.set_attribute(OPERATION_TTFT, operation_ttft_ms)
                        operation_ttft_recorded = True
                    yield AnswerChunk(text=delta)
            used_ids = {hit.id for hit in result.hits}
            grounding = tuple(hit for hit in hits if hit.id in used_ids)
            # Reinforcement is part of answering, so it stays inside the operation and runs
            # before the terminal chunk: a caller that sees the result sees a settled store.
            self._reinforce_answered(grounding)
            # An abstention is the answering plane reporting that the evidence did not support
            # an answer, which is the same durable signal as an empty search: the memory the
            # question needed is missing or unreachable.
            self._retrieval.note_query_failure(prepared.text, failed=result.abstained)
            return AnswerResult(
                answer=result.answer,
                hits=grounding,
                abstained=result.abstained,
                abstention_reason=result.abstention_reason,
            )

    def _route_generation(
        self,
        prepared: PreparedContent,
        operation: OperationAssets,
    ) -> ModelInput:
        value = self._hydrator.model_input(prepared)
        missing = value.modalities - self._backends.generation_capabilities
        if (
            missing == {Modality.TEXT}
            and (prepared.audio_transcript or prepared.visual_description)
            and value.modalities - {Modality.TEXT} <= self._backends.generation_capabilities
        ):
            return ModelInput(assets=value.assets)
        unsupported = fallback_unsupported(
            prepared,
            self._backends.generation_capabilities,
            "generation",
        )
        if self._speech.answer_speech_assets(prepared.assets):
            prepared = self._speech.with_speech_identities(prepared, operation)
        if Modality.AUDIO in unsupported and not prepared.audio_transcript:
            require_audio_transcription(self._backends.transcription_capabilities)
            prepared = self._speech.with_audio_transcripts(prepared, operation)
        value = self._hydrator.model_input(prepared)
        unsupported = value.modalities - self._backends.generation_capabilities
        if not unsupported:
            return value
        text = derived_text(value.text, prepared.assets)
        fallback = {Modality.AUDIO}
        if prepared.visual_description:
            fallback.update((Modality.IMAGE, Modality.VIDEO))
        assets = tuple(asset for asset in value.assets if asset.modality not in unsupported)
        if unsupported <= fallback and (text or assets):
            routed = ModelInput(
                text=text,
                assets=assets,
            )
            if not routed.modalities - self._backends.generation_capabilities:
                return routed
        names = ", ".join(sorted(modality.value for modality in unsupported))
        raise ModelError(
            f"configured generation model does not support: {names}",
            reason="unsupported_modality",
        )

    def _route_generation_hits(
        self,
        hits: Sequence[SearchHit],
        operation: OperationAssets,
        *,
        link_identities: bool = True,
    ) -> tuple[SearchHit, ...]:
        asset_ids = tuple(asset.id for hit in hits for asset in hit.assets)
        with (
            self._trace("mindbridge.storage.hydrate", kind="stage"),
            translate_storage_errors("hydrate media for answer generation"),
        ):
            stored = self._store.media.read_assets(asset_ids)
        by_id = {asset.asset_id: asset for asset in stored}
        prepared_hits = []
        for hit in hits:
            try:
                assets = tuple(by_id[asset.id] for asset in hit.assets)
            except KeyError:
                raise StorageError(
                    "memory references missing media metadata", reason="asset_unavailable"
                ) from None
            prepared_hits.append(
                PreparedContent(
                    text=hit.content,
                    assets=assets,
                    modality=memory_modality(assets),
                    canonical_parts=(),
                    audio_transcript=has_stream_transcript(hit.content, assets),
                    visual_description=has_stream_description(hit.content, assets),
                )
            )
        if Modality.AUDIO not in self._backends.generation_capabilities:
            for prepared in prepared_hits:
                fallback_unsupported(
                    prepared,
                    self._backends.generation_capabilities,
                    "generation",
                )
        speech_assets = self._speech.answer_speech_assets(
            tuple(asset for prepared in prepared_hits for asset in prepared.assets)
        )
        if speech_assets:
            self._speech.recognize(speech_assets, operation)
        elif Modality.AUDIO not in self._backends.generation_capabilities:
            self._speech.cache_audio_transcripts(
                tuple(
                    asset
                    for prepared in prepared_hits
                    if not prepared.audio_transcript
                    for asset in prepared.assets
                ),
                operation,
            )
        face_assets = self._vision.answer_face_assets(
            tuple(asset for prepared in prepared_hits for asset in prepared.assets)
        )
        if face_assets:
            self._vision.recognize(face_assets, operation, link_identities=link_identities)
        routed = []
        for hit, prepared in zip(hits, prepared_hits, strict=True):
            if self._vision.answer_face_assets(prepared.assets):
                prepared = self._vision.with_face_identities(
                    prepared, operation, link_identities=link_identities
                )
            model_input = self._route_generation(prepared, operation)
            routed.append(
                replace(
                    hit,
                    content=model_input.text,
                    assets=model_input.assets,
                    modality=model_input.modality,
                )
            )
        return tuple(routed)

    def _answer_rounds(
        self,
        context: _RecallContext,
    ) -> Generator[str, None, tuple[AnswerResult, tuple[SearchHit, ...]]]:
        """Answer this question, replanning once when the first round's own answer asked for it.

        A round another round may replace is held rather than streamed. Two complete answers on
        one wire with no boundary between them is not something a caller can render, and the
        terminal result has to be the text that was streamed -- so the deltas of a replannable
        round are buffered and yielded only if that round is the one that stands. Every other
        round, the common case and always the last one, streams as the provider produces it.
        """
        attempted = ""
        for attempt in range(self._settings.recall_rounds if self._settings.recall_planning else 1):
            held: builtins.list[str] = []
            replannable = self._replan_possible(context, attempt=attempt)
            with closing(self._recall_round(context, attempted=attempted)) as deltas:
                while True:
                    try:
                        delta = next(deltas)
                    except StopIteration as complete:
                        result, hits = complete.value
                        break
                    if replannable:
                        held.append(delta)
                    else:
                        yield delta
            if not (result.abstained and replannable):
                yield from held
                break
            attempted = _attempt_note(hits, result)
        return result, hits

    def _recall_round(
        self,
        context: _RecallContext,
        *,
        attempted: str,
    ) -> Generator[str, None, tuple[AnswerResult, tuple[SearchHit, ...]]]:
        """Ground one attempt at this question and yield its answer deltas.

        With `recall_planning` off there is exactly one attempt and this is the sequence `ask`
        has always run: the ranking's own grounding window, the routed question with the
        reference clock last, then the answer.
        """
        program = self._recall_program(context, attempted=attempted)
        hits, note = self._grounded_recall(program, context)
        routed = context.routed
        question = (
            _with_reference_time(_with_recall_note(routed, note), context.reference)
            if routed.text
            else routed
        )
        routed_hits = (
            self._route_generation_hits(
                hits,
                context.assets,
                link_identities=context.link_identities,
            )
            if hits
            else ()
        )
        self._speech.persist_transcripts(context.assets)
        with closing(
            self._answer_chunks(
                question,
                routed_hits,
                answer_policy=context.answer_policy,
                # The reader cannot see that these rows are a predicate's whole set in time order
                # rather than a ranking, and the note above says they are, so the prompt's own
                # description of their order has to agree with it.
                exhaustive=program is not None and program.plan.exhaustive,
            )
        ) as deltas:
            result = yield from deltas
        return result, hits

    def _recall_program(self, context: _RecallContext, *, attempted: str) -> RecallResult | None:
        """Plan and run this question's reads, or None when planning is off.

        Inside its own stage: the `mindbridge.retrieve` leg closes on the ranked hits, and the
        planning call and the exhaustive reads happen after it -- a model call and a set of table
        scans that a trace without this span attributes to nothing, leaving the operation's own
        duration as the only evidence they ran at all.
        """
        if not self._settings.recall_planning:
            return None
        with self._trace("mindbridge.recall", kind="stage") as span:
            question, k = context.prepared.text, context.limit
            plan = self._recall_plan(
                question,
                reference_at=context.reference,
                k=k,
                attempted=attempted,
            )
            program = execute(
                plan,
                _RecallReads(self._store, self._hydrator, context),
                limit=k,
                # The digest the planner was already handed, recomputed only after a commit, so
                # the selectivity of every step is decided against one cached read per question
                # rather than a count per step.
                active_records=self._store.recall.recall_digest().records,
            )
            span.set_attributes(
                {
                    RECALL_SHAPE: plan.shape,
                    RECALL_OPS: tuple(step.op for step in plan.steps),
                    RECALL_EXHAUSTIVE_ROWS: len(program.exhaustive),
                    RECALL_COMPLETE: program.complete,
                    RECALL_NON_SELECTIVE_STEPS: program.non_selective_steps,
                    RECALL_REPLAN: bool(attempted),
                    # Every way planning can fail resolves to this same plan, so equality with it
                    # is the only signal that says the planner did not decide this question --
                    # which is what a result claiming the feature ran has to be read against.
                    RECALL_FALLBACK: plan == fallback_plan(question, k=k),
                }
            )
            return program

    def _grounded_recall(
        self,
        program: RecallResult | None,
        context: _RecallContext,
    ) -> tuple[tuple[SearchHit, ...], str | None]:
        """Widen the grounding window by whatever the question's shape asks for beyond it.

        A ranking question keeps the window the ranking earned. A question that asked for every
        matching record adds that set to it, up to `recall_set_budget_chars`, and carries a line
        saying what the set is -- because a reader cannot see the predicate that produced its
        evidence, and a count over a silently truncated set is a wrong answer stated confidently.

        The set is added to the ranked window, never substituted for it. Grounding a plan on its
        exhaustive rows alone cost every question whose reads returned few rows the window it
        already had: measured on ATM-Hard, 12 grounded records down to 1, 2, 4 and 5, and one
        down to a refusal the unplanned path had answered. A plan says what else is relevant; it
        never says the ranking was wrong.
        """
        if program is None or not program.plan.exhaustive:
            budget = self._settings.evidence_budget
            return grounding_hits(context.ranked, context.limit, budget_chars=budget), None
        hits, omitted = _budgeted_recall(
            program,
            # `evidence_budget_chars` is what a caller who cares about prompt size sets, and a
            # set plan used to walk straight past it. The set budget may narrow that ceiling and
            # never widen it; with no ceiling set, the set budget is the only bound.
            self._settings.recall_set_budget
            if self._settings.evidence_budget is None
            else min(self._settings.recall_set_budget, self._settings.evidence_budget),
            media_limit=_RECALL_MEDIA_FACTOR * context.limit,
            max_rows=self._settings.recall_set_max_rows,
            # Exactly what the unplanned path grounds, this question's modality floor included,
            # so no plan can cost a question the evidence it already had.
            required=grounding_hits(context.ranked, context.limit),
        )
        return hits, recall_note(program, omitted=omitted)

    def _replan_possible(self, context: _RecallContext, *, attempt: int) -> bool:
        """Whether a later round could replace this one, which is what makes it unstreamable.

        Another round is spent only on the failure a round's own answer reports, and only when
        the caller asked for a committed answer: under `strict` that same signal is already the
        refusal they were given, and re-reading behind their back would answer a question they
        were told could not be answered. This is everything but the failure, which is what a
        round has to know before it starts.
        """
        return (
            self._settings.recall_planning
            and context.answer_policy == "best_effort"
            and attempt + 1 < self._settings.recall_rounds
        )

    def _recall_plan(
        self,
        question: str,
        *,
        reference_at: datetime,
        k: int,
        attempted: str = "",
    ) -> RecallPlan:
        """Ask the answerer how to read for this question, or keep today's single search.

        Every way this can go wrong lands on the same fallback -- a backend that cannot plan, a
        planner that fails, an empty completion, a plan the kernel will not run -- so the cost
        of a bad planner is one model call, never a different answer. Model output is data: it
        reaches `parse_recall_plan`, which decides what MindBridge does.
        """
        planner = self._backends.answerer
        if not isinstance(planner, RecallPlanningBackend) or not question.strip():
            return fallback_plan(question, k=k)
        model = getattr(planner, "generation_model", None)
        with self._model_trace(
            "generation",
            "plan",
            model=model if isinstance(model, str) else None,
            batch_size=1,
            modalities=(Modality.TEXT,),
        ):
            mark_model_requests(1)
            # The protocol is `runtime_checkable`, so `isinstance` proves the method exists and
            # not that it takes this keyword. A backend written against the shorter signature
            # keeps planning as long as the first round asks for nothing new.
            previous: builtins.dict[str, str] = {} if not attempted else {"attempted": attempted}
            try:
                payload = planner.plan_recall(
                    question,
                    reference_at=reference_at,
                    corpus_digest=self._corpus_digest(),
                    **previous,
                )
            except ModelError:
                # Planning is not answering. A provider that refused this call has said nothing
                # about whether it can answer, and the fallback plan is the one `ask` has always
                # run, so the question continues instead of failing on a hint.
                return fallback_plan(question, k=k)
        if payload is None:
            return fallback_plan(question, k=k)
        return parse_recall_plan(payload, reference_at=reference_at) or fallback_plan(question, k=k)

    def _corpus_digest(self) -> str:
        """Describe the active corpus in one line, so a plan cannot ask for what is not there."""
        digest = self._store.recall.recall_digest()
        if not digest.records:
            return "empty"
        span = (
            "no dated records"
            if digest.earliest is None or digest.latest is None
            else f"{datetime_text(digest.earliest)} to {datetime_text(digest.latest)}"
        )
        return (
            f"{digest.records} records; {span}; "
            f"modalities {', '.join(digest.modalities)}; "
            f"{digest.named_identities} named identities"
        )

    def _answer_chunks(  # noqa: C901 - streaming and non-streaming validation share one span
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> Generator[str, None, AnswerResult]:
        """Yield answer deltas in provider order and return the validated grounded result.

        A backend without `stream_answer` yields its whole answer as one delta, so every caller
        sees the same shape and `ask_stream()` never requires a streaming provider.
        """
        if self._backends.answerer is None:
            raise ModelError(
                "answer backend is not configured",
                reason="backend_not_configured",
                stage="generate",
            )
        model = getattr(self._backends.answerer, "generation_model", None)
        with self._model_trace(
            "generation",
            "chat",
            model=model if isinstance(model, str) else None,
            batch_size=1,
            modalities=(
                modality
                for value in (question, *(ModelInput(hit.content, hit.assets) for hit in hits))
                for modality in value.modalities
            ),
        ):
            mark_model_requests(1)
            buffered = False
            # The protocols are `runtime_checkable`, so `isinstance` proves the method exists but
            # not that it takes these keywords. A backend written against the two-argument
            # signature keeps working as long as the defaults ask for nothing new, so each one is
            # sent only when the caller or the plan moved it off its default.
            policy: dict[str, Any] = (
                {} if answer_policy == "strict" else {"answer_policy": answer_policy}
            )
            if exhaustive:
                policy["exhaustive"] = True
            try:
                if isinstance(self._backends.answerer, StreamingGenerationBackend):
                    started = perf_counter()
                    parts: builtins.list[str] = []
                    stream = iter(self._backends.answerer.stream_answer(question, hits, **policy))
                    used_hits: object = None
                    try:
                        while True:
                            try:
                                part = next(stream)
                            except StopIteration as completed:
                                used_hits = completed.value
                                break
                            if not isinstance(part, str):
                                raise ModelError(
                                    "generation model returned an invalid answer chunk",
                                    reason="response_invalid",
                                )
                            if part:
                                if not parts and current_model_request_count():
                                    trace.get_current_span().set_attribute(
                                        MODEL_TTFT, perf_counter() - started
                                    )
                                parts.append(part)
                                yield part
                    finally:
                        # This iterator owns the provider's streaming response. An abandoned
                        # `ask_stream()` unwinds to here, and leaving the close to the cleared
                        # frame's last reference is interpreter-dependent: 3.12 collects it at
                        # once, 3.10 does not, so the response stayed open there. Closing it
                        # here makes the release deterministic on every supported version.
                        release = getattr(stream, "close", None)
                        if callable(release):
                            release()
                    answer = "".join(parts)
                    if not answer.strip():
                        raise ModelError(
                            "generation model returned an invalid answer", reason="response_invalid"
                        )
                    if isinstance(used_hits, AnswerResult):
                        if used_hits.answer != answer:
                            raise ModelError(
                                "generation model returned an invalid answer",
                                reason="response_invalid",
                            )
                        result = used_hits
                    elif used_hits is None:
                        grounded = tuple(hits)
                    elif isinstance(used_hits, tuple) and all(
                        isinstance(hit, SearchHit) for hit in used_hits
                    ):
                        grounded = used_hits
                        # A backend may report an answer that is not the concatenated deltas:
                        # under `best_effort` the stream carries a low-confidence marker line
                        # that belongs to `abstained`, not to the prose the caller shows. The
                        # default path never consults `.answer`, so a hit sequence that happens
                        # to carry that attribute cannot replace a caller's deltas there.
                        reported = (
                            None
                            if answer_policy == "strict"
                            else getattr(used_hits, "answer", None)
                        )
                        if reported is not None:
                            if not isinstance(reported, str) or not reported.strip():
                                raise ModelError(
                                    "generation model returned an invalid answer",
                                    reason="response_invalid",
                                )
                            answer = reported
                        abstention_reason = getattr(used_hits, "abstention_reason", None)
                        if abstention_reason is not None and not isinstance(
                            abstention_reason, AbstentionReason
                        ):
                            raise ModelError(
                                "generation model returned invalid abstention status",
                                reason="response_invalid",
                            )
                    else:
                        raise ModelError(
                            "generation model returned invalid grounding hits",
                            reason="response_invalid",
                        )
                    if not isinstance(used_hits, AnswerResult):
                        reason = abstention_reason if isinstance(used_hits, tuple) else None
                        result = AnswerResult(
                            answer=answer,
                            hits=grounded,
                            abstained=reason is not None,
                            abstention_reason=reason,
                        )
                else:
                    result = self._backends.answerer.answer(question, hits, **policy)
                    buffered = True
            except MindBridgeError:
                raise
            except Exception as error:
                raise ModelError(
                    "failed to generate a grounded answer", reason="model_failed"
                ) from error
            if not isinstance(result, AnswerResult):
                raise ModelError(
                    "generation model returned an invalid answer", reason="response_invalid"
                )
            if buffered:
                yield result.answer
            return result

    def _reinforce_answered(self, hits: Sequence[SearchHit]) -> None:
        """Count the evidence an answer cited, so `access_count` is not permanently zero.

        `ranking_signals` scales a candidate by its access count and, when `decay_half_life_days`
        is set, ages it by the time since it was last read. Nothing but the explicit `reinforce`
        operation ever wrote either field, so for an agent driving MindBridge over MCP or REST the
        reinforcement factor was pinned at 1.0 for ever and enabling decay degraded memories
        purely by age, with no usage signal to hold up the ones that keep answering questions.
        Answering is that signal, and this set is already narrowed to what the model cited.

        `_write_lock` is reentrant and `ask` holds none of it at this point, so taking it here
        cannot deadlock against the searches above, which release it before returning. A failure
        is swallowed on purpose: this is bookkeeping, and losing it must not discard an answer
        that has already been generated and paid for.

        `reinforce_on_answer=False` turns it off, which measurement needs: reinforcing mid-run
        makes one question's retrieval depend on which earlier questions answered, and under
        concurrency on the order their updates committed.
        """
        if not hits or not self._settings.reinforce_on_answer:
            return
        with suppress(Exception), self._write_lock:
            self._store.records.reinforce_memories(
                tuple(hit.id for hit in hits),
                accessed_at=datetime.now(timezone.utc),
            )
