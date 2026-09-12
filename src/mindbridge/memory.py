"""Developer-facing local memory API.

`Memory` is the facade over the kernel in `mindbridge.kernel`. It validates local policy and
backend contracts once, opens the store and the index in the documented order, wires one object
per plane -- write, retrieval, answering, compilation, identity, control, records -- with the
dependencies each declares, and forwards every public operation to the plane that owns it. The
planes share nothing implicitly: the store, the media store, the locks, the backends, and the
settings travel as explicit constructor arguments.
"""

from __future__ import annotations

import asyncio
import builtins
import logging
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from contextvars import copy_context
from dataclasses import fields
from datetime import datetime
from functools import partial
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import cast

from opentelemetry import trace
from opentelemetry.trace import Tracer

from mindbridge._telemetry import TRACER_NAME
from mindbridge.configuration import MindBridgeConfig, resolve_memory_config
from mindbridge.exceptions import (
    IndexUnavailableError,
    MindBridgeError,
    StorageError,
    ValidationError,
)
from mindbridge.infrastructure.local.assets import AssetStore
from mindbridge.infrastructure.local.zvec_index import ZvecIndex
from mindbridge.kernel.answering import ASYNC_QUEUE_TIME_MS, Answering
from mindbridge.kernel.compilation import Compilation
from mindbridge.kernel.contracts import (
    STORE_METADATA_KEYS,
    close_quietly,
    declared_capabilities,
    resolve_backends,
    unique_resources,
)
from mindbridge.kernel.control_plane import ControlPlane
from mindbridge.kernel.embedding import Embedding
from mindbridge.kernel.formation import Formation
from mindbridge.kernel.hydration import Hydrator
from mindbridge.kernel.identity import Identities
from mindbridge.kernel.ingestion import Ingestion
from mindbridge.kernel.lifecycle import Lifecycle
from mindbridge.kernel.materialization import Materializer
from mindbridge.kernel.projection import Projection
from mindbridge.kernel.records import Records
from mindbridge.kernel.retrieval import Retrieval
from mindbridge.kernel.runtime import (
    Index,
    Storage,
    ensure_store_metadata,
    open_store,
    translate_storage_errors,
)
from mindbridge.kernel.settings import resolve_settings
from mindbridge.kernel.speech import Speech
from mindbridge.kernel.validation import strict_bool
from mindbridge.kernel.vision import Vision
from mindbridge.models.base import (
    ConsolidationBackend,
    EmbeddingBackend,
    FaceBackend,
    FormationBackend,
    GenerationBackend,
    RecallPlanningBackend,
    SpeechBackend,
    TranscriptionBackend,
    VisionDescriptionBackend,
)
from mindbridge.plugins import MemoryConfig, MemoryPlugins
from mindbridge.types import (
    AnswerChunk,
    AnswerPolicy,
    AnswerResult,
    ConsentState,
    ConsolidationCandidate,
    ConsolidationReport,
    ContentInput,
    ContextBudget,
    ContextBundle,
    DeliberationReport,
    ExportBundle,
    FaceObservation,
    IdentityErasure,
    IdentityProfile,
    IndexQuantization,
    MemoryCapabilities,
    MemoryOperation,
    MemoryOperationRecord,
    MemoryOutcome,
    MemoryRecord,
    MemoryTrigger,
    MemoryType,
    ObservationContext,
    Page,
    PendingCapture,
    RetentionPolicy,
    RetentionReport,
    RetrievalMode,
    RetrievalScope,
    SearchHit,
    SpeakerSegment,
    StreamInput,
    TracedSearchResult,
)

_LOGGER = logging.getLogger(__name__)
_DEFAULT_CONFIG = MemoryConfig()
_STREAM_EXHAUSTED = object()


class Memory:
    """Persist and retrieve native text, image, video, audio, and omni memories."""

    def __init__(
        self,
        data_dir: str | Path = ".mindbridge",
        *,
        embedder: EmbeddingBackend,
        answerer: GenerationBackend | None = None,
        transcriber: SpeechBackend | TranscriptionBackend | None = None,
        vision_describer: VisionDescriptionBackend | None = None,
        face_analyzer: FaceBackend | None = None,
        former: FormationBackend | None = None,
        consolidator: ConsolidationBackend | None = None,
        index_speech: bool = _DEFAULT_CONFIG.index_speech,
        index_quantization: IndexQuantization = _DEFAULT_CONFIG.index_quantization,
        retrieval_mode: RetrievalMode = _DEFAULT_CONFIG.retrieval_mode,
        minimum_relevance: float = _DEFAULT_CONFIG.minimum_relevance,
        ambiguity_margin: float = _DEFAULT_CONFIG.ambiguity_margin,
        evidence_budget_chars: int | None = _DEFAULT_CONFIG.evidence_budget_chars,
        recall_planning: bool = _DEFAULT_CONFIG.recall_planning,
        recall_set_budget_chars: int = _DEFAULT_CONFIG.recall_set_budget_chars,
        recall_set_max_rows: int = _DEFAULT_CONFIG.recall_set_max_rows,
        recall_rounds: int = _DEFAULT_CONFIG.recall_rounds,
        decay_half_life_days: float | None = _DEFAULT_CONFIG.decay_half_life_days,
        reinforce_on_answer: bool = _DEFAULT_CONFIG.reinforce_on_answer,
        speaker_similarity: float = _DEFAULT_CONFIG.speaker_similarity,
        speaker_margin: float = _DEFAULT_CONFIG.speaker_margin,
        face_similarity: float = _DEFAULT_CONFIG.face_similarity,
        face_margin: float = _DEFAULT_CONFIG.face_margin,
        identity_link_min_assets: int = _DEFAULT_CONFIG.identity_link_min_assets,
        memory_budget_records: int | None = _DEFAULT_CONFIG.memory_budget_records,
        query_failure_window_seconds: float = _DEFAULT_CONFIG.query_failure_window_seconds,
        query_failure_history: int = _DEFAULT_CONFIG.query_failure_history,
        retention: RetentionPolicy = _DEFAULT_CONFIG.retention,
        tracer: Tracer | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        tracer = trace.get_tracer(TRACER_NAME) if tracer is None else tracer
        self._settings = resolve_settings(
            index_speech=index_speech,
            index_quantization=index_quantization,
            retrieval_mode=retrieval_mode,
            minimum_relevance=minimum_relevance,
            ambiguity_margin=ambiguity_margin,
            evidence_budget_chars=evidence_budget_chars,
            recall_planning=recall_planning,
            recall_set_budget_chars=recall_set_budget_chars,
            recall_set_max_rows=recall_set_max_rows,
            recall_rounds=recall_rounds,
            decay_half_life_days=decay_half_life_days,
            reinforce_on_answer=reinforce_on_answer,
            speaker_similarity=speaker_similarity,
            speaker_margin=speaker_margin,
            face_similarity=face_similarity,
            face_margin=face_margin,
            identity_link_min_assets=identity_link_min_assets,
            memory_budget_records=memory_budget_records,
            query_failure_window_seconds=query_failure_window_seconds,
            query_failure_history=query_failure_history,
            retention=retention,
        )
        if (
            self._settings.recall_planning
            and answerer is not None
            and not isinstance(answerer, RecallPlanningBackend)
        ):
            # Every planning failure resolves to the fallback plan on purpose, so an answerer that
            # never declared `plan_recall` -- a wiring bug, not a planner failing -- would read
            # exactly like a planner choosing nothing. Said once, here, not once per question.
            _LOGGER.warning(
                "recall_planning is on but the answerer cannot plan; every question runs the "
                "fallback plan"
            )

        self._store = open_store(self.data_dir)
        injected = (
            embedder,
            transcriber,
            vision_describer,
            face_analyzer,
            former,
            consolidator,
            answerer,
        )
        try:
            self._backends = resolve_backends(
                index_quantization=self._settings.index_quantization,
                embedder=embedder,
                answerer=answerer,
                transcriber=transcriber,
                vision_describer=vision_describer,
                face_analyzer=face_analyzer,
                former=former,
                consolidator=consolidator,
            )
            assets = AssetStore(self.data_dir)
            self._storage = Storage(
                store=self._store,
                assets=assets,
                write_lock=RLock(),
                formation_lock=RLock(),
                settle_lock=RLock(),
            )
            self._write_lock = self._storage.write_lock
            # The planes that need no index open first, so the store can be upgraded and
            # re-embedded before the index that projects it exists.
            self._lifecycle = Lifecycle(
                storage=self._storage,
            )
            self._hydrator = Hydrator(
                storage=self._storage,
            )
            materializer = Materializer(
                storage=self._storage,
                lifecycle=self._lifecycle,
            )
            speech = Speech(
                tracer=tracer,
                storage=self._storage,
                backends=self._backends,
                settings=self._settings,
                hydrator=self._hydrator,
            )
            embedding = Embedding(
                tracer=tracer,
                storage=self._storage,
                backends=self._backends,
                settings=self._settings,
                hydrator=self._hydrator,
                speech=speech,
            )
            self._lifecycle.collect_orphan_assets(scan_physical=True)
            index_path = self.data_dir / "zvec"
            index_missing = not index_path.exists()
            index_rebuild, embedding_rebuild = ensure_store_metadata(
                self._store, self._backends, self._settings, index_path
            )
            if embedding_rebuild:
                _LOGGER.warning(
                    "re-embedding stored memories for space %s", self._backends.space_id
                )
                embedding.reembed_memories()
                self._store.set_metadata(STORE_METADATA_KEYS["space"], self._backends.space_id)
                self._store.set_metadata(STORE_METADATA_KEYS["index"], self._settings.index_recipe)
            if index_missing or index_rebuild:
                _LOGGER.info(
                    "rebuilding the search index (missing=%s, recipe changed=%s)",
                    index_missing,
                    index_rebuild,
                )
                with translate_storage_errors("checkpoint a missing search index"):
                    self._store.index.queue_all_embeddings()
        except BaseException:
            close_quietly(*injected, self._store)
            raise

        try:
            self._index: Index = ZvecIndex(
                index_path,
                dimension=self._backends.embedding_dimension,
                quantization=self._settings.index_quantization,
            )
        except Exception as error:
            close_quietly(*injected, self._store)
            raise IndexUnavailableError(
                "failed to open the local search index", stage="open", reason="index_missing"
            ) from error

        self._projection = Projection(
            tracer=tracer,
            storage=self._storage,
            index=self._index,
            lifecycle=self._lifecycle,
        )
        self._vision = Vision(
            tracer=tracer,
            storage=self._storage,
            backends=self._backends,
            settings=self._settings,
            lifecycle=self._lifecycle,
            hydrator=self._hydrator,
            materializer=materializer,
            speech=speech,
            embedding=embedding,
            projection=self._projection,
        )
        self._retrieval = Retrieval(
            tracer=tracer,
            storage=self._storage,
            index=self._index,
            backends=self._backends,
            settings=self._settings,
            lifecycle=self._lifecycle,
            hydrator=self._hydrator,
            materializer=materializer,
            speech=speech,
            embedding=embedding,
            projection=self._projection,
        )
        self._formation = Formation(
            tracer=tracer,
            storage=self._storage,
            backends=self._backends,
            materializer=materializer,
            embedding=embedding,
            projection=self._projection,
        )
        self._identities = Identities(
            tracer=tracer,
            storage=self._storage,
            backends=self._backends,
            lifecycle=self._lifecycle,
            materializer=materializer,
            speech=speech,
            embedding=embedding,
            projection=self._projection,
            vision=self._vision,
            formation=self._formation,
        )
        self._ingestion = Ingestion(
            tracer=tracer,
            storage=self._storage,
            backends=self._backends,
            settings=self._settings,
            lifecycle=self._lifecycle,
            hydrator=self._hydrator,
            materializer=materializer,
            speech=speech,
            embedding=embedding,
            projection=self._projection,
            vision=self._vision,
            formation=self._formation,
            identities=self._identities,
        )
        self._answering = Answering(
            tracer=tracer,
            storage=self._storage,
            backends=self._backends,
            settings=self._settings,
            lifecycle=self._lifecycle,
            hydrator=self._hydrator,
            materializer=materializer,
            speech=speech,
            vision=self._vision,
            retrieval=self._retrieval,
        )
        self._compilation = Compilation(
            tracer=tracer,
            storage=self._storage,
            backends=self._backends,
            lifecycle=self._lifecycle,
            hydrator=self._hydrator,
            materializer=materializer,
            speech=speech,
            retrieval=self._retrieval,
        )
        self._control = ControlPlane(
            tracer=tracer,
            storage=self._storage,
            backends=self._backends,
            settings=self._settings,
            lifecycle=self._lifecycle,
            hydrator=self._hydrator,
            materializer=materializer,
            projection=self._projection,
            retrieval=self._retrieval,
            formation=self._formation,
            identities=self._identities,
        )
        self._records = Records(
            tracer=tracer,
            storage=self._storage,
            settings=self._settings,
            lifecycle=self._lifecycle,
            hydrator=self._hydrator,
            projection=self._projection,
            identities=self._identities,
        )
        self._lifecycle.mark_open()
        try:
            with self._write_lock:
                self._projection.drain()
        except BaseException:
            self._lifecycle.mark_closed()
            self._close_resources()
            raise

    @classmethod
    def from_plugins(
        cls,
        data_dir: str | Path = ".mindbridge",
        *,
        plugins: MemoryPlugins,
        config: MemoryConfig | None = None,
        tracer: Tracer | None = None,
    ) -> Memory:
        """Open memory from an explicit capability bundle and local policy."""
        if not isinstance(plugins, MemoryPlugins):
            raise ValidationError("plugins must be a MemoryPlugins value")
        if config is None:
            config = MemoryConfig()
        elif not isinstance(config, MemoryConfig):
            raise ValidationError("config must be a MemoryConfig value")
        # The constructor takes exactly these two dataclasses' fields flattened, so unpacking them
        # is the whole translation. A field added to either reaches the constructor without a line
        # here.
        return cls(
            data_dir,
            tracer=tracer,
            **{field.name: getattr(plugins, field.name) for field in fields(plugins)},
            **{field.name: getattr(config, field.name) for field in fields(config)},
        )

    @classmethod
    def from_config(
        cls,
        config: MindBridgeConfig | Mapping[str, object],
        *,
        tracer: Tracer | None = None,
    ) -> Memory:
        """Open memory from validated declarative configuration."""
        resolved = resolve_memory_config(config)
        try:
            return cls.from_plugins(
                resolved.data_dir,
                plugins=resolved.plugins,
                config=resolved.settings,
                tracer=tracer,
            )
        except BaseException:
            resolved.close()
            raise

    def __enter__(self) -> Memory:
        self._lifecycle.require_open()
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def add(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord:
        """Add one native or mixed-modal memory and return its stable record."""
        return self._ingestion.add(
            content,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        occurred_at: Sequence[datetime | None] | None = None,
        occurred_end: Sequence[datetime | None] | None = None,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: Sequence[ObservationContext | None] | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """Add memories in one model batch and one SQLite transaction."""
        return self._ingestion.add_many(
            contents,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    def capture(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord:
        """Commit one memory without calling any model; `settle()` makes it searchable.

        The returned record is the content-addressed record `add()` returns for the same input, so
        capturing and then adding the same content is one memory. It is durable and readable
        through `get()` and `list()` immediately and invisible to `search()` until settled.

        `settle()` appends the text its models derive to `content`; it never rewrites what the
        caller supplied. See `MemoryRecord` for how derived sections are marked and where the raw
        evidence lives.
        """
        return self._ingestion.capture(
            content,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    def settle(
        self,
        *,
        limit: int = 100,
        max_attempts: int = 3,
        memory_ids: Sequence[str] | None = None,
    ) -> int:
        """Enrich, embed, index, and form up to `limit` captured records in enqueue order.

        Every readable record is attempted: a failing one keeps its queue row, its attempt count,
        and its reason, and the records behind it still settle. The first failure is raised once
        the batch is done, with `subject` naming its record; the rest are readable through
        `pending_captures()`. A record that has already failed `max_attempts` times is skipped
        rather than retried, so one poisoned capture cannot block the queue forever. It stays
        queued and visible, and raising the ceiling is what retries it.

        Pass `memory_ids` to settle only those records. Naming a record is the host asking for it
        by hand, so the retry ceiling does not apply and `max_attempts` is ignored: that is how a
        record parked at the ceiling is retried, quarantined by simply never being named, or
        settled ahead of the queue. IDs that are not queued are skipped.

        One settlement runs at a time per `Memory`. A concurrent call waits rather than paying
        for the same model work twice, and usually then finds the queue already drained.
        """
        return self._ingestion.settle(limit=limit, max_attempts=max_attempts, memory_ids=memory_ids)

    def pending_captures(
        self,
        *,
        limit: int = 100,
        memory_ids: Sequence[str] | None = None,
    ) -> tuple[PendingCapture, ...]:
        """Return up to `limit` records whose deferred work is not finished, oldest first.

        With a formation backend, `add()` holds a row between its commit and formation, so a
        queued record may already be searchable and owe formation only. Pass `memory_ids` to ask
        whether specific records are still waiting: one that is absent from the result is not
        pending, which means it is settled or was never stored, and `get()` tells the two apart.
        """
        return self._ingestion.pending_captures(limit=limit, memory_ids=memory_ids)

    def add_stream(
        self,
        contents: Iterable[ContentInput | StreamInput],
        *,
        capture: bool = False,
    ) -> Iterator[MemoryRecord]:
        """Add a lazy omni stream one durable, searchable observation at a time.

        Index applies are batched in bounded groups, so a fast source pays one outbox drain per
        group instead of one per item; a reader always drains first, so a committed item is
        searchable before the group closes.

        With `capture=True` each item commits through `capture()` instead: every yielded record is
        durable and readable but has no vectors, so the stream keeps its acknowledgement off the
        model path and the host owes the matching `settle()` before anything is searchable.
        """
        return self._ingestion.add_stream(contents, capture=capture)

    def search(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> tuple[SearchHit, ...]:
        """Return ranked memories for a native or mixed-modal query."""
        return self._retrieval.search(
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
        )

    def search_with_trace(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> TracedSearchResult:
        """Return ranked memories plus an opt-in candidate trace without evidence content."""
        return self._retrieval.search_with_trace(
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
        )

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
        """Answer a native or mixed-modal question only from retrieved memories.

        `answer_policy` decides what happens when the retrieved evidence is thin. The default,
        `"strict"`, refuses: `answer` is a fixed sentence and `abstained` is true. With
        `"best_effort"` the answerer commits to the most likely answer the evidence supports --
        for a multiple-choice question, always one of the options -- and still reports the same
        `abstained` and `abstention_reason`, so a caller whose protocol gives no credit for
        "unknown" gets a usable answer without losing the confidence signal.

        `link_identities` gates the one write `ask` can otherwise reach: when a retrieved image
        or video corroborates a voice-and-face pair, face recognition still runs to identify who
        answers the question, but with `link_identities=False` the corroborated bind is never
        committed -- no MERGE row, no new identity link. A host that withholds embodied MCP
        tools passes `link_identities=False` here too, so recall access alone never carries
        merge authority.
        """
        return self._answering.ask(
            question,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            link_identities=link_identities,
            answer_policy=answer_policy,
        )

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
        """Answer as `ask()` does, but yield the answer while the model is still producing it.

        Retrieval runs to completion before the first chunk; only generation is incremental. The
        terminal chunk carries the same `AnswerResult` `ask()` returns, and a backend without
        `stream_answer` still yields its whole answer as one delta. `link_identities` means what
        it means on `ask()`, which is this method drained.

        Reading to the terminal chunk releases everything the answer held, so stopping there
        needs no cleanup. Abandoning the stream mid-answer leaves one operation open until the
        generator is closed or collected, and `close()` waits for open operations.
        """
        return self._answering.ask_stream(
            question,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            link_identities=link_identities,
            answer_policy=answer_policy,
        )

    def compile(
        self,
        goal: ContentInput,
        *,
        budget: ContextBudget | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        allow_partial_sources: bool = False,
    ) -> ContextBundle:
        """Compile one bounded, structured context bundle for a goal."""
        return self._compilation.compile(
            goal,
            budget=budget,
            reference_at=reference_at,
            scope=scope,
            allow_partial_sources=allow_partial_sources,
        )

    @property
    def capabilities(self) -> MemoryCapabilities:
        """What this composition supports, from the same declarations routing reads.

        Published so a caller does not have to construct a probe write to find out, and so a
        transport can report the composition instead of a bare liveness flag.
        """
        return declared_capabilities(
            embedder=self._backends.embedder,
            answerer=self._backends.answerer,
            transcriber=self._backends.transcriber,
            vision_describer=self._backends.vision_describer,
            face_analyzer=self._backends.face_analyzer,
            former=self._backends.former,
            consolidator=self._backends.consolidator,
        )

    def get(self, memory_id: str) -> MemoryRecord:
        """Return one memory or raise `MemoryNotFoundError`."""
        return self._records.get(memory_id)

    def speech(self, memory_id: str) -> tuple[SpeakerSegment, ...]:
        """Transcribe speech and resolve stable local speaker identities."""
        return self._identities.speech(memory_id)

    def faces(self, memory_id: str) -> tuple[FaceObservation, ...]:
        """Detect faces and resolve stable local face-and-voice identities."""
        return self._identities.faces(memory_id)

    def register_speaker(
        self,
        speaker_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        """Assign or replace a human-readable name for one recognized speaker."""
        return self._identities.register_speaker(speaker_id, name, relationship=relationship)

    def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        """Assign or replace the name, and optionally the relationship, of one identity.

        Omitting ``relationship`` leaves any recorded relationship intact, so renaming a person
        never silently discards it. There is deliberately no way to clear one here.
        """
        return self._identities.register_identity(identity_id, name, relationship=relationship)

    def identity(self, identity_id: str) -> IdentityProfile | None:
        """Return one identity's recorded name and relationship, or None when it does not exist.

        Face and speech observations carry an ``identity_id``; this resolves that id, following
        a merge alias, to whatever a caller has registered for the person.
        """
        return self._identities.identity(identity_id)

    def record_consent(
        self,
        identity_id: str,
        state: ConsentState,
        *,
        note: str | None = None,
    ) -> MemoryOperationRecord | None:
        """Record what one recognized person said about being processed as a recognized person.

        Consent is an assertion in the same family as naming: an identity-bound, evidence-bearing
        claim that stands until a later one supersedes it, recorded as `USER_STATEMENT` because
        only a person can give it. A model may not propose one -- the kernel refuses a proposed
        `CONSENT` operation as `unauthorized` the way it refuses a proposed `MERGE` -- so this is
        the only path that writes one.

        `WITHHELD` and `WITHDRAWN` restrain the kernel identically from the next observation on:
        `faces()`, `speech()`, and `ask(link_identities=True)` stop enrolling new face and voice
        exemplars for this person and stop merging their voice and face identities, and
        `compile()` leaves them out of the actors section and says so with a `CONSENT_WITHHELD`
        unknown. Recognition against exemplars already held is deliberately untouched: a
        question about a photo the host already has is not new processing of a person, and
        destroying what is already held is `forget_identity`, which this is not.

        Returns the logged operation, or `None` when the same statement already stands, so the
        decision is auditable through `operations()` and reversible through `rollback()`, which
        restores whatever was recorded before it.
        """
        return self._identities.record_consent(identity_id, state, note=note)

    def consent(self, identity_id: str) -> ConsentState | None:
        """Return the consent state one identity's standing assertion projects, or None.

        `None` means nobody has recorded a statement, which is neither consent nor a refusal:
        the kernel restrains itself only on a statement that was actually made.
        """
        return self._identities.consent(identity_id)

    def unlink_identity(self, alias_id: str) -> str | None:
        """Reverse one recorded face-and-voice merge, restoring ``alias_id`` as its own identity.

        Cross-modal binding infers that one voice and one face belong to the same person from
        their co-occurrence, which corroboration makes unlikely to be wrong but cannot make
        impossible. Returns the restored identity ID, or None when the merge is not reversible
        because no record names which modality was contributed. The restored identity keeps no
        name or relationship; the identity it was merged into keeps both.

        This resets the pair's accumulated evidence; it does not suppress the pair. If the same
        voice and face keep co-occurring, they will be corroborated and merged again.

        A split is the `CORRECT` half of the control plane's "correct or split" intent, so it
        commits with its own log row: it shows up in `operations()` and `rollback()` of that row
        re-merges the pair under the identity that survived.
        """
        return self._identities.unlink_identity(alias_id)

    def forget_identity(self, identity_id: str) -> IdentityErasure:
        """Erase a person: their biometric template, their aliases, and their indexed name.

        `delete` removes a memory; this removes a *person* from every memory. Both are needed,
        because they answer different requests: "forget that evening" and "forget me". Erasing
        the identity rows alone would not honour the second one, since a registered name is
        written into the indexed document, so the search projection would still answer to it.

        Memories, their content and their media survive, and a transcript keeps its words with
        the speaker attribution dropped. Forgetting a person is not forgetting the evening.

        This does **not** stop a later encounter from minting a fresh identity for the same
        person: recognising someone as previously-forgotten would require keeping the template
        this destroys. A deployment that wants "never recognise this person again" needs a
        retained blocklist, which is the opposite of a deletion and must not be spelled like one.
        """
        return self._identities.forget_identity(identity_id)

    def reinforce(self, memory_ids: Sequence[str]) -> int:
        """Record explicit positive feedback for existing memories."""
        return self._records.reinforce(memory_ids)

    # -- Agentic memory control plane ----------------------------------------------------------
    # One bounded memory-management loop (gate 3 of docs/context-os.md). The backend sees a
    # bounded evidence set and only proposes; every field is validated here, each accepted
    # operation commits with its own append-only log row, and `rollback()` reverses it. Physical
    # deletion is not an intent: it stays on `delete()` under host authority. None of these
    # methods is exposed on REST or MCP.

    def consolidation_candidates(
        self,
        *,
        limit: int = 32,
        idle: bool = False,
    ) -> tuple[ConsolidationCandidate, ...]:
        """Ask what needs deliberation, at most `limit` rows, interleaved across triggers.

        This is the durable trigger the slow loop runs on: every row is derived from state
        already committed -- evidence links, lineage disagreement, recorded confirmations,
        recorded recall failures, the configured record budget -- rather than from a clock. Hand
        a row's `memory_ids` straight to `consolidate(evidence_ids=...)` with the row's
        `trigger`, or let `deliberate()` do it.

        `idle` is the operator declaring an approved idle or charging window. The kernel does not
        guess it from a clock: whether now is a good time to spend the device's battery on
        reasoning is the host's knowledge.
        """
        return self._control.consolidation_candidates(limit=limit, idle=idle)

    def deliberate(
        self,
        *,
        limit: int = 32,
        max_rounds: int = 4,
        idle: bool = False,
    ) -> DeliberationReport:
        """Run the memory-management loop to a fixed point, or to `max_rounds`.

        One round asks `consolidation_candidates()` what is due and runs `consolidate()` over
        each row with the row's own trigger. Applying operations can make further work due --
        a derived record gains evidence, a corrected lineage stops disagreeing -- so the loop
        repeats until nothing is due or the ceiling is reached.

        It terminates on a backend that proposes nothing, because a pass records that it weighed
        its evidence set whatever it yielded, and candidate derivation excludes a candidate
        weighed since its own signal. The report then says so: rounds and `weighed` non-zero,
        `applied` zero.
        """
        return self._control.deliberate(limit=limit, max_rounds=max_rounds, idle=idle)

    def consolidate(
        self,
        *,
        evidence_ids: Sequence[str] | None = None,
        query: ContentInput | None = None,
        limit: int = 32,
        trigger: MemoryTrigger = MemoryTrigger.MANUAL,
    ) -> ConsolidationReport:
        """Deliberate over a bounded evidence set and apply the operations policy accepts."""
        return self._control.consolidate(
            evidence_ids=evidence_ids, query=query, limit=limit, trigger=trigger
        )

    def apply(self, operation: MemoryOperation) -> MemoryOperationRecord:
        """Apply one operation the host supplies, through the same kernel validation.

        This is the public replay surface: a logged `MemoryOperation` re-applied against a fresh
        store reproduces the derived state, without a `ConsolidationBackend` and without a model.
        Nothing is trusted because the host supplied it. The window is the operation's own cited
        evidence and named targets, eligibility is checked exactly as it is for a proposal, and
        the apply transaction re-checks every target so one that moved meanwhile is refused as
        stale rather than half-applied.

        Raises `ValidationError` naming the kernel's own rejection reason when the operation is
        refused, because a single operation the caller chose has nowhere else to report it.
        """
        return self._control.apply(operation)

    def record_outcome(
        self,
        operation_id: int,
        outcome: MemoryOutcome,
        *,
        note: str | None = None,
    ) -> bool:
        """Record what later evidence said about one applied operation.

        Post-hoc and purely for measurement: the kernel never reads it back into a decision, and
        recording `REFUTED` does not reverse anything -- `rollback()` does that. It exists so the
        slow loop's quality is derivable from the log rather than only its rollback success:
        consolidation precision and contradiction recovery are `CONFIRMED` rates over
        `CONSOLIDATE` and `CORRECT` rows, and false retirement is the `REFUTED` rate over rows
        that forgot something.

        Reports `False` for an unknown `operation_id`. A later call replaces an earlier
        judgement, because later evidence supersedes earlier evidence here as everywhere.
        """
        return self._control.record_outcome(operation_id, outcome, note=note)

    def forget(self, memory_ids: Sequence[str]) -> MemoryOperationRecord | None:
        """Cognitively forget memories: recall skips them, `get()` and `list()` keep them.

        This is not deletion. `MemoryRecord.forgotten_at` stays readable for audit and
        `rollback()` of the returned operation is what restores recall -- re-adding the same
        content does not, because content addressing returns the record that already exists and
        leaves its `forgotten_at` standing. Use `delete()` to remove a record and its media.

        All or nothing: an unknown ID raises `MemoryNotFoundError` the way `get()` and `delete()`
        do. The host names the IDs here, so partially applying them and logging the rest would
        make the operation log claim an effect that never happened.

        `None` means nothing changed and nothing was logged: no IDs were named, a record was
        already forgotten, the set named a bound naming assertion (only `rollback()` of the
        operation that asserted a name may retire it), a target moved under the proposal, or an
        identical operation is already in the log.
        """
        return self._control.forget(memory_ids)

    def rollback(self, operation_id: int) -> bool:
        """Reverse one applied operation, newest first on a lineage.

        Reports `False` for an unknown or already-reversed `operation_id`, and for one a later
        standing operation has built on: when a second consolidation superseded the record this
        one put in force, reversing this one first would leave two current versions in the
        lineage, so the newer operation must be rolled back first.

        Also reports `False`, without reversing anything, for the one operation that is not
        reversible: the erasure of a person. A `FORGET` row carrying an identity is physical
        forgetting, and its log row is the audit trail that it happened rather than a copy of
        what it destroyed.
        """
        return self._control.rollback(operation_id)

    def operations(self, *, limit: int = 100) -> tuple[MemoryOperationRecord, ...]:
        """List logged control-plane operations, newest first."""
        return self._control.operations(limit=limit)

    def list(self, *, limit: int = 100, cursor: str | None = None) -> Page:
        """List newest memories with an opaque stable keyset cursor."""
        return self._records.list(limit=limit, cursor=cursor)

    def delete(self, memory_id: str) -> bool:
        """Delete one memory and garbage-collect media no memory still references.

        A naming assertion is an ordinary record, so deleting one is allowed and moves the
        projection it fed. The registry and the indexed text that quoted the name are rebuilt
        in the same commit, because a name nothing asserts must not stay searchable.
        """
        return self._records.delete(memory_id)

    def export(
        self,
        *,
        identity_id: str | None = None,
        memory_ids: Sequence[str] | None = None,
    ) -> ExportBundle:
        """Return everything this memory holds about one data subject, in transferable form.

        Name exactly one subject: `identity_id` for a recognized person -- every memory they
        occur in, every name and consent statement asserted about them, their registry row, and
        every logged operation that moved any of it -- or `memory_ids` for a set of records a
        caller already knows, with the log rows that touched them.

        Records come back in every version, including retired ones and ones cognitively
        forgotten, because the question this answers is what is held rather than what is
        current. Media travels as asset identity, size, and digest on each record; the bytes
        stay on disk, so copying them is the host's decision and not an accident of asking.

        This reads only. Nothing here forgets, deletes, or acknowledges anything, and a merged
        alias resolves to its canonical identity exactly as every other identity read does.
        """
        return self._records.export(identity_id=identity_id, memory_ids=memory_ids)

    def apply_retention(self, *, dry_run: bool = False) -> RetentionReport:
        """Delete what the declared retention policy says has outlived its purpose.

        Physical forgetting under a deterministic policy: aged media and every memory that
        still references it, records cognitively forgotten longer ago than the policy allows,
        and capture-queue rows whose repeated failures have aged out. Deletion runs through
        `delete()`, so media is garbage-collected and the naming projection is rebuilt exactly
        as they are for a hand-deleted record.

        An unset field in `RetentionPolicy` is not a zero-day policy: it does nothing at all.
        A policy that names no age therefore makes this a no-op, which is the right default for
        an operation whose effect cannot be undone.

        `dry_run=True` reports the same identifiers and deletes nothing, so a policy can be
        read back against a real store before it is allowed to run.
        """
        return self._records.apply_retention(dry_run=dry_run)

    def reindex(self) -> int:
        """Rebuild the disposable Zvec collection from authoritative SQLite rows."""
        return self._projection.reindex()

    def optimize(self) -> None:
        """Merge staged Zvec vectors into the configured index."""
        return self._projection.optimize()

    def _add_stream_input(self, item: StreamInput) -> MemoryRecord:
        return self._ingestion.add_stream_input(item)

    def _capture_stream_input(self, item: StreamInput) -> MemoryRecord:
        return self._ingestion.capture_stream_input(item)

    def close(self) -> None:
        """Close model, index, and SQLite resources; repeated calls are harmless."""
        # A constructor that failed before wiring already closed what it had opened and left
        # nothing here to close, so this is the same no-op a closed instance gets.
        lifecycle: Lifecycle | None = getattr(self, "_lifecycle", None)
        if lifecycle is None or not lifecycle.begin_close():
            return
        try:
            with self._write_lock:
                failures = []
                try:
                    self._lifecycle.cleanup_pending_assets()
                except Exception as error:
                    failures.append(error)
                # Rows a failed flush leaves pending are durable in SQLite and replay on open.
                try:
                    self._projection.flush_pending()
                except Exception as error:
                    failures.append(error)
                failures.extend(self._close_resources())
        finally:
            self._lifecycle.mark_closed()
        if not failures:
            return
        first = failures[0]
        if isinstance(first, MindBridgeError):
            raise first
        raise StorageError(
            "failed to close local memory resources", reason="io_failed", stage="close"
        ) from first

    def _close_resources(self) -> builtins.list[Exception]:
        failures: builtins.list[Exception] = []
        for resource in unique_resources((*self._backends.resources(), self._index, self._store)):
            try:
                resource.close()
            except Exception as error:
                failures.append(error)
        return failures


class AsyncMemory:
    """Async facade over one open synchronous `Memory`, which it owns and closes."""

    def __init__(self, memory: Memory) -> None:
        if not isinstance(memory, Memory):
            raise ValidationError("memory must be a Memory instance")
        self._memory = memory

    @classmethod
    def from_plugins(
        cls,
        data_dir: str | Path = ".mindbridge",
        *,
        plugins: MemoryPlugins,
        config: MemoryConfig | None = None,
        tracer: Tracer | None = None,
    ) -> AsyncMemory:
        """Open async memory from an explicit capability bundle and local policy."""
        return cls(Memory.from_plugins(data_dir, plugins=plugins, config=config, tracer=tracer))

    @classmethod
    def from_config(
        cls,
        config: MindBridgeConfig | Mapping[str, object],
        *,
        tracer: Tracer | None = None,
    ) -> AsyncMemory:
        """Open async memory from validated declarative configuration."""
        return cls(Memory.from_config(config, tracer=tracer))

    async def __aenter__(self) -> AsyncMemory:
        self._memory._lifecycle.require_open()
        return self

    async def __aexit__(self, *_error: object) -> None:
        await self.close()

    async def add(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord:
        return await asyncio.to_thread(
            self._memory.add,
            content,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    async def _add_stream_input(self, item: StreamInput) -> MemoryRecord:
        return await asyncio.to_thread(self._memory._add_stream_input, item)

    async def _capture_stream_input(self, item: StreamInput) -> MemoryRecord:
        return await asyncio.to_thread(self._memory._capture_stream_input, item)

    async def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        occurred_at: Sequence[datetime | None] | None = None,
        occurred_end: Sequence[datetime | None] | None = None,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: Sequence[ObservationContext | None] | None = None,
    ) -> tuple[MemoryRecord, ...]:
        return await asyncio.to_thread(
            self._memory.add_many,
            contents,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    async def capture(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord:
        return await asyncio.to_thread(
            self._memory.capture,
            content,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    async def settle(
        self,
        *,
        limit: int = 100,
        max_attempts: int = 3,
        memory_ids: Sequence[str] | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self._memory.settle,
            limit=limit,
            max_attempts=max_attempts,
            memory_ids=memory_ids,
        )

    async def pending_captures(
        self,
        *,
        limit: int = 100,
        memory_ids: Sequence[str] | None = None,
    ) -> tuple[PendingCapture, ...]:
        return await asyncio.to_thread(
            self._memory.pending_captures, limit=limit, memory_ids=memory_ids
        )

    async def add_stream(
        self,
        contents: AsyncIterable[ContentInput | StreamInput],
        *,
        capture: bool = False,
    ) -> AsyncIterator[MemoryRecord]:
        """Add an async omni stream one durable, searchable observation at a time.

        With `capture=True` each item commits through `capture()` and owes a later `settle()`.
        """
        if not isinstance(contents, AsyncIterable):
            raise ValidationError("contents must be an async iterable of memory inputs")
        capture = strict_bool(capture, "capture")
        iterator = aiter(contents)
        index = 0
        while True:
            try:
                content = await anext(iterator)
            except StopAsyncIteration:
                return
            except MindBridgeError as error:
                if error.subject is None:
                    error.subject = f"contents[{index}]"
                raise
            try:
                if isinstance(content, StreamInput):
                    record = (
                        await self._capture_stream_input(content)
                        if capture
                        else await self._add_stream_input(content)
                    )
                elif capture:
                    record = await self.capture(content)
                else:
                    record = await self.add(content)
            except MindBridgeError as error:
                if error.subject is None:
                    error.subject = f"contents[{index}]"
                raise
            yield record
            index += 1

    async def search(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> tuple[SearchHit, ...]:
        return await asyncio.to_thread(
            self._memory.search,
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
        )

    async def search_with_trace(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> TracedSearchResult:
        return await asyncio.to_thread(
            self._memory.search_with_trace,
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
        )

    async def ask(
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
        queued_at = perf_counter()

        def answer() -> AnswerResult:
            token = ASYNC_QUEUE_TIME_MS.set((perf_counter() - queued_at) * 1_000.0)
            try:
                return self._memory.ask(
                    question,
                    limit=limit,
                    memory_type=memory_type,
                    reference_at=reference_at,
                    scope=scope,
                    link_identities=link_identities,
                    answer_policy=answer_policy,
                )
            finally:
                ASYNC_QUEUE_TIME_MS.reset(token)

        return await asyncio.to_thread(answer)

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
    ) -> AsyncGenerator[AnswerChunk, None]:
        """Answer as `ask()` does, yielding chunks as the model produces them.

        Every step of the underlying synchronous stream runs in a worker thread, so the event
        loop stays free while the provider is generating. Abandoning or cancelling the stream
        closes it, which releases the operation the answer holds open.

        This is a plain function returning an async generator, not an `async def` generator, so
        that arguments are rejected at the call exactly as `Memory.ask_stream` rejects them; an
        `async def` generator would defer them to the first `__anext__`.
        """
        # Validation and generator construction only, no I/O, so the event loop is not blocked.
        stream = self._memory.ask_stream(
            question,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            link_identities=link_identities,
            answer_policy=answer_policy,
        )
        return self._pump(stream)

    async def _pump(
        self,
        stream: Generator[AnswerChunk, None, AnswerResult],
    ) -> AsyncGenerator[AnswerChunk, None]:
        loop = asyncio.get_running_loop()
        context = copy_context()
        # One worker, not `asyncio.to_thread`: every step has to run on the same thread so the
        # span and operation contextvars the generator attaches on its first step are the ones
        # it detaches on its last. `to_thread` runs each call in a fresh context copy, which
        # would strand both. `next` is given a sentinel because a `StopIteration` crossing a
        # future becomes `RuntimeError` rather than ending the loop.
        #
        # ponytail: thread affinity makes this one thread per in-flight stream, outside the cap
        # the default executor puts on every other `AsyncMemory` call. Fine while concurrent
        # streams are counted in tens; give the class a shared pool of pinned workers if a host
        # ever runs enough of them for the thread count to bind before the provider does.
        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mindbridge-ask")
        first_advance = True

        def advance(queued_at: float, record_queue: bool) -> object:
            token = None
            if record_queue:
                token = ASYNC_QUEUE_TIME_MS.set((perf_counter() - queued_at) * 1_000.0)
            try:
                return next(stream, _STREAM_EXHAUSTED)
            finally:
                if token is not None:
                    ASYNC_QUEUE_TIME_MS.reset(token)

        try:
            while True:
                queued_at = perf_counter()
                record_queue = first_advance
                first_advance = False
                chunk = await loop.run_in_executor(
                    worker,
                    context.run,
                    advance,
                    queued_at,
                    record_queue,
                )
                if chunk is _STREAM_EXHAUSTED:
                    return
                yield cast(AnswerChunk, chunk)
        finally:
            # A running executor call cannot be cancelled, so the close always completes even
            # when the awaiting task is gone; `wait=False` keeps that off the event loop.
            closing_stream = loop.run_in_executor(worker, context.run, stream.close)
            with suppress(asyncio.CancelledError):
                await asyncio.shield(closing_stream)
            worker.shutdown(wait=False)

    async def compile(
        self,
        goal: ContentInput,
        *,
        budget: ContextBudget | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        allow_partial_sources: bool = False,
    ) -> ContextBundle:
        return await asyncio.to_thread(
            self._memory.compile,
            goal,
            budget=budget,
            reference_at=reference_at,
            scope=scope,
            allow_partial_sources=allow_partial_sources,
        )

    @property
    def capabilities(self) -> MemoryCapabilities:
        """The composition's declared capabilities; read from memory, so no thread is needed."""
        return self._memory.capabilities

    async def get(self, memory_id: str) -> MemoryRecord:
        return await asyncio.to_thread(self._memory.get, memory_id)

    async def speech(self, memory_id: str) -> tuple[SpeakerSegment, ...]:
        return await asyncio.to_thread(self._memory.speech, memory_id)

    async def faces(self, memory_id: str) -> tuple[FaceObservation, ...]:
        return await asyncio.to_thread(self._memory.faces, memory_id)

    async def register_speaker(
        self,
        speaker_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            partial(
                self._memory.register_speaker,
                speaker_id,
                name,
                relationship=relationship,
            )
        )

    async def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            partial(
                self._memory.register_identity,
                identity_id,
                name,
                relationship=relationship,
            )
        )

    async def identity(self, identity_id: str) -> IdentityProfile | None:
        return await asyncio.to_thread(self._memory.identity, identity_id)

    async def unlink_identity(self, alias_id: str) -> str | None:
        return await asyncio.to_thread(self._memory.unlink_identity, alias_id)

    async def record_consent(
        self,
        identity_id: str,
        state: ConsentState,
        *,
        note: str | None = None,
    ) -> MemoryOperationRecord | None:
        return await asyncio.to_thread(
            partial(self._memory.record_consent, identity_id, state, note=note)
        )

    async def consent(self, identity_id: str) -> ConsentState | None:
        return await asyncio.to_thread(self._memory.consent, identity_id)

    async def export(
        self,
        *,
        identity_id: str | None = None,
        memory_ids: Sequence[str] | None = None,
    ) -> ExportBundle:
        return await asyncio.to_thread(
            partial(self._memory.export, identity_id=identity_id, memory_ids=memory_ids)
        )

    async def apply_retention(self, *, dry_run: bool = False) -> RetentionReport:
        return await asyncio.to_thread(partial(self._memory.apply_retention, dry_run=dry_run))

    async def reinforce(self, memory_ids: Sequence[str]) -> int:
        return await asyncio.to_thread(self._memory.reinforce, memory_ids)

    async def consolidation_candidates(
        self,
        *,
        limit: int = 32,
        idle: bool = False,
    ) -> tuple[ConsolidationCandidate, ...]:
        return await asyncio.to_thread(
            partial(self._memory.consolidation_candidates, limit=limit, idle=idle)
        )

    async def deliberate(
        self,
        *,
        limit: int = 32,
        max_rounds: int = 4,
        idle: bool = False,
    ) -> DeliberationReport:
        return await asyncio.to_thread(
            partial(
                self._memory.deliberate,
                limit=limit,
                max_rounds=max_rounds,
                idle=idle,
            )
        )

    async def apply(self, operation: MemoryOperation) -> MemoryOperationRecord:
        return await asyncio.to_thread(self._memory.apply, operation)

    async def record_outcome(
        self,
        operation_id: int,
        outcome: MemoryOutcome,
        *,
        note: str | None = None,
    ) -> bool:
        return await asyncio.to_thread(
            partial(self._memory.record_outcome, operation_id, outcome, note=note)
        )

    async def consolidate(
        self,
        *,
        evidence_ids: Sequence[str] | None = None,
        query: ContentInput | None = None,
        limit: int = 32,
        trigger: MemoryTrigger = MemoryTrigger.MANUAL,
    ) -> ConsolidationReport:
        return await asyncio.to_thread(
            partial(
                self._memory.consolidate,
                evidence_ids=evidence_ids,
                query=query,
                limit=limit,
                trigger=trigger,
            )
        )

    async def forget(self, memory_ids: Sequence[str]) -> MemoryOperationRecord | None:
        return await asyncio.to_thread(self._memory.forget, memory_ids)

    async def rollback(self, operation_id: int) -> bool:
        return await asyncio.to_thread(self._memory.rollback, operation_id)

    async def operations(self, *, limit: int = 100) -> tuple[MemoryOperationRecord, ...]:
        return await asyncio.to_thread(partial(self._memory.operations, limit=limit))

    async def list(self, *, limit: int = 100, cursor: str | None = None) -> Page:
        return await asyncio.to_thread(self._memory.list, limit=limit, cursor=cursor)

    async def delete(self, memory_id: str) -> bool:
        return await asyncio.to_thread(self._memory.delete, memory_id)

    async def reindex(self) -> int:
        return await asyncio.to_thread(self._memory.reindex)

    async def optimize(self) -> None:
        await asyncio.to_thread(self._memory.optimize)

    async def close(self) -> None:
        await asyncio.to_thread(self._memory.close)
