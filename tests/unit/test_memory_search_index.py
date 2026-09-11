"""`Memory` driven against the real Zvec index rather than a stand-in.

The API tests install a fake index so they can steer retrieval, which means they cannot see any
behaviour the real index actually implements. One such rule cost a release: a memory's full-text
document is written on its ``object_part == 0`` row alone, so a memory stored without that row is
reachable by the dense route only. A test asserting the memory was "still searchable" passed
against the fake while the real index could not match it at all.

These tests therefore construct `Memory` with no index substitution. Only the models are faked;
the store, the outbox, and the index are the real ones. `zvec` is a base dependency, so this runs
wherever the suite runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar, cast

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mindbridge import Memory, RetrievalScope
from mindbridge._telemetry import (
    IDENTITY_NAMES_BOUND,
    IDENTITY_NAMES_REFUSED,
    VISION_BATCHES_FAILED,
    VISION_BATCHES_RETRIED,
)
from mindbridge.exceptions import ModelError
from mindbridge.infrastructure.local.store import LocalStore
from mindbridge.memory import (
    _LEXICAL_FULL_COVERAGE,
    _LEXICAL_FULL_COVERAGE_RELEVANCE,
    _MAX_DESCRIBE_CONTEXT_CHARACTERS,
    _MAX_TEXT_CHARACTERS,
    _lexical_query_terms,
    _lexical_relevance,
    _speaker_labels,
    _speech_evidence,
    _speech_retrieval_text,
)
from mindbridge.models.base import (
    EmbedTask,
    ModelInput,
    SpeakerEmbedding,
    SpeechAnalysis,
    SpeechTurn,
)
from mindbridge.types import (
    AnswerPolicy,
    AnswerResult,
    AssetRef,
    Blob,
    MemoryIntent,
    Modality,
    SearchHit,
    SpeakerSegment,
)

_ALL_INPUT_MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})


class _Embedder:
    """A deterministic embedder that can refuse a named asset the way a provider limit does."""

    embedding_model = "fake-real-index"
    embedding_space = "fake-real-index:2:test"
    embedding_dimension = 2
    # Optional, duck-typed by `Memory` (`getattr(..., "_legacy_embedding_spaces", frozenset())`);
    # declared here only so a test assigning it type-checks.
    _legacy_embedding_spaces: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self.embedding_capabilities = _ALL_INPUT_MODALITIES
        self.oversized_assets: frozenset[str] = frozenset()
        self.document_inputs: list[ModelInput] = []
        # Per call, not only accumulated: an assertion over every input ever embedded passes on
        # the strength of the first write, while what a later rebuild of the same document keys
        # is exactly what a reindex can break.
        self.document_batches: list[tuple[ModelInput, ...]] = []

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        batch = tuple(inputs)
        if any(asset.id in self.oversized_assets for value in batch for asset in value.assets):
            raise ModelError(
                "encoded inline model media item exceeds the limit",
                reason="payload_too_large",
            )
        if task is EmbedTask.DOCUMENT:
            self.document_inputs.extend(batch)
            self.document_batches.append(batch)
        # One vector for everything: dense relevance is deliberately uninformative so that a
        # lexical assertion cannot pass on the strength of the dense route.
        return tuple((1.0, 0.0) for _ in batch)

    def close(self) -> None:
        return None


class _Describer:
    """A vision backend that reports one fixed sentence, so the index document is predictable."""

    vision_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    vision_model = "fake-describer"
    vision_space = "fake-describer:caption-v1"

    def __init__(self, description: str = "a red bicycle leaning on the fence") -> None:
        self.description = description
        self.calls = 0

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        batch = tuple(inputs)
        self.calls += 1
        return tuple(self.description for _ in batch)

    def close(self) -> None:
        return None


class _CountingDescriber:
    """A describer that never repeats itself, so a reused caption is distinguishable from a call.

    Modelled on the measured endpoint, which returns a different completion for the same image on
    every request: a fixed-caption fake would pass every cache assertion under a broken cache.
    """

    vision_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    vision_model = "fake-describer"

    def __init__(self, prefix: str = "caption") -> None:
        self.vision_space = "fake-describer:caption-v1"
        self.calls = 0
        self.described: list[str] = []
        self.captions: list[str] = []
        self._prefix = prefix

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        batch = tuple(inputs)
        self.calls += 1
        self.described.extend(str(value.assets[0].sha256) for value in batch)
        fresh = tuple(
            f"{self._prefix} bicycle number {len(self.captions) + index}"
            for index in range(len(batch))
        )
        self.captions.extend(fresh)
        return fresh

    def close(self) -> None:
        return None


def _caption(content: str) -> str:
    """Read back the one derived caption a memory's stored document carries."""
    marker = "[visual description:"
    assert marker in content, content
    return content.split("]\n", 1)[1].strip()


class _DirectionalEmbedder:
    """Places a memory at a chosen cosine to the query so a floor can be read back as a cosine.

    The query is always the unit vector on the first axis. A document whose text contains one of
    the named markers is placed at that marker's cosine, which makes `minimum_relevance` directly
    checkable against the similarity a caller would reason about.
    """

    embedding_model = "fake-directional"
    embedding_space = "fake-directional:2:test"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT})

    def __init__(self, cosines: Mapping[str, float]) -> None:
        self._cosines = dict(cosines)
        self.tasks: list[EmbedTask] = []

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        self.tasks.append(task)
        vectors = []
        for value in inputs:
            cosine = 1.0
            for marker, target in self._cosines.items():
                if marker in value.text:
                    cosine = target
                    break
            vectors.append((cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))))
        return tuple(vectors)

    def close(self) -> None:
        return None


def _lexical_matches(memory: Memory, query: str) -> set[str]:
    """Memory ids the full-text route matched, ignoring stale-index candidates with no parent."""
    traced = memory.search_with_trace(query)
    return {
        candidate.memory_id
        for candidate in traced.trace.candidates
        if candidate.lexical_match and candidate.memory_id is not None
    }


def test_the_real_index_matches_a_memory_on_its_own_words(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_Embedder()) as memory:
        kitchen = memory.add("the kitchen at dusk")
        garden = memory.add("the garden at noon")

        # Establishes that these tests exercise real BM25: the dense route returns one vector for
        # every memory, so only the full-text route can tell these two apart.
        matched = _lexical_matches(memory, "kitchen")
        assert kitchen.id in matched
        assert garden.id not in matched


def test_a_memory_whose_aggregate_key_was_elided_keeps_its_full_text_document(
    tmp_path: Path,
) -> None:
    embedder = _Embedder()
    with Memory(tmp_path, embedder=embedder) as memory:
        probe = memory.add(("the kitchen at dusk", Blob(b"oversized-clip", "video/mp4")))
        oversized = probe.assets[0].id
        memory.delete(probe.id)
        embedder.oversized_assets = frozenset({oversized})

        elided = memory.add(("the kitchen at dusk", Blob(b"oversized-clip", "video/mp4")))
        # A second document, so the index scores against a corpus instead of degenerately
        # matching its only entry.
        intact = memory.add("the kitchen at dawn")

        # The retrieval key holding the media is also the aggregate key, which is part 0 -- the
        # only row the index writes a full-text document on. Dropping it without renumbering the
        # survivors leaves this memory with an empty document, and the query its own text answers
        # cannot reach it.
        matched = _lexical_matches(memory, "kitchen")
        assert intact.id in matched
        assert elided.id in matched

        # The memory kept its media and the write did not fail; only the key was degraded.
        assert memory.get(elided.id).assets[0].id == oversized


def test_a_memory_with_no_carriable_key_never_reaches_the_index(tmp_path: Path) -> None:
    embedder = _Embedder()
    with Memory(tmp_path, embedder=embedder) as memory:
        # Two assets and no text, so the memory has several retrieval keys and every one of them
        # carries refused media. A single-asset memory would have one key and take the earlier
        # path that re-raises before any degradation is attempted, leaving the guard that decides
        # a memory is unreachable untested.
        clips = (Blob(b"first-clip", "video/mp4"), Blob(b"second-clip", "video/mp4"))
        probe = memory.add(clips)
        assert len(probe.assets) == 2
        embedder.oversized_assets = frozenset(asset.id for asset in probe.assets)
        memory.delete(probe.id)

        # Degrading every key would store a memory no query could reach, so the write fails
        # rather than leaving an unreachable row behind.
        with pytest.raises(ModelError) as failure:
            memory.add(clips)
        assert failure.value.reason == "payload_too_large"

        assert memory.list().items == ()


def test_an_image_only_memory_is_lexically_reachable_through_its_description(
    tmp_path: Path,
) -> None:
    embedder = _Embedder()
    describer = _Describer()
    with Memory(tmp_path, embedder=embedder, vision_describer=describer) as memory:
        # The embedder declares image natively, which is the composition the product recommends.
        # The derived description is therefore not needed to route the media, and used to be
        # skipped entirely -- leaving the empty string as this memory's whole BM25 document.
        picture = memory.add(Blob(b"bicycle-frame", "image/png"))
        # A second document so the full-text index scores against a corpus.
        other = memory.add("the garden at noon")

        assert describer.calls == 1
        matched = _lexical_matches(memory, "bicycle")
        assert picture.id in matched
        assert other.id not in matched


def test_a_described_image_still_reaches_the_embedder_as_an_image(tmp_path: Path) -> None:
    embedder = _Embedder()
    with Memory(tmp_path, embedder=embedder, vision_describer=_Describer()) as memory:
        record = memory.add(Blob(b"bicycle-frame", "image/png"))

        # The description is added to the index document; it does not replace the native route.
        asset_id = record.assets[0].id
        assert any(
            asset.id == asset_id for value in embedder.document_inputs for asset in value.assets
        )


def test_an_image_only_memory_derives_nothing_without_a_describer(tmp_path: Path) -> None:
    # Deriving text is a paid model call, so it has to follow from configuration. With no
    # describer the write must stay exactly as cheap as it is today.
    with Memory(tmp_path, embedder=_Embedder()) as memory:
        record = memory.add(Blob(b"bicycle-frame", "image/png"))

        assert memory.get(record.id).content == ""


def test_the_relevance_floor_rejects_the_similarity_it_reports(tmp_path: Path) -> None:
    cosines = {"kepler": 0.15, "orbital": 0.9}
    for floor, expected in ((0.55, False), (0.10, True)):
        directory = tmp_path / f"floor-{floor}"
        with Memory(
            directory,
            embedder=_DirectionalEmbedder(cosines),
            minimum_relevance=floor,
        ) as memory:
            middling = memory.add("kepler transit candidate list")
            memory.add("orbital resonance survey notes")

            # The query must contain no marker of its own, so that it stays the reference vector.
            hits = memory.search("resonance survey window")
            found = {hit.id for hit in hits}
            # A floor of 0.55 used to admit cosine 0.15, because the gate read (1 + cos) / 2 while
            # the caller read cosine back on `SearchHit.score`.
            assert (middling.id in found) is expected
            # With no rank prior enabled the score *is* the gated relevance, so the floor reads
            # back exactly. Enable `decay_half_life_days` or ask a dated question and the priors
            # move `score` below the floor on purpose — see
            # `test_decay_demotes_an_old_memory_without_evicting_it`.
            assert all(hit.score >= floor for hit in hits)


def test_a_weak_lexical_match_cannot_smuggle_an_anticorrelated_memory_past_the_floor(
    tmp_path: Path,
) -> None:
    cosines = {"anticorrelated": -1.0, "ordinary": 0.2}
    query = "zibaldone quokka nephoscope"
    for floor in (0.55, 0.10):
        directory = tmp_path / f"floor-{floor}"
        with Memory(
            directory,
            embedder=_DirectionalEmbedder(cosines),
            minimum_relevance=floor,
        ) as memory:
            opposite = memory.add("anticorrelated zibaldone drift")
            memory.add("ordinary weather notes")

            traced = memory.search_with_trace(query)
            by_id = {hit.id: hit for hit in traced.hits}
            # One shared rare term used to earn a flat 0.6 gate confidence regardless of how weak
            # the match was, so a document pointing the other way cleared the default floor. It is
            # now scored on its real strength, which for a single partial term is ~0.075 -- below
            # even the permissive floor, so it is rejected at both. That is a stronger guarantee
            # than this test originally asserted: it used to be admitted at 0.10 and merely
            # scored honestly. A full-coverage lexical match still survives a floor well above
            # the old constant, which `test_the_lexical_route_survives_...` pins.
            assert opposite.id not in by_id
            assert all(hit.score >= floor for hit in traced.hits)
            candidate = next(
                trace for trace in traced.trace.candidates if trace.memory_id == opposite.id
            )
            assert candidate.lexical_match is True, "it did match lexically"
            assert candidate.gate_relevance is not None and candidate.gate_relevance < 0.10


def test_the_lexical_route_survives_a_floor_above_the_old_flat_constant(tmp_path: Path) -> None:
    # Every full-text match used to be gated at exactly 0.6, so any floor above that deleted the
    # lexical route wholesale. A memory covering every distinctive query term is the case the
    # dense+lexical union exists to catch and must still be reachable.
    with Memory(
        tmp_path,
        embedder=_DirectionalEmbedder({"anticorrelated": -1.0, "ordinary": 0.2}),
        minimum_relevance=0.7,
    ) as memory:
        covering = memory.add("anticorrelated zibaldone quokka")
        memory.add("ordinary weather notes")

        hits = memory.search("zibaldone quokka")

        assert [hit.id for hit in hits] == [covering.id]
        assert hits[0].score >= 0.7


def test_partial_lexical_candidate_outside_the_ann_window_gets_its_persisted_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = _DirectionalEmbedder({"distractor": 0.81, "rescue-target": 0.8})
    requested: list[tuple[str, ...]] = []
    original_read = LocalStore.iter_memory_embedding_vectors

    def observed_read(
        store: LocalStore,
        memory_ids: Sequence[str],
        *,
        space_id: str,
        task: str,
    ) -> Iterator[tuple[str, tuple[float, ...]]]:
        requested.append(tuple(memory_ids))
        return original_read(store, memory_ids, space_id=space_id, task=task)

    monkeypatch.setattr(LocalStore, "iter_memory_embedding_vectors", observed_read)
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0.1) as memory:
        memory.add_many([f"distractor ordinary weather note {index}" for index in range(100)])
        known_before_target = datetime.now(timezone.utc)
        target = memory.add("rescue-target zibaldone drift")
        target_index_id = memory._store.read_memory_index_documents((target.id,))[
            0
        ].embedding.embedding_id

        dense_window = memory._index.search(
            (1.0, 0.0),
            limit=100,
            space_id=embedder.embedding_space,
            task=EmbedTask.DOCUMENT.value,
        )
        assert target_index_id not in {hit.id for hit in dense_window}

        before_search_tasks = tuple(embedder.tasks)
        scoped = memory.search(
            "zibaldone quokka nephoscope",
            scope=RetrievalScope(known_at=known_before_target),
        )
        assert target.id not in {hit.id for hit in scoped}
        assert requested == [], "historically invisible parents must not be score-completed"

        traced = memory.search_with_trace("zibaldone quokka nephoscope")

        assert traced.hits[0].id == target.id
        candidate = next(item for item in traced.trace.candidates if item.memory_id == target.id)
        assert candidate.lexical_match is True
        assert candidate.lexical_relevance == 0.0, "the query has only partial term coverage"
        assert candidate.dense_relevance == pytest.approx(0.8)
        assert candidate.dense_confidence == pytest.approx(0.9)
        assert requested == [(target.id,)]
        assert embedder.tasks[len(before_search_tasks) :] == [
            EmbedTask.QUERY,
            EmbedTask.QUERY,
        ]

        memory.forget((target.id,))
        requested.clear()
        assert target.id not in {hit.id for hit in memory.search("zibaldone quokka nephoscope")}
        assert requested == [], "forgotten parents must not be score-completed"


def test_dense_candidates_keep_the_index_score_without_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = _DirectionalEmbedder({"target": 0.6, "other": 0.2})

    def unexpected_read(
        *_args: object, **_kwargs: object
    ) -> Iterator[tuple[str, tuple[float, ...]]]:
        raise AssertionError("a parent already scored by the dense route was completed again")

    monkeypatch.setattr(LocalStore, "iter_memory_embedding_vectors", unexpected_read)
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0.1) as memory:
        target = memory.add("target zibaldone drift")
        memory.add("other weather note")

        traced = memory.search_with_trace("zibaldone quokka nephoscope")

        candidate = next(item for item in traced.trace.candidates if item.memory_id == target.id)
        assert candidate.dense_relevance == pytest.approx(0.6)
        assert candidate.dense_confidence == pytest.approx(0.8)


def test_a_text_only_embedder_reaches_an_image_through_the_describer(tmp_path: Path) -> None:
    # The composition `VisionDescriptionBackend` exists for. `add` never asked for a description
    # on any code path, so this write used to fail outright with `unsupported_modality`.
    embedder = _Embedder()
    embedder.embedding_capabilities = frozenset({Modality.TEXT})
    with Memory(tmp_path, embedder=embedder, vision_describer=_Describer()) as memory:
        picture = memory.add(Blob(b"bicycle-frame", "image/png"))
        other = memory.add("the garden at noon")

        matched = _lexical_matches(memory, "bicycle")
        assert picture.id in matched
        assert other.id not in matched
        # No key carries the image, because this embedder cannot take one.
        assert all(not value.assets for value in embedder.document_inputs)


def test_a_second_ingest_of_the_same_image_reuses_the_stored_caption(tmp_path: Path) -> None:
    """Two ingests of one picture must build the same document and pay once.

    The measured describe endpoint returns a different caption for the same image on every call
    even at temperature 0 with a fixed seed, so without a store-side cache a re-ingest -- or a
    re-derive after a crash -- silently rewrites what a memory's full-text document says, and pays
    for every image again. The caption is keyed by the asset's own SHA-256, so a *different*
    memory over the same bytes reuses it too.
    """
    describer = _CountingDescriber()
    picture = Blob(b"bicycle-frame", "image/png", "bicycle.png")
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as memory:
        first = memory.add(("morning note", picture))
        assert describer.calls == 1

        repeat = memory.add(("afternoon note", picture))
        second_instance = memory.add(picture)

        assert describer.calls == 1, "the same bytes were described again"
        first_caption = _caption(memory.get(first.id).content)
        assert first_caption == describer.captions[0]
        assert _caption(memory.get(repeat.id).content) == first_caption
        assert _caption(memory.get(second_instance.id).content) == first_caption

    # A fresh `Memory` over the same data_dir is the crash-and-re-derive case.
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as reopened:
        after_restart = reopened.add(("evening note", picture))
        assert describer.calls == 1
        assert _caption(reopened.get(after_restart.id).content) == first_caption


def test_a_describer_in_another_vision_space_does_not_reuse_captions(tmp_path: Path) -> None:
    """The cache is keyed by recipe, not by model name, so a new prompt re-describes.

    A caption lands inside the searchable document, where a stale one written under a superseded
    prompt is invisible. Keying on the model alone would serve it forever.
    """
    first_describer = _CountingDescriber(prefix="early")
    second_describer = _CountingDescriber(prefix="revised")
    second_describer.vision_space = "fake-describer:caption-v2"
    picture = Blob(b"bicycle-frame", "image/png", "bicycle.png")

    with Memory(tmp_path, embedder=_Embedder(), vision_describer=first_describer) as memory:
        original = memory.add(("morning note", picture))
        assert _caption(memory.get(original.id).content).startswith("early")
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=second_describer) as memory:
        revised = memory.add(("afternoon note", picture))
        assert second_describer.calls == 1, "a new recipe must not read the old space"
        assert _caption(memory.get(revised.id).content).startswith("revised")
        # The first space keeps its own caption rather than being evicted by the second.
        assert _caption(memory.get(original.id).content).startswith("early")
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=first_describer) as memory:
        back = memory.add(("evening note", picture))
        assert first_describer.calls == 1
        assert _caption(memory.get(back.id).content).startswith("early")


def test_two_memories_over_one_picture_cost_one_description(tmp_path: Path) -> None:
    """Derive-early: one describe per asset *content* within a write, not per memory."""
    describer = _CountingDescriber()
    picture = Blob(b"bicycle-frame", "image/png", "bicycle.png")
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as memory:
        records = memory.add_many([("morning note", picture), ("afternoon note", picture)])

        assert describer.calls == 1
        assert len(describer.described) == 1, "one visual reached the model, not two"
        captions = {_caption(memory.get(record.id).content) for record in records}
        assert captions == {describer.captions[0]}


def test_reindexing_does_not_re_describe_stored_visuals(tmp_path: Path) -> None:
    """Zvec is rebuildable from SQLite alone, so a rebuild must not spend description tokens."""
    describer = _CountingDescriber()
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as memory:
        record = memory.add(("morning note", Blob(b"bicycle-frame", "image/png", "bicycle.png")))
        caption = _caption(memory.get(record.id).content)
        assert describer.calls == 1

        assert memory.reindex() == 1

        assert describer.calls == 1
        assert _caption(memory.get(record.id).content) == caption


def test_decay_demotes_an_old_memory_without_evicting_it(tmp_path: Path) -> None:
    """`minimum_relevance` gates evidence quality, not the recency priors applied after it.

    The priors are bounded below by `_RANK_FLOOR = 0.3`, so a *perfectly* relevant memory decays
    to `0.30`, and to `0.09` once a temporal window also misses it — under the `0.10` default. A
    floor that included the priors therefore turned "prefer recent" into "hide old", which is the
    wrong failure for a companion that is asked about last year. Evidence relevance is what the
    caller set a floor on; how much a memory is preferred is a separate question.
    """
    long_ago = datetime.now(timezone.utc) - timedelta(days=4000)
    with Memory(
        tmp_path,
        embedder=_DirectionalEmbedder({"heirloom": 1.0}),
        minimum_relevance=0.35,
        decay_half_life_days=30.0,
    ) as memory:
        aged = memory.add("heirloom clock wound every sunday", occurred_at=long_ago)

        hits = memory.search("clock wound every sunday")

        assert [hit.id for hit in hits] == [aged.id], "decay must demote, never evict"
        # The reported score carries the decay, so it sits below the evidence floor by design.
        assert hits[0].score < 0.35
        candidate = next(
            trace
            for trace in memory.search_with_trace("clock wound every sunday").trace.candidates
            if trace.memory_id == aged.id
        )
        assert candidate.gate_relevance is not None
        assert candidate.gate_relevance >= 0.35, "the gated quantity is the evidence relevance"
        assert candidate.retention_factor is not None
        assert candidate.retention_factor < 1.0, "decay was actually applied"


def test_a_description_that_does_not_fit_is_omitted_not_fatal(tmp_path: Path) -> None:
    """A derived description must never cost the caller the write it was decorating.

    The description is convenience, not content: the asset is stored and embedded either way. So a
    memory whose own text nearly fills `_MAX_TEXT_CHARACTERS` still stores, minus the description,
    rather than failing and leaving the caller no option but to drop the describer entirely.
    """
    describer = _Describer()
    # Inside the limit on its own, but with no room left for a description plus its separator.
    long_text = ("kitchen " * 8_192)[: _MAX_TEXT_CHARACTERS - 32]
    assert len(long_text) < _MAX_TEXT_CHARACTERS
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as memory:
        record = memory.add((long_text, Blob(b"kitchen-photo", "image/png", "kitchen.png")))

        assert record.content == long_text.strip()
        assert "[visual description:" not in record.content
        assert record.assets, "the asset itself is still stored"
        # The short-text case still receives it, so the omission is length-driven and not a
        # silently broken describer.
        short = memory.add(("kitchen note", Blob(b"other-photo", "image/png", "other.png")))
        assert "[visual description:" in memory.get(short.id).content


def test_indexed_speech_is_prose_and_not_a_json_blob(tmp_path: Path) -> None:
    """The index document must carry the words, not the schema around them.

    `index_speech` is on by default, so every speech-bearing memory's full-text document was a
    JSON object: `start_ms`, `end_ms`, `speaker_id` and `identity_score` all became BM25 tokens
    while the spoken words sat among them. Index content is the only lever that moves the R@20
    ceiling, so what lands in the document is the whole point of the flag.

    Stored content keeps the JSON: the answering model reads it as structured evidence, and only
    the derived index and embedding projections are prose.
    """
    projected = _speech_retrieval_text(
        json.dumps(
            {
                "asset_id": "asset-1",
                "segments": [
                    {
                        "start_ms": 0,
                        "end_ms": 900,
                        "text": "the sourdough needs another hour",
                        "speaker_id": "identity_9f2c",
                        "speaker_name": "Mum",
                        "identity_score": None,
                    },
                    {
                        "start_ms": 900,
                        "end_ms": 1500,
                        "text": "I will set a timer",
                        "speaker_id": "identity_44ab",
                        "speaker_name": None,
                        "identity_score": 0.91,
                    },
                ],
            }
        ),
        "asset-1",
    )

    assert projected is not None
    assert "the sourdough needs another hour" in projected
    assert "Mum" in projected
    for noise in ("start_ms", "end_ms", "identity_score", "asset_id", "segments"):
        assert noise not in projected, f"{noise} must not be a lexical token"
    # An unstable per-run identity is still projected to a stable alias, which is what kept the
    # document from changing every time the recognizer re-minted a person.
    assert "identity_9f2c" not in projected
    assert "identity_44ab" not in projected
    assert "speaker_" in projected, "an unnamed speaker keeps a stable alias"


class _NoVideoEmbedder(_Embedder):
    """An embedder that takes every modality except video, which is a real provider shape."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE, Modality.AUDIO})


class _VideoTranscriber:
    """Declares VIDEO, as the cloud transcriber now does once it demuxes locally."""

    transcription_model = "fake-video-transcriber"
    transcription_space = "fake-video-transcriber:v1"
    transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
        self.calls += 1
        return tuple("the kettle whistled twice" for _ in assets)

    def close(self) -> None:
        return None


_ANSWER = "用户最喜欢的饮料是可乐"
_DECOY = "今天的天气是晴朗的"
# Every distinctive character of the question, none of its adjacent pairs but 用户.
_SHUFFLED = "最新饮品, 料酒和喜茶, 用户很欢迎"


class _ChineseDense:
    """An embedder that scores the decoy above the answer, so only coverage can reorder them."""

    embedding_model = "fake-chinese-dense"
    embedding_space = "fake-chinese-dense:2:test"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT})

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        cosines = {_ANSWER: 0.50, _DECOY: 0.74, _SHUFFLED: 0.40}
        return tuple(
            (cosine, math.sqrt(1.0 - cosine * cosine))
            for cosine in (
                next(
                    (value for text, value in cosines.items() if text in single.text),
                    1.0,
                )
                for single in inputs
            )
        )

    def close(self) -> None:
        return None


def test_a_transcribed_video_falls_back_instead_of_failing_the_write(tmp_path: Path) -> None:
    """A route exists, so the write must take it rather than fail before inference.

    The transcript was already being derived -- the transcriber ran -- but the fallback set only
    admitted VIDEO when a *visual description* existed, never when a transcript did. So the video
    asset still reached an embedder that had just declared it could not take one, and the whole
    write failed with `unsupported_modality` while its speech sat there unused.

    The visual content is not embedded in this composition, which is the honest cost of the
    fallback; the asset is still stored and still on the record.
    """
    embedder = _NoVideoEmbedder()
    transcriber = _VideoTranscriber()
    with Memory(tmp_path, embedder=embedder, transcriber=transcriber) as memory:
        record = memory.add(Blob(b"a video clip", "video/mp4", "clip.mp4"))

        assert transcriber.calls == 1
        assert "the kettle whistled twice" in record.content
        assert record.assets, "the video asset is still stored"
        # The speech reached the embedder as text, and the video asset did not reach it at all.
        document = embedder.document_inputs[-1]
        assert "the kettle whistled twice" in document.text
        assert Modality.VIDEO not in document.modalities
        # And it is lexically reachable by the words that were spoken.
        assert record.id in _lexical_matches(memory, "kettle whistled")


def test_a_chinese_question_can_reach_full_lexical_coverage(tmp_path: Path) -> None:
    """A Chinese query must be able to fuse the two routes, which needs full term coverage.

    `\\w+` matched an entire Chinese sentence as one token, so every multi-character Chinese
    query carried a term that is by construction the rarest in the corpus and can never match.
    It took the largest IDF weight into the coverage ratio, which put `_LEXICAL_FULL_COVERAGE`
    permanently out of reach -- and that lift is the only term that performs cross-route fusion.
    Chinese retrieval was therefore stuck on the demoted rank proxy and lost to any mediocre
    dense neighbour, in the product's largest market.

    The dense route here is deliberately wrong: the decoy shares only function characters yet
    scores the higher cosine. Only the coverage lift can rescue the memory that answers.

    The third memory is why single characters are not enough on their own. It reuses every
    distinctive character of the question in unrelated words, so a character-only tokenizer
    scores it as complete a match as the answer. Adjacent-character bigrams are what tell them
    apart without a segmenter.
    """
    with Memory(tmp_path, embedder=_ChineseDense()) as memory:
        answer = memory.add(_ANSWER)
        decoy = memory.add(_DECOY)
        shuffled = memory.add(_SHUFFLED)

        traced = memory.search_with_trace("用户最喜欢的饮料是什么")
        scored = {candidate.memory_id: candidate for candidate in traced.trace.candidates}

        # The decoy is the stronger dense candidate and shares only 的 and 是 with the question.
        decoy_dense = scored[decoy.id].dense_relevance
        answer_dense = scored[answer.id].dense_relevance
        assert decoy_dense is not None
        assert answer_dense is not None
        assert decoy_dense > answer_dense

        # The full-coverage floor, not the rank proxy: the answer covers every distinctive term
        # of the question, so its lexical contribution is the near-certain one.
        assert scored[answer.id].lexical_relevance == pytest.approx(
            _LEXICAL_FULL_COVERAGE_RELEVANCE
        )
        # Same characters, different words: a match, but not a complete one. Read the coverage
        # ratio itself, because the score the rank proxy carries it into is also small.
        assert scored[shuffled.id].lexical_match
        coverage = _lexical_relevance(
            "用户最喜欢的饮料是什么",
            memory._store.read_memories([answer.id, shuffled.id]),
        )
        assert coverage[answer.id] == pytest.approx(1.0)
        assert coverage[shuffled.id] < _LEXICAL_FULL_COVERAGE
        assert [hit.content for hit in traced.hits] == [_ANSWER, _DECOY, _SHUFFLED]


def test_query_terms_mirror_what_the_index_can_actually_match() -> None:
    """Rerank terms have to be the terms the full-text index produces, or coverage cannot fuse.

    Three separate ways they used not to be. Thai reached `\\w+`, whose word class excludes the
    combining marks Thai writes its vowels and tones with, so a query shredded at every mark into
    fragments whose boundaries no index term shares. The unsegmented-script list that would have
    routed it to bigrams did not include Thai, or Lao, Khmer or Myanmar. And every run emitted its
    single characters beside its bigrams, which the index cannot produce at `ngram_min` 2 and
    which made the term count -- the denominator `_LEXICAL_FULL_COVERAGE` is measured against --
    a property of the language rather than of the question.
    """
    thai = _lexical_query_terms("ร้านเบเกอรี่")
    assert thai
    assert all(len(term) == 2 for term in thai)
    # The index breaks its ngrams at the mark, so a bigram may not span one: measured against
    # Zvec, "ยว" matches a document containing "เบเกอรี่ยว" and the pair spanning the mark before
    # it does not.
    assert "รย" not in thai

    for text in ("ອາຫານລາວ", "អាហារខ្មែរ", "ထမင်းဟင်း"):
        assert _lexical_query_terms(text), text

    # No single-character terms, in any script, and a Chinese question no longer carries several
    # times the terms of the English one it translates.
    chinese = _lexical_query_terms("爱丽丝面包店")
    assert all(len(term) == 2 for term in chinese)
    assert len(chinese) <= 2 * len(_lexical_query_terms("Alice bakery Tuesday"))
    # A lone character yields nothing rather than a term the index can never answer, which drops
    # the lexical route instead of running it to guaranteed emptiness.
    assert _lexical_query_terms("猫") == frozenset()


def test_temporal_factor_boosts_overlap_and_never_penalises() -> None:
    """A memory outside the asked range keeps its relevance; only an overlapping one is lifted.

    The penalty this replaces pushed a perfectly relevant memory that happened at the wrong time
    under merely well-timed ones and, inside the gate, could delete it; replay showed the penalty
    recovered no gold on 810 paired questions while the boost alone won or tied on all of them.
    """
    from mindbridge.memory import _RANK_CEILING, _temporal_factor

    start = datetime(2024, 3, 1, tzinfo=timezone.utc)
    until = datetime(2024, 4, 1, tzinfo=timezone.utc)
    inside = datetime(2024, 3, 15, tzinfo=timezone.utc)
    far_before = datetime(2019, 1, 1, tzinfo=timezone.utc)

    assert _temporal_factor(inside, None, (start, until)) == _RANK_CEILING
    assert _temporal_factor(far_before, None, (start, until)) == 1.0
    assert _temporal_factor(None, None, (start, until)) == 1.0


class _HashedDense:
    """Places each text at its own angle in the first quadrant, so ranks are not all ties."""

    embedding_model = "fake-hashed"
    embedding_space = "fake-hashed:2:test"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT})

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        del task
        vectors = []
        for value in inputs:
            digest = hashlib.blake2b(value.text.encode(), digest_size=2).digest()
            angle = int.from_bytes(digest, "big") / 65536.0 * (math.pi / 3.0)
            vectors.append((math.cos(angle), math.sin(angle)))
        return tuple(vectors)

    def close(self) -> None:
        return None


def test_a_narrow_limit_returns_the_prefix_of_a_wide_one_with_the_same_scores(
    tmp_path: Path,
) -> None:
    """The requested `limit` selects results; it must not change what they are worth.

    Both mechanisms that used to break this are in play here. Every index route was truncated
    to `max(_RERANK_CANDIDATES, limit * 3)`, so a memory's dense relevance -- the maximum over
    the routes that reached it -- depended on the depth; and `_lexical_relevance` counted its
    document frequencies over the candidate pool that depth produced, so the rerank bonus moved
    with `limit` even for a memory both runs found. With 200 records, limit 5 read 100 documents
    per route and limit 100 read 300.
    """
    with Memory(tmp_path, embedder=_HashedDense()) as memory:
        memory.add_many(
            (
                *(f"lantern harbour drift {index:03d}" for index in range(197)),
                "lantern harbour ledger alpha",
                "lantern harbour ledger beta",
                "harbour ledger only",
            )
        )

        narrow = memory.search("lantern harbour ledger", limit=5)
        wide = memory.search("lantern harbour ledger", limit=100)

    assert len(narrow) == 5
    assert len(wide) == 100
    assert [hit.id for hit in narrow] == [hit.id for hit in wide[:5]]
    assert [hit.score for hit in narrow] == [hit.score for hit in wide[:5]]


_TRACE_ORDER_PROBE = '''
import hashlib
import json
import math
import sys
from pathlib import Path

from mindbridge import Memory
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.types import Modality


class _Sha256Embedder:
    """Deterministic across processes: the vector is a hash of the text, not of an object id."""

    embedding_model = "fake-sha256"
    embedding_space = "fake-sha256:8:test"
    embedding_dimension = 8
    embedding_capabilities = frozenset({Modality.TEXT})

    def embed(self, inputs, task=EmbedTask.DOCUMENT):
        del task
        vectors = []
        for value in inputs:
            digest = hashlib.sha256(value.text.encode("utf-8")).digest()
            raw = [byte / 255.0 - 0.5 for byte in digest[:8]]
            norm = math.sqrt(sum(component * component for component in raw)) or 1.0
            vectors.append(tuple(component / norm for component in raw))
        return tuple(vectors)

    def close(self):
        return None


directory = Path(sys.argv[1])
query = "which note mentions the harbour wall"
fresh = not directory.exists()
with Memory(directory, embedder=_Sha256Embedder()) as memory:
    if fresh:
        memory.add_many(
            [
                f"note {index} about the harbour wall, ledger {index} and roadmap {index}"
                for index in range(150)
            ]
        )
    traced = memory.search_with_trace(query, limit=20)
print(
    json.dumps(
        {
            "candidates": [candidate.memory_id for candidate in traced.trace.candidates],
            "hits": [hit.id for hit in traced.hits],
        }
    )
)
'''


def test_the_candidate_trace_order_does_not_depend_on_the_process_hash_seed(
    tmp_path: Path,
) -> None:
    """`search_with_trace`'s candidate order must be reproducible across processes.

    A trace is a debugging surface and an arm-comparison input, so two runs of one query on one
    library must print it in one order. Hydration used to be ordered through `lexical_matches`, a
    `set` of memory ids, whose iteration order follows the interpreter's string hash seed.

    This test is the reason that is stated narrowly. It was written to reproduce a candidate-dump
    difference observed between two replays, and it **passes on the pre-fix code**: `read_memories`
    does not order by its argument, so the ranked trace never followed the set, and the observed
    difference turned out to be two genuinely different ranking rules rather than a seed effect.
    What remains is the general guarantee, which is what is worth pinning: the corpus deliberately
    exceeds the route depth so the union of the two routes is larger than either, which is the
    only situation in which any set-derived ordering could reach the trace at all.
    """
    script = tmp_path / "probe.py"
    script.write_text(_TRACE_ORDER_PROBE, encoding="utf-8")
    library = tmp_path / "library"
    runs = []
    for seed in ("0", "1"):
        completed = subprocess.run(
            [sys.executable, str(script), str(library)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        runs.append(json.loads(completed.stdout))

    assert len(runs[0]["candidates"]) > 100, "the corpus must exceed the route depth"
    assert runs[0]["hits"] == runs[1]["hits"], "the ranking must not depend on the hash seed"
    assert runs[0]["candidates"] == runs[1]["candidates"]


def test_the_modality_floor_never_evicts_the_top_hit(tmp_path: Path) -> None:
    """A floor promotes into spare slots; it does not rebuild the window out of last places."""
    from mindbridge.memory import _grounding_hits
    from mindbridge.types import SearchHit

    created = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)
    media_types = {
        Modality.IMAGE: "image/jpeg",
        Modality.VIDEO: "video/mp4",
        Modality.AUDIO: "audio/wav",
    }

    def _hit(name: str, modality: Modality, score: float) -> SearchHit:
        assets: tuple[AssetRef, ...] = ()
        if modality is not Modality.TEXT:
            path = tmp_path / f"{name}-asset"
            path.write_bytes(b"x")
            assets = (
                AssetRef(
                    id=hashlib.sha256(name.encode()).hexdigest(),
                    modality=modality,
                    media_type=media_types[modality],
                    size_bytes=1,
                    sha256=hashlib.sha256(b"x").hexdigest(),
                    name=path.name,
                    path=path,
                ),
            )
        return SearchHit(
            id=name,
            content=name,
            score=score,
            created_at=created,
            assets=assets,
            modality=modality,
        )

    ranking = tuple(
        _hit(name, modality, score)
        for name, modality, score in (
            ("t1", Modality.TEXT, 0.9),
            ("t2", Modality.TEXT, 0.8),
            ("i1", Modality.IMAGE, 0.7),
            ("v1", Modality.VIDEO, 0.6),
            ("a1", Modality.AUDIO, 0.5),
        )
    )

    grounded = {limit: [hit.id for hit in _grounding_hits(ranking, limit)] for limit in range(1, 6)}

    assert grounded == {
        1: ["t1"],
        2: ["t1", "i1"],
        3: ["t1", "i1", "v1"],
        4: ["t1", "i1", "v1", "a1"],
        5: ["t1", "t2", "i1", "v1", "a1"],
    }
    # The floor it promises: every modality the ranking holds is present once the window has a
    # slot for each of them, and the ranking's own first hit is in every window.
    for limit in range(4, 6):
        assert {hit.modality for hit in _grounding_hits(ranking, limit)} == {
            Modality.TEXT,
            Modality.IMAGE,
            Modality.VIDEO,
            Modality.AUDIO,
        }


def test_evidence_budget_skips_an_oversized_extra_and_keeps_a_later_fit() -> None:
    """The optional budget extension is not closed by one expensive lower-ranked record."""
    from mindbridge.context import evidence_cost
    from mindbridge.memory import _grounding_hits
    from mindbridge.types import SearchHit

    created = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)
    ranking = tuple(
        SearchHit(id=name, content=content, score=score, created_at=created)
        for name, content, score in (
            ("required", "a", 0.9),
            ("oversized", "0123456789", 0.8),
            ("fits", "bc", 0.7),
        )
    )

    grounded = _grounding_hits(ranking, limit=1, budget_chars=3)

    assert [hit.id for hit in grounded] == ["required", "fits"]
    # This is the live ask budget's unit: text characters plus any text-equivalent media cost.
    # It intentionally is not represented as a model-token guarantee.
    assert sum(evidence_cost(hit) for hit in grounded) == 3


class _StructuredDescriber:
    """A describer that answers in the labelled-line shape and records the context it was given.

    Every real caption crosses the write path as one string per visual, so the labelled lines and
    the `Fact:` lines arrive together and are cut apart by `_split_description`. This fake keeps
    that shape, and keeps `ModelInput.text` so a test can assert what the model was shown.
    """

    vision_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    vision_model = "fake-describer"
    vision_space = "fake-describer:structured-v2"

    _DEFAULT = (
        "Shown: two people at a kitchen table with a laptop",
        'Text: "Hongqiao" on a station sign',
        "Counts: 2 people, 1 laptop",
        "Place: kitchen, sign reads Hongqiao",
        "When: wall clock at 14:30",
        "Tags: kitchen, laptop, station sign",
        "Fact: speaker_1 is called Lily",
        "Fact: Lily is allergic to peanuts",
    )

    def __init__(self, *lines: str) -> None:
        self.lines = lines or self._DEFAULT
        self.context: list[str] = []
        self.calls = 0

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        batch = tuple(inputs)
        self.calls += 1
        self.context.extend(value.text for value in batch)
        return tuple("\n".join(self.lines) for _ in batch)

    def close(self) -> None:
        return None


class _Diarised:
    """A speech backend answering with two distinguishable speakers per asset."""

    transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})
    transcription_model = "fake-funasr"
    transcription_space = "fake-funasr:speech:test"

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        batch = tuple(assets)
        self.calls += 1
        return tuple(
            SpeechAnalysis(
                turns=(
                    SpeechTurn(0, 900, "my name is Lily", "0"),
                    SpeechTurn(900, 1800, "and I cannot eat peanuts", "0"),
                    SpeechTurn(1800, 2700, "noted", "1"),
                ),
                speakers=(
                    SpeakerEmbedding("0", (1.0, 0.0)),
                    SpeakerEmbedding("1", (0.0, 1.0)),
                ),
            )
            for _asset in batch
        )

    def close(self) -> None:
        return None


_LONG_TURN_TEXT = "the quick brown fox jumps over the lazy dog by the riverbank at dawn today"


class _VerboseDiarised:
    """A speech backend whose one speaker talks long enough to exceed the describe-context cap."""

    transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})
    transcription_model = "fake-funasr-verbose"
    transcription_space = "fake-funasr-verbose:speech:test"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        turns = tuple(
            SpeechTurn(index * 900, (index + 1) * 900, _LONG_TURN_TEXT, "0") for index in range(150)
        )
        return tuple(
            SpeechAnalysis(turns=turns, speakers=(SpeakerEmbedding("0", (1.0, 0.0)),))
            for _asset in assets
        )

    def close(self) -> None:
        return None


def _section(content: str, marker: str) -> str:
    """Read back one derived section of a stored document by its marker line."""
    for section in content.split("\n\n"):
        head, separator, body = section.partition("\n")
        if separator and head == marker:
            return body
    raise AssertionError(f"{marker} missing from {content!r}")


def test_a_described_clip_carries_its_durable_facts_as_their_own_section(tmp_path: Path) -> None:
    """The distilled half of a caption is indexed, and indexed separately from the observed half.

    Media reach the index only through their description, and a question asked days later is
    about the durable claim ("who is allergic to peanuts"), not about what a frame showed. Both
    halves are unioned into the document -- the asset stays attached and authoritative -- and
    they are separate sections so a reader can tell the distillation from the observation.
    """
    embedder = _Embedder()
    describer = _StructuredDescriber()
    with Memory(
        tmp_path,
        embedder=embedder,
        transcriber=_Diarised(),
        vision_describer=describer,
    ) as memory:
        record = memory.add(Blob(b"kitchen-clip", "video/mp4", "kitchen.mp4"))
        stored = memory.get(record.id).content
        asset_id = record.assets[0].id

        described = _section(stored, f"[visual description:{asset_id}]")
        assert described.startswith("Shown: two people at a kitchen table")
        assert "Hongqiao" in described
        assert "Fact:" not in described, "a fact must not stay inside the description"
        # The label is projected to the name this same write's own facts just staged, the same
        # projection transcript prose gets -- see `_project_fact_labels`. The binding line reads
        # a little redundant rendered that way ("Lily is called Lily"), which is the accepted
        # cost of one write-time pass rather than a second special case for the line that is the
        # assertion itself.
        assert _section(stored, f"[facts:{asset_id}]") == (
            "Lily is called Lily\nLily is allergic to peanuts"
        )
        # The verbatim asset stays attached: the sections are additional keys, not a replacement.
        assert record.assets[0].sha256 is not None

        # Both halves reach the lexical route. "peanuts" is only ever said in the facts.
        assert record.id in _lexical_matches(memory, "peanuts")
        assert record.id in _lexical_matches(memory, "Hongqiao")
        # And both reach the embedder as their own retrieval key -- in the *last* batch, which
        # is the reindex the stated name triggered, not the original write. Rebuilding a
        # document from its stored text has to recover the same atomic parts `add()` embedded;
        # rebuilding it as one merged string keys 2048-character windows of the whole document
        # instead, and the per-section keys are lost on exactly the clips that name somebody.
        keys = [value.text for value in embedder.document_batches[-1]]
        assert any(text.startswith(f"[facts:{asset_id}]") for text in keys)
        assert any(text.startswith(f"[visual description:{asset_id}]") for text in keys)


def test_a_re_embedding_migration_keeps_each_derived_section_as_its_own_key(
    tmp_path: Path,
) -> None:
    """An embedder swap re-keys every stored row, and must key them the way `add()` did.

    The migration reads `content` back as one string. Handing that string over as a single
    canonical part throws away the section boundaries the write path embedded separately, so a
    corpus that migrates loses the atomic facts key it had before the swap.
    """
    describer = _StructuredDescriber("Shown: a red bicycle", "Fact: the bicycle lives in the shed")
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as memory:
        record = memory.add(Blob(b"bicycle-frame", "image/png"))
        asset_id = record.assets[0].id

    upgraded = _Embedder()
    upgraded.embedding_space = "fake-real-index:2:test-v2"
    upgraded._legacy_embedding_spaces = frozenset({_Embedder.embedding_space})
    with Memory(tmp_path, embedder=upgraded, vision_describer=describer):
        keys = [value.text for value in upgraded.document_batches[-1]]

        assert any(text.startswith(f"[facts:{asset_id}]") for text in keys)
        assert any(text.startswith(f"[visual description:{asset_id}]") for text in keys)


def test_a_described_clip_is_shown_the_transcript_under_the_indexed_labels(tmp_path: Path) -> None:
    """A durable fact about a person is in the words, so the words travel with the stills.

    The labels the describer is shown have to be the labels the document prints, or a fact
    naming `speaker_1` names nobody the reader can find. This clip's facts state no name, so the
    document still prints the labels it was described under; a clip that does state one is
    reindexed under that name, which is `test_a_stated_name_carries_to_every_later_clip`.
    """
    describer = _StructuredDescriber("Shown: two people at a kitchen table", "Fact: nobody cooks")
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        transcriber=_Diarised(),
        vision_describer=describer,
    ) as memory:
        record = memory.add(Blob(b"kitchen-clip", "video/mp4", "kitchen.mp4"))
        asset_id = record.assets[0].id
        projected = _speech_retrieval_text(
            _section(memory.get(record.id).content, f"[speech identities:{asset_id}]"),
            asset_id,
        )

    assert describer.context == [
        "speaker_1: my name is Lily\nspeaker_1: and I cannot eat peanuts\nspeaker_2: noted"
    ]
    assert describer.context[0] == projected, "the describer saw labels the document does not use"


def test_a_long_transcript_shown_to_the_describer_is_capped_at_a_line_boundary(
    tmp_path: Path,
) -> None:
    """One long clip's own words must not dominate the token budget every visual in it pays.

    Cut at `_MAX_DESCRIBE_CONTEXT_CHARACTERS`, and at a line boundary -- each line is one
    speaker turn -- so the cut context still ends on a whole turn.
    """
    describer = _StructuredDescriber("Shown: a long conversation")
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        transcriber=_VerboseDiarised(),
        vision_describer=describer,
    ) as memory:
        memory.add(Blob(b"long-clip", "video/mp4", "long.mp4"))

    assert len(describer.context) == 1
    shown = describer.context[0]
    assert len(shown) <= _MAX_DESCRIBE_CONTEXT_CHARACTERS
    assert shown, "some context should still survive the cap"
    assert all(line == f"speaker_1: {_LONG_TURN_TEXT}" for line in shown.split("\n"))


def test_a_described_image_is_shown_no_transcript_and_still_yields_a_description(
    tmp_path: Path,
) -> None:
    """An image has no words, so it travels as pixels alone, exactly as it did before facts."""
    describer = _StructuredDescriber("Shown: a red bicycle", "Tags: bicycle, fence")
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as memory:
        record = memory.add(Blob(b"bicycle-frame", "image/png"))
        stored = memory.get(record.id).content

    assert describer.context == [""]
    assert (
        stored
        == f"[visual description:{record.assets[0].id}]\nShown: a red bicycle\nTags: bicycle, fence"
    )
    assert "[facts:" not in stored


def test_a_media_memory_derives_no_section_at_all_without_a_describer(tmp_path: Path) -> None:
    """Route by capability: with no describer the document is what it was before facts existed.

    Deriving text is a paid model call per visual, so it must follow from configuration and from
    nothing else. A speech backend is configured here, which is the composition most likely to
    grow a section by accident, since the transcript is exactly what the describer would be shown.
    """
    embedder = _Embedder()
    with Memory(tmp_path, embedder=embedder, transcriber=_Diarised()) as memory:
        record = memory.add(Blob(b"kitchen-clip", "video/mp4", "kitchen.mp4"))
        stored = memory.get(record.id).content

    asset_id = record.assets[0].id
    assert "[facts:" not in stored
    assert "[visual description:" not in stored
    assert stored.startswith(f"[speech identities:{asset_id}]")
    assert all("[facts:" not in value.text for value in embedder.document_inputs)


def test_a_speaker_label_shown_to_the_describer_is_the_label_the_index_prints() -> None:
    """`_speaker_labels` resolves a fact's label, so it must not drift from the projection.

    The projection is what writes `speaker_2` into a searchable document; the label map is what
    turns a distilled `speaker_2 is called Lily` back into a person. Two rules would silently
    name the wrong one.
    """
    asset_id = "a" * 64
    segments = (
        SpeakerSegment(asset_id, 0, 900, "hello", speaker_id="identity_second"),
        SpeakerSegment(asset_id, 900, 1800, "hi", speaker_id="identity_first"),
        SpeakerSegment(asset_id, 1800, 2700, "again", speaker_id="identity_second"),
    )
    labels = _speaker_labels(segments)
    projected = _speech_retrieval_text(json.dumps(_speech_evidence(asset_id, segments)), asset_id)

    assert labels == {"identity_second": "speaker_1", "identity_first": "speaker_2"}
    assert projected == "speaker_1: hello\nspeaker_2: hi\nspeaker_1: again"


def _speaker_identity(memory: Memory, memory_id: str, label: str) -> str:
    """Resolve the identity behind one `speaker_N` label of a stored memory's own clip."""
    segments = memory.speech(memory_id)
    labels = _speaker_labels(segments)
    return next(identity for identity, alias in labels.items() if alias == label)


def test_a_stated_name_carries_to_every_later_clip_of_the_same_voice(tmp_path: Path) -> None:
    """The lever the M3-Agent ablations measure: stable identity across clips, worth -11.2 to lose.

    A recognizer mints an unstable `identity_*` per run, and the index prints `speaker_1`. Once
    the dialogue states the name, the person is keyed under it everywhere -- including clips that
    say nothing about who is talking, which is where the retrieval question is actually asked.
    """
    describer = _StructuredDescriber(
        "Shown: two people at a kitchen table",
        "Fact: speaker_1 is called Lily",
    )
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        transcriber=_Diarised(),
        vision_describer=describer,
    ) as memory:
        named = memory.add(Blob(b"kitchen-clip", "video/mp4", "kitchen.mp4"))
        identity_id = _speaker_identity(memory, named.id, "speaker_1")
        assert memory.identity(identity_id) is not None
        assert memory.identity(identity_id).name == "Lily"  # type: ignore[union-attr]

        # A later clip of the same voice, whose own facts say nothing about a name.
        describer.lines = ("Shown: the same table, later",)
        anonymous = memory.add(Blob(b"kitchen-clip-two", "video/mp4", "later.mp4"))
        asset_id = anonymous.assets[0].id
        projected = _speech_retrieval_text(
            _section(memory.get(anonymous.id).content, f"[speech identities:{asset_id}]"),
            asset_id,
        )

        assert projected is not None
        assert projected.startswith("Lily: my name is Lily")
        assert "speaker_1" not in projected
        # And the naming is an ordinary auditable operation, not a hidden write.
        assert [record.operation.intent for record in memory.operations()] == [
            MemoryIntent.IDENTIFY
        ]


def test_a_stated_name_never_replaces_the_one_a_host_registered(tmp_path: Path) -> None:
    """A name is what the index keys a person under, so a mishearing must not propagate.

    The fact came from a model reading a transcript. Letting it overwrite a standing name would
    rewrite every document that person appears in, from one wrong word.
    """
    describer = _StructuredDescriber("Shown: a kitchen table")
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        transcriber=_Diarised(),
        vision_describer=describer,
    ) as memory:
        first = memory.add(Blob(b"kitchen-clip", "video/mp4", "kitchen.mp4"))
        identity_id = _speaker_identity(memory, first.id, "speaker_1")
        memory.register_identity(identity_id, "Mei")

        describer.lines = ("Shown: a kitchen table", "Fact: speaker_1 is called Lily")
        memory.add(Blob(b"kitchen-clip-two", "video/mp4", "later.mp4"))

        assert memory.identity(identity_id).name == "Mei"  # type: ignore[union-attr]
        # Re-stating the name a person already carries is a no-op, not a second assertion.
        describer.lines = ("Shown: a kitchen table", "Fact: speaker_1 is called Mei")
        memory.add(Blob(b"kitchen-clip-three", "video/mp4", "later-still.mp4"))
        assert memory.identity(identity_id).name == "Mei"  # type: ignore[union-attr]
        assert [record.operation.intent for record in memory.operations()] == [
            MemoryIntent.IDENTIFY
        ]


def test_a_stated_name_with_a_few_words_is_bound_in_full(tmp_path: Path) -> None:
    """A short multi-word name is not truncated to its first word.

    The binding is bounded by length, not by word count: up to four words is still one name.
    """
    describer = _StructuredDescriber(
        "Shown: two people at a kitchen table",
        "Fact: speaker_1 is called Mary Jane",
    )
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        transcriber=_Diarised(),
        vision_describer=describer,
    ) as memory:
        named = memory.add(Blob(b"kitchen-clip", "video/mp4", "kitchen.mp4"))
        identity_id = _speaker_identity(memory, named.id, "speaker_1")

        assert memory.identity(identity_id).name == "Mary Jane"  # type: ignore[union-attr]


def test_a_stated_name_does_not_swallow_the_clause_that_follows_it(tmp_path: Path) -> None:
    """A name is a handful of words, not everything up to the next period.

    The old pattern captured every character up to a period or semicolon, so a fact that kept
    talking past the name in the same sentence ("...and works in marketing") bound the whole
    clause as the name. Bounded to a few words, that sentence fails to match at all, and nothing
    is bound rather than something wrong.
    """
    describer = _StructuredDescriber(
        "Shown: two people at a kitchen table",
        "Fact: speaker_1 is called Lily and works in marketing",
    )
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        transcriber=_Diarised(),
        vision_describer=describer,
    ) as memory:
        named = memory.add(Blob(b"kitchen-clip", "video/mp4", "kitchen.mp4"))
        identity_id = _speaker_identity(memory, named.id, "speaker_1")

        assert memory.identity(identity_id).name is None  # type: ignore[union-attr]
        assert memory.operations() == ()


def test_a_fact_naming_a_label_nobody_produced_is_counted_not_silently_dropped(
    tmp_path: Path,
) -> None:
    """A fact naming `speaker_5` when the clip only produced two speakers names nobody.

    That drop used to vanish with no trace once the label failed to resolve to an identity.
    It is the same kind of event as the "already named" refusal `_bind_speaker_names` already
    counted, so both land on `IDENTITY_NAMES_REFUSED`.
    """
    describer = _StructuredDescriber(
        "Shown: two people at a kitchen table",
        "Fact: speaker_5 is called Nobody",
    )
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        transcriber=_Diarised(),
        vision_describer=describer,
        tracer=provider.get_tracer("test"),
    ) as memory:
        memory.add(Blob(b"kitchen-clip", "video/mp4", "kitchen.mp4"))
    provider.shutdown()

    spans = [
        span for span in exporter.get_finished_spans() if span.name == "mindbridge.identity.names"
    ]
    assert len(spans) == 1
    attributes = spans[0].attributes or {}
    assert attributes.get(IDENTITY_NAMES_BOUND) == 0
    assert attributes.get(IDENTITY_NAMES_REFUSED) == 1


def test_a_stated_name_binds_nothing_without_a_speech_backend(tmp_path: Path) -> None:
    """No recognizer, no labels, nobody to name: the fact stays indexed text and nothing else.

    Route by capability. A label is only resolvable through the diarisation that produced it, so
    a composition with no speech backend must not invent a person to hang the name on.
    """
    describer = _StructuredDescriber(
        "Shown: two people at a kitchen table",
        "Fact: speaker_1 is called Lily",
    )
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as memory:
        record = memory.add(Blob(b"kitchen-clip", "video/mp4", "kitchen.mp4"))
        stored = memory.get(record.id).content

        assert _section(stored, f"[facts:{record.assets[0].id}]") == "speaker_1 is called Lily"
        assert memory.operations() == ()


class _ThrottledDescriber:
    """A describer that refuses a set number of batches the way a loaded endpoint does."""

    vision_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    vision_model = "fake-describer"
    vision_space = "fake-describer:throttled-v2"

    def __init__(self, refusals: int, error: ModelError) -> None:
        self.refusals = refusals
        self.error = error
        self.calls = 0

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        batch = tuple(inputs)
        self.calls += 1
        if self.calls <= self.refusals:
            raise self.error
        return tuple("Shown: a red bicycle" for _ in batch)

    def close(self) -> None:
        return None


def _rate_limited() -> ModelError:
    return ModelError("slow down", reason="rate_limited", stage="describe")


def _out_of_quota() -> ModelError:
    return ModelError("no quota", reason="quota_exhausted", stage="describe")


class _UpstreamFailure(Exception):
    """Stands in for the provider SDK's own 5xx exception, which carries a status code."""

    status_code = 503


def _server_error() -> ModelError:
    """A 5xx as it really arrives: unclassified, with the status only on the provider's cause."""
    error = ModelError("upstream", stage="describe")
    error.__cause__ = _UpstreamFailure()
    return error


class _AbortedJsonGeneration(Exception):
    """Stands in for the provider SDK's `BadRequestError` when it aborts mid-response.

    The message is the inner-prism gateway's own, verbatim, from a live ATM raw-arm run: a 400
    that an identical retry 5s later clears, unlike an ordinary rejected request.
    """

    status_code = 400
    body: ClassVar[dict[str, object]] = {
        "message": (
            "<400> InternalError.Algo.InvalidParameter: Model output became abnormal while "
            "generating a JSON response for response_format. The generation was aborted because "
            "the partial output may be incomplete or invalid JSON. Please retry the request or "
            "adjust your prompt or JSON schema."
        ),
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_parameter_error",
    }


def _aborted_json_generation() -> ModelError:
    error = ModelError(
        "vision description request failed", reason="request_rejected", stage="describe"
    )
    error.__cause__ = _AbortedJsonGeneration()
    return error


class _RejectedImage(Exception):
    """An ordinary 400: the same status code, a permanent cause, no aborted-generation wording."""

    status_code = 400
    body: ClassVar[dict[str, object]] = {
        "message": "invalid image",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_parameter_error",
    }


def _rejected_image() -> ModelError:
    """A 400 that really is permanent, to prove the narrow message match does not over-retry."""
    error = ModelError(
        "vision description request failed", reason="request_rejected", stage="describe"
    )
    error.__cause__ = _RejectedImage()
    return error


@pytest.mark.parametrize("failure", [_rate_limited, _server_error, _aborted_json_generation])
def test_a_throttled_describe_is_retried_before_the_caption_is_given_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: Callable[[], ModelError],
) -> None:
    """A provider throttling an ingest throttles it for minutes; fail-open costs the whole arm.

    The SDK client has already spent its own retry budget by the time the failure arrives here,
    so without a wait every asset in the burst is stored with no caption behind one log line. A
    5xx is not in the closed retryable vocabulary and has to be recognized by its status code.
    """
    monkeypatch.setattr("mindbridge.memory._VISION_RETRY_BACKOFF", (0.0, 0.0, 0.0))
    describer = _ThrottledDescriber(2, failure())
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as memory:
        record = memory.add(Blob(b"bicycle-frame", "image/png"))

        assert describer.calls == 3
        assert _caption(memory.get(record.id).content) == "Shown: a red bicycle"


def test_a_permanent_describe_refusal_is_not_retried_and_still_fails_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exhausted billing retried forever is worse than a missing caption, and costs per attempt.

    The memory is the caller's; the caption is derived convenience. So the write still lands with
    an empty document, uncached, and a later ingest retries the description.
    """
    monkeypatch.setattr("mindbridge.memory._VISION_RETRY_BACKOFF", (0.0, 0.0, 0.0))
    describer = _ThrottledDescriber(
        4, ModelError("no quota", reason="quota_exhausted", stage="describe")
    )
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as memory:
        record = memory.add(Blob(b"bicycle-frame", "image/png"))

        assert describer.calls == 1
        assert memory.get(record.id).content == ""


def test_an_ordinary_400_is_still_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The narrow message match for the aborted-JSON-generation quirk must not swallow every 400.

    Same status code as the transient case, a permanent cause: an unsupported or rejected image
    is not going to describe successfully on a second try, so this must still fail open on
    attempt one, the way it did before the aborted-generation case was recognized.
    """
    monkeypatch.setattr("mindbridge.memory._VISION_RETRY_BACKOFF", (0.0, 0.0, 0.0))
    describer = _ThrottledDescriber(4, _rejected_image())
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as memory:
        record = memory.add(Blob(b"bicycle-frame", "image/png"))

        assert describer.calls == 1
        assert memory.get(record.id).content == ""


def test_a_sustained_refusal_gives_up_after_a_bounded_number_of_waits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("mindbridge.memory._VISION_RETRY_BACKOFF", (0.0, 0.0))
    describer = _ThrottledDescriber(99, _rate_limited())
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as memory:
        record = memory.add(Blob(b"bicycle-frame", "image/png"))

        assert describer.calls == 3, "one attempt per wait, plus the last one"
        assert memory.get(record.id).content == ""


def test_waiting_out_a_throttled_describe_is_counted_apart_from_losing_the_caption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`failed_batches` answers "how many memories lost their caption", so an attempt is not one.

    Retrying inside the fail-open turned one lost batch into three of them, and a counter that
    sums attempts cannot tell a provider that throttled an ingest from one that ate it. The
    attempts are still worth counting -- they are what a long ingest pays -- under their own name.
    """
    monkeypatch.setattr("mindbridge.memory._VISION_RETRY_BACKOFF", (0.0, 0.0))
    cases = (
        (_ThrottledDescriber(2, _rate_limited()), 0, 2),
        (_ThrottledDescriber(99, _rate_limited()), 1, 2),
        (_ThrottledDescriber(99, _out_of_quota()), 1, 0),
        (_ThrottledDescriber(1, _aborted_json_generation()), 0, 1),
    )
    for index, (describer, failed, retried) in enumerate(cases):
        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with Memory(
            tmp_path / f"case-{index}",
            embedder=_Embedder(),
            vision_describer=describer,
            tracer=provider.get_tracer("test"),
        ) as memory:
            memory.add(Blob(b"bicycle-frame", "image/png"))
        provider.shutdown()

        counted = {VISION_BATCHES_FAILED: 0, VISION_BATCHES_RETRIED: 0}
        for span in exporter.get_finished_spans():
            for name in counted:
                counted[name] += cast(int, (span.attributes or {}).get(name, 0))

        assert counted[VISION_BATCHES_FAILED] == failed, describer.error.reason
        assert counted[VISION_BATCHES_RETRIED] == retried, describer.error.reason


def test_a_caption_that_is_only_facts_is_not_described_or_appended_twice(tmp_path: Path) -> None:
    """A facts-only caption leaves no description marker, which used to mean "not yet described".

    Both derived sections have to count as evidence that the asset was described, or a second
    write citing the same picture pays for the same text again and appends the facts twice.
    """
    describer = _StructuredDescriber("Fact: the yoga mat lives in the storage room")
    picture = Blob(b"storage-room", "image/png", "storage.png")
    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as memory:
        first = memory.add(("morning note", picture))
        stored = memory.get(first.id).content

        assert "[visual description:" not in stored
        assert stored.count("[facts:") == 1
        assert describer.calls == 1

    with Memory(tmp_path, embedder=_Embedder(), vision_describer=describer) as reopened:
        again = reopened.add(("evening note", picture))

        assert describer.calls == 1, "the same bytes were described again"
        assert reopened.get(again.id).content.count("[facts:") == 1


class _TextOnlyAnswerer:
    """A generation backend that can read text and nothing else, the way a text LLM does."""

    generation_capabilities = frozenset({Modality.TEXT})

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        return AnswerResult(answer="; ".join(hit.content for hit in hits) or "nothing found")

    def close(self) -> None:
        return None


def test_a_facts_only_caption_still_lets_a_text_only_answerer_read_the_image(
    tmp_path: Path,
) -> None:
    """`_has_stream_description` has to count a facts-only caption, or `ask` refuses the image.

    A caption with no visible half sets no `[visual description:]` marker, only `[facts:]`.
    `_route_generation` falls an image its answerer cannot take back to derived text only when
    `_has_stream_description` says the image was described. The write-path test above covers the
    asset cache, a different mechanism, and would pass unchanged even with the `[facts:]` clause
    missing from `_has_stream_description`; a text-only answerer here would instead raise
    `unsupported_modality` on exactly that gap.
    """
    describer = _StructuredDescriber("Fact: the yoga mat lives in the storage room")
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        vision_describer=describer,
        answerer=_TextOnlyAnswerer(),
    ) as memory:
        memory.add(Blob(b"storage-room", "image/png", "storage.png"))
        result = memory.ask("where does the yoga mat live")

    assert "storage room" in result.answer
