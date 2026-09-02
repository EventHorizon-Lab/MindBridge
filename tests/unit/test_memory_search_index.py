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

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindbridge import Memory
from mindbridge.exceptions import ModelError
from mindbridge.memory import _MAX_TEXT_CHARACTERS, _speech_retrieval_text
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.types import AssetRef, Blob, Modality

_ALL_INPUT_MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})


class _Embedder:
    """A deterministic embedder that can refuse a named asset the way a provider limit does."""

    embedding_model = "fake-real-index"
    embedding_space = "fake-real-index:2:test"
    embedding_dimension = 2

    def __init__(self) -> None:
        self.embedding_capabilities = _ALL_INPUT_MODALITIES
        self.oversized_assets: frozenset[str] = frozenset()
        self.document_inputs: list[ModelInput] = []

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
        # One vector for everything: dense relevance is deliberately uninformative so that a
        # lexical assertion cannot pass on the strength of the dense route.
        return tuple((1.0, 0.0) for _ in batch)

    def close(self) -> None:
        return None


class _Describer:
    """A vision backend that reports one fixed sentence, so the index document is predictable."""

    vision_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    vision_model = "fake-describer"

    def __init__(self, description: str = "a red bicycle leaning on the fence") -> None:
        self.description = description
        self.calls = 0

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        batch = tuple(inputs)
        self.calls += 1
        return tuple(self.description for _ in batch)

    def close(self) -> None:
        return None


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

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        del task
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
    for floor, expected in ((0.55, False), (0.10, True)):
        directory = tmp_path / f"floor-{floor}"
        with Memory(
            directory,
            embedder=_DirectionalEmbedder(cosines),
            minimum_relevance=floor,
        ) as memory:
            opposite = memory.add("anticorrelated zibaldone drift")
            memory.add("ordinary weather notes")

            hits = memory.search(query)
            by_id = {hit.id: hit for hit in hits}
            # One shared rare term used to earn a flat 0.6 gate confidence regardless of how weak
            # the match was, so a document pointing the other way cleared the default floor.
            assert (opposite.id in by_id) is expected
            assert all(hit.score >= floor for hit in hits)
            if expected:
                # Admitted, but at its real strength rather than a confident-looking constant.
                assert by_id[opposite.id].score < 0.5


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
