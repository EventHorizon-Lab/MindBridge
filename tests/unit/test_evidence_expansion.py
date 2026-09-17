"""Evidence expansion: widening a grounded window along the edges the kernel already wrote.

The measured failure these tests pin is the one a ranking cannot fix on its own. A question needs
several records; similarity finds the one that shares its words and leaves the rest unreached --
the reply to a retrieved question, the other observation from the same capture. Expansion follows
an authoritative column instead of a cosine, so it costs one indexed read and no model call.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from mindbridge import (
    AnswerPolicy,
    AnswerResult,
    AssetRef,
    EmbedTask,
    Memory,
    Modality,
    ModelInput,
    ObservationContext,
    SearchHit,
)
from mindbridge.exceptions import ValidationError
from mindbridge.infrastructure.local.store import RECALL_MAX_ROWS
from mindbridge.kernel.answering import _packed_grounding

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


class _ScriptedEmbedder:
    """Places records at chosen angles from the query, so a test can dictate the ranking.

    The gate compares linked rows against the candidates the ask ranked, and that pool is
    `min(100, limit * 3)` deep. A hash embedder cannot say which record lands inside it, so a test
    about the gate has to choose: text mapped to an angle, the query at zero, and cosine order is
    the order the angles were written.
    """

    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "scripted-test"
    embedding_space = "scripted-test:2:l2-v1"
    embedding_dimension = 2

    def __init__(self, angles: Mapping[str, float]) -> None:
        self._angles = dict(angles)

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        del task
        vectors = []
        for value in inputs:
            angle = next(
                (turn for text, turn in self._angles.items() if text in value.text),
                math.pi / 2,
            )
            vectors.append((math.cos(angle), math.sin(angle)))
        return tuple(vectors)

    def close(self) -> None:
        return None


class _RecordingAnswerer:
    """An answerer that reports exactly the evidence it was handed, in the order it arrived."""

    generation_capabilities = frozenset({Modality.TEXT})
    generation_model = "recording-answerer"

    def __init__(self) -> None:
        self.grounded: list[tuple[SearchHit, ...]] = []
        self.questions: list[ModelInput] = []

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        del answer_policy, exhaustive
        grounded = tuple(hits)
        self.questions.append(question)
        self.grounded.append(grounded)
        return AnswerResult(answer="answered", hits=grounded)

    def close(self) -> None:
        return None


def _memory(
    tmp_path: Path,
    answerer: _RecordingAnswerer,
    *,
    evidence_expansion: bool = False,
    evidence_budget_chars: int | None = 24_000,
    evidence_expansion_max_rows: int = 24,
    embedder: object | None = None,
) -> Memory:
    return Memory(
        tmp_path / "store",
        embedder=embedder or TinyEmbedder(),  # type: ignore[arg-type]
        answerer=answerer,
        minimum_relevance=0.0,
        ambiguity_margin=0.0,
        reinforce_on_answer=False,
        evidence_expansion=evidence_expansion,
        evidence_expansion_max_rows=evidence_expansion_max_rows,
        evidence_budget_chars=evidence_budget_chars,
    )


def _capture(memory: Memory, source: str, *texts: str, minute: int = 0) -> tuple[str, ...]:
    """Commit several observations under one capture, the way a host's own session does."""
    return tuple(
        memory.add(
            text,
            occurred_at=NOW + timedelta(minutes=minute + offset),
            context=ObservationContext(source_id=source),
        ).id
        for offset, text in enumerate(texts)
    )


def _grounded_ids(answerer: _RecordingAnswerer) -> tuple[str, ...]:
    return tuple(hit.id for hit in answerer.grounded[-1])


def test_expansion_recovers_the_partner_record_similarity_left_behind(tmp_path: Path) -> None:
    """The reply to a retrieved question is reached through its capture, not through its words.

    Which record the ranking puts first is the embedder's business and not this test's, so the
    corpus is three captures of two, the ranking is asked what it found, and the claim is about
    what expansion did with it: the window keeps its top hit and gains that hit's own partner,
    and gains nothing from the captures the top hit is not in.
    """
    answerer = _RecordingAnswerer()
    with _memory(tmp_path, answerer, evidence_expansion=True, evidence_budget_chars=None) as memory:
        captures = {
            "S1": _capture(memory, "S1", "how long into red art", "seven years now", minute=0),
            "S2": _capture(memory, "S2", "the blue crate", "in the shed", minute=60),
            "S3": _capture(memory, "S3", "a plain wooden bench", "beside the door", minute=120),
        }
        ranked = memory.search("how long into red art?", limit=1)
        top = ranked[0].id
        partner = next(
            other for ids in captures.values() if top in ids for other in ids if other != top
        )

        memory.ask("how long into red art?", limit=1)

    grounded = _grounded_ids(answerer)
    assert grounded[0] == top, "the window the ranking earned stays first"
    assert grounded == (top, partner), "exactly the top hit's own capture partner is recovered"


def test_expansion_is_off_by_default_and_changes_nothing(tmp_path: Path) -> None:
    """With the setting off the grounded window is exactly what it has always been."""
    answerer = _RecordingAnswerer()
    with _memory(tmp_path, answerer, evidence_budget_chars=None) as memory:
        question_id, _answer_id = _capture(
            memory,
            "S1",
            "how long have you been into red art",
            "seven years now",
        )
        memory.ask("how long into red art?", limit=1)

    assert _grounded_ids(answerer) == (question_id,)
    assert "linked to the evidence above" not in (answerer.questions[-1].text or "")


# One capture whose partner the ranking never proposes: `limit=1` ranks three candidates, and the
# partner is placed fifth. Distractors fill the pool so the gate has something to discard.
_FAR_PARTNER = {
    "a red wrench on the bench": 0.0,
    "distractor one": 0.2,
    "distractor two": 0.3,
    "distractor three": 0.4,
    "and a red hammer beside it": 1.2,
}


def _far_partner_store(memory: Memory) -> tuple[str, str]:
    anchor, partner = _capture(
        memory,
        "S1",
        "a red wrench on the bench",
        "and a red hammer beside it",
    )
    _capture(memory, "S2", "distractor one", "distractor two", "distractor three", minute=30)
    return anchor, partner


def test_the_note_says_which_edge_linked_the_rows_and_refuses_a_total(tmp_path: Path) -> None:
    """A linked record is not a matched record, and a reader that cannot tell states wrong totals."""
    answerer = _RecordingAnswerer()
    with _memory(
        tmp_path,
        answerer,
        evidence_expansion=True,
        evidence_budget_chars=None,
        embedder=_ScriptedEmbedder(_FAR_PARTNER),
    ) as memory:
        _far_partner_store(memory)
        memory.ask("a red wrench on the bench", limit=1)

    note = answerer.questions[-1].text or ""
    assert "linked to the evidence above (capture)" in note
    assert "do not state a total over them" in note


def test_an_edge_that_selects_the_corpus_is_dropped_whole(tmp_path: Path) -> None:
    """One capture holding the corpus says nothing about the question, so it contributes nothing."""
    answerer = _RecordingAnswerer()
    with _memory(tmp_path, answerer, evidence_expansion=True, evidence_budget_chars=None) as memory:
        ids = _capture(memory, "S1", *(f"record {index} about red things" for index in range(12)))

        memory.ask("red things?", limit=1)

    grounded = _grounded_ids(answerer)
    assert grounded[0] in ids
    # The ceiling is four times the ask's own limit, and this capture links to eleven more rows.
    assert "linked to the evidence above" not in (answerer.questions[-1].text or "")


def test_expansion_never_displaces_the_window_the_ranking_earned(tmp_path: Path) -> None:
    """The required window is mandatory: expansion competes with the budget's tail, never with it."""
    answerer = _RecordingAnswerer()
    with _memory(tmp_path, answerer, evidence_expansion=True, evidence_budget_chars=None) as memory:
        _capture(memory, "S1", "a red wrench", "a red hammer", minute=0)
        _capture(memory, "S2", "a plain wooden crate", "a tin of nails", minute=10)
        window = tuple(hit.id for hit in memory.search("red wrench?", limit=2))

        memory.ask("red wrench?", limit=2)

    grounded = _grounded_ids(answerer)
    assert grounded[: len(window)] == window, "the ranked window keeps its place and its order"


def test_related_memories_reports_its_edges_and_excludes_its_anchors(tmp_path: Path) -> None:
    """Attribution is the claim: which edge recovered the evidence, and which edge was refused."""
    answerer = _RecordingAnswerer()
    with _memory(tmp_path, answerer) as memory:
        anchor, sibling = _capture(memory, "S1", "first observation", "second observation")
        (other,) = _capture(memory, "S2", "unrelated observation", minute=60)

        read = memory._store.recall.related_memories((anchor,), ceiling=10)

        assert tuple(row.memory_id for row in read) == (sibling,)
        assert read.edges["capture"] == 1
        assert read.dropped == ()
        assert other not in {row.memory_id for row in read}

        refused = memory._store.recall.related_memories((anchor,), ceiling=0)
        assert tuple(refused) == ()
        assert "capture" in refused.dropped


def test_related_memories_rejects_an_unknown_edge(tmp_path: Path) -> None:
    """Edge names are a closed set, so a typo is refused rather than silently reading nothing."""
    answerer = _RecordingAnswerer()
    with _memory(tmp_path, answerer) as memory:
        (anchor,) = _capture(memory, "S1", "an observation")

        # The store raises its own boundary error; the kernel is what translates one.
        with pytest.raises(ValueError, match="edges must be chosen from"):
            memory._store.recall.related_memories((anchor,), edges=("sesion",))


def test_evidence_expansion_max_rows_must_be_positive(tmp_path: Path) -> None:
    """Policy is validated once, at wiring, like every other setting."""
    with pytest.raises(ValidationError):
        Memory(
            tmp_path / "store",
            embedder=TinyEmbedder(),
            evidence_expansion=True,
            evidence_expansion_max_rows=0,
        )


def test_evidence_expansion_max_rows_is_bounded_by_the_store_read_it_becomes(
    tmp_path: Path,
) -> None:
    """The value goes straight to a recall read's own `max_rows`, which refuses 501.

    Unbounded at wiring, a caller who set it high opened a memory that worked and then raised a
    bare store `ValueError` out of every `ask` -- policy failing at query time instead of once.
    """
    with pytest.raises(ValidationError):
        Memory(
            tmp_path / "store",
            embedder=TinyEmbedder(),
            evidence_expansion=True,
            evidence_expansion_max_rows=RECALL_MAX_ROWS + 1,
        )


# One anchor whose capture partner and whose place-mate are both linked, and only one slot for
# them: the place-mate is older, so corpus order gives it the slot and the capture supplies
# nothing the reader ends up holding.
_ONE_SLOT = {
    "an older note from the kitchen": 0.8,
    "a red wrench on the bench": 0.0,
    "and a red hammer beside it": 1.2,
}


def test_the_note_names_only_the_edges_that_supplied_a_row(tmp_path: Path) -> None:
    """Attribution is a claim to the reader, so it is counted over the rows it actually got.

    Counted over what the edges linked to instead, scope and `max_rows` drop rows between the two
    and the prompt tells a reader a record is there because it shares a capture when the capture
    edge supplied nothing at all.
    """
    answerer = _RecordingAnswerer()
    with _memory(
        tmp_path,
        answerer,
        evidence_expansion=True,
        evidence_budget_chars=None,
        evidence_expansion_max_rows=1,
        embedder=_ScriptedEmbedder(_ONE_SLOT),
    ) as memory:
        older = memory.add(
            "an older note from the kitchen",
            occurred_at=NOW,
            context=ObservationContext(source_id="S2", place_id="kitchen"),
        ).id
        for offset, text in enumerate(("a red wrench on the bench", "and a red hammer beside it")):
            memory.add(
                text,
                occurred_at=NOW + timedelta(minutes=10 + offset),
                context=ObservationContext(source_id="S1", place_id="kitchen"),
            )

        memory.ask("a red wrench on the bench", limit=1)

    assert _grounded_ids(answerer)[-1] == older, "the one slot went to the older place-mate"
    note = answerer.questions[-1].text or ""
    assert "linked to the evidence above (place)" in note


def test_a_saturated_budget_adds_nothing_and_says_nothing(tmp_path: Path) -> None:
    """The note counts what expansion admitted, not what it happened to be linked to.

    A linked record is usually also a ranked record, so intersecting the window with the
    expansion read credited expansion for every row the ranked tail had already admitted. On a
    budget the ranking fills, that was a prompt telling the reader linked records followed its
    evidence when none had been added.
    """
    answerer = _RecordingAnswerer()
    with _memory(tmp_path, answerer, evidence_expansion=True, evidence_budget_chars=40) as memory:
        _capture(memory, "S1", "a red wrench", "a red hammer", "a red clamp")

        memory.ask("red?", limit=1)

    # The budget fits the window and nothing more, so the ranked tail spends it and expansion
    # admits no row -- and the prompt must not claim otherwise.
    assert "linked to the evidence above" not in (answerer.questions[-1].text or "")


def test_linked_rows_fill_only_what_the_ranking_left(tmp_path: Path) -> None:
    """Expansion goes last, so it spends leftover budget and never the ranking's own rows."""
    answerer = _RecordingAnswerer()
    with _memory(
        tmp_path,
        answerer,
        evidence_expansion=True,
        evidence_budget_chars=None,
        embedder=_ScriptedEmbedder(_FAR_PARTNER),
    ) as memory:
        anchor, partner = _far_partner_store(memory)

        memory.ask("a red wrench on the bench", limit=1)

    grounded = _grounded_ids(answerer)
    assert grounded[0] == anchor, "the window the ranking earned stays first"
    assert partner in grounded, "the capture partner fills the budget-free tail"


def test_a_linked_row_the_ranking_also_ranked_is_still_offered(tmp_path: Path) -> None:
    """Pool membership is not window membership, so it cannot decide what expansion admits.

    Discarding a linked row that appears anywhere in the candidate pool was tried and measured
    harmful: with no character budget the pool is `min(100, limit * 3)` and the window is `limit`,
    so a gold row at rank 20 is in the pool and will never be in the window. What removes the rows
    a wider window would have held anyway is `_packed_grounding`'s dedup, which is the only place
    that knows what the window holds.
    """
    answerer = _RecordingAnswerer()
    with _memory(
        tmp_path,
        answerer,
        evidence_expansion=True,
        evidence_budget_chars=None,
        embedder=_ScriptedEmbedder(_FAR_PARTNER),
    ) as memory:
        anchor, partner = _far_partner_store(memory)
        # The partner is inside the pool the ask ranks and outside the window it grounds.
        pool = {hit.id for hit in memory.search("a red wrench on the bench", limit=5)}

        memory.ask("a red wrench on the bench", limit=1)

    grounded = _grounded_ids(answerer)
    assert {anchor, partner} <= pool, "the fixture must rank both inside the pool"
    assert grounded[0] == anchor
    assert partner in grounded, "a ranked-but-unwindowed row is still worth admitting"


def test_a_linked_row_the_ranking_never_proposed_is_admitted(tmp_path: Path) -> None:
    """A capture partner outside the ranked pool entirely still reaches the window."""
    answerer = _RecordingAnswerer()
    with _memory(
        tmp_path,
        answerer,
        evidence_expansion=True,
        evidence_budget_chars=None,
        embedder=_ScriptedEmbedder(_FAR_PARTNER),
    ) as memory:
        _anchor, partner = _far_partner_store(memory)
        pool = {hit.id for hit in memory.search("a red wrench on the bench", limit=3)}

        memory.ask("a red wrench on the bench", limit=1)

    grounded = _grounded_ids(answerer)
    assert partner not in pool, "the fixture must place the partner beyond the ask's own window"
    assert partner in grounded, "a record the window would not have held reached it"


def _hit(identifier: str, *, media: bool = False) -> SearchHit:
    asset = AssetRef(
        id=f"asset-{identifier}",
        modality=Modality.IMAGE,
        media_type="image/png",
        size_bytes=1,
        sha256="0" * 64,
        path=Path(f"/nonexistent/{identifier}.png"),
    )
    return SearchHit(
        id=identifier,
        content=identifier,
        score=0.5,
        created_at=NOW,
        modality=Modality.IMAGE if media else Modality.TEXT,
        assets=(asset,) if media else (),
    )


def test_expansion_media_rows_stop_at_the_cap_a_plan_s_media_rows_stop_at() -> None:
    """Linked clips cost recognition writes exactly as a plan's matched clips do.

    A capture on a photo corpus is a day of photographs, so an unbounded expansion could follow a
    `limit`-row window with `evidence_expansion_max_rows` clips -- each one paying face and speech
    recognition before the answer call and again on every replan round, which is the cost
    `_budgeted_recall` already refuses to let a set read run up. The cap is a packing pass like
    every other bound here, so a text row behind a refused clip still gets in.
    """
    window = (_hit("w1"),)
    expansion = (_hit("m1", media=True), _hit("m2", media=True), _hit("t1"))

    packed, added = _packed_grounding(window, expansion, (), None, media_limit=1)

    assert [hit.id for hit in packed] == ["w1", "m1", "t1"]
    assert added == 2, "the clip over the cap is refused; the text row behind it is not"


def test_the_media_cap_never_trims_the_window_or_the_budget_s_own_tail() -> None:
    """The cap bounds what expansion added. The ranking is never trimmed for carrying media."""
    window = (_hit("w1", media=True),)
    tail = (_hit("r1", media=True), _hit("r2", media=True))

    packed, added = _packed_grounding(window, (), tail, 100_000, media_limit=0)

    assert [hit.id for hit in packed] == ["w1", "r1", "r2"]
    assert added == 0
