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
    EmbedTask,
    Memory,
    Modality,
    ModelInput,
    ObservationContext,
    SearchHit,
)
from mindbridge.exceptions import ValidationError

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


def test_a_linked_row_the_ranking_already_found_is_not_paid_for_twice(tmp_path: Path) -> None:
    """Expansion pays only for evidence the ranking has no route to.

    A linked row inside the candidate pool is a row a wider budget reaches, so admitting it here
    buys nothing and costs a window slot. What the edges are for is the record the ranking never
    proposed -- and on a corpus where the ranking already spans the capture, that is none of them.
    """
    answerer = _RecordingAnswerer()
    with _memory(tmp_path, answerer, evidence_expansion=True, evidence_budget_chars=None) as memory:
        # Two records, one capture: with `limit=2` the ranking proposes both, so the partner is
        # reachable and the gate discards it.
        _capture(memory, "S1", "a red wrench", "a red hammer")

        memory.ask("red?", limit=2)

    assert len(_grounded_ids(answerer)) == 2
    assert "linked to the evidence above" not in (answerer.questions[-1].text or "")


def test_a_linked_row_the_ranking_never_proposed_is_admitted(tmp_path: Path) -> None:
    """The same capture, a window too narrow to contain it, and the partner survives the gate."""
    answerer = _RecordingAnswerer()
    with _memory(
        tmp_path,
        answerer,
        evidence_expansion=True,
        evidence_budget_chars=None,
        embedder=_ScriptedEmbedder(_FAR_PARTNER),
    ) as memory:
        _anchor, partner = _far_partner_store(memory)
        # The pool the ask ranks is `min(100, limit * 3)` deep; the partner is placed outside it.
        pool = {hit.id for hit in memory.search("a red wrench on the bench", limit=3)}

        memory.ask("a red wrench on the bench", limit=1)

    grounded = _grounded_ids(answerer)
    assert partner not in pool, "the fixture must place the partner beyond the ranked pool"
    assert partner in grounded, "a record the ranking did not propose reached the window"
