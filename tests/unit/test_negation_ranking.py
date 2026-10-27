"""Literal evidence without inferring grammatical polarity from words or bigrams."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge import EmbedTask, Memory, Modality, ModelInput
from mindbridge.infrastructure.local.store import StoredMemory
from mindbridge.kernel.ranking import lexical_query_terms, lexical_relevance


def _stored(memory_id: str, content: str) -> StoredMemory:
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    return StoredMemory(
        memory_id=memory_id,
        content=content,
        metadata_json="{}",
        created_at=now,
        updated_at=now,
    )


@pytest.mark.parametrize("negator", tuple("不没未无非别"))
def test_chinese_negation_remains_a_contextual_term(negator: str) -> None:
    query = f"{negator}购买红茶"
    memories = (
        _stored("negated", query),
        _stored("positive", "购买红茶"),
        _stored("elsewhere", f"{negator}喝\uff0c购买红茶"),
    )

    coverage = lexical_relevance(query, memories)

    assert coverage["negated"] > coverage["positive"]
    assert coverage["negated"] > coverage["elsewhere"]
    assert coverage["negated"] == pytest.approx(1.0)
    assert all(len(term) == 2 for term in lexical_query_terms(query))
    assert lexical_query_terms(negator) == frozenset()

    balanced = lexical_relevance(
        f"{negator}购买", (_stored("negated", f"{negator}购"), _stored("positive", "购买"))
    )
    assert balanced["negated"] == pytest.approx(0.5)
    assert balanced["positive"] == pytest.approx(0.5)


@pytest.mark.parametrize("boundary", ("\uff0c", " ", "a"))
def test_chinese_negation_does_not_cross_term_boundaries(boundary: str) -> None:
    query = f"不{boundary}喜欢"

    assert "不喜" not in lexical_query_terms(query)
    assert lexical_relevance(query, (_stored("positive", query),))["positive"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "term", ("no", "not", "never", "neither", "nor", "without", "非常", "别喜", "非洲", "无锡")
)
def test_equal_frequency_terms_receive_equal_weight(term: str) -> None:
    memories = (_stored("term", term), _stored("anchor", "anchor"))

    coverage = lexical_relevance(f"{term} anchor", memories)

    assert coverage == pytest.approx({"term": 0.5, "anchor": 0.5})


@pytest.mark.parametrize("apostrophe", ("'", "\u2019"))
def test_english_contractions_keep_the_negation_signal(apostrophe: str) -> None:
    query = f"Mira didn{apostrophe}t buy tea"
    memories = (
        _stored("negated", "Mira did not buy tea"),
        _stored("positive", "Mira did buy tea"),
    )

    coverage = lexical_relevance(query, memories)

    assert "not" in lexical_query_terms(query)
    assert coverage["negated"] > coverage["positive"]


class _NegationEmbedder:
    """Keep model evidence fixed while comparing lexical ranking policies."""

    embedding_model = "synthetic-negation-v2"
    embedding_space = "synthetic-negation-v2:2"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT})

    def __init__(
        self,
        preferred: str | None = None,
        *,
        cosine: float = 0.70,
        preferred_cosine: float = 0.72,
    ) -> None:
        self.preferred = preferred
        self.cosine = cosine
        self.preferred_cosine = preferred_cosine

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        if task is EmbedTask.QUERY:
            return tuple((1.0, 0.0) for _ in inputs)
        cosines = (
            self.preferred_cosine if value.text == self.preferred else self.cosine
            for value in inputs
        )
        return tuple((cosine, math.sqrt(1.0 - cosine**2)) for cosine in cosines)

    def close(self) -> None:
        return None


@pytest.mark.parametrize(
    ("negated", "positive"),
    (
        ("不喜欢红茶", "喜欢红茶"),
        ("没喝过红茶", "喝过红茶"),
        ("未购买红茶", "购买红茶"),
        ("无红茶供应", "红茶供应"),
        ("非红茶饮料", "红茶饮料"),
        ("别购买红茶", "购买红茶"),
    ),
)
@pytest.mark.parametrize("subject", ("小林", "Alice", "\uff21\uff2c\uff29\uff23\uff25"))
def test_literal_chinese_negation_keeps_complete_evidence_first(
    tmp_path: Path, negated: str, positive: str, subject: str
) -> None:
    with Memory(tmp_path, embedder=_NegationEmbedder()) as memory:
        answer = memory.add(f"{subject}{negated}\uff0c早餐喝豆浆")
        contradiction = memory.add(f"{subject}{positive}\uff0c周二早餐喝豆浆")
        misplaced = memory.add(f"{subject}{positive}\uff0c周二早餐不喝豆浆")

        result = memory.search_with_trace(f"{subject}{negated}", limit=3)

        assert result.hits[0].id == answer.id
        assert {hit.id for hit in result.hits} == {answer.id, contradiction.id, misplaced.id}
        assert all(0.0 < hit.score < 1.0 for hit in result.hits)
        assert all(candidate.lexical_match for candidate in result.trace.candidates)


_COUNTEREXAMPLES = (
    ("非常安静的酒店", "安静的酒店\uff0c适合休息", "非常热闹的酒店\uff0c适合聚会"),
    ("特别喜欢红茶", "喜欢红茶", "特别喜欢绿茶"),
    ("不喜欢喝浓咖啡", "不爱喝浓咖啡", "不喜欢喝茶"),
    ("not prefer strong coffee", "dislike strong coffee", "not prefer tea"),
    ("never select red tea", "avoid red tea", "never select coffee"),
    ("非常に静かな宿", "静かな宿", "非常に騒がしい宿"),
)


@pytest.mark.parametrize(("query", "answer", "distractor"), _COUNTEREXAMPLES)
def test_surface_fragments_do_not_gain_polarity_privilege(
    tmp_path: Path, query: str, answer: str, distractor: str
) -> None:
    with Memory(tmp_path, embedder=_NegationEmbedder(answer)) as memory:
        desired = memory.add(answer)
        other = memory.add(distractor)

        result = memory.search_with_trace(query, limit=2)

        assert [hit.id for hit in result.hits] == [desired.id, other.id]
        assert all(0.0 < hit.score < 1.0 for hit in result.hits)


def test_an_incidental_surface_fragment_cannot_gain_relevance_admission(tmp_path: Path) -> None:
    with Memory(
        tmp_path, embedder=_NegationEmbedder(cosine=0.45), minimum_relevance=0.55
    ) as memory:
        memory.add("安静的酒店\uff0c适合休息")
        memory.add("非常热闹的酒店\uff0c适合聚会")

        result = memory.search_with_trace("非常安静的酒店", limit=2)

        assert result.hits == ()
        assert len(result.trace.candidates) == 2


@pytest.mark.parametrize(
    ("query", "opposite"),
    (
        ("Mira did not buy red tea", "Mira did buy red tea"),
        ("Mira never buys red tea", "Mira buys red tea"),
        ("Mira orders tea without sugar", "Mira orders tea with sugar"),
    ),
)
def test_literal_english_negation_remains_evidence(
    tmp_path: Path, query: str, opposite: str
) -> None:
    with Memory(tmp_path, embedder=_NegationEmbedder()) as memory:
        desired = memory.add(query)
        other = memory.add(opposite)

        assert [hit.id for hit in memory.search(query, limit=2)] == [desired.id, other.id]
