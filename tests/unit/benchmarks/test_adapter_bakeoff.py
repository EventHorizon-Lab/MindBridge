"""Checks for the consolidation adapter bake-off runner."""

import math
from pathlib import Path
from typing import cast

import pytest

from mindbridge.benchmarks.adapter_bakeoff import (
    AdapterScore,
    LabeledPair,
    PairCorpus,
    ThresholdOutcome,
    _best_threshold,
    _roc_auc,
    _score_adapter,
    run_adapter_bakeoff,
)
from mindbridge.core import EmbeddingSpaceReference, ModelReference
from mindbridge.models import Embedding, EmbedRequest, EmbedResult
from mindbridge.models.jina import SentenceTransformersEmbedder

CORPUS_PATH = Path(__file__).parents[3] / "benchmarks" / "adapter-bakeoff-pairs.json"


def test_committed_pair_corpus_matches_current_schema() -> None:
    """The reviewable corpus cannot drift from the runner that consumes it."""
    corpus = PairCorpus.model_validate_json(CORPUS_PATH.read_text(encoding="utf-8"))

    labels = [pair.label for pair in corpus.pairs]
    assert labels.count("paraphrase") >= 4
    assert labels.count("contradiction") >= 4
    assert labels.count("unrelated") >= 4
    assert all(pair.left != pair.right for pair in corpus.pairs)


def test_roc_auc_scores_perfect_and_inverted_separation() -> None:
    assert _roc_auc([0.9, 0.8], [0.2, 0.1]) == 1.0
    assert _roc_auc([0.1, 0.2], [0.8, 0.9]) == 0.0
    assert _roc_auc([0.5], [0.5]) == 0.5


def test_roc_auc_rejects_a_corpus_missing_one_side() -> None:
    with pytest.raises(ValueError, match="one positive and one negative"):
        _roc_auc([0.9], [])


def test_best_threshold_separates_a_cleanly_split_corpus() -> None:
    threshold, youden_j = _best_threshold([0.9, 0.85], [0.4, 0.3])

    assert threshold == pytest.approx(0.85)
    assert youden_j == pytest.approx(1.0)


async def test_score_adapter_reports_production_threshold_outcomes() -> None:
    """A contradiction scoring above the shipped cut must surface as a false merge."""
    corpus = PairCorpus(
        description="unit corpus",
        pairs=(
            LabeledPair(left="a", right="a-restated", label="paraphrase"),
            LabeledPair(left="b", right="b-negated", label="contradiction"),
            LabeledPair(left="c", right="c-unrelated", label="unrelated"),
            LabeledPair(left="d", right="d-restated", label="paraphrase"),
        ),
    )
    # Cosine of each pair is fully determined by the angle assigned per text.
    angles = {
        "a": 0.0,
        "a-restated": 0.0,  # similarity 1.00 -> merges at both thresholds
        "b": 0.0,
        "b-negated": 0.60,  # similarity ~0.83 -> false merge at 0.7 and 0.8
        "c": 0.0,
        "c-unrelated": 1.4,  # similarity ~0.17 -> correctly rejected
        "d": 0.0,
        "d-restated": 0.90,  # similarity ~0.62 -> missed merge at both thresholds
    }

    score = await _score_adapter(
        cast(SentenceTransformersEmbedder, _AngleEmbedder(angles)), corpus, "unit-model"
    )

    assert _outcome(score, 0.7).false_merge_rate_contradiction == pytest.approx(1.0)
    assert _outcome(score, 0.7).false_merge_rate_unrelated == pytest.approx(0.0)
    assert _outcome(score, 0.7).missed_merge_rate == pytest.approx(0.5)
    assert _outcome(score, 0.8).false_merge_rate_contradiction == pytest.approx(1.0)
    assert score.mean_contradiction_similarity > score.mean_unrelated_similarity
    # Paraphrase 0.62 sits below contradiction 0.83, so ranking is partly inverted.
    assert score.contradiction_auc == pytest.approx(0.5)


async def test_the_bakeoff_opts_both_adapters_into_their_repository_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both Jina adapters keep their weights behind repository code.

    Loading them without `trust_remote_code` leaves transformers refusing the config with
    "requires you to execute the configuration file", after the corpus has been read and the
    device chosen -- and no CI job installs this extra, so nothing else would report it.
    """
    loads: list[dict[str, object]] = []

    class _Loader:
        @staticmethod
        def load(**kwargs: object) -> _AngleEmbedder:
            loads.append(kwargs)
            return _AngleEmbedder(
                {
                    "a": 0.0,
                    "a-restated": 0.0,
                    "b": 0.0,
                    "b-negated": 0.60,
                    "c": 0.0,
                    "c-unrelated": 1.4,
                    "d": 0.0,
                    "d-restated": 0.90,
                }
            )

    monkeypatch.setattr(
        "mindbridge.benchmarks.adapter_bakeoff.SentenceTransformersEmbedder", _Loader
    )

    await run_adapter_bakeoff(
        corpus=PairCorpus(
            description="unit corpus",
            pairs=(
                LabeledPair(left="a", right="a-restated", label="paraphrase"),
                LabeledPair(left="b", right="b-negated", label="contradiction"),
                LabeledPair(left="c", right="c-unrelated", label="unrelated"),
                LabeledPair(left="d", right="d-restated", label="paraphrase"),
            ),
        ),
        retrieval_model_id="jinaai/jina-embeddings-v5-omni-small-retrieval",
        text_matching_model_id="jinaai/jina-embeddings-v5-omni-small-text-matching",
        text_matching_model_revision="0123456789abcdef0123456789abcdef01234567",
        device="cpu",
        dimension=1_024,
    )

    assert [load["trust_remote_code"] for load in loads] == [True, True]
    assert [load["revision"] for load in loads] == [
        None,
        "0123456789abcdef0123456789abcdef01234567",
    ]


def _outcome(score: AdapterScore, threshold: float) -> ThresholdOutcome:
    return next(item for item in score.thresholds if item.threshold == threshold)


class _AngleEmbedder:
    """Returns unit vectors whose pairwise cosine is exactly cos(angle delta)."""

    def __init__(self, angles: dict[str, float]) -> None:
        self._angles = angles

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        vectors = []
        for input_value in request.inputs:
            text = "".join(part.text for part in input_value.parts if hasattr(part, "text"))
            angle = self._angles[text]
            vectors.append((math.cos(angle), math.sin(angle)) + (0.0,) * 1_022)
        return EmbedResult(
            tuple(
                Embedding(
                    vector,
                    ModelReference(model_id="unit-model"),
                    EmbeddingSpaceReference(space_id="unit-space"),
                )
                for vector in vectors
            )
        )
