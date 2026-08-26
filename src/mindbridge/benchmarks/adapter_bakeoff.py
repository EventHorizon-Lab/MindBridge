"""Measure which Jina task adapter should score consolidation similarity.

Consolidation compares two stored documents with each other and merges above a
threshold. That is symmetric pairwise scoring, which is what the text-matching
adapter is built for, while the deployed vectors come from the retrieval adapter
trained for query-to-document proximity. This runner quantifies the gap on
labeled pairs before anyone pays for a second serving instance.
"""

from __future__ import annotations

import asyncio
import json
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Literal

from pydantic import AwareDatetime, Field

from mindbridge.benchmarks.cli import parser as build_parser
from mindbridge.benchmarks.runtime import dot_product
from mindbridge.contracts import ContractModel, NonEmptyString
from mindbridge.models import EmbedRequest, EmbedTask, ModelInput, TextPart
from mindbridge.models.jina import JinaEmbedder

PairLabel = Literal["paraphrase", "contradiction", "unrelated"]

# The thresholds consolidation ships with today, so the report speaks in the
# same units as production instead of an abstract score.
PRODUCTION_THRESHOLDS = (0.7, 0.8)


class LabeledPair(ContractModel):
    """Two stored statements plus the merge decision a human would make."""

    left: NonEmptyString
    right: NonEmptyString
    label: PairLabel


class PairCorpus(ContractModel):
    """A reviewable, versioned set of consolidation decisions."""

    description: NonEmptyString
    pairs: tuple[LabeledPair, ...] = Field(min_length=4)


class ThresholdOutcome(ContractModel):
    """What one production threshold would actually do to this corpus."""

    threshold: float
    missed_merge_rate: float
    false_merge_rate_contradiction: float
    false_merge_rate_unrelated: float


class AdapterScore(ContractModel):
    """One adapter measured as a symmetric consolidation scorer."""

    model_id: NonEmptyString
    separation_auc: float
    contradiction_auc: float
    mean_paraphrase_similarity: float
    mean_contradiction_similarity: float
    mean_unrelated_similarity: float
    best_threshold: float
    best_threshold_youden_j: float
    thresholds: tuple[ThresholdOutcome, ...]


class AdapterBakeoffResult(ContractModel):
    """Output needed to reproduce an adapter comparison."""

    created_at: AwareDatetime
    corpus_description: NonEmptyString
    pair_count: int
    paraphrase_count: int
    contradiction_count: int
    unrelated_count: int
    device: NonEmptyString
    dimension: int
    retrieval: AdapterScore
    text_matching: AdapterScore
    contradiction_auc_gain: float
    python_version: NonEmptyString
    torch_version: NonEmptyString
    sentence_transformers_version: NonEmptyString


async def run_adapter_bakeoff(
    *,
    corpus: PairCorpus,
    retrieval_model_id: str,
    text_matching_model_id: str,
    device: str,
    dimension: int,
) -> AdapterBakeoffResult:
    """Score every pair with both adapters using the symmetric document side."""
    scores: list[AdapterScore] = []
    for model_id in (retrieval_model_id, text_matching_model_id):
        # This sweep loads whichever repository the caller named, and `load` resolves the pin
        # from the model id: the bundled repository still gets its commit -- which is the
        # default of `--retrieval-model-id`, so the common invocation stays pinned under
        # `trust_remote_code=True` -- and any other repository, which that commit could not
        # resolve against, gets its default branch.
        embedder = JinaEmbedder.load(model_id=model_id, device=device, dimension=dimension)
        try:
            scores.append(await _score_adapter(embedder, corpus, model_id))
        finally:
            await _close(embedder)

    retrieval, text_matching = scores
    labels = [pair.label for pair in corpus.pairs]
    return AdapterBakeoffResult(
        created_at=datetime.now(timezone.utc),
        corpus_description=corpus.description,
        pair_count=len(corpus.pairs),
        paraphrase_count=labels.count("paraphrase"),
        contradiction_count=labels.count("contradiction"),
        unrelated_count=labels.count("unrelated"),
        device=device,
        dimension=dimension,
        retrieval=retrieval,
        text_matching=text_matching,
        contradiction_auc_gain=text_matching.contradiction_auc - retrieval.contradiction_auc,
        python_version=platform.python_version(),
        torch_version=version("torch"),
        sentence_transformers_version=version("sentence-transformers"),
    )


async def _score_adapter(
    embedder: JinaEmbedder,
    corpus: PairCorpus,
    model_id: str,
) -> AdapterScore:
    """Encode both sides as documents, because consolidation has no query side."""
    texts = [side for pair in corpus.pairs for side in (pair.left, pair.right)]
    result = await embedder.embed(
        EmbedRequest(
            inputs=tuple(ModelInput((TextPart(text),)) for text in texts),
            task=EmbedTask.DOCUMENT,
        )
    )
    vectors = [embedding.values for embedding in result.embeddings]
    similarities = [
        dot_product(vectors[index * 2], vectors[index * 2 + 1])
        for index in range(len(corpus.pairs))
    ]
    by_label: dict[PairLabel, list[float]] = {
        "paraphrase": [],
        "contradiction": [],
        "unrelated": [],
    }
    for pair, similarity in zip(corpus.pairs, similarities, strict=True):
        by_label[pair.label].append(similarity)

    positives = by_label["paraphrase"]
    negatives = by_label["contradiction"] + by_label["unrelated"]
    best_threshold, best_j = _best_threshold(positives, negatives)
    return AdapterScore(
        model_id=model_id,
        separation_auc=_roc_auc(positives, negatives),
        contradiction_auc=_roc_auc(positives, by_label["contradiction"]),
        mean_paraphrase_similarity=fmean(positives),
        mean_contradiction_similarity=fmean(by_label["contradiction"]),
        mean_unrelated_similarity=fmean(by_label["unrelated"]),
        best_threshold=best_threshold,
        best_threshold_youden_j=best_j,
        thresholds=tuple(
            ThresholdOutcome(
                threshold=threshold,
                missed_merge_rate=1.0 - _rate_at_least(positives, threshold),
                false_merge_rate_contradiction=_rate_at_least(by_label["contradiction"], threshold),
                false_merge_rate_unrelated=_rate_at_least(by_label["unrelated"], threshold),
            )
            for threshold in PRODUCTION_THRESHOLDS
        ),
    )


def _roc_auc(positives: list[float], negatives: list[float]) -> float:
    """Rank-based AUC; ties count as half a win."""
    if not positives or not negatives:
        raise ValueError("AUC needs at least one positive and one negative pair")
    # ponytail: O(n*m) is fine for a curated corpus; sort-and-rank if it ever grows.
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _best_threshold(positives: list[float], negatives: list[float]) -> tuple[float, float]:
    """Return the cut maximizing true-merge rate minus false-merge rate."""
    best_threshold, best_j = 0.0, -1.0
    for candidate in sorted({*positives, *negatives}):
        youden_j = _rate_at_least(positives, candidate) - _rate_at_least(negatives, candidate)
        if youden_j > best_j:
            best_threshold, best_j = candidate, youden_j
    return best_threshold, best_j


def _rate_at_least(values: list[float], threshold: float) -> float:
    """Share of pairs this threshold would merge."""
    if not values:
        return 0.0
    return sum(1 for value in values if value >= threshold) / len(values)


async def _close(embedder: object) -> None:
    close = getattr(embedder, "close", None)
    if close is not None:
        await close()


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Compare both adapters and emit a machine-readable manifest."""
    parser = build_parser(prog=prog, description=__doc__)
    parser.add_argument(
        "--pairs", type=Path, required=True, help="prepared pair corpus both adapters answer"
    )
    parser.add_argument(
        "--retrieval-model-id",
        default="jinaai/jina-embeddings-v5-omni-small-retrieval",
        help="Hugging Face repository of the retrieval adapter",
    )
    parser.add_argument(
        "--text-matching-model-id",
        default="jinaai/jina-embeddings-v5-omni-small-text-matching",
        help="Hugging Face repository of the text-matching adapter",
    )
    parser.add_argument("--device", default="cuda", help="torch device to load both adapters on")
    parser.add_argument(
        "--dimension", type=int, default=1_024, help="Matryoshka dimension to compare at"
    )
    arguments = parser.parse_args(argv)

    corpus = PairCorpus.model_validate_json(arguments.pairs.read_text(encoding="utf-8"))
    result = asyncio.run(
        run_adapter_bakeoff(
            corpus=corpus,
            retrieval_model_id=arguments.retrieval_model_id,
            text_matching_model_id=arguments.text_matching_model_id,
            device=arguments.device,
            dimension=arguments.dimension,
        )
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
