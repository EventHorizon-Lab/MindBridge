"""Bootstrap acceleration must preserve draws, weighting, and floating-point results."""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from mindbridge.benchmarks import eval_statistics as statistics


def _reference_draws(
    clusters: Sequence[Sequence[float]], *, seed: int, samples: int
) -> tuple[float, ...]:
    generator = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = tuple(clusters[generator.randrange(len(clusters))] for _ in clusters)
        flattened = tuple(score for cluster in selected for score in cluster)
        draws.append(sum(flattened) / len(flattened))
    return tuple(sorted(draws))


@pytest.mark.parametrize("seed", (0, 7, 1234))
@pytest.mark.parametrize(
    "clusters",
    (
        ((1.0, 0.0, 1.0), (0.0,), (1.0, 1.0)),
        ((-2.0, 5.0), (2.0, -1.0, 0.0), (3.0,)),
        ((0.1, 0.3, 0.7), (0.2,), (1 / 3, 2 / 3)),
        ((1e16, 1.0, -1e16), (1.0,), (-1e16, 1e16, -1.0)),
        ((float(2**53 // 3), 1.0), (-float(2**53 // 3), -1.0), (1.0,)),
    ),
)
def test_bootstrap_matches_original_draws_exactly(
    clusters: tuple[tuple[float, ...], ...], seed: int
) -> None:
    assert statistics._cluster_bootstrap(clusters, seed=seed, samples=257) == _reference_draws(
        clusters, seed=seed, samples=257
    )


def test_paired_probability_and_interval_share_one_set_of_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = (
        statistics.ScoredValue("a", "first", 1.0),
        statistics.ScoredValue("b", "first", 0.0),
        statistics.ScoredValue("c", "second", 0.0),
        statistics.ScoredValue("d", "third", 1.0),
    )
    baseline = tuple(
        statistics.ScoredValue(value.sample_id, value.cluster_id, 0.5) for value in candidate
    )
    draws = _reference_draws(((0.5, -0.5), (-0.5,), (0.5,)), seed=7, samples=257)
    calls = 0
    original = statistics._cluster_bootstrap

    def bootstrap(
        clusters: Sequence[Sequence[float]], *, seed: int, samples: int
    ) -> tuple[float, ...]:
        nonlocal calls
        calls += 1
        return original(clusters, seed=seed, samples=samples)

    monkeypatch.setattr(statistics, "_cluster_bootstrap", bootstrap)
    result = statistics.paired_comparison(candidate, baseline, seed=7, bootstrap_samples=257)

    assert calls == 1
    assert result["confidence_interval_95"] == [
        statistics.percentile(draws, 0.025, presorted=True),
        statistics.percentile(draws, 0.975, presorted=True),
    ]
    assert result["probability_of_improvement"] == sum(draw > 0 for draw in draws) / len(draws)
