"""Deterministic benchmark scoring, uncertainty, and paired comparisons."""

from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

_DIRECT_CHOICE = re.compile(r"(?:option\s+)?[([]?([A-J])[]).]?", re.IGNORECASE)
_EXPLICIT_CHOICE = re.compile(
    r"\b(?:answer|choice|option|final)(?:\s+is|\s*[:=])?\s*[([]?([A-J])\b",
    re.IGNORECASE,
)
_LEADING_CHOICE = re.compile(
    r"^\s*[([]?([A-J])(?:[]).,:-]|\s+(?=(?:because|as|since)\b))", re.IGNORECASE
)
_CORRECT_CHOICE = re.compile(
    r"\b([A-J])\b\s+(?:is|was|would be)\s+(?:the\s+)?(?:answer|correct)\b", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class ScoredValue:
    """One finite score and the independent unit it belongs to."""

    sample_id: str
    cluster_id: str
    value: float

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.cluster_id.strip():
            raise ValueError("sample and cluster IDs must be non-empty")
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise ValueError("sample scores must be numeric")
        if not math.isfinite(self.value):
            raise ValueError("sample scores must be finite")


def parse_choice(response: str, choices: Sequence[str] = ()) -> str | None:
    """Extract an option label, falling back to an exact option-text match."""
    normalized = response.strip()
    allowed = "ABCDEFGHIJ"[: len(choices)] if choices else "ABCDEFGHIJ"
    direct = _DIRECT_CHOICE.fullmatch(normalized.strip("*_`"))
    if direct is not None and direct.group(1).upper() in allowed:
        return direct.group(1).upper()
    for pattern in (_EXPLICIT_CHOICE, _LEADING_CHOICE, _CORRECT_CHOICE):
        matches = pattern.findall(normalized)
        for match in reversed(matches):
            label = str(match).upper()
            if label in allowed:
                return label
    answer = normalize_text(normalized)
    for label, choice in zip("ABCDEFGHIJ", choices, strict=False):
        if answer == normalize_text(choice):
            return label
    return None


def token_f1(prediction: str, references: Sequence[str]) -> float:
    """Return the best deterministic bag-of-token F1 across accepted references."""
    predicted = normalize_text(prediction).split()
    if not references:
        raise ValueError("token F1 needs at least one reference")
    return max(_token_f1(predicted, normalize_text(reference).split()) for reference in references)


def exact_match(prediction: str, references: Sequence[str]) -> float:
    """Return normalized exact match across accepted references."""
    normalized = normalize_text(prediction)
    return float(any(normalized == normalize_text(reference) for reference in references))


def normalize_text(value: str) -> str:
    """Normalize case, Unicode, punctuation, and whitespace without language-specific stemming."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalized).split()
    )


def summarize(
    values: Sequence[ScoredValue],
    *,
    seed: int,
    bootstrap_samples: int,
    clamp: tuple[float, float] | None = (0.0, 1.0),
) -> dict[str, object]:
    """Summarize a mean with cluster-robust SE and a deterministic cluster bootstrap CI."""
    if not values:
        return {
            "sample_count": 0,
            "cluster_count": 0,
            "mean": None,
            "cluster_standard_error": None,
            "confidence_interval_95": None,
            "normal_confidence_interval_95": None,
            "bootstrap_samples": bootstrap_samples,
        }
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    scores = tuple(value.value for value in values)
    mean = sum(scores) / len(scores)
    clusters = _clusters(values)
    standard_error = _cluster_standard_error(clusters, mean, len(scores))
    interval = None
    if len(clusters) > 1:
        bootstrapped = _cluster_bootstrap(clusters, seed=seed, samples=bootstrap_samples)
        low = cast(float, percentile(bootstrapped, 0.025))
        high = cast(float, percentile(bootstrapped, 0.975))
        if clamp is not None:
            low, high = max(clamp[0], low), min(clamp[1], high)
        interval = [low, high]
    normal = None
    if standard_error is not None:
        normal_low = mean - 1.959963984540054 * standard_error
        normal_high = mean + 1.959963984540054 * standard_error
        if clamp is not None:
            normal_low = max(clamp[0], normal_low)
            normal_high = min(clamp[1], normal_high)
        normal = [normal_low, normal_high]
    return {
        "sample_count": len(scores),
        "cluster_count": len(clusters),
        "mean": mean,
        "cluster_standard_error": standard_error,
        "confidence_interval_95": interval,
        "normal_confidence_interval_95": normal,
        "bootstrap_samples": bootstrap_samples,
    }


def paired_comparison(
    candidate: Sequence[ScoredValue],
    baseline: Sequence[ScoredValue],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    """Join identical samples and report a paired, cluster-aware score delta."""
    previous = {value.sample_id: value for value in baseline}
    current = {value.sample_id: value for value in candidate}
    if len(previous) != len(baseline) or len(current) != len(candidate):
        raise ValueError("candidate and baseline sample IDs must be unique")
    if current.keys() != previous.keys():
        raise ValueError("candidate and baseline must contain identical scored samples")
    if any(current[key].cluster_id != previous[key].cluster_id for key in current):
        raise ValueError("candidate and baseline cluster IDs differ")
    paired = tuple(
        ScoredValue(
            value.sample_id, value.cluster_id, value.value - previous[value.sample_id].value
        )
        for value in candidate
    )
    if not paired:
        raise ValueError("candidate and baseline contain no scored samples")
    deltas = tuple(value.value for value in paired)
    summary = summarize(
        paired,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        clamp=None,
    )
    clusters = _clusters(paired)
    draws = (
        _cluster_bootstrap(clusters, seed=seed, samples=bootstrap_samples)
        if len(clusters) > 1
        else ()
    )
    return {
        **summary,
        "paired_sample_count": len(paired),
        "win_count": sum(delta > 0 for delta in deltas),
        "tie_count": sum(delta == 0 for delta in deltas),
        "loss_count": sum(delta < 0 for delta in deltas),
        "probability_of_improvement": (
            sum(draw > 0 for draw in draws) / len(draws) if draws else None
        ),
    }


def _token_f1(predicted: Sequence[str], reference: Sequence[str]) -> float:
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0
    remaining: dict[str, int] = defaultdict(int)
    for token in reference:
        remaining[token] += 1
    overlap = 0
    for token in predicted:
        if remaining[token] > 0:
            remaining[token] -= 1
            overlap += 1
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _clusters(values: Iterable[ScoredValue]) -> tuple[tuple[float, ...], ...]:
    grouped: dict[str, list[float]] = {}
    for value in values:
        grouped.setdefault(value.cluster_id, []).append(value.value)
    return tuple(tuple(grouped[key]) for key in sorted(grouped))


def _cluster_standard_error(
    clusters: Sequence[Sequence[float]], mean: float, sample_count: int
) -> float | None:
    count = len(clusters)
    if count < 2:
        return None
    residuals = tuple(sum(cluster) - len(cluster) * mean for cluster in clusters)
    variance = count / (count - 1) * sum(value * value for value in residuals) / sample_count**2
    return math.sqrt(max(variance, 0.0))


def _cluster_bootstrap(
    clusters: Sequence[Sequence[float]], *, seed: int, samples: int
) -> tuple[float, ...]:
    generator = random.Random(seed)
    count = len(clusters)
    draws = []
    for _ in range(samples):
        selected = tuple(clusters[generator.randrange(count)] for _ in range(count))
        flattened = tuple(score for cluster in selected for score in cluster)
        draws.append(sum(flattened) / len(flattened))
    return tuple(sorted(draws))


def percentile(values: Sequence[float], probability: float) -> float | None:
    """Return the linearly interpolated quantile, or ``None`` for no observations."""
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
