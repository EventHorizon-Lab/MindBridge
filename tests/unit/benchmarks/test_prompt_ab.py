"""Validation of evidence labels and strict selector parsing, without model calls."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def driver() -> ModuleType:
    path = Path(__file__).parents[3] / "benchmarks/prompt_ab.py"
    spec = importlib.util.spec_from_file_location("prompt_ab", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "ratings",
    [
        [{"id": "C1", "score": 5}, {"id": "C1", "score": 2}],
        [{"id": "C1", "score": True}, {"id": "C2", "score": 2}],
        [{"id": "C1", "score": 5}, {"id": "C3", "score": 2}],
        [{"id": "C1", "score": 5}],
    ],
)
def test_rejects_invalid_selector_output(ratings: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError):
        driver().parse_ratings(json.dumps({"ratings": ratings}), 2)


def test_support_accepts_equivalent_copies_but_requires_complement() -> None:
    module = driver()
    assert module.complete_support("reference", [1, 3, 7, 8])
    assert not module.complete_support("reference", [0, 2, 3, 4])
    assert module.complete_support("single_fact", [3, 5, 6, 7])
    assert not module.complete_support("correction", [1, 2, 3, 4])


def test_fixture_has_frozen_size_and_unique_scenarios() -> None:
    cases = driver().scenarios()
    assert len(cases) == len({case["id"] for case in cases}) == 36
    assert all(len(case["records"]) == 10 for case in cases)
    assert len({case["family"] for case in cases}) == 6
