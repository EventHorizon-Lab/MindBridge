"""Provider billing remains unknown until complete usage has actually been observed."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import httpx2 as httpx
import pytest


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = Path(__file__).parents[3] / "benchmarks/support_coverage.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("support_coverage_billing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module._reset_billing()
    return module


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"usage": None},
        {"usage": []},
        {"usage": {"completion_tokens": 3}},
        {"usage": {"prompt_tokens": None}},
        {"usage": {"prompt_tokens": "5"}},
        {"usage": {"prompt_tokens": 1.5}},
        {"usage": {"prompt_tokens": True}},
        {"usage": {"prompt_tokens": -1}},
        [],
    ],
)
def test_missing_or_malformed_billing_is_unknown(driver: ModuleType, payload: object) -> None:
    driver._billing_hook(httpx.Response(200, json=payload))
    assert driver._billed_tokens() is None


def test_unobserved_billing_is_unknown(driver: ModuleType) -> None:
    assert driver._billed_tokens() is None


def test_observed_zero_is_valid_and_counts_accumulate(driver: ModuleType) -> None:
    driver._billing_hook(httpx.Response(200, json={"usage": {"prompt_tokens": 0}}))
    assert driver._billed_tokens() == 0
    driver._billing_hook(httpx.Response(200, json={"usage": {"prompt_tokens": 7}}))
    assert driver._billed_tokens() == 7


@pytest.mark.parametrize("counts", [(5, None), (None, 5), (5, None, 8)])
def test_partial_billing_stays_unknown(driver: ModuleType, counts: tuple[int | None, ...]) -> None:
    for count in counts:
        driver._billing_hook(httpx.Response(200, json={"usage": {"prompt_tokens": count}}))
    assert driver._billed_tokens() is None


def test_non_json_usage_does_not_fail_answer(driver: ModuleType) -> None:
    driver._billing_hook(httpx.Response(200, content=b"not json"))
    assert driver._billed_tokens() is None


def test_unexpected_hook_bug_propagates(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = httpx.Response(200, json={})

    def fail() -> bytes:
        raise RuntimeError("local accounting failed")

    monkeypatch.setattr(response, "read", fail)
    with pytest.raises(RuntimeError, match="local accounting failed"):
        driver._billing_hook(response)


@pytest.mark.parametrize("counts, expected", [((None, 8), None), ((0, 8), 4.0), ((0, 0), 0.0)])
def test_billing_summary_preserves_unknowns(
    driver: ModuleType, counts: tuple[int | None, ...], expected: float | None
) -> None:
    outcomes = [
        driver.Outcome(
            question_id=str(index),
            arm="window",
            support="complete_in_window",
            gold=1,
            found=1,
            window_rows=1,
            window_chars=10,
            expansion_rows=0,
            prompt_chars=10,
            evidence_cost=10,
            prompt_tokens=count,
            media_rows=0,
            correct=1.0,
        )
        for index, count in enumerate(counts)
    ]
    assert driver._summarize(outcomes)["mean_prompt_tokens"] == expected
