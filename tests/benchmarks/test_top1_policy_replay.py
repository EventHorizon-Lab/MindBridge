"""Independent denominators and real SDK isolation for selective retrieval policy replay."""

from pathlib import Path

import pytest

from mindbridge.benchmarks.top1_ambiguity import (
    CASES,
    Candidate,
    Observation,
    accepts,
    observe,
    summarize,
)


def test_risk_and_correct_retention_do_not_reward_false_rejections() -> None:
    first = Candidate(0, 0.9, True, frozenset({"Alice"}))
    distant = Candidate(1, 0.4, True, frozenset({"Alice"}))
    close = Candidate(1, 0.899, True, frozenset({"Alice"}))
    terms = frozenset({"Alice"})
    observations = (
        Observation("correct", terms, (0,), (first, distant), True),
        Observation("unsafe", terms, (), (first, distant), True),
        Observation("missed", terms, (0,), (first, close), False),
        Observation("empty", terms, (), (), False),
    )

    assert summarize(observations, "no_exemption", 0.01) == {
        "policy": "no_exemption",
        "margin": 0.01,
        "total": 4,
        "accepted": 2,
        "correct": 1,
        "unsafe": 1,
        "coverage": 0.5,
        "risk": 0.5,
        "correct_retention": 0.5,
        "false_rejections": 1,
        "correct_top_rejections": 1,
    }
    assert summarize(observations, "no_exemption", 1.0)["risk"] is None


def test_coverage_exemption_requires_a_term_absent_from_all_close_rivals() -> None:
    terms = frozenset({"alice", "bakery", "tuesday"})
    first = Candidate(0, 0.99, True, terms)
    second = Candidate(1, 0.989, True, frozenset({"alice", "bakery"}))
    third = Candidate(2, 0.988, True, frozenset({"alice", "tuesday"}))
    observation = Observation("distributed-coverage", terms, (), (first, second, third), False)
    assert accepts(observation, "route_exemption", 0.01)
    assert not accepts(observation, "coverage_exemption", 0.01)


@pytest.mark.parametrize("name", ("same-name", "current-validity", "question-echo"))
def test_public_sdk_probe_is_independent_of_insertion_order(tmp_path: Path, name: str) -> None:
    case = next(case for case in CASES if case.name == name)
    forward = observe(case, tmp_path / "forward")
    backward = observe(case, tmp_path / "backward", reverse=True)
    assert [(item.document, item.lexical) for item in forward.ranked] == [
        (item.document, item.lexical) for item in backward.ranked
    ]
    assert forward.sdk_accepted == backward.sdk_accepted
    assert forward.sdk_accepted == accepts(forward, "no_exemption", 0.01)
