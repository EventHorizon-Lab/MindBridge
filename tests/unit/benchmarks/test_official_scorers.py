"""Parity fixtures for benchmark-native deterministic and judge scorers."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

import mindbridge.benchmarks.eval as eval_module
from mindbridge.benchmarks.eval import SampleResult, _apply_judges, _metrics
from mindbridge.benchmarks.eval_adapters import EvalQuestion, EvalUnit, LoadedTask
from mindbridge.benchmarks.official_scorers import (
    JudgeMessage,
    JudgePlan,
    judge_model_is_official,
    judge_plan,
    local_scores,
    parse_judge_response,
)
from mindbridge.benchmarks.task_catalog import TaskSpec


def _scores(
    task: str,
    prediction: str,
    reference: str,
    metadata: dict[str, object],
    *,
    question: str = "What happened?",
    evidence: tuple[str, ...] = (),
) -> dict[str, float]:
    return local_scores(
        task,
        score_kind="text",
        prediction=prediction,
        parsed_choice=None,
        expected_choice=None,
        references=(reference,),
        question=question,
        metadata=metadata,
        evidence_source_ids=evidence,
    )


def test_deterministic_scorers_match_upstream_edge_cases() -> None:
    assert _scores("locomo-refined", "A shell necklace!", "A shell necklace", {}) == {
        "token_f1": pytest.approx(6 / 7),
        "bleu_1": pytest.approx(0.75),
    }
    assert (
        _scores(
            "atm-bench-main",
            "2025-07-25",
            "July 25, 2025",
            {"qtype": "number"},
        )["accuracy"]
        == 1.0
    )
    assert _scores(
        "atm-bench-main",
        "email000000000001 and email000000000002 and email000000000003",
        "email000000000001, email000000000002",
        {"qtype": "list_recall"},
    )["accuracy"] == pytest.approx(2 / 3)
    gallery = _scores(
        "mem-gallery",
        "running",
        "runs",
        {"clue_ids": ("r1", "r2")},
        evidence=("r1", "wrong"),
    )
    assert gallery == {
        "f1": 1.0,
        "bleu": 0.0,
        "bleu_1": 0.0,
        "bleu_2": 0.0,
        "exact_match": 0.0,
        "retrieval_precision@10": 0.5,
        "retrieval_recall@10": 0.5,
        "retrieval_hit@10": 1.0,
    }

    atm = _scores(
        "atm-bench-main",
        "correct answer",
        "correct answer",
        {"qtype": "number", "evidence_ids": ("a", "b")},
        evidence=("a", "wrong", "b"),
    )
    assert atm == {
        "accuracy": 1.0,
        "retrieval_recall@1": 0.5,
        "retrieval_recall@5": 1.0,
        "retrieval_recall@10": 1.0,
        "retrieval_recall@gt": 0.5,
        "retrieval_hit@1": 1.0,
        "joint_partial@5": 1.0,
        "joint_strict@5": 1.0,
        "joint_partial@10": 1.0,
        "joint_strict@10": 1.0,
    }


def test_official_judge_response_parsers_keep_upstream_mappings() -> None:
    message = ((JudgeMessage("user", "prompt"),),)
    assert parse_judge_response(JudgePlan("p", "locomo", message), '{"label":"CORRECT"}') == {
        "llm_judge": 1.0
    }
    assert parse_judge_response(JudgePlan("p", "m3", message), "Yes") == {"accuracy": 1.0}
    assert parse_judge_response(
        JudgePlan("p", "egotempo", message),
        "{'pred': 'correct', 'score': 4, 'reason': 'ok'}",
    ) == {"accuracy": 1.0, "judge_score_0_5": 4.0}
    assert parse_judge_response(JudgePlan("p", "memlens", message), '{"answer_score": 1}') == {
        "accuracy": 1.0
    }
    assert parse_judge_response(
        JudgePlan("p", "mm_lifelong", message), "Analysis: partial\nFinal Score: 3"
    ) == {"answer_accuracy": 0.5, "judge_score_0_5": 3.0}
    assert parse_judge_response(
        JudgePlan("p", "atm", message), '{"accuracy": true, "explanation": "ok"}'
    ) == {"accuracy": 1.0}
    # The released Mem-Gallery parser collapses its five-level prompt to 0/0.5/1.
    assert parse_judge_response(
        JudgePlan("p", "gallery", message), '{"score": 0.75, "reasoning": "good"}'
    ) == {"llm_judge": 1.0}
    assert parse_judge_response(JudgePlan("p", "gallery", message), '{"score": "0.5"}') == {
        "llm_judge": 0.5
    }


def test_memlens_and_locomo_plans_preserve_official_protocol_details() -> None:
    locomo = judge_plan(
        "locomo-refined",
        question="What did I buy?",
        references=("A shell", "A necklace"),
        prediction="A shell necklace",
        metadata={},
    )
    assert locomo is not None
    assert len(locomo.calls) == 2
    assert locomo.extra_body == {"enable_thinking": False}
    assert '"label"' in locomo.calls[0][0].content

    memlens = judge_plan(
        "memlens-32k",
        question="What is the latest answer?",
        references=("New",),
        prediction="New",
        metadata={
            "question_type": "knowledge_update",
            "question_subtype": "",
            "old_answer": "Old",
        },
    )
    assert memlens is not None
    assert memlens.details == {"task_key": "KU_KnowledgeUpdate"}
    assert "<Old (Outdated) Answer>: Old" in memlens.calls[0][0].content
    assert judge_model_is_official("locomo-refined", "Qwen/Qwen3-14B")
    assert not judge_model_is_official("locomo-refined", "qwen3.8-27b")

    assert _scores(
        "locomo-refined",
        r"reasoning\nFinal Answer: ignored \boxed{A {shell} necklace}",
        "A {shell} necklace",
        {},
    ) == {"llm_judge": 1.0, "token_f1": 1.0, "bleu_1": 1.0}


def test_m3_plan_preserves_the_official_system_message() -> None:
    plan = judge_plan(
        "m3-bench-robot",
        question="What happened?",
        references=("The robot moved.",),
        prediction="It moved.",
        metadata={},
    )

    assert plan is not None
    assert plan.calls[0][0] == JudgeMessage("system", "You are an expert in video understanding.")
    assert plan.calls[0][1].role == "user"


@pytest.mark.asyncio
async def test_atm_official_judge_uses_minimal_reasoning() -> None:
    calls: list[dict[str, object]] = []

    class Responses:
        async def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(output_text='{"accuracy": true}')

    plan = judge_plan(
        "atm-bench-main",
        question="What happened?",
        references=("An event.",),
        prediction="An event occurred.",
        metadata={"qtype": "open_end"},
    )
    assert plan is not None
    sample = SampleResult(
        task="atm-bench-main",
        benchmark="ATM-Bench",
        dataset_sha256="1" * 64,
        evaluation_sha256="2" * 64,
        unit_id="u1",
        question_id="q1",
        prediction="An event occurred.",
        parsed_choice=None,
        score=None,
        exact_match=None,
        latency_ms=1.0,
        confidence=0.0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=None,
        metadata={"qtype": "open_end"},
    )

    scores, _, _ = await eval_module._judge_call(
        cast(Any, SimpleNamespace(responses=Responses())),
        plan.calls[0],
        sample=sample,
        plan=plan,
        call_index=0,
        cache=None,
        semaphore=asyncio.Semaphore(1),
        config=eval_module._JudgeConfig(
            model="gpt-5-mini",
            base_url="https://api.openai.com/v1",
        ),
    )

    assert scores == {"accuracy": 1.0}
    assert calls[0]["reasoning"] == {"effort": "minimal"}
    assert calls[0]["max_output_tokens"] == 600


def test_locomo_uses_all_metrics_from_the_llm_selected_reference() -> None:
    plan = judge_plan(
        "locomo-refined",
        question="What did I buy?",
        references=("shell", "shell necklace"),
        prediction="shell necklace today",
        metadata={},
    )
    assert plan is not None
    scores = (
        {"llm_judge": 1.0},
        {"llm_judge": 0.0},
    )

    from mindbridge.benchmarks.official_scorers import combine_judge_scores

    assert combine_judge_scores(plan, scores) == {
        "llm_judge": 1.0,
        "token_f1": 0.5,
        "bleu_1": pytest.approx(1 / 3),
    }


def test_results_mark_proxy_judge_metrics_nonofficial(tmp_path: Path) -> None:
    question = EvalQuestion(
        "q1",
        ("prompt",),
        ("answer",),
        source_question="question",
    )
    task = LoadedTask(
        TaskSpec(
            "locomo-refined",
            "LoCoMo-Refined",
            "fixture.json",
            "v1",
            "owner/repo",
            "0" * 40,
        ),
        tmp_path / "fixture.json",
        "1" * 64,
        (EvalUnit("u1", (), (question,)),),
    )
    sample = SampleResult(
        task="locomo-refined",
        benchmark="LoCoMo-Refined",
        dataset_sha256="1" * 64,
        evaluation_sha256="2" * 64,
        unit_id="u1",
        question_id="q1",
        prediction="answer",
        parsed_choice=None,
        score=1.0,
        exact_match=None,
        latency_ms=1.0,
        confidence=0.0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=None,
        metadata={},
        metrics={"llm_judge": 1.0, "token_f1": 1.0},
        scorer_protocol="locomo_refined_judge_887091190789",
        judge_model="qwen3.8-27b",
    )
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(seed=7, bootstrap_samples=20),
    )

    result = _metrics(task, (sample,), arguments)
    metrics = cast(dict[str, dict[str, object]], result["metrics"])

    assert result["official_metric"] is False
    assert result["judge_model_official"] is False
    assert metrics["llm_judge"]["official_metric"] is False
    assert metrics["token_f1"]["official_metric"] is True


@pytest.mark.asyncio
async def test_unified_eval_applies_and_records_the_official_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class Completions:
        async def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"CORRECT"}'))]
            )

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=Completions())

        async def close(self) -> None:
            return None

    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = Client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    question = EvalQuestion(
        "q1",
        ("generated prompt",),
        ("A shell necklace",),
        source_question="What did I buy?",
    )
    spec = TaskSpec(
        "locomo-refined",
        "LoCoMo-Refined",
        "fixture.json",
        "v1",
        "owner/repo",
        "0" * 40,
    )
    task = LoadedTask(spec, tmp_path / "fixture.json", "1" * 64, (EvalUnit("u1", (), (question,)),))
    sample = SampleResult(
        task="locomo-refined",
        benchmark="LoCoMo-Refined",
        dataset_sha256="1" * 64,
        evaluation_sha256="2" * 64,
        unit_id="u1",
        question_id="q1",
        prediction="A shell necklace!",
        parsed_choice=None,
        score=None,
        exact_match=None,
        latency_ms=1.0,
        confidence=0.0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=None,
        metadata={},
        metrics={"token_f1": 1.0, "bleu_1": 1.0},
        scorer_protocol="locomo_refined_judge_887091190789",
    )
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(quiet=True, use_cache=None, run_id="run", log_samples=True),
    )
    judged = await _apply_judges(
        (task,),
        (sample,),
        arguments=arguments,
        config=eval_module._JudgeConfig(
            model="qwen3-14b",
            base_url="https://judge.example/v1",
            api_key="EMPTY",
        ),
    )

    assert judged[0].score == 1.0
    assert judged[0].metrics["llm_judge"] == 1.0
    assert judged[0].judge_model == "qwen3-14b"
    assert judged[0].judge_response == '["{\\"label\\":\\"CORRECT\\"}"]'
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["extra_body"] == {"enable_thinking": False}
