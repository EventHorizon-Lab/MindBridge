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
    combine_judge_scores,
    judge_model_is_official,
    judge_plan,
    local_scores,
    official_judge_model,
    parse_judge_response,
    task_family,
    task_primary_metric,
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


def _judged(
    task: str,
    *,
    question: str,
    references: tuple[str, ...],
    prediction: str,
    metadata: dict[str, object],
    replies: tuple[str, ...],
) -> tuple[JudgePlan, dict[str, float]]:
    plan = judge_plan(
        task,
        question=question,
        references=references,
        prediction=prediction,
        metadata=metadata,
    )
    assert plan is not None
    assert len(plan.calls) == len(replies)
    outcomes = tuple(parse_judge_response(plan, reply) for reply in replies)
    return plan, combine_judge_scores(plan, outcomes)


def test_longmemeval_judge_picks_the_template_its_question_type_names() -> None:
    plan, scores = _judged(
        "longmemeval-s",
        question="What degree did I graduate with?",
        references=("Business Administration",),
        prediction="Business Administration.",
        metadata={"question_type": "temporal-reasoning", "abstention": False},
        replies=("Yes.",),
    )
    # Only the temporal template forgives off-by-one day counts.
    assert "do not penalize off-by-one errors" in plan.calls[0][0].content
    assert scores == {"accuracy": 1.0}

    plan, scores = _judged(
        "longmemeval-s",
        question="Which gate?",
        references=("The itinerary never states it.",),
        prediction="I cannot tell from our conversations.",
        metadata={"question_type": "multi-session", "abstention": True},
        replies=("no",),
    )
    # Abstention swaps in a different template keyed off the `_abs` question ID.
    assert "unanswerable" in plan.calls[0][0].content
    assert scores == {"accuracy": 0.0}

    # The verdict is read as upstream reads it -- `"yes" in lowered` -- which a
    # reply mentioning both words resolves differently from any "no" test.
    _, mixed = _judged(
        "longmemeval-s",
        question="What degree did I graduate with?",
        references=("Business Administration",),
        prediction="Business Administration.",
        metadata={"question_type": "multi-session", "abstention": False},
        replies=("Yes, no doubt about it.",),
    )
    assert mixed == {"accuracy": 1.0}


def test_clbench_judge_is_binary_and_skips_an_empty_answer() -> None:
    _, scores = _judged(
        "clbench",
        question="What do Sighting cards do?",
        references=("Define a Sighting card.", "Name its trigger.", "State the Myth award."),
        prediction="They award Myth and can move Humans.",
        metadata={},
        replies=(
            '{"Grading Rationale": "r", '
            '"List of Requirement Satisfaction Status": ["yes", "no", "yes"], '
            '"Overall Score": 1}',
        ),
    )
    assert scores["solving_rate"] == 1.0
    assert scores["requirement_ratio"] == pytest.approx(2 / 3)

    # `process_single_item` scores an empty output 0 without calling the judge.
    assert (
        judge_plan(
            "clbench",
            question="q",
            references=("r",),
            prediction="   ",
            metadata={},
        )
        is None
    )


def test_beam_judges_each_rubric_item_and_truncates_partial_credit() -> None:
    rubric = ("first", "second", "third", "fourth")
    replies = ('{"score": 1.0}', '{"score": 0.5}', '{"score": 0.0}', '{"score": 1.0}')
    plan, scores = _judged(
        "beam-100k",
        question="Summarise the project",
        references=rubric,
        prediction="A summary.",
        metadata={"category": "summarization"},
        replies=replies,
    )
    # One call per rubric item, and the prompt keeps upstream's unsubstituted
    # `<question>` placeholder.
    assert len(plan.calls) == len(rubric)
    assert "<question>" in plan.calls[0][0].content
    assert "first" in plan.calls[0][0].content
    # Nine categories accumulate with `int(score)`, so 0.5 truncates to 0.
    assert scores == {"llm_judge_score": 0.5}

    _, ordering = _judged(
        "beam-100k",
        question="Order the events",
        references=("first", "second"),
        prediction="An ordering.",
        metadata={"category": "event_ordering"},
        replies=('{"score": 0.5}', '{"score": 1.0}'),
    )
    # `event_ordering` alone accumulates with `float(score)`.
    assert ordering == {"llm_judge_score": 0.75}


def test_personamem_rubric_applies_the_official_deterministic_aggregation() -> None:
    metadata: dict[str, object] = {
        "task_type": "chatbot_personalized_response",
        "groundtruth_preference": "Enjoys affectionate couple-life content.",
        "distractor_preferences": ("Interested in DIY nursery projects",),
        "rubric_tags": ("(+) Reflect the user's relevant preferences.",),
        "judge_evidence": {},
    }
    clean = (
        '{"preference_alignment": 9, "telegraph_avoidance": 4, "avoid_leak_violated": false, '
        '"privacy_leak_violated": false, "stale_preference_use_violated": false}'
    )
    plan, scores = _judged(
        "personamem-v3",
        question="what should i cook tonight",
        references=("pepper stew",),
        prediction="A pepper stew bowl.",
        metadata=metadata,
        replies=(clean,),
    )
    # This task has one positive dim, so the prompt states the single-target
    # wording rather than the 80/20 split, and the judge returns per-dimension
    # scores only -- the arithmetic is applied in code.
    prompt = plan.calls[0][0].content
    assert "The MAIN SCORE for this task is **preference_alignment** alone" in prompt
    assert "what should i cook tonight" in prompt
    assert "Enjoys affectionate couple-life content." in prompt
    # main 9 - 0.5 x (10 - 4) telegraph deduction = 6.
    assert scores["pr_combined_personalization_score"] == 6.0
    # This task's headline excludes the telegraph deduction.
    assert scores["pr_preference_alignment_score_gated"] == 9.0
    assert scores["personamem_score"] == pytest.approx(0.9)

    leaked = clean.replace('"privacy_leak_violated": false', '"privacy_leak_violated": true')
    _, violated = _judged(
        "personamem-v3",
        question="what should i cook tonight",
        references=("pepper stew",),
        prediction="A pepper stew bowl.",
        metadata=metadata,
        replies=(leaked,),
    )
    # A hard rule is one-strike: it zeroes the row whatever the dims say.
    assert violated["pr_combined_personalization_score"] == 0.0
    assert violated["personamem_score"] == 0.0


def test_personamem_ranking_is_deterministic_and_reads_both_answer_shapes() -> None:
    metadata: dict[str, object] = {
        "task_type": "personalized_recommendation",
        "candidate_count": 5,
        "positive_indexes": (2,),
        "negative_indexes": (0, 1),
    }
    # Ranking rows never reach a judge.
    assert (
        judge_plan(
            "personamem-v3",
            question="rank",
            references=("gold",),
            prediction="Ranked indexes: [2, 3, 4, 1, 0]",
            metadata=metadata,
        )
        is None
    )
    hit = _scores("personamem-v3", "Ranked indexes: [2, 3, 4, 1, 0]", "gold", dict(metadata))
    assert hit["recall@1"] == 1.0
    assert hit["personamem_score"] == 1.0
    assert hit["negative_in_top3"] == 0.0

    # The evaluation repository's current prompt asks for this shape instead.
    # The order is deliberately not the identity one, so failing to read it
    # cannot be mistaken for the fallback below.
    reversed_json = '```json\n{"ranked_indices": [4, 3, 2, 1, 0]}\n```'
    miss = _scores("personamem-v3", reversed_json, "gold", dict(metadata))
    assert miss["recall@1"] == 0.0
    assert miss["mrr"] == pytest.approx(1 / 3)
    assert miss["negative_in_top1"] == 0.0
    # The headline is upstream's graded nDCG@5, not top-1: burying the gold at
    # rank 3 costs a position discount rather than the whole score, which is
    # exactly what separates `ndcg_graded@5` from `recall@1` here.
    assert miss["ndcg_graded@5"] == pytest.approx(0.5)
    assert miss["personamem_score"] == pytest.approx(0.5)

    # A reply that is not a permutation of the slate is unusable, and so is an
    # answer with no ranking in it. Both fall back to the identity order, which
    # is what `slate_ranking.run_task_a` does.
    identity = _scores("personamem-v3", "no idea", "gold", dict(metadata))
    assert identity["recall@1"] == 0.0
    assert identity["negative_in_top1"] == 1.0
    assert _scores("personamem-v3", "Ranked indexes: [2, 2, 3]", "gold", dict(metadata)) == identity
    assert _scores("personamem-v3", "Ranked indexes: [2, 0]", "gold", dict(metadata)) == identity


@pytest.mark.parametrize(
    "task_type",
    [
        # No judge family at all.
        "proactive_close_friend_update",
        "proactive_trending_feed_react",
        "restraint_sensitive_event_silence",
        "proactive_overactive_check",
        "active_mistake_prevention",
        # In the rubric's applicability map, but upstream keeps that output as
        # a diagnostic and computes the headline a different way: a leak-set
        # composite, fatigue counters, or a paired-row delta.
        "new_suggestions_chatbot",
        "new_suggestions_recsys",
        "local_recommendation_geo_shift",
        "over_personalization_repetition_chatbot",
        "over_personalization_repetition_recsys",
        # Ranks a slate like the other three, but upstream scores it with a
        # delta across two paired rows that a single row cannot carry.
        "short_vs_long_term_lifecycle",
    ],
)
def test_personamem_leaves_families_it_cannot_reproduce_unscored(task_type: str) -> None:
    metadata: dict[str, object] = {
        "task_type": task_type,
        # Slate fields are supplied so a row cannot be left unscored merely for
        # want of a candidate list.
        "candidate_count": 4,
        "positive_indexes": (1,),
        "negative_indexes": (0,),
        "judge_evidence": {},
    }
    scored = _scores("personamem-v3", "Ranked indexes: [1, 0, 2, 3]", "gold", dict(metadata))
    # A slate-ranking task with no reproducible headline still reports its
    # deterministic diagnostics; it just must not enter the aggregate.
    assert "personamem_score" not in scored
    if task_type != "short_vs_long_term_lifecycle":
        assert scored == {}
    assert (
        judge_plan(
            "personamem-v3",
            question="react?",
            references=("gold",),
            prediction="I would stay quiet.",
            metadata=metadata,
        )
        is None
    )


def test_personamem_scores_exactly_the_reproducible_task_types() -> None:
    """The judged set is EVAL.md's headline table, not the applicability map."""
    from mindbridge.benchmarks._official.personamem_v3_scoring import (
        APPLICABILITY,
        OWN_JUDGE_TASK_TYPES,
        RUBRIC_HEADLINE_TASK_TYPES,
    )

    assert RUBRIC_HEADLINE_TASK_TYPES.issubset(APPLICABILITY)
    assert set(APPLICABILITY) - RUBRIC_HEADLINE_TASK_TYPES
    assert len(RUBRIC_HEADLINE_TASK_TYPES) == 13
    for task_type in RUBRIC_HEADLINE_TASK_TYPES | OWN_JUDGE_TASK_TYPES:
        assert (
            judge_plan(
                "personamem-v3",
                question="q",
                references=("gold",),
                prediction="answer",
                metadata={"task_type": task_type, "judge_evidence": {}},
            )
            is not None
        ), task_type


def test_openeqa_llm_match_selects_its_prompt_and_trims_the_prediction() -> None:
    plain = judge_plan(
        "openeqa-hm3d",
        question="What color is the rug?",
        references=("tan with pink and blue",),
        # `evaluate-predictions.py` cuts after the last period when that period
        # is not already the final character, so the judge never sees the
        # trailing unterminated sentence.
        prediction="brown with pink and blue. I think it",
        metadata={"category": "attribute recognition"},
    )
    assert plain is not None
    prompt = plain.calls[0][0].content
    assert prompt.endswith("Response: brown with pink and blue.\n")
    assert "Extra Answers:" not in prompt
    # `openai_max_tokens=32` in `get_llm_match_score`.
    assert plain.max_tokens == 32
    assert len(plain.calls) == 1

    extra = judge_plan(
        "openeqa-scannet",
        question="Where is the mirror?",
        references=("Next to the staircase",),
        prediction="by the stairs",
        metadata={"extra_answers": ("On the wall", "")},
    )
    assert extra is not None
    extra_prompt = extra.calls[0][0].content
    # Upstream renders the list with `str()`, so the judge sees a Python repr
    # including the one blank entry the release ships.
    assert "Extra Answers: ['On the wall', '']" in extra_prompt
    assert "extra answers that are also correct" in extra_prompt

    assert task_family("openeqa-hm3d") == "openeqa"
    assert task_primary_metric("openeqa-scannet") == "llm_match"
    assert official_judge_model("openeqa-hm3d") == "gpt-4-1106-preview"
    # No deterministic half: LLM-Match is the whole protocol.
    assert _scores("openeqa-hm3d", "brown", "tan", {}) == {}


def test_openeqa_marks_scale_and_clip_exactly_as_upstream() -> None:
    plan = JudgePlan("p", "openeqa", ((JudgeMessage("user", "prompt"),),))

    assert parse_judge_response(plan, "1") == {"llm_match": 0.0, "llm_match_score_1_5": 1.0}
    assert parse_judge_response(plan, "3") == {"llm_match": 0.5, "llm_match_score_1_5": 3.0}
    assert parse_judge_response(plan, "5") == {"llm_match": 1.0, "llm_match_score_1_5": 5.0}
    # The tagged branch reads only up to the next newline, so a judge that
    # explains itself afterwards still parses.
    assert parse_judge_response(plan, "Your mark: 4\nbecause the color matches") == {
        "llm_match": 0.75,
        "llm_match_score_1_5": 4.0,
    }
    # `evaluate-predictions.py` clips instead of validating: 0 is upstream's own
    # sentinel for a missing prediction and scores zero points, and an
    # overshooting judge is capped rather than discarding the question.
    assert parse_judge_response(plan, "0") == {"llm_match": 0.0, "llm_match_score_1_5": 1.0}
    assert parse_judge_response(plan, "9") == {"llm_match": 1.0, "llm_match_score_1_5": 5.0}
    with pytest.raises(ValueError, match="Invalid output string"):
        parse_judge_response(plan, "the response is good")

    assert combine_judge_scores(plan, (parse_judge_response(plan, "5"),)) == {
        "llm_match": 1.0,
        "llm_match_score_1_5": 5.0,
    }
