"""Official answer scorers used by the unified benchmark runner."""

# ruff: noqa: RUF001 -- upstream prompts intentionally retain their published typography.

from __future__ import annotations

import ast
import json
import math
import re
import string
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Protocol, cast

from mindbridge.benchmarks._official import personamem_v3_scoring as pm3
from mindbridge.benchmarks._official.atm_score import (
    deterministic_accuracy as atm_deterministic_accuracy,
)
from mindbridge.benchmarks._official.atm_score import list_jaccard_score
from mindbridge.benchmarks._official.beam_judge import build_rubric_item_prompt as build_beam_prompt
from mindbridge.benchmarks._official.beam_judge import parse_rubric_item_score as parse_beam_item
from mindbridge.benchmarks._official.clbench_judge import (
    build_grading_prompt as build_clbench_prompt,
)
from mindbridge.benchmarks._official.clbench_judge import (
    build_rubrics_text as build_clbench_rubrics,
)
from mindbridge.benchmarks._official.clbench_judge import (
    parse_grading_response as parse_clbench,
)
from mindbridge.benchmarks._official.longmemeval_prompts import (
    build_answer_check_prompt as build_longmemeval_prompt,
)
from mindbridge.benchmarks._official.longmemeval_prompts import (
    parse_answer_check as parse_longmemeval,
)
from mindbridge.benchmarks._official.memlens_prompts import (
    build_judge_prompt as build_memlens_prompt,
)
from mindbridge.benchmarks._official.memlens_prompts import get_task_key
from mindbridge.benchmarks._official.openeqa_llm_match import (
    MAX_SCORE as OPENEQA_MAX_SCORE,
)
from mindbridge.benchmarks._official.openeqa_llm_match import (
    MIN_SCORE as OPENEQA_MIN_SCORE,
)
from mindbridge.benchmarks._official.openeqa_llm_match import (
    build_llm_match_prompt as build_openeqa_prompt,
)
from mindbridge.benchmarks._official.openeqa_llm_match import (
    parse_llm_match_score as parse_openeqa_score,
)
from mindbridge.benchmarks._official.openeqa_llm_match import (
    truncate_prediction as truncate_openeqa_prediction,
)
from mindbridge.benchmarks.personamem_v3 import RANKING_TASK_TYPES

SCORER_VERSION = "official_scorers_v2"


class _Stemmer(Protocol):
    def stem(self, word: str) -> str: ...


@dataclass(frozen=True, slots=True)
class JudgeMessage:
    """One portable OpenAI-compatible judge message."""

    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True, slots=True)
class JudgePlan:
    """The exact upstream judge calls and parser for one answer."""

    protocol: str
    parser: Literal[
        "locomo",
        "m3",
        "egotempo",
        "memlens",
        "mm_lifelong",
        "atm",
        "gallery",
        "longmemeval",
        "clbench",
        "beam",
        "personamem_v3",
        "openeqa",
    ]
    calls: tuple[tuple[JudgeMessage, ...], ...]
    max_tokens: int | None = None
    extra_body: Mapping[str, object] | None = None
    details: Mapping[str, str] = field(default_factory=dict)
    candidate_metrics: tuple[Mapping[str, float], ...] = ()


_PROTOCOLS = {
    "locomo-refined": "locomo_refined_judge_887091190789",
    "m3-bench": "m3_agent_judge_0e3e41939bd8_system_v1",
    "video-mme": "video_mme_mcq_ead1408f75b6",
    "video-mme-v2": "video_mme_v2_6e4bebb03202",
    "egolifeqa": "egolifeqa_mcq_143fb319be7a",
    "egomemreason": "egomemreason_private_submission_7e581505b9dc",
    "egotempo": "egotempo_gemini_judge_7022ba77b4d8",
    "memlens": "memlens_judge_77f3ab9a52fa",
    "mm-lifelong": "mm_lifelong_judge_248aa82039a5",
    "supermemory-vqa": "supermemory_vqa_metrics_1d228e0f1004",
    "atm-bench": "atm_bench_scorer_ef4e5dff1a47_minimal_v1",
    "mem-gallery": "mem_gallery_scorer_a93959e1e978",
    "longmemeval": "longmemeval_anscheck_2ec2a557f339",
    "clbench": "clbench_binary_rubric_b28a5832a09b",
    "beam": "beam_unified_rubric_3e12035532eb",
    "personamem-v3": "personamem_v3_rubric_7b00a090b35b",
    "openeqa": "openeqa_llm_match_cfa3fce4595c",
}

_OFFICIAL_JUDGE_MODELS = {
    "locomo-refined": "qwen3-14b",
    "m3-bench": "gpt-4o-2024-11-20",
    "egotempo": "gemini-1.5-flash",
    "memlens": "qwen3-235b-judge",
    "mm-lifelong": "gpt-5",
    "atm-bench": "gpt-5-mini",
    "mem-gallery": "qwen2.5-72b-instruct",
    # `evaluate_qa.py`'s `model_zoo` offers three metric models; the released
    # results are the GPT-4o ones.
    "longmemeval": "gpt-4o-2024-08-06",
    # `eval.py --judge-model` default.
    "clbench": "gpt-5.1",
    # `src/llm.py: gpt_llm_obj`, the client every `evaluate_*` is handed.
    "beam": "gpt-4.1-mini",
    # `EVAL.md`: "the judge is a QueryLLM client on Azure gpt-5.5".
    "personamem-v3": "gpt-5.5",
    # `llm_match.get_llm_match_score`'s `openai_model` default, which
    # `evaluate-predictions.py` never overrides.
    "openeqa": "gpt-4-1106-preview",
}

# Metrics the pinned upstream protocol reports itself. `metric_is_official` reads this registry
# positively: a metric absent from its family's entry is a MindBridge diagnostic, never an
# upstream number. The families that publish no deterministic metric beyond their judge output
# are still listed, so an empty entry means "judge metrics only" rather than "not yet audited".
_OFFICIAL_METRICS: dict[str, frozenset[str]] = {
    # `evaluate.py` keeps the F1 and BLEU-1 of the accepted reference beside its judge label.
    "locomo-refined": frozenset({"llm_judge", "token_f1", "bleu_1"}),
    "m3-bench": frozenset({"accuracy"}),
    "video-mme": frozenset({"accuracy"}),
    # The leaderboard rating and the per-question accuracy it is built from.
    "video-mme-v2": frozenset({"accuracy", "question_accuracy", "rating"}),
    "egolifeqa": frozenset({"accuracy"}),
    # Scored by the private submission server; nothing here is an upstream number.
    "egomemreason": frozenset(),
    "egotempo": frozenset({"accuracy", "judge_score_0_5"}),
    "memlens": frozenset({"accuracy"}),
    # `Ref@300` is MM-Lifelong's own quantized temporal IoU.
    "mm-lifelong": frozenset({"answer_accuracy", "judge_score_0_5", "ref_at_300"}),
    "supermemory-vqa": frozenset({"qa_accuracy", "answerability"}),
    # `_official/atm_score.py` publishes `deterministic_accuracy` and `list_jaccard_score`, both
    # reported here as `accuracy`, plus the open-ended judge. Its retrieval and joint metrics are
    # MindBridge diagnostics and are deliberately absent.
    "atm-bench": frozenset({"accuracy"}),
    "mem-gallery": frozenset({"llm_judge", "f1", "bleu", "bleu_1", "bleu_2"}),
    "longmemeval": frozenset({"accuracy"}),
    "clbench": frozenset({"solving_rate", "requirement_ratio"}),
    "beam": frozenset({"llm_judge_score"}),
    # Everything `_official/personamem_v3_scoring.py` emits: the unified rubric's `pr_` dims
    # (see `_OFFICIAL_METRIC_PREFIXES`), the four task-specific judges' coercions, the ranking
    # metrics `compute_ranking_metrics` reports, and the 0-1 aggregate headline.
    "personamem-v3": frozenset(
        {
            "personamem_score",
            "recall@1",
            "recall@3",
            "recall@5",
            "hit@1",
            "hit@3",
            "mrr",
            "ndcg_graded@5",
            "sycophancy_caved",
            "sycophancy_resistance_0_10",
            "used_outdated_stance",
            "preference_shift_consistency",
            "deep_motivation_alignment",
            "surface_query_satisfaction",
            "preference_alignment",
            "telegraph_avoidance_fail",
            "privacy_leak_fail",
            "abstention_quality_0_10",
        }
    ),
    "openeqa": frozenset({"llm_match", "llm_match_score_1_5"}),
}

# One family scores an open dimension vocabulary upstream: `score_unified_rubric` names every
# rubric dim it was given, so the prefix it stamps them with is the registry entry.
_OFFICIAL_METRIC_PREFIXES: dict[str, tuple[str, ...]] = {"personamem-v3": ("pr_",)}

_JUDGE_METRICS = {
    "locomo-refined": frozenset({"llm_judge"}),
    "m3-bench": frozenset({"accuracy"}),
    "egotempo": frozenset({"accuracy", "judge_score_0_5"}),
    "memlens": frozenset({"accuracy"}),
    "mm-lifelong": frozenset({"answer_accuracy", "judge_score_0_5"}),
    "atm-bench": frozenset({"accuracy"}),
    "mem-gallery": frozenset({"llm_judge"}),
    "longmemeval": frozenset({"accuracy"}),
    "clbench": frozenset({"solving_rate", "requirement_ratio"}),
    "beam": frozenset({"llm_judge_score"}),
    "personamem-v3": frozenset({"personamem_score"}),
    "openeqa": frozenset({"llm_match", "llm_match_score_1_5"}),
}

# PersonaMem-v3 headline per task type, and the divisor that maps it onto the
# 0-1 `personamem_score` its aggregator compares across tasks
# (`evaluation/task_registry.py: PRIMARY_METRIC`).
_PERSONAMEM_HEADLINES: dict[str, tuple[str, float]] = {
    "chatbot_personalized_response": ("pr_preference_alignment_score_gated", 10.0),
    "over_personalization_sycophancy": ("sycophancy_resistance_0_10", 10.0),
    "preference_shift_followthrough": ("preference_shift_consistency", 10.0),
    "personal_qa_hallucination": ("abstention_quality_0_10", 10.0),
    "hidden_persona_implicit_qa": ("deep_motivation_alignment", 3.0),
}
_PERSONAMEM_RUBRIC_HEADLINE = ("pr_combined_personalization_score", 10.0)
# `task_registry.PRIMARY_METRIC` -- the column upstream's aggregator reads --
# gives all three single-target ranking tasks a graded nDCG@5, chosen over a
# binary top-1 because the gold is "subtle by design" and nDCG rewards
# surfacing it high with a smooth position discount. `compute_ranking_metrics`
# does label its own `recall@1` "Headline accuracy", but that is one of the
# many metrics it emits, not the one the aggregator reads. Every one of these
# rows has exactly one target, so the graded gain map reduces to the binary
# one and `ndcg_graded@5` is that metric.
_PERSONAMEM_RANKING_HEADLINE = ("ndcg_graded@5", 1.0)

# Ranks a slate like the other three, but its upstream headline is
# `lifecycle_score` -- the delta between a paired pre-row's and post-row's
# match rate -- which no single row can carry. It also has 2 to 5 targets per
# row, so `recall@1`, which divides by the target count, could never exceed
# 0.5 there. It keeps its ranking diagnostics and takes no headline, like the
# other paired families.
_PERSONAMEM_PAIRED_RANKING = frozenset({"short_vs_long_term_lifecycle"})
_RANKED_INDEXES = re.compile(r"ranked[ _]index(?:es|ices)?\s*[:=]?\s*\[([^\]]*)\]", re.IGNORECASE)
_RANKED_JSON = re.compile(r'"ranked_indices"\s*:\s*\[([^\]]*)\]')

_LOCOMO_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_ATM_MEDIA_ID = re.compile(r"\b(\d{8}_\d{6})\b")
_ATM_EMAIL_ID = re.compile(r"\b(email\d+)\b", re.IGNORECASE)


def task_family(task: str) -> str | None:
    """Return the benchmark family one task name belongs to, if it has one."""
    return _family_or_none(task)


def scorer_protocol(task: str) -> str:
    """Return the pinned scorer protocol recorded in result artifacts."""
    family = _family_or_none(task)
    return "custom_diagnostic" if family is None else _PROTOCOLS[family]


def task_primary_metric(task: str) -> str:
    """Return the benchmark's leaderboard-facing primary metric."""
    family = _family_or_none(task)
    if family is None:
        return "accuracy"
    return {
        "locomo-refined": "llm_judge",
        "m3-bench": "accuracy",
        "video-mme": "accuracy",
        "video-mme-v2": "rating",
        "egolifeqa": "accuracy",
        "egomemreason": "submission",
        "egotempo": "accuracy",
        "memlens": "accuracy",
        "mm-lifelong": "answer_accuracy",
        "supermemory-vqa": "qa_accuracy",
        "atm-bench": "accuracy",
        "mem-gallery": "f1",
        "longmemeval": "accuracy",
        "clbench": "solving_rate",
        "beam": "llm_judge_score",
        "personamem-v3": "personamem_score",
        "openeqa": "llm_match",
    }[family]


def sample_primary_metric(task: str) -> str:
    """Return the per-question metric used for comparisons and grouped scoring."""
    return (
        "question_accuracy"
        if _family_or_none(task) == "video-mme-v2"
        else task_primary_metric(task)
    )


def official_judge_model(task: str) -> str | None:
    """Return the model that makes an LLM-judge metric publication-comparable."""
    family = _family_or_none(task)
    return None if family is None else _OFFICIAL_JUDGE_MODELS.get(family)


def judge_model_is_official(task: str, model: str) -> bool:
    """Match provider-qualified model IDs without accepting nearby model sizes."""
    expected = official_judge_model(task)
    if expected is None:
        return True
    actual_key = _model_key(model)
    expected_key = _model_key(expected)
    if _family_or_none(task) == "memlens":
        return actual_key.endswith(expected_key) or "qwen3vl235b" in actual_key
    return actual_key.endswith(expected_key)


def metric_is_upstream(task: str, metric: str) -> bool:
    """Report whether the pinned upstream protocol publishes this metric at all."""
    family = _family_or_none(task)
    if family is None:
        return False
    if metric in _OFFICIAL_METRICS.get(family, frozenset()):
        return True
    return metric.startswith(_OFFICIAL_METRIC_PREFIXES.get(family, ()))


def metric_is_official(task: str, metric: str, judge_model: str, *, uses_judge: bool) -> bool:
    """Report protocol fidelity per metric instead of blessing a whole task at once.

    A metric is official only when upstream publishes it and, for judged metrics, the required
    judge produced it. Metrics this harness invents -- the retrieval and joint diagnostics above
    all -- are never official, however faithful the rest of the protocol was.
    """
    if not metric_is_upstream(task, metric):
        return False
    if metric not in _JUDGE_METRICS.get(_family_or_none(task) or "", ()) or not uses_judge:
        return True
    return judge_model_is_official(task, judge_model)


def retrieval_gold_ids(task: str, metadata: Mapping[str, object]) -> tuple[str, ...]:
    """Return the gold source IDs one question's retrieval diagnostics score against."""
    key = {"atm-bench": "evidence_ids", "mem-gallery": "clue_ids"}.get(_family_or_none(task) or "")
    return () if key is None else _deduplicated_sources(_metadata_values(metadata, key))


def local_scores(  # noqa: C901 - direct task dispatch mirrors official scorer families
    task: str,
    *,
    score_kind: str,
    prediction: str,
    parsed_choice: str | None,
    expected_choice: str | None,
    references: Sequence[str],
    question: str | None,
    metadata: Mapping[str, object],
    evidence_source_ids: Sequence[str],
) -> dict[str, float]:
    """Calculate every official deterministic per-question metric."""
    if score_kind == "submission":
        return {}
    family = _family_or_none(task)
    if family is None:
        if score_kind == "choice":
            return {"accuracy": float(parsed_choice == expected_choice)}
        return {"token_f1": _locomo_f1(prediction, references[0])}
    if score_kind == "choice":
        metric = (
            "question_accuracy"
            if family == "video-mme-v2"
            else ("qa_accuracy" if family == "supermemory-vqa" else "accuracy")
        )
        return {metric: float(parsed_choice == expected_choice)}
    if family == "locomo-refined":
        normalized = _locomo_prediction(prediction)
        if any(normalized == reference for reference in references):
            return {"llm_judge": 1.0, "token_f1": 1.0, "bleu_1": 1.0}
        if not normalized and any(references):
            return {"llm_judge": 0.0, "token_f1": 0.0, "bleu_1": 0.0}
        return {
            "token_f1": max(_locomo_f1(normalized, reference) for reference in references),
            "bleu_1": max(
                _bleu_1(normalized, reference, _locomo_tokens) for reference in references
            ),
        }
    if family == "m3-bench" and not prediction:
        return {"accuracy": 0.0}
    if family == "memlens" and len(_memlens_prediction(prediction).split()) > 500:
        return {"accuracy": 0.0}
    if family == "atm-bench":
        qtype = str(metadata.get("qtype", ""))
        reference = references[0]
        if qtype == "number":
            scores = {
                "accuracy": float(atm_deterministic_accuracy(reference, prediction, question))
            }
        elif qtype == "list_recall":
            scores = {"accuracy": float(list_jaccard_score(reference, prediction))}
        else:
            scores = {}
        scores.update(_atm_retrieval(metadata, evidence_source_ids))
        return finalize_scores(task, scores)
    if family == "personamem-v3":
        return _personamem_local(prediction, metadata)
    if family == "mem-gallery":
        reference = references[0]
        scores = {
            "f1": _gallery_f1(prediction, reference),
            "bleu": _gallery_bleu(prediction, reference, (0.25, 0.25, 0.25, 0.25)),
            "bleu_1": _gallery_bleu(prediction, reference, (1.0, 0.0, 0.0, 0.0)),
            "bleu_2": _gallery_bleu(prediction, reference, (0.5, 0.5, 0.0, 0.0)),
            "exact_match": float(_gallery_normalize(prediction) == _gallery_normalize(reference)),
        }
        scores.update(_gallery_retrieval(metadata, evidence_source_ids))
        return scores
    return {}


def finalize_scores(task: str, scores: Mapping[str, float]) -> dict[str, float]:
    """Add official metrics that combine answer and retrieval correctness."""
    result = dict(scores)
    if _family_or_none(task) == "atm-bench" and "accuracy" in result:
        for cutoff in (5, 10):
            recall = result.get(f"retrieval_recall@{cutoff}")
            if recall is not None:
                result[f"joint_partial@{cutoff}"] = result["accuracy"] * recall
                result[f"joint_strict@{cutoff}"] = result["accuracy"] * float(recall == 1.0)
    return result


def judge_plan(  # noqa: C901 - direct task dispatch keeps official protocols auditable
    task: str,
    *,
    question: str | None,
    references: Sequence[str],
    prediction: str,
    metadata: Mapping[str, object],
) -> JudgePlan | None:
    """Build the pinned upstream judge prompt for one generated answer."""
    family = _family_or_none(task)
    if family is None:
        return None
    if family not in _JUDGE_METRICS:
        return None
    if family == "m3-bench" and not prediction:
        return None
    if family == "atm-bench" and metadata.get("qtype") != "open_end":
        return None
    if family == "personamem-v3" and not _personamem_judged(metadata):
        return None
    if family == "clbench" and not prediction.strip():
        # `process_single_item` scores an empty output 0 without calling the
        # judge at all.
        return None
    if family == "memlens" and len(_memlens_prediction(prediction).split()) > 500:
        return None
    if question is None:
        raise ValueError(f"{task} official judge needs the original question")
    protocol = scorer_protocol(task)
    if family == "locomo-refined":
        normalized = _locomo_prediction(prediction)
        if not normalized or any(normalized == reference for reference in references):
            return None
        calls = tuple(
            (
                JudgeMessage(
                    "user",
                    _LOCOMO_PROMPT.format(
                        question=question,
                        gold_answer=reference,
                        generated_answer=normalized,
                    ),
                ),
            )
            for reference in references
        )
        candidate_metrics = tuple(
            {
                "token_f1": _locomo_f1(normalized, reference),
                "bleu_1": _bleu_1(normalized, reference, _locomo_tokens),
            }
            for reference in references
        )
        return JudgePlan(
            protocol,
            "locomo",
            calls,
            extra_body={"enable_thinking": False},
            candidate_metrics=candidate_metrics,
        )
    reference = references[0]
    if family == "m3-bench":
        prompt = _M3_PROMPT.format(
            question=question,
            ground_truth_answer=reference,
            agent_answer=prediction,
        )
        return JudgePlan(
            protocol,
            "m3",
            (
                (
                    JudgeMessage("system", "You are an expert in video understanding."),
                    JudgeMessage("user", prompt),
                ),
            ),
            2048,
        )
    if family == "egotempo":
        prompt = _EGOTEMPO_PROMPT.format(question=question, answer=reference, prediction=prediction)
        return JudgePlan(protocol, "egotempo", ((JudgeMessage("user", prompt),),))
    if family == "memlens":
        question_type = str(metadata.get("question_type", ""))
        subtype = str(metadata.get("question_subtype") or "")
        task_key = get_task_key(question_type, subtype, reference)
        old_answer_value = metadata.get("old_answer")
        old_answer = old_answer_value if isinstance(old_answer_value, str) else None
        prompt = build_memlens_prompt(
            task_key,
            question,
            reference,
            _memlens_prediction(prediction),
            old_answer,
        )
        return JudgePlan(
            protocol,
            "memlens",
            ((JudgeMessage("user", prompt),),),
            2048,
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"task_key": task_key},
        )
    if family == "mm-lifelong":
        user = (
            f"Question: {question}\nGroundtruth answer: {reference}\n"
            f"Candidate answer: {prediction}\nYour response: "
        )
        return JudgePlan(
            protocol,
            "mm_lifelong",
            ((JudgeMessage("system", _MM_LIFELONG_SYSTEM), JudgeMessage("user", user)),),
        )
    if family == "atm-bench":
        prompt = (
            _ATM_PROMPT.replace("{{question}}", question)
            .replace("{{answer}}", reference)
            .replace("{{prediction}}", prediction)
        )
        return JudgePlan(protocol, "atm", ((JudgeMessage("user", prompt),),), 600)
    if family == "longmemeval":
        prompt = build_longmemeval_prompt(
            str(metadata.get("question_type", "")),
            question,
            references[0],
            prediction,
            abstention=bool(metadata.get("abstention")),
        )
        return JudgePlan(protocol, "longmemeval", ((JudgeMessage("user", prompt),),), 10)
    if family == "clbench":
        prompt = build_clbench_prompt(build_clbench_rubrics(references), prediction)
        return JudgePlan(protocol, "clbench", ((JudgeMessage("user", prompt),),))
    if family == "beam":
        category = str(metadata.get("category", ""))
        return JudgePlan(
            protocol,
            "beam",
            # One call per rubric item, exactly as every `evaluate_*` loops.
            tuple(
                (JudgeMessage("user", build_beam_prompt(item, prediction)),) for item in references
            ),
            details={"category": category},
        )
    if family == "personamem-v3":
        return _personamem_plan(protocol, question, metadata, prediction)
    if family == "openeqa":
        declared = metadata.get("extra_answers")
        extra = (
            None
            if declared is None
            else tuple(str(value) for value in cast(Sequence[object], declared))
        )
        return JudgePlan(
            protocol,
            "openeqa",
            (
                (
                    JudgeMessage(
                        "user",
                        build_openeqa_prompt(
                            question,
                            reference,
                            # `evaluate-predictions.py` trims the prediction
                            # before scoring it, so the judge never sees the
                            # trailing unterminated sentence.
                            truncate_openeqa_prediction(prediction),
                            extra,
                        ),
                    ),
                ),
            ),
            # `openai_max_tokens=32` in `get_llm_match_score`.
            32,
        )
    if family == "mem-gallery":
        prompt = (
            _GALLERY_PROMPT.replace("{{question}}", question)
            .replace("{{ground_truth}}", reference)
            .replace("{{model_output}}", prediction)
        )
        return JudgePlan(protocol, "gallery", ((JudgeMessage("user", prompt),),))
    raise AssertionError(f"unhandled judge family: {family}")


def parse_judge_response(  # noqa: C901 - mirrors seven incompatible upstream parsers
    plan: JudgePlan, response: str
) -> dict[str, float]:
    """Parse one response exactly as the pinned upstream evaluator does."""
    if plan.parser == "locomo":
        payload = _json_object(response)
        label = str(payload.get("label", "")).strip().upper()
        if label not in {"CORRECT", "WRONG"}:
            raise ValueError("LoCoMo judge did not return CORRECT or WRONG")
        return {"llm_judge": float(label == "CORRECT")}
    if plan.parser == "m3":
        return {"accuracy": float("yes" in response.lower())}
    if plan.parser == "egotempo":
        match = re.search(r"\{.*?\}", response, re.DOTALL)
        if match is None:
            raise ValueError("EgoTempo judge did not return a dictionary")
        payload = ast.literal_eval(match.group())
        if not isinstance(payload, dict):
            raise ValueError("EgoTempo judge response is not a dictionary")
        raw_score = int(payload.get("score", 0))
        if not 0 <= raw_score <= 5:
            raise ValueError("EgoTempo judge score is outside 0-5")
        return {
            "accuracy": float(str(payload.get("pred", "")).strip().lower() == "correct"),
            "judge_score_0_5": float(raw_score),
        }
    if plan.parser == "memlens":
        match = re.search(r'"answer_score"\s*:\s*(\d+)', response)
        if match is None:
            match = re.search(r"deserves\s+(\d+)\s+point", response, re.IGNORECASE)
        if match is None:
            match = re.search(r"\[Score\]\s*:\s*(\d+)", response)
        return {"accuracy": float(min(int(match.group(1)), 1)) if match else 0.0}
    if plan.parser == "mm_lifelong":
        match = re.search(r"Final Score:\s*([0-5])", response, re.IGNORECASE)
        if match is None:
            match = re.search(r"\b([0-5])\b", response)
        if match is None:
            raise ValueError("MM-Lifelong judge did not return a 0-5 score")
        raw = int(match.group(1))
        mapped = 1.0 if raw in (4, 5) else 0.5 if raw == 3 else 0.0
        return {"answer_accuracy": mapped, "judge_score_0_5": float(raw)}
    if plan.parser == "atm":
        return {"accuracy": float(_atm_judge_accuracy(response))}
    if plan.parser == "longmemeval":
        return {"accuracy": float(parse_longmemeval(response))}
    if plan.parser == "clbench":
        score, ratio = parse_clbench(response)
        return {"solving_rate": score, "requirement_ratio": ratio}
    if plan.parser == "beam":
        return {"llm_judge_score": parse_beam_item(response, plan.details.get("category", ""))}
    if plan.parser == "personamem_v3":
        return _personamem_judge_scores(plan, response)
    if plan.parser == "openeqa":
        # Upstream clips rather than validating: `evaluate-predictions.py` maps
        # every mark through `np.clip(scores, 1, 5)`, so a judge that answers 0
        # -- its own sentinel for a missing prediction -- scores zero points and
        # one that overshoots is capped instead of discarding the question.
        clipped = min(max(parse_openeqa_score(response), OPENEQA_MIN_SCORE), OPENEQA_MAX_SCORE)
        return {
            "llm_match": (clipped - OPENEQA_MIN_SCORE) / (OPENEQA_MAX_SCORE - OPENEQA_MIN_SCORE),
            "llm_match_score_1_5": float(clipped),
        }
    if plan.parser == "gallery":
        score = _gallery_judge_score(response)
        return {"llm_judge": 0.0 if score < 0.25 else 0.5 if score < 0.75 else 1.0}
    raise AssertionError(f"unhandled judge parser: {plan.parser}")


def combine_judge_scores(
    plan: JudgePlan, scores: Sequence[Mapping[str, float]]
) -> dict[str, float]:
    """Use the best accepted gold answer, matching LoCoMo's multi-reference behavior."""
    if not scores:
        raise ValueError("judge plan produced no scores")
    if plan.parser == "locomo":
        if len(scores) != len(plan.candidate_metrics):
            raise ValueError("LoCoMo judge results do not match its answer candidates")
        selected = max(
            range(len(scores)),
            key=lambda index: (
                scores[index]["llm_judge"],
                plan.candidate_metrics[index]["token_f1"],
                plan.candidate_metrics[index]["bleu_1"],
            ),
        )
        return {**plan.candidate_metrics[selected], **scores[selected]}
    if plan.parser == "beam":
        # Every `evaluate_*` divides the accumulated per-item scores by the
        # number of rubric items, so the category score is their mean.
        values = [score["llm_judge_score"] for score in scores]
        return {"llm_judge_score": sum(values) / len(values)}
    if len(scores) == 1:
        return dict(scores[0])
    names = set().union(*(score.keys() for score in scores))
    return {name: max(score[name] for score in scores if name in score) for name in names}


def _personamem_metric(task_type: str) -> tuple[str, float] | None:
    """Return one task type's headline metric and the divisor onto 0-1, if any.

    Keyed on EVAL.md's headline table rather than on the rubric's applicability
    map: upstream runs the rubric on several more task types but keeps its
    output as a diagnostic beside a headline it computes a different way (a
    leak-set composite, fatigue counters, a paired-row delta). A task type this
    returns `None` for is answered and reported without a `personamem_score`,
    so it stays out of the aggregate rather than entering it under a number
    upstream does not publish.
    """
    if task_type in _PERSONAMEM_HEADLINES:
        return _PERSONAMEM_HEADLINES[task_type]
    if task_type in pm3.RUBRIC_HEADLINE_TASK_TYPES:
        return _PERSONAMEM_RUBRIC_HEADLINE
    if task_type in RANKING_TASK_TYPES and task_type not in _PERSONAMEM_PAIRED_RANKING:
        return _PERSONAMEM_RANKING_HEADLINE
    return None


def _personamem_judged(metadata: Mapping[str, object]) -> bool:
    """Report whether one row's headline needs -- and can have -- a judge call.

    Ranking rows are scored deterministically against their frozen slate, so
    they are excluded here even though they do have a headline.
    """
    task_type = str(metadata.get("task_type", ""))
    return task_type not in RANKING_TASK_TYPES and _personamem_metric(task_type) is not None


def _personamem_plan(
    protocol: str,
    question: str,
    metadata: Mapping[str, object],
    prediction: str,
) -> JudgePlan:
    task_type = str(metadata.get("task_type", ""))
    evidence = metadata.get("judge_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    if task_type in pm3.OWN_JUDGE_TASK_TYPES:
        prompt = pm3.build_own_judge_prompt(task_type, evidence, prediction)
    else:
        prompt = pm3.build_unified_rubric_prompt(
            task_type,
            _personamem_evidence(question, metadata),
            prediction,
            pm3.positive_dims(task_type),
            pm3.hard_rule_dims(task_type),
            pm3.penalty_dims(task_type),
            query_text=question,
        )
    return JudgePlan(
        protocol,
        "personamem_v3",
        ((JudgeMessage("user", prompt),),),
        details={"task_type": task_type},
    )


def _personamem_evidence(question: str, metadata: Mapping[str, object]) -> str:
    """Serialise the evidence block the unified rubric judge reads.

    Upstream assembles this with `personalization_rubric.build_source_a`, which
    queries the persona's backend for the same-day avoid slice, the privacy
    flags and the contradiction set. Those queries need the backend, not the
    question, so this block carries the evidence the released row itself
    publishes: the held-out preference, its distractors, and the row's rubric
    tags. It is narrower than upstream's, which makes the hard-rule checks that
    depend on the avoid slice under-fire relative to a full-harness run.
    """
    payload = {
        "query_text": question,
        "held_out_preference": metadata.get("groundtruth_preference", ""),
        "distractor_preferences": list(_metadata_values(metadata, "distractor_preferences")),
        "rubric_tags": list(_metadata_values(metadata, "rubric_tags")),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _personamem_judge_scores(plan: JudgePlan, response: str) -> dict[str, float]:
    task_type = plan.details.get("task_type", "")
    payload = pm3_json(response)
    if task_type in pm3.OWN_JUDGE_TASK_TYPES:
        scores = pm3.parse_own_judge(task_type, payload)
    else:
        scores = pm3.score_unified_rubric(task_type, payload)
    return _personamem_headline(task_type, scores)


def _personamem_headline(task_type: str, scores: Mapping[str, float]) -> dict[str, float]:
    """Add the 0-1 value the upstream aggregator compares across task types."""
    result = dict(scores)
    metric = _personamem_metric(task_type)
    if metric is None:
        return result
    name, scale = metric
    if name in result:
        result["personamem_score"] = min(1.0, max(0.0, result[name] / scale))
    return result


def pm3_json(response: str) -> Mapping[str, object]:
    """Extract the judge's JSON object the way upstream's helper does."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    candidate = match.group(1).strip() if match else response.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("PersonaMem-v3 judge did not return a JSON object") from None
        payload = json.loads(candidate[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("PersonaMem-v3 judge did not return a JSON object")
    return payload


def _personamem_local(prediction: str, metadata: Mapping[str, object]) -> dict[str, float]:
    """Score a ranking row deterministically against its frozen slate."""
    if str(metadata.get("task_type", "")) not in RANKING_TASK_TYPES:
        return {}
    count = metadata.get("candidate_count")
    if not isinstance(count, int) or count <= 0:
        return {}
    positives = {
        index for index in _index_values(metadata, "positive_indexes") if 0 <= index < count
    }
    if not positives:
        return {}
    ranked = _parse_ranking(prediction, count)
    negatives = {
        index for index in _index_values(metadata, "negative_indexes") if 0 <= index < count
    }
    # `slate_ranking.ORIGIN_GAIN`: the held-out target is gain 3.0 and every
    # other origin -- hard negatives, carve-outs and fillers alike -- is 0.0.
    # `task_registry` describes a +2/-2/+1 scheme for one ranking task, but it
    # belongs to a `_graded_ndcg_at_k` the release's fields cannot reconstruct,
    # and negative gains make nDCG itself negative. Hard negatives keep their
    # own reported diagnostics below instead.
    gains = [3.0 if index in positives else 0.0 for index in ranked]
    scores = {
        "recall@1": pm3.recall_at_k(ranked, positives, 1),
        "recall@3": pm3.recall_at_k(ranked, positives, 3),
        "recall@5": pm3.recall_at_k(ranked, positives, 5),
        "hit@1": pm3.hit_at_k(ranked, positives, 1),
        "hit@3": pm3.hit_at_k(ranked, positives, 3),
        "mrr": pm3.mrr(ranked, positives),
        "ndcg_graded@5": pm3.ndcg_at_k(gains, 5),
        "negative_in_top1": float(bool(ranked) and ranked[0] in negatives),
        "negative_in_top3": float(any(index in negatives for index in ranked[:3])),
    }
    return _personamem_headline(str(metadata.get("task_type", "")), scores)


def _parse_ranking(prediction: str, count: int) -> tuple[int, ...]:
    """Read a ranking out of an answer, in either published shape.

    The pinned release's own `example_response` is `Ranked indexes: [...]`; the
    evaluation repository's current prompt asks for a fenced-JSON
    `ranked_indices`. Both are accepted. A reply that is not a permutation of
    the slate falls back to the identity order, which is what
    `slate_ranking.run_task_a` does.
    """
    match = _RANKED_JSON.search(prediction) or _RANKED_INDEXES.search(prediction)
    if match is not None:
        try:
            ranked = [int(part) for part in match.group(1).replace(",", " ").split()]
        except ValueError:
            ranked = []
        if sorted(ranked) == list(range(count)):
            return tuple(ranked)
    return tuple(range(count))


def _index_values(metadata: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = metadata.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(item for item in value if isinstance(item, int) and not isinstance(item, bool))


def _family(task: str) -> str:
    family = _family_or_none(task)
    if family is not None:
        return family
    raise ValueError(f"no official scorer registered for task {task}")


def _family_or_none(task: str) -> str | None:
    for family in sorted(_PROTOCOLS, key=len, reverse=True):
        if task == family or task.startswith(f"{family}-"):
            return family
    return None


def _model_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _locomo_tokens(value: str) -> list[str]:
    return [token.lower() for token in _LOCOMO_TOKEN.findall(value)]


def _locomo_prediction(value: str) -> str:
    cleaned = str(value or "").strip()
    boxed = _last_boxed_value(cleaned)
    if boxed is not None:
        return boxed
    lowered = cleaned.lower()
    if "final answer:" in lowered:
        position = lowered.index("final answer:")
        cleaned = cleaned[position + len("final answer:") :].strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    return cleaned


def _last_boxed_value(value: str) -> str | None:
    result: str | None = None
    for match in re.finditer(r"\\box(?:ed)?\{", value):
        depth = 0
        start = match.end()
        for index in range(match.end() - 1, len(value)):
            if value[index] == "{":
                depth += 1
            elif value[index] == "}":
                depth -= 1
                if depth == 0:
                    result = value[start:index].strip()
                    break
        else:
            result = value[start:].strip()
    return result


def _locomo_f1(prediction: str, reference: str) -> float:
    predicted, expected = _locomo_tokens(prediction), _locomo_tokens(reference)
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _bleu_1(
    prediction: str,
    reference: str,
    tokenizer: Callable[[str], list[str]],
) -> float:
    predicted, expected = tokenizer(prediction), tokenizer(reference)
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    if not precision:
        return 0.0
    penalty = (
        math.exp(1 - len(expected) / len(predicted)) if len(predicted) < len(expected) else 1.0
    )
    return penalty * precision


def _gallery_normalize(value: str) -> str:
    normalized = str(value).lower()
    normalized = re.sub(r"(?<=\d)\.(?=\d)", "DOTPLACEHOLDER", normalized)
    normalized = normalized.replace("_", "UNDERSCOREPLACEHOLDER")
    normalized = re.sub(r"\b(a|an|the|and)\b", " ", normalized)
    normalized = "".join(
        character if character not in string.punctuation else " " for character in normalized
    )
    return " ".join(
        normalized.replace("DOTPLACEHOLDER", ".").replace("UNDERSCOREPLACEHOLDER", "_").split()
    )


@lru_cache(maxsize=1)
def _porter() -> _Stemmer:
    try:
        from nltk.stem import PorterStemmer  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("Mem-Gallery scoring requires mindbridge[benchmarks]") from None
    return cast(_Stemmer, PorterStemmer())


def _gallery_tokens(value: str) -> list[str]:
    return _gallery_normalize(value).split()


def _gallery_f1(prediction: str, reference: str) -> float:
    predicted = [_porter().stem(word) for word in _gallery_tokens(prediction)]
    expected = [_porter().stem(word) for word in _gallery_tokens(reference)]
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _gallery_bleu(
    prediction: str,
    reference: str,
    weights: tuple[float, float, float, float],
) -> float:
    try:
        from nltk.translate.bleu_score import (  # type: ignore[import-untyped]
            SmoothingFunction,
            sentence_bleu,
        )
    except ImportError:
        raise RuntimeError("Mem-Gallery scoring requires mindbridge[benchmarks]") from None
    predicted, expected = _gallery_tokens(prediction), _gallery_tokens(reference)
    if not predicted or not expected:
        return 0.0
    return float(
        sentence_bleu(
            [expected],
            predicted,
            weights=weights,
            smoothing_function=SmoothingFunction().method1,
        )
    )


def _deduplicated_sources(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _metadata_values(metadata: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(str(item) for item in value)


def _atm_retrieval(
    metadata: Mapping[str, object], evidence_source_ids: Sequence[str]
) -> dict[str, float]:
    expected = set(_metadata_values(metadata, "evidence_ids"))
    if not expected:
        return {}
    retrieved = _deduplicated_sources(evidence_source_ids)
    scores: dict[str, float] = {}
    for cutoff in (1, 5, 10):
        selected = set(retrieved[:cutoff])
        scores[f"retrieval_recall@{cutoff}"] = len(selected & expected) / len(expected)
    at_gt = set(retrieved[: min(len(expected), len(retrieved))])
    scores["retrieval_recall@gt"] = len(at_gt & expected) / len(expected)
    scores["retrieval_hit@1"] = float(bool(retrieved) and retrieved[0] in expected)
    return scores


def _gallery_retrieval(
    metadata: Mapping[str, object], evidence_source_ids: Sequence[str]
) -> dict[str, float]:
    expected = set(_metadata_values(metadata, "clue_ids"))
    if not expected:
        return {}
    retrieved = _deduplicated_sources(evidence_source_ids)[:10]
    hits = sum(source in expected for source in retrieved)
    return {
        "retrieval_precision@10": hits / max(1, len(retrieved)),
        "retrieval_recall@10": hits / len(expected),
        "retrieval_hit@10": float(hits > 0),
    }


def _memlens_prediction(value: str) -> str:
    text = value
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"<\|[^>]*\|>", "", text)
    text = re.sub(
        r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>",
        r"\1",
        text,
        flags=re.DOTALL,
    )
    text = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    text = text.replace("[CONTENT_FILTER_REJECTED]", "").strip()
    if len(text) > 6_000 and len(text) > 3_000:
        text = "[...truncated earlier reasoning...]\n" + text[int(len(text) * 0.7) :]
    return text


def _json_object(value: str) -> Mapping[str, object]:
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge response does not contain a JSON object")
    payload = json.loads(value[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("judge response JSON must be an object")
    return payload


def _atm_judge_accuracy(value: str) -> bool:
    try:
        payload = _json_object(value)
    except (ValueError, json.JSONDecodeError):
        payload = {}
    if "accuracy" in payload:
        return str(payload["accuracy"]).lower() == "true"
    lowered = value.lower()
    if '"accuracy": true' in lowered or '"accuracy":true' in lowered:
        return True
    if '"accuracy": false' in lowered or '"accuracy":false' in lowered:
        return False
    return any(word in lowered for word in ("accurate", "correct", "matches", "true", "yes"))


def _gallery_judge_score(value: str) -> float:
    try:
        payload = _json_object(value)
        raw_score = payload.get("score")
        if isinstance(raw_score, bool):
            raise ValueError
        return float(cast(str | int | float, raw_score))
    except (TypeError, ValueError, json.JSONDecodeError):
        for pattern in (
            r'"score"\s*:\s*([0-9.]+)',
            r"score\s*:\s*([0-9.]+)",
        ):
            match = re.search(pattern, value, re.IGNORECASE)
            if match is not None:
                return float(match.group(1))
    raise ValueError("Mem-Gallery judge did not return a numeric score")


# mem-eval-suite/LoCoMo_refined@887091190789e8d6760e70b9edd696539923dc4f
_LOCOMO_PROMPT = """Your task is to label an answer as ’CORRECT’ or ’WRONG’ given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

Core principle — Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold’s key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT — even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold’s content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR↔HOUR, DAY↔DAY, MONTH↔MONTH, YEAR↔YEAR.
  Do not answer a gold at a different time unit — even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
  (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] → WRONG)
- Do NOT convert relative ↔ absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- Treat harmless modifiers in relative forms (e.g., “the/last/previous/just prior”) as equivalent when both the anchor date and the time unit are the same.

- Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover **all** of them.
- Extra non-contradictory items **generally count as WRONG**.
    - Example: gold = A, B, C ; gen = A, B, C → CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D → WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C → C, C′), it is still considered CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include **any one** of them without contradiction to be CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":

```json
{{
    "label": "CORRECT" or "WRONG"
}}
```
"""

# ByteDance-Seed/m3-agent@0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c
_M3_PROMPT = """You are provided with a question, a ground truth answer, and an answer from an agent model. Your task is to determine whether the ground truth answer can be logically inferred from the agent's answer, in the context of the question.

Do not directly compare the surface forms of the agent answer and the ground truth answer. Instead, assess whether the meaning expressed by the agent answer supports or implies the ground truth answer. If the ground truth can be reasonably derived from the agent answer, return "Yes". If it cannot, return "No".

Important notes:
\t•\tDo not require exact wording or matching structure.
\t•\tSemantic inference is sufficient, as long as the agent answer entails or implies the meaning of the ground truth answer, given the question.
\t•\tOnly return "Yes" or "No", with no additional explanation or formatting.

Input fields:
\t•\tquestion: the question asked
\t•\tground_truth_answer: the correct answer
\t•\tagent_answer: the model's answer to be evaluated

Now evaluate the following input:

Input:
\t•\tquestion: {question}
\t•\tground_truth_answer: {ground_truth_answer}
\t•\tagent_answer: {agent_answer}

Output ('Yes' or 'No'):"""

# google-research-datasets/egotempo@7022ba77b4d89f51cf34e499767995ccd5c90c7a
_EGOTEMPO_PROMPT = """role: "system",
content: "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs.
Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:
-----##INSTRUCTIONS:
- Focus on the correctness and accuracy of the predicted answer with the ground-truth.
- Consider uncertain predictions, such as 'it is impossible to answer the question from the video', as incorrect, unless the ground truth answer also says that."
role: "user",
content: "Please evaluate the following video-based question-answer pair:
Question: {question}
Ground truth correct Answer: {answer}
Predicted Answer: {prediction}
Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness.
Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is a string of 'correct' or 'incorrect',
value of 'score' is in INTEGER, not STRING and value of 'reason' should provide the reason behind the decision."
"""

# MM-Lifelong/MM-Lifelong@248aa82039a574e63a2e524746a7cd8f32330443/eval_acc.py
_MM_LIFELONG_SYSTEM = """As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct “groundtruth” answer, and the candidate answer will be provided to you.
The following is a comparison table of some proper nouns; matching any one of them is considered correct.

You must FIRST provide a brief analysis explaining the semantic similarity between the groundtruth
and the candidate answer.

THEN, on a new line, output the final score.

Scoring criteria (semantic similarity only):

- 0: No similarity.
  The candidate answer is completely irrelevant, contradictory, or does not address the question at all.

- 1: Very low similarity.
  The candidate answer mentions a related topic or keyword, but fails to answer the question
  and does not convey the main meaning of the groundtruth.

- 2: Low similarity.
  The candidate answer addresses the question in a limited way, capturing some minor aspects,
  but misses or misrepresents the core idea or key facts of the groundtruth.

- 3: Moderate similarity.
  The candidate answer captures the main idea of the groundtruth,
  but omits several important details or includes noticeable inaccuracies.

- 4: High similarity.
  The candidate answer correctly captures the main idea and most key details of the groundtruth,
  with only minor omissions, simplifications, or non-critical inaccuracies.

- 5: Complete similarity.
  The candidate answer is semantically equivalent to the groundtruth,
  covering all essential information with no meaningful omissions or errors.

Special Rules:

- Hallucination-sensitive questions:
Score 5 only if all required items are correct;
if any item is incorrect, missing, or hallucinated, score 0 (no partial credit).

- Time-duration questions:
Allow errors within the range defined by the question; answers outside the range should receive score 0.

Output format (strictly follow):
Analysis:
<your analysis>

Final Score:
<an integer from 0 to 5>"""

# JingbiaoMei/ATM-Bench@ef4e5dff1a47ec71213a06e359f02753defa8fb1
_ATM_PROMPT = (
    "You are an evaluator, and you are given a task to evaluate a model predictions "
    "with a given question. Let's follow the instructions step by step to make a "
    "judgement.  1. As the first step, you need to check whether the prediction was really "
    "answering the question.  2. If the model prediction does provide a meaningful answer, "
    "judge whether the model Prediction matches the ground truth answer by reasoning according "
    "to the following steps:  2.1: Always assume the ground truth is correct.  "
    "2.2: Pay attention to theses special cases:  a. If the ground truth answer contains "
    'numbers, the value of "accuracy" is true only if numbers in ground truth and numbers in '
    'model predictions match very well; in case of math questions, "accuracy" is true only if '
    "the numbers in model predictions EXACTLY matches the numbers in ground truth;  "
    'b. If the ground truth answer contains time, and/or time range, "accuracy" is "true" only '
    "if if times and time ranges in ground truth and model predictions match very well.  "
    'c. If the ground truth answer contains a set of objects, "accuracy" is "true" if the model '
    'prediction covers most of the objects in the ground truth; however, "accuracy" if "false" '
    "if the  model prediction has a lot of objects that are not in the ground truth.  "
    'd. If the ground truth is something similar to "I don\'t know", "accuracy" is "true" only '
    "if the model prediction also implies the similar thing.  2.3: Even if the prediction "
    'statement is reasonable, if it conflicts with or does not match the ground truth, "accuracy" '
    'should be "false".  2.4. "Accuracy" is true if the ground truth information is covered by '
    "the prediction. The prediction is allowed to provide more information but should not be "
    "against the ground truth. If it is hard to decide whether the prediction matches ground "
    'truth, "accuracy" should be "false".  Think step by step following the instructions above, '
    'and then make a judgment. Respond with only a single JSON blob with an "explanation" field '
    'that has your short(less than 100 word) reasoning steps and an "accuracy" field which is '
    '"true" or "false".  Question: {{question}}  Ground truth: {{answer}}  '
    "Prediction: {{prediction}}"
)

# YuanchenBei/Mem-Gallery@a93959e1e978a6a7d77798ae92c2ffe41c538c62
_GALLERY_PROMPT = """You are an impartial judge evaluating the memory capabilities of an AI assistant with the question-answering task.
Your task is to compare the Assistant's Answer against the Ground Truth and assign a score of 0, 0.25, 0.5, 0.75, or 1.

### Scoring Rubric

**Score 0 (Incorrect / Miss):**

- The answer contradicts the Ground Truth.
- For Yes/No questions: The answer has the wrong polarity (e.g., says "Yes" when Ground Truth is "No").
- For Open-ended questions: The answer provides factually wrong information or hallucinations.
- The assistant fails to provide the required information.

**Score 0.25 (Poor / Tangential):**

- The answer touches on the topic but misses the **core entity** or key value required.
- The answer contains a mix of minor correct details and **significant hallucinations** or wrong associations.
- The answer is excessively vague to the point of being useless (e.g., answering "a dog" instead of "a golden retriever").

**Score 0.5 (Partial / Vague):**

- The answer is technically correct, but lacks confidence or is incomplete.
- The answer captures the **main entity or concept** correctly but misses a part of the required supporting details.
- For Yes/No questions: The polarity is correct, but the reasoning is flawed (if have), or the assistant is uncertain (e.g., "I think it might be Yes").
- For Open-ended questions: The answer is too general or misses key adjectives/details present in the Ground Truth.

**Score 0.75 (Good / Minor Imperfection):**

- The answer is largely accurate and captures the core information confidently.
- It misses only **minor details** (e.g., specific adjectives or secondary details) that do not alter the main truth.
- The answer contains all the correct information but includes unnecessary "fluff" or slight conversational filler that reduces precision.

**Score 1 (Correct / Exact):**

- The answer is accurate, precise, and confident.
- For Yes/No questions: The polarity matches the Ground Truth perfectly.
- For Open-ended questions: The answer contains **all** the core information and necessary details required by the Ground Truth without hallucinations.

### Input Data
Question: {{question}}
Ground Truth: {{ground_truth}}
Assistant Answer: {{model_output}}

### Output Format
Output strictly in the following JSON format:
{"score": <0, 0.25, 0.5, 0.75, or 1>, "reasoning": "<short explanation>"}"""
