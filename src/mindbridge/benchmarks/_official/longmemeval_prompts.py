"""LongMemEval answer-check prompts pinned from the official evaluator.

Transcribed from `xiaowu0162/LongMemEval@main:src/evaluation/evaluate_qa.py`
(`get_anscheck_prompt`). The judge is asked for a bare yes/no and upstream
scores `label = 'yes' in eval_response.lower()`.
"""

from __future__ import annotations

_SHARED = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
)

_TAIL = (
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
    "\n\nIs the model response correct? Answer yes or no only."
)

# `single-session-user`, `single-session-assistant` and `multi-session` share
# one template upstream; the remaining types each add one clause.
_DEFAULT = _SHARED + _TAIL

_TEMPORAL = (
    _SHARED
    + "In addition, do not penalize off-by-one errors for the number of days. If the question "
    "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
    "(e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
    + _TAIL
)

_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, the "
    "response should be considered as correct as long as the updated answer is the required "
    "answer." + _TAIL
)

# The preference template swaps "Correct Answer" for "Rubric": the gold field
# holds a rubric for a desired personalized response rather than a fact.
_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
    "answer no. The model does not need to reflect all the points in the rubric. The response is "
    "correct as long as it recalls and utilizes the user's personal information correctly."
    "\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {response}"
    "\n\nIs the model response correct? Answer yes or no only."
)

_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. The model "
    "could say that the information is incomplete, or some other information is given but the "
    "asked information is not."
    "\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {response}"
    "\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
)

_TEMPLATES = {
    "single-session-user": _DEFAULT,
    "single-session-assistant": _DEFAULT,
    "multi-session": _DEFAULT,
    "temporal-reasoning": _TEMPORAL,
    "knowledge-update": _KNOWLEDGE_UPDATE,
    "single-session-preference": _PREFERENCE,
}


def build_answer_check_prompt(
    question_type: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool,
) -> str:
    """Return the official answer-check prompt for one question."""
    if abstention:
        return _ABSTENTION.format(question=question, answer=answer, response=response)
    template = _TEMPLATES.get(question_type)
    if template is None:
        # Upstream raises `NotImplementedError` on an unrecognised type rather
        # than falling back to a generic template, so neither does this.
        raise ValueError(f"LongMemEval has no judge template for {question_type}")
    return template.format(question=question, answer=answer, response=response)


def parse_answer_check(response: str) -> bool:
    """Score one judge reply exactly as upstream does: `'yes' in lowered`."""
    return "yes" in response.lower()
