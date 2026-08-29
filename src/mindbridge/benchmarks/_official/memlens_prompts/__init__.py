"""MemLens judge prompts pinned from the official evaluator."""

from .answer_refusal import CRITERIA as AR_CRITERIA
from .answer_refusal import EXAMPLES as AR_EXAMPLES
from .base import QA_BENCHMARK_JUDGE_PROMPT
from .information_extraction import CRITERIA as IE_CRITERIA
from .information_extraction import EXAMPLES as IE_EXAMPLES
from .knowledge_update import CRITERIA as KU_CRITERIA
from .knowledge_update import EXAMPLES as KU_EXAMPLES
from .multi_session_reasoning import CRITERIA as MSR_CRITERIA
from .multi_session_reasoning import EXAMPLES as MSR_EXAMPLES
from .temporal_reasoning import CRITERIA as TR_CRITERIA
from .temporal_reasoning import EXAMPLES as TR_EXAMPLES

TASK_CRITERIA = {
    **IE_CRITERIA,
    **KU_CRITERIA,
    **TR_CRITERIA,
    **MSR_CRITERIA,
    **AR_CRITERIA,
}
TASK_EXAMPLES = {
    **IE_EXAMPLES,
    **KU_EXAMPLES,
    **TR_EXAMPLES,
    **MSR_EXAMPLES,
    **AR_EXAMPLES,
}


def get_task_key(  # noqa: C901 - pinned upstream rubric dispatch
    question_type: str, question_subtype: str, reference: str
) -> str:
    """Map official MemLens question metadata to its judge rubric."""
    if question_type == "multi_session_reasoning":
        if question_subtype == "arithmetic":
            return "MSR_Arithmetic"
        if question_subtype == "counting":
            return "MSR_Counting"
        if question_subtype == "entity_resolution":
            return "MSR_YesNo" if reference.strip() in ("Yes", "No") else "MSR_Counting"
        return "MSR_Counting"
    if question_type == "temporal_reasoning":
        if question_subtype == "duration_comparison":
            return "TR_DurationComparison"
        if question_subtype == "order_ranking":
            return "TR_OrderRanking"
        return "TR_DateExtraction"
    if question_type == "knowledge_update":
        return "KU_KnowledgeUpdate"
    if question_type == "answer_refusal":
        return "AR_AnswerRefusal"
    if question_type == "information_extraction" and question_subtype == "previnfo":
        return "IE_PreviousInfo"
    return "IE_Entity"


def build_judge_prompt(
    task_type: str,
    question: str,
    reference: str,
    prediction: str,
    old_answer: str | None = None,
) -> str:
    """Build the official grading-teacher prompt."""
    prompt = QA_BENCHMARK_JUDGE_PROMPT.format(
        task_criteria=TASK_CRITERIA.get(task_type, ""),
        examples=TASK_EXAMPLES.get(task_type, ""),
    )
    if task_type == "KU_KnowledgeUpdate" and old_answer:
        current = f"""[Current Case]
<Question>: {question}
<Standard Answer>: {reference}
<Old (Outdated) Answer>: {old_answer}
<Student Answer>: {prediction}

"""
    elif task_type == "AR_AnswerRefusal":
        current = f"""[Current Case]
<Question>: {question}
<Standard Answer>: Insufficient information
<Student Answer>: {prediction}

"""
    else:
        current = f"""[Current Case]
<Question>: {question}
<Standard Answer>: {reference}
<Student Answer>: {prediction}

"""
    return prompt + current + "[Scoring Rationale]:"
