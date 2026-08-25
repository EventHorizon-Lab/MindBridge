"""Official query wordings owned by the benchmark adapters, not by the product.

These reproduce each dataset's published instruction so a run is comparable with the
leaderboard. They are deliberately kept out of `mindbridge.prompts`: nothing on the
observe/recall path may read them, and they must never influence the shared answer policy.
"""

from mindbridge.prompts import PromptSpec

EGOMEM_REASON_QUERY_PROMPT = PromptSpec(
    name="egomem_reason_query",
    version="egomem_reason_query_v1",
    purpose="Add the official query-time reference to an EgoMemReason question.",
    used_by="mindbridge.benchmarks.egolife_runner._answer_egomem_question",
    text="Query time reference: {query_time}\n{question_with_options}",
)

MEMLENS_QUERY_PROMPT = PromptSpec(
    name="memlens_query",
    version="memlens_query_v1",
    purpose="Add MEMLENS question time and its required abstention response.",
    used_by="mindbridge.benchmarks.memlens_runner._question_query",
    text=(
        "Question date: {question_date}\n{question}\n\n"
        'If the memories are insufficient, answer exactly "Insufficient information".'
    ),
)

VIDEO_MME_QUERY_PROMPT = PromptSpec(
    name="video_mme_query",
    version="video_mme_query_v1",
    purpose="Apply the official Video-MME multiple-choice answer instruction.",
    used_by="mindbridge.benchmarks.video_mme._answer_question",
    text=(
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        "{question}\n{options}\nThe best answer is:"
    ),
)

EGOTEMPO_QUERY_PROMPT = PromptSpec(
    name="egotempo_query",
    version="egotempo_query_v1",
    purpose="Apply the official EgoTempo open-question instruction.",
    used_by="mindbridge.benchmarks.egotempo._answer_question",
    text=(
        "These are frames from a video that I want to upload. Use the visual cues to answer the "
        "question: {question}. You need to answer the question in any case and not demand "
        "additional context information. Note: All actions mentioned refer to the person "
        "recording the video."
    ),
)

ATM_BENCH_QUERY_PROMPT = PromptSpec(
    name="atm_bench_query",
    version="atm_bench_query_v1",
    purpose="Apply the official ATM-Bench per-type answer instruction.",
    used_by="mindbridge.benchmarks.atm_bench_runner._question_query",
    text="{question}\n{format_constraint}",
)

ATM_BENCH_NUMBER_FORMAT_PROMPT = PromptSpec(
    name="atm_bench_number_format",
    version="atm_bench_number_format_v1",
    purpose=(
        "Ask an ATM-Bench `number` question for a bare, machine-parseable numeric answer. "
        "MindBridge-authored, not transcribed from ATM-Bench: unlike the Mem-Gallery "
        "constraints, no released ATM-Bench prompt file names this wording -- the Oracle "
        "baseline gives every qtype one shared system prompt (`oracle/config.py`'s "
        "ORACLE_SYSTEM) rather than a `number`-specific instruction of its own."
    ),
    used_by="mindbridge.benchmarks.prompts.atm_format_constraint",
    text="Answer with the number alone, including its unit or currency symbol.",
)

ATM_BENCH_LIST_RECALL_FORMAT_PROMPT = PromptSpec(
    name="atm_bench_list_recall_format",
    version="atm_bench_list_recall_format_v1",
    purpose=(
        "Ask an ATM-Bench `list_recall` question for a bare, comma-separated evidence-ID "
        "answer. MindBridge-authored, not transcribed from ATM-Bench: the Oracle baseline's "
        "only list-recall wording is one clause inside its shared ORACLE_SYSTEM prompt, not a "
        "`list_recall`-specific instruction of its own."
    ),
    used_by="mindbridge.benchmarks.prompts.atm_format_constraint",
    text=("Answer with the matching evidence IDs alone, separated by commas, and nothing else."),
)

ATM_BENCH_OPEN_END_FORMAT_PROMPT = PromptSpec(
    name="atm_bench_open_end_format",
    version="atm_bench_open_end_format_v1",
    purpose=(
        "Ask an ATM-Bench `open_end` question for a concise, evidence-grounded answer. "
        "MindBridge-authored, not transcribed from ATM-Bench: the Oracle baseline has no "
        "`open_end`-specific wording at all, only its shared ORACLE_SYSTEM prompt."
    ),
    used_by="mindbridge.benchmarks.prompts.atm_format_constraint",
    text="Answer concisely, using only what the memories support.",
)

_ATM_BENCH_CONSTRAINTS = {
    "number": ATM_BENCH_NUMBER_FORMAT_PROMPT,
    "list_recall": ATM_BENCH_LIST_RECALL_FORMAT_PROMPT,
    "open_end": ATM_BENCH_OPEN_END_FORMAT_PROMPT,
}


def atm_format_constraint(qtype: str) -> str:
    """Return one ATM-Bench question type's format instruction.

    Unlike `mem_gallery_format_constraint`, every one of ATM's three `qtype`s gets an
    instruction -- there is no "no constraint" case to fall back to -- so an unrecognised
    qtype is a caller bug and raises `KeyError` rather than silently returning empty text.
    """
    return _ATM_BENCH_CONSTRAINTS[qtype].text


MEM_GALLERY_QUERY_PROMPT = PromptSpec(
    name="mem_gallery_query",
    version="mem_gallery_query_v1",
    purpose="Apply the official Mem-Gallery concise-answer instruction.",
    used_by="mindbridge.benchmarks.mem_gallery_runner._question_query",
    text=(
        "Your task is to answer the question about the conversation between {speaker_a} "
        "and {speaker_b} in a concise manner with the help of memory content.\n"
        "Please only provide the content of the answer, without including introductory "
        "phrases like 'answer:'.\n"
        "For questions that require answering a date or time, strictly follow the format "
        "and provide a specific date or time whenever possible.\n"
        "Generate answers primarily concise, yet complete enough to accurately answer the "
        "questions.\n\n"
        "The current question is as follows:\n{question} {format_constraint}"
    ),
)

MEM_GALLERY_REFUSAL_PROMPT = PromptSpec(
    name="mem_gallery_refusal",
    version="mem_gallery_refusal_v1",
    purpose="Carry the official Mem-Gallery answer-refusal constraint for `AR` questions.",
    used_by="mindbridge.benchmarks.prompts.mem_gallery_format_constraint",
    text=(
        "Provide your answer based on the information in the conversation. Only if the "
        "information about the question is not present in the conversation, reply with: "
        "“Not mentioned.”"
    ),
)

MEM_GALLERY_CONFLICT_PROMPT = PromptSpec(
    name="mem_gallery_conflict",
    version="mem_gallery_conflict_v1",
    purpose="Carry the official Mem-Gallery conflict-detection constraint for `CD` questions.",
    used_by="mindbridge.benchmarks.prompts.mem_gallery_format_constraint",
    text=(
        "Please check whether this information conflicts with the conversation, and reply "
        "strictly with either “Yes.” or “No.”"
    ),
)

MEM_GALLERY_SEARCH_PROMPT = PromptSpec(
    name="mem_gallery_search",
    version="mem_gallery_search_v1",
    purpose="Carry the official Mem-Gallery visual-search constraint for `VS` questions.",
    used_by="mindbridge.benchmarks.prompts.mem_gallery_format_constraint",
    text=(
        "Return the image_id of the image(s). If there are multiple images, sort them in "
        "ascending order and separate them by commas. Format example: “D2:IMG_003, "
        "D2:IMG_010, D10:IMG_002” (for format reference only)."
    ),
)

_MEM_GALLERY_CONSTRAINTS = {
    "AR": MEM_GALLERY_REFUSAL_PROMPT,
    "CD": MEM_GALLERY_CONFLICT_PROMPT,
    "VS": MEM_GALLERY_SEARCH_PROMPT,
}


def mem_gallery_format_constraint(point: str) -> str:
    """Return one task type's official constraint, already separated, or empty text.

    The official runner applies a constraint file for `AR`, `CD` and `VS` only; the other
    six task types are asked without one, and adding one would change the task. It also
    prefixes the constraint with a blank line -- `format_constraint_str = "\\n\\n" +
    format_constraint` -- and interpolates it after a literal space, so a constrained
    question renders as its own paragraph. The separator belongs here rather than in the
    template, because this is the only place that knows whether a constraint exists at all,
    and it keeps each `PromptSpec.text` byte-identical to the upstream `.txt` file it
    reproduces.
    """
    prompt = _MEM_GALLERY_CONSTRAINTS.get(point.upper())
    return f"\n\n{prompt.text}" if prompt is not None else ""


BENCHMARK_PROMPTS = (
    EGOMEM_REASON_QUERY_PROMPT,
    MEMLENS_QUERY_PROMPT,
    VIDEO_MME_QUERY_PROMPT,
    EGOTEMPO_QUERY_PROMPT,
    ATM_BENCH_QUERY_PROMPT,
    ATM_BENCH_NUMBER_FORMAT_PROMPT,
    ATM_BENCH_LIST_RECALL_FORMAT_PROMPT,
    ATM_BENCH_OPEN_END_FORMAT_PROMPT,
    MEM_GALLERY_QUERY_PROMPT,
    MEM_GALLERY_REFUSAL_PROMPT,
    MEM_GALLERY_CONFLICT_PROMPT,
    MEM_GALLERY_SEARCH_PROMPT,
)
