"""Official query wordings owned by the benchmark adapters, not by the product.

These reproduce each dataset's published instruction so a run is comparable with the
leaderboard. They are deliberately kept out of `mindbridge.prompts`: nothing on the
observe/recall path may read them, and they must never influence the shared answer policy.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Versioned benchmark prompt kept out of the product prompt contract."""

    name: str
    version: str
    purpose: str
    used_by: str
    text: str
    # The exact refusal a dataset mandates, when it mandates one of its own. `AnswerResult`
    # reports `abstained` for the product's own refusal sentence only, which is correct: the
    # product cannot know what wording a caller told the model to use instead. A task that
    # overrides the wording has to declare it here, or its run reports a refusal rate of zero
    # while the model refuses. MEMLENS did exactly that: `results.json` said 0 abstentions on a
    # dev split where 16 of 59 answers were refusals.
    refusal: str | None = None


EGOMEM_REASON_QUERY_PROMPT = PromptSpec(
    name="egomem_reason_query",
    version="egomem_reason_query_v1",
    purpose="Add the official query-time reference to an EgoMemReason question.",
    used_by="mindbridge.benchmarks.egolife_runner._answer_egomem_question",
    text="Query time reference: {query_time}\n{question_with_options}",
)

MEMLENS_QUERY_PROMPT = PromptSpec(
    name="memlens_query",
    version="memlens_query_v2",
    purpose="Add the required MEMLENS abstention response.",
    used_by="mindbridge.benchmarks.memlens_runner._question_query",
    text=(
        '{question}\n\nIf the memories are insufficient, answer exactly "Insufficient information".'
    ),
    refusal="Insufficient information",
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

VIDEO_MME_V2_QUERY_PROMPT = PromptSpec(
    name="video_mme_v2_query",
    version="video_mme_v2_query_v1",
    purpose="Apply the official Video-MME-v2 multiple-choice answer instruction.",
    used_by="mindbridge.benchmarks.video_mme_v2._answer_question",
    # The published prompt, minus its "These are the frames of a video." opener: MindBridge
    # answers from memory rather than from a frame stack handed to a VLM, and claiming frames
    # that are not in the context is the kind of drift that makes a score unquotable. Note the
    # released standalone script emits the instruction *after* the question while the README
    # documents it before; the README wording is the one the benchmark publishes as canonical.
    # Every question is offered A through H here even where it lists fewer options, because
    # that is what the official instruction says regardless of the option count.
    text=(
        "Select the best answer to the following multiple-choice question based on the video.\n"
        "Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option.\n"
        "Question: {question}\n{options}"
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

OPENEQA_QUERY_PROMPT = PromptSpec(
    name="openeqa_query",
    version="openeqa_query_v1",
    purpose="Apply the official OpenEQA EM-EQA baseline instruction.",
    used_by="mindbridge.benchmarks.eval_adapters._openeqa",
    # Transcribed from `prompts/gpt4v.txt`, which OpenEQA ships byte-identically
    # as `claude3-vision.txt` and `gemini-pro-vision.txt`: one shared wording for
    # every EM-EQA vision baseline. The second line describes that baseline's
    # input medium -- a set of images pasted into one request -- and is kept
    # verbatim anyway, because the published wording is what makes a number
    # comparable with the leaderboard. MindBridge instead retrieves the episode's
    # own stored observations.
    text=(
        "You are an intelligent question answering agent. I will ask you questions about an "
        "indoor space and you must provide an answer.\n"
        "You will be shown a set of images that have been collected from a single location.\n"
        "Given a user query, you must output `text` to answer to the question asked by the "
        "user.\n"
        "User Query: {question}"
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
    refusal="Not mentioned.",
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


CLBENCH_QUERY_PROMPT = PromptSpec(
    name="clbench_query",
    version="clbench_query_v1",
    purpose="Carry the CL-Bench record's own system prompt alongside its question.",
    used_by="mindbridge.benchmarks.eval_adapters._clbench",
    # CL-Bench publishes no answer-generation prompt: `infer.py` posts the
    # record's `messages` unchanged, so the system turn is the only instruction
    # the graded model ever sees. Its rubrics grade formatting and style rules
    # stated there, so dropping it would fail rubric items the response never
    # had a chance to satisfy. The reference document travels through memory
    # instead of through this prompt.
    text="{system_prompt}\n\n{question}",
)

BEAM_QUERY_PROMPT = PromptSpec(
    name="beam_query",
    version="beam_query_v1",
    purpose="Apply BEAM's official retrieval-answer instruction.",
    used_by="mindbridge.benchmarks.eval_adapters._beam",
    # `src/prompts.py: answer_generation_for_rag`, minus its `CONTEXT:\n<context>`
    # block. That slot is where BEAM's own harness pastes the passages its
    # retriever returned; MindBridge answers from its own recall, and pasting an
    # empty context section would tell the model to answer from nothing. Every
    # instruction that shapes the answer -- answer only from context, do not use
    # internal knowledge, be direct and concise, output only the answer -- is
    # kept verbatim, because the rubric judge grades against them.
    text=(
        "You are an assistant that MUST answer questions using ONLY the information provided "
        "in the context below. \n\n"
        "STRICT INSTRUCTIONS:\n"
        "1. Answer ONLY based on the provided context\n"
        "2. Do NOT use your internal knowledge\n\n"
        "QUESTION:\n{question}\n\n"
        "ANSWER REQUIREMENTS:\n"
        "- Be direct and concise\n"
        "- Only output the answer to the question without any explanation \n\n"
        "RESPONSE:"
    ),
)

PERSONAMEM_V3_RANKING_PROMPT = PromptSpec(
    name="personamem_v3_ranking",
    version="personamem_v3_ranking_v1",
    purpose="Ask a PersonaMem-v3 slate question in the release's own answer format.",
    used_by="mindbridge.benchmarks.eval_adapters._personamem_v3",
    # The answer format is taken from the pinned release rather than from the
    # evaluation repository: every ranking row's `example_response` is
    # `"Ranked indexes: [13, 5, ...]"`, while the repository's current
    # `slate_ranking_prompt` asks for a fenced-JSON `ranked_indices`. The two
    # have drifted; the released gold is what the released data can be scored
    # against. The scorer accepts either shape, so a model that answers in the
    # repository's format is not penalised.
    text=(
        "{query}\n\n"
        "## Candidate slate (order is random)\n{slate}\n\n"
        "## Your job\n"
        "Rank the {count} candidates from most to least likely that **this specific user**, at "
        "this moment, would positively engage with, using only the memories. Penalize "
        "candidates the user has disliked or would find irrelevant right now.\n\n"
        "## Output\n"
        "Reply with one line and nothing else, listing every index exactly once:\n"
        "Ranked indexes: [<idx>, <idx>, ...]"
    ),
)

PERSONAMEM_V3_QUERY_PROMPT = PromptSpec(
    name="personamem_v3_query",
    version="personamem_v3_query_v1",
    purpose="Ask a non-ranking PersonaMem-v3 query as the released row states it.",
    used_by="mindbridge.benchmarks.eval_adapters._personamem_v3",
    # PersonaMem-v3's runners build a different prompt per task family, and the
    # released rows carry the already-rendered `user_query` those runners hand
    # to the model -- including the bracketed scenario labels the proactive and
    # ranking rows use. It is passed through unchanged; only the prior
    # conversation, where a row has one, is prepended as context.
    text="{question}",
)

BENCHMARK_PROMPTS = (
    EGOMEM_REASON_QUERY_PROMPT,
    MEMLENS_QUERY_PROMPT,
    VIDEO_MME_QUERY_PROMPT,
    VIDEO_MME_V2_QUERY_PROMPT,
    EGOTEMPO_QUERY_PROMPT,
    OPENEQA_QUERY_PROMPT,
    ATM_BENCH_QUERY_PROMPT,
    ATM_BENCH_NUMBER_FORMAT_PROMPT,
    ATM_BENCH_LIST_RECALL_FORMAT_PROMPT,
    ATM_BENCH_OPEN_END_FORMAT_PROMPT,
    MEM_GALLERY_QUERY_PROMPT,
    MEM_GALLERY_REFUSAL_PROMPT,
    MEM_GALLERY_CONFLICT_PROMPT,
    MEM_GALLERY_SEARCH_PROMPT,
    CLBENCH_QUERY_PROMPT,
    BEAM_QUERY_PROMPT,
    PERSONAMEM_V3_RANKING_PROMPT,
    PERSONAMEM_V3_QUERY_PROMPT,
)
