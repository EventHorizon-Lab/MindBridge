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

BENCHMARK_PROMPTS = (
    EGOMEM_REASON_QUERY_PROMPT,
    MEMLENS_QUERY_PROMPT,
    VIDEO_MME_QUERY_PROMPT,
    EGOTEMPO_QUERY_PROMPT,
)
