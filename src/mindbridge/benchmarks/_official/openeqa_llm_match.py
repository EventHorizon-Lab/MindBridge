"""OpenEQA LLM-Match prompts and parser pinned from the official evaluator.

Transcribed from `facebookresearch/open-eqa@cfa3fce4595c`:
`prompts/mmbench.txt`, `prompts/mmbench-extra.txt`,
`openeqa/evaluation/llm_match.py` and `evaluate-predictions.py`.

The judge returns one integer 1-5. Upstream's headline is
`mean(100 * (clip(score, 1, 5) - 1) / 4)`, so the clip -- not a range check --
is what a score outside 1-5 meets.
"""

from __future__ import annotations

from collections.abc import Sequence

# `prompts/mmbench.txt`, used when the release publishes no `extra_answers`.
_MMBENCH = """You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer.
1 means that the response is completely different from the answer.

Example 1:
Question: Is it overcast?
Answer: no
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Response: {prediction}
"""

# `prompts/mmbench-extra.txt`, used when it does. The three worked examples
# differ from `mmbench.txt` only by their `Extra Answers` line, whose quoting is
# upstream's own -- `' it's sunny'` keeps its leading space and `'doesn't look
# like it'` its apostrophe.
_MMBENCH_EXTRA = """You are an AI assistant who will help me to evaluate the response given the question, the correct answer, and extra answers that are also correct.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer or any of the extra answers.
1 means that the response is completely different from the answer and all of the extra answers.

Example 1:
Question: Is it overcast?
Answer: no
Extra Answers: ['doesn't look like it', 'no',' it's sunny']
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Extra Answers: ['a woman', 'a lady', 'woman']
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Extra Answers: ['yes, there are drapes', 'yeah', 'the drapes are to the right of the king bed']
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Extra Answers: {extra_answers}
Response: {prediction}
"""

SCORE_TAG = "Your mark:"
MIN_SCORE = 1
MAX_SCORE = 5


def build_llm_match_prompt(
    question: str,
    answer: str,
    prediction: str,
    extra_answers: Sequence[str] | None,
) -> str:
    """Fill the prompt `get_llm_match_score` selects for this question."""
    if extra_answers is None:
        return _MMBENCH.format(question=question, answer=answer, prediction=prediction)
    # `prompt.format(extra_answers=extra_answers)` renders the list with
    # `str()`, so the judge sees a Python list repr including any blank entry.
    return _MMBENCH_EXTRA.format(
        question=question,
        answer=answer,
        prediction=prediction,
        extra_answers=list(extra_answers),
    )


def parse_llm_match_score(output: str, tag: str = SCORE_TAG) -> int:
    """Read the judge's mark exactly as `llm_match.parse_score` does.

    `output.isdigit()` is applied to the unstripped response, and the tagged
    branch takes the text between the tag and the next newline -- so a judge
    that explains itself on later lines still parses.
    """
    if output.isdigit():
        return int(output)
    start_index = output.find(tag)
    if start_index == -1:
        raise ValueError(f"Invalid output string: {output}")
    end_index = output.find("\n", start_index)
    if end_index == -1:
        return int(output[start_index:].replace(tag, "").strip())
    return int(output[start_index:end_index].replace(tag, "").strip())


def truncate_prediction(prediction: str) -> str:
    """Drop a trailing unterminated sentence, as `evaluate-predictions.py` does.

    Upstream cuts after the last `.` only when that period is not already the
    final character, so `"A rug. It is"` becomes `"A rug."` while `"A rug."` is
    left alone. An empty prediction is passed through untouched.
    """
    if not prediction:
        return prediction
    end_index = prediction.rfind(".")
    if end_index >= 0 and end_index + 1 < len(prediction):
        return prediction[: end_index + 1]
    return prediction
