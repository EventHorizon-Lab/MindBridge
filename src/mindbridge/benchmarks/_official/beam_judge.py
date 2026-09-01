"""BEAM probing-question judge pinned from the official evaluator.

Transcribed from
`mohammadtavakoli78/BEAM@3e12035532eb85768f1a7cd779832b650c4b2ef9`:
`src/prompts.py` (`unified_llm_judge_base_prompt`) and
`src/evaluation/compute_metrics.py` (the ten `evaluate_*` functions plus
`parse_json_response`).

All ten categories run the same loop -- one judge call per rubric item, and the
category score is the mean of those per-item scores. Two upstream details are
reproduced deliberately rather than tidied up:

* The prompt declares a `<question>` placeholder, but `evaluate_*` substitutes
  only `<rubric_item>` and `<llm_response>`. The judge therefore reads the
  literal text `<question>` where the question would go. Substituting the real
  question would make every score incomparable with the published numbers.
* Nine of the ten categories accumulate with `int(response['score'])`, so a 0.5
  "partial compliance" verdict truncates to 0. Only `event_ordering` uses
  `float(...)` and keeps the half point.

Not reproduced: upstream's `event_ordering` composite
`final_score = tau_norm x f1`. It needs a second, differently shaped model call
that semantically aligns the response's event list against the rubric before
Kendall's tau can be computed, and the runner issues one judge shape per
question. That composite is reported absent rather than approximated; the
per-rubric `llm_judge_score` upstream also computes for that category is what
this module returns.
"""

from __future__ import annotations

import json
import re

# Categories whose per-item accumulation truncates a 0.5 verdict to 0, which is
# every category except `event_ordering`.
_TRUNCATING_CATEGORIES = frozenset(
    {
        "abstention",
        "contradiction_resolution",
        "information_extraction",
        "instruction_following",
        "knowledge_update",
        "multi_session_reasoning",
        "preference_following",
        "summarization",
        "temporal_reasoning",
    }
)

_FENCED = re.compile(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", re.DOTALL)
_EMBEDDED = re.compile(r"(\{.*?\}|\[.*?\])", re.DOTALL)

UNIFIED_LLM_JUDGE_BASE_PROMPT = (
    "\n"
    "You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.\n"
    "\n"
    "## EVALUATION INPUTS\n"
    "- QUESTION (what the user asked): <question>\n"
    "- RUBRIC CRITERION (what to check): <rubric_item>\n"
    "- RESPONSE TO EVALUATE: <llm_response>\n"
    "\n"
    "## EVALUATION RUBRIC:\n"
    "The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. \n"
    "\n"
    "**IMPORTANT**: Pay careful attention to whether the rubric specifies:\n"
    "- **Positive requirements** (things the response SHOULD include/do)\n"
    '- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")\n'
    "\n"
    "## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)\n"
    "A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.\n"
    "- If the response does not address the QUESTION, score **0.0** and stop.\n"
    "- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.\n"
    "\n"
    "## SEMANTIC TOLERANCE RULES:\n"
    "Judge by meaning, not exact wording.\n"
    "- Accept **paraphrases** and **synonyms** that preserve intent.\n"
    "- **Case/punctuation/whitespace** differences must be ignored.\n"
    "- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.\n"
    "- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.\n"
    "\n"
    "## STYLE NEUTRALITY (prevents style contamination):\n"
    "Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).\n"
    "- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.\n"
    "- Only evaluate format when the rubric **explicitly** mandates it.\n"
    "\n"
    "## SCORING SCALE:\n"
    "- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.\n"
    "  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).\n"
    "  - Negative: prohibited element **absent** AND response is **responsive**.\n"
    "  \n"
    "- **0.5 (Partial Compliance)**: Partially complies.\n"
    "  - Positive: element present but minor inaccuracies/incomplete execution.\n"
    "  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.\n"
    "  \n"
    "- **0.0 (No Compliance)**: Fails to comply.\n"
    "  - Positive: required element missing or incorrect.\n"
    "  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.\n"
    "\n"
    "## EVALUATION INSTRUCTIONS:\n"
    "1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).\n"
    "\n"
    '2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:\n'
    "   - **All elements** must be present for full compliance (1.0)\n"
    "   - **Some elements** present indicates partial compliance (0.5)\n"
    "   - **No elements** present indicates no compliance (0.0)\n"
    "   \n"
    "3. **Check Compliance**: \n"
    "   - For positive requirements: Look for the presence and quality of the required element\n"
    "   - For negative constraints: Look for the absence of the prohibited element\n"
    "\n"
    "4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.\n"
    "\n"
    "5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.\n"
    "\n"
    "## OUTPUT FORMAT:\n"
    "Return your evaluation in JSON format with two fields:\n"
    "\n"
    "{\n"
    '   "score": [your score: 1.0, 0.5, or 0.0],\n'
    '   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"\n'
    "}\n"
    "\n"
    "NOTE: ONLY output the json object, without any explanation before or after that\n"
)


def build_rubric_item_prompt(rubric_item: str, llm_response: str) -> str:
    """Return the judge prompt for one rubric item.

    Only the two placeholders upstream substitutes are substituted; see the
    module docstring for why `<question>` is left standing.
    """
    return UNIFIED_LLM_JUDGE_BASE_PROMPT.replace("<rubric_item>", rubric_item).replace(
        "<llm_response>", llm_response
    )


def parse_json_response(response: str) -> object:
    """Parse a judge reply the way upstream's `parse_json_response` does."""
    text = response.strip()
    if text.startswith("```"):
        match = _FENCED.search(text)
        if match is not None:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _EMBEDDED.search(text)
    if match is None:
        raise ValueError("No valid JSON found in response.")
    return json.loads(match.group(1))


def parse_rubric_item_score(response: str, category: str) -> float:
    """Return one rubric item's contribution under its category's accumulation."""
    payload = parse_json_response(response)
    if not isinstance(payload, dict) or "score" not in payload:
        raise ValueError("BEAM judge did not return a score")
    score = payload["score"]
    if isinstance(score, bool) or not isinstance(score, int | float | str):
        raise ValueError("BEAM judge returned a non-numeric score")
    value = float(score)
    return float(int(value)) if category in _TRUNCATING_CATEGORIES else value
