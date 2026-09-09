"""WorldMemArena's released checkpoint-QA judge prompt."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence


def build_qa_prompt(
    question: str, reference_answer: str, key_memory_points: Sequence[str], response: str
) -> str:
    points = "\n".join(point for point in key_memory_points if point) or "No evidence available."
    return f"""You are an **evaluation expert for AI memory system question answering**.
Based **only** on the provided **\"Question\"**, **\"Reference Answer\"**, and **\"Key Memory Points\"**, strictly evaluate the **accuracy** of the **\"Memory System Response.\"** Classify it as one of **\"Correct\"**, **\"Hallucination\"**, or **\"Omission.\"** Do **not** use any external knowledge or subjective inference.

# Evaluation Criteria

### 1. Correct
* The response accurately answers the question and is **semantically equivalent** to the Reference Answer.
* No contradictions with Key Memory Points or Reference Answer.
* Synonyms, paraphrasing, and reasonable summarization are acceptable.

### 2. Hallucination
* The response includes information that **contradicts** the Reference Answer or Key Memory Points.
* When the Reference Answer is *unknown/uncertain*, yet the response provides a specific fact.

### 3. Omission
* The response is **incomplete** compared to the Reference Answer.
* It states \"don't know\" or \"no related memory\" even though relevant information exists.
* For multi-element questions, missing **any** element counts as Omission.

## Priority Rules
* Both missing info AND fabricated info -> **Hallucination**.
* No fabrication but missing info -> **Omission**.
* Fully equivalent -> **Correct**.

# Information

* **Question:** {question}
* **Reference Answer:** {reference_answer}
* **Key Memory Points:** {points}
* **Memory System Response:** {response}

# Output

```json
{{
  \"reasoning\": \"Concise evaluation rationale\",
  \"evaluation_result\": \"Correct | Hallucination | Omission\"
}}
```"""


def parse_qa_response(response: str) -> Mapping[str, float]:
    candidate = response.strip()
    if candidate.startswith("```"):
        candidate = candidate.removeprefix("```json").removeprefix("```").removesuffix("```")
    payload = json.loads(candidate.strip())
    if not isinstance(payload, dict):
        raise ValueError("WorldMemArena judge did not return a JSON object")
    label = str(payload.get("evaluation_result", "Omission")).strip()
    if label not in {"Correct", "Hallucination", "Omission"}:
        label = "Omission"
    return {
        "correct_ratio": float(label == "Correct"),
        "hallucination_ratio": float(label == "Hallucination"),
        "omission_ratio": float(label == "Omission"),
    }
