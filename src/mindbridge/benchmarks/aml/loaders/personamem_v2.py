"""PersonaMem v2 -> AML case loader.

Turns the official `benchmark.csv` plus per-persona chat-history JSON files
into the benchmark-neutral `AmlCase` model (see
`mindbridge.benchmarks.aml.cases`). v2 is a different dataset from v1, not
another split of it: different columns, a different id-less CSV shape, and
a different judge-selection payload key (`preference`) that
`evaluate-narrow` reads to pick between two judge prompts. This loader does
not chunk messages and does not evaluate anything -- a later driver ingests
each case's `messages` into MindBridge under its `user_id`, then feeds the
retrieved context, plus each question's payload, to the vendored
`personamem/pipeline_v2.py`.

### Modelling tension: same hazard as v1, different shape

v1 needed splitting because several questions were generated against
different *prefixes* of one shared history. v2's version of the same hazard
would be several questions generated against different *snapshots* of one
persona -- `chat_history_32k_link` is file-specific, and two on-disk files
do exist for persona 0 under `data/chat_history_32k/` (different generation
timestamps). Measured directly against the real
`benchmark/text/benchmark.csv` (5000 rows, 200 personas), though: every row
for a given `persona_id` points at exactly the same `chat_history_32k_link`
-- 0 of 200 personas differ. So unlike v1, `persona_id` alone is already
the correct, un-leaky retrieval scope for this release, and `user_id` needs
no snapshot suffix. `_group_rows` still asserts this invariant per persona
(raising loudly rather than silently picking one link) so that if a future
corpus revision does violate it, loading fails fast instead of quietly
leaking one snapshot's content into another's questions.

### No id column

The CSV has no `id`/`question_id`/`qid`/`sample_id` column at all. Ids are
synthesized as `f"persona{persona_id}-{row_index}"`, where `row_index` is
the row's 0-based position in file-read order across the whole CSV -- this
is stable across reruns (CSV row order is deterministic), which the
pipeline's `answer` and `evaluate-*` steps both require in order to agree
on the same ids.
"""

from __future__ import annotations

import csv
import json
from ast import literal_eval as parse_python_literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion


class _RawRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    persona_id: str
    chat_history_32k_link: str
    user_query: str
    correct_answer: str
    incorrect_answers: str
    preference: str


def load(benchmark_csv: Path, data_root: Path) -> tuple[AmlCase, ...]:
    """Load the official PersonaMem v2 corpus into benchmark-neutral AML cases."""
    rows = _read_rows(benchmark_csv)
    grouped, links = _group_rows(rows)
    histories = {persona_id: _read_history(data_root, link) for persona_id, link in links.items()}
    return tuple(
        _case(persona_id, indexed_rows, histories[persona_id])
        for persona_id, indexed_rows in grouped.items()
    )


def _read_rows(path: Path) -> list[_RawRow]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return TypeAdapter(list[_RawRow]).validate_python(rows)


def _group_rows(
    rows: list[_RawRow],
) -> tuple[dict[str, list[tuple[int, _RawRow]]], dict[str, str]]:
    """Group rows by `persona_id`, in file-read order, tracking each persona's link.

    Also verifies the measured real-corpus invariant that one `persona_id`
    never spans more than one `chat_history_32k_link` -- see the module
    docstring. If that ever stops holding, this raises rather than silently
    ingesting the wrong (or a merged) history for the group.
    """
    grouped: dict[str, list[tuple[int, _RawRow]]] = {}
    links: dict[str, str] = {}
    for index, row in enumerate(rows):
        seen_link = links.setdefault(row.persona_id, row.chat_history_32k_link)
        if seen_link != row.chat_history_32k_link:
            raise ValueError(
                f"persona {row.persona_id!r} has more than one "
                f"chat_history_32k_link ({seen_link!r} vs "
                f"{row.chat_history_32k_link!r}) -- this loader scopes cases "
                "by persona_id alone, which the real corpus supports (0/200 "
                "personas differed), but this row breaks that assumption. "
                "Scope by (persona_id, chat_history_32k_link) instead, the "
                "way personamem_v1 scopes by (shared_context_id, end_index)."
            )
        grouped.setdefault(row.persona_id, []).append((index, row))
    return grouped, links


def _read_history(data_root: Path, link: str) -> tuple[dict[str, object], ...]:
    raw = json.loads((data_root / link).read_text())
    return tuple(
        {"role": turn["role"], "content": turn["content"], "timestamp": None}
        for turn in raw["chat_history"]
    )


def _case(
    persona_id: str,
    indexed_rows: list[tuple[int, _RawRow]],
    history: tuple[dict[str, object], ...],
) -> AmlCase:
    questions = tuple(_question(persona_id, index, row) for index, row in indexed_rows)
    return AmlCase(
        user_id=f"personamem-v2:{persona_id}",
        messages=history,
        questions=questions,
    )


def _question(persona_id: str, row_index: int, row: _RawRow) -> AmlQuestion:
    question_id = f"persona{persona_id}-{row_index}"
    return AmlQuestion(
        question_id=question_id,
        question=_question_text(row.user_query),
        payload={
            "id": question_id,
            "user_query": row.user_query,
            "correct_answer": row.correct_answer,
            "incorrect_answers": _incorrect_answers(row.incorrect_answers),
            "persona_id": persona_id,
            "preference": row.preference,
        },
    )


def _question_text(user_query: str) -> str:
    """Extract the plain question text from the CSV's Python-repr dict string.

    `user_query` is written as a single-quoted dict literal (e.g.
    `"{'role': 'user', 'content': '...'}"`), not JSON -- matches the
    pipeline's own `ast.literal_eval` fallback in `user_query_text()`
    (imported here under an alias only to keep the call site readable; it
    is the same safe, literal-only parser, never the `eval` builtin). The
    raw string is *also* kept verbatim in the payload's `user_query` key,
    since the vendored pipeline parses it itself at answer/evaluate time.
    """
    parsed = parse_python_literal(user_query)
    return str(parsed["content"])


def _incorrect_answers(raw: str) -> list[str]:
    """Parse the CSV's JSON-encoded `incorrect_answers` list.

    Verified against the real corpus: always valid JSON, always a
    non-empty list of strings (5000/5000 rows). The vendored pipeline
    raises `TypeError` on an empty or missing list, so a malformed row is
    not defended against further here -- it should fail loudly rather than
    silently produce a broken MCQ.
    """
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"incorrect_answers must be a non-empty list, got {raw!r}")
    return [str(item) for item in parsed]
