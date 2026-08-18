"""Cross-seam test: every loader's messages must satisfy the AML add contract.

Blocking 1 (final review, 2026-08-17): `AmlAddRequest`/`AmlMessage`
(`mindbridge.api.aml_contracts`) type `timestamp` as `int | None` -- AML's
wire contract, epoch milliseconds. Three loaders (`locomo_refined`, `longmemeval`,
`beam`) emitted free-text dates instead (`"1:56 pm on 8 May, 2023"`,
`"2023/05/20 (Sat) 02:21"`, `"March-15-2024"`), so the very first
`/aml/add` of a real run 422'd.

Every existing loader test asserts only the loader's own output shape, and
every route test asserts only the contract's shape -- nothing crossed the
seam between "what a loader emits" and "what the wire contract accepts",
which is exactly why this shipped unnoticed. This module feeds each of the
six loaders' real output through `AmlAddRequest.model_validate`, the same
validation `POST /aml/add` performs, so a loader that regresses to a
non-contract timestamp fails here -- in CI, in milliseconds -- rather than
on the first request of a real benchmark run.

Fixtures are hand-built (small, inline) rather than read from
`.benchmarks/`, which is gitignored and not available in CI.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mindbridge.api.aml_contracts import AmlAddRequest
from mindbridge.benchmarks.aml.cases import AmlCase, chunk_messages
from mindbridge.benchmarks.aml.loaders import (
    beam,
    clbench,
    locomo_refined,
    longmemeval,
    personamem_v1,
    personamem_v2,
)


def _assert_wire_compatible(case: AmlCase) -> None:
    """Replay exactly what `driver.run_case` sends to `/aml/add`."""
    for index, chunk in enumerate(chunk_messages(case.messages)):
        AmlAddRequest.model_validate(
            {
                "request_id": f"{case.user_id}:chunk-{index}",
                "messages": list(chunk),
                "user_id": case.user_id,
                "session_id": case.user_id,
            }
        )


def test_locomo_refined_loader_output_satisfies_the_add_wire_contract(
    tmp_path: Path,
) -> None:
    fixture = [
        {
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_1": [
                    {"speaker": "Alice", "dia_id": "D1:1", "text": "hi from alice"},
                    {"speaker": "Bob", "dia_id": "D1:2", "text": "hi from bob"},
                ],
            },
            "qa": [{"question": "Q1?", "answer": ["A1"], "evidence": ["D1:1"], "category": 1}],
        }
    ]
    path = tmp_path / "locomo_refined.json"
    path.write_text(json.dumps(fixture))

    for case in locomo_refined.load(path):
        _assert_wire_compatible(case)


def test_longmemeval_loader_output_satisfies_the_add_wire_contract(tmp_path: Path) -> None:
    fixture = [
        {
            "question_id": "e47becba",
            "question_type": "single-session-user",
            "question": "What degree did I graduate with?",
            "answer": "Business Administration",
            "question_date": "2023/05/30 (Tue) 23:40",
            "haystack_dates": ["2023/05/20 (Sat) 02:21"],
            "haystack_session_ids": ["sess_0"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I graduated with a degree in Biology"},
                    {"role": "assistant", "content": "Congratulations!"},
                ]
            ],
            "answer_session_ids": ["sess_0"],
        }
    ]
    path = tmp_path / "longmemeval_s"
    path.write_text(json.dumps(fixture))

    for case in longmemeval.load(path):
        _assert_wire_compatible(case)


def test_beam_loader_output_satisfies_the_add_wire_contract(tmp_path: Path) -> None:
    chat = [
        {
            "batch_number": 1,
            "turns": [
                [
                    {
                        "role": "user",
                        "id": 0,
                        "time_anchor": "March-15-2024",
                        "content": "I'm planning a Flask app.",
                    },
                    {"role": "assistant", "id": 1, "content": "Great, let's start."},
                ]
            ],
        }
    ]
    questions = {
        "abstention": [
            {
                "question": "What database migrations have I run?",
                "rubric": ["no information about database migrations"],
            }
        ]
    }
    conv_dir = tmp_path / "100K" / "1"
    conv_dir.mkdir(parents=True)
    chat_path = conv_dir / "chat.json"
    chat_path.write_text(json.dumps(chat))
    pq_dir = conv_dir / "probing_questions"
    pq_dir.mkdir()
    questions_path = pq_dir / "probing_questions.json"
    questions_path.write_text(json.dumps(questions))

    for case in beam.load(chat_path, questions_path):
        _assert_wire_compatible(case)


def test_clbench_loader_output_satisfies_the_add_wire_contract(tmp_path: Path) -> None:
    record = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Some background.\n\nWhat do Sightings Cards do?"},
        ],
        "rubrics": ["mentions Sightings Cards"],
        "metadata": {"task_id": "task-1"},
    }
    path = tmp_path / "CL-bench.jsonl"
    path.write_text(json.dumps(record) + "\n")

    for case in clbench.load(path):
        _assert_wire_compatible(case)


def test_personamem_v1_loader_output_satisfies_the_add_wire_contract(tmp_path: Path) -> None:
    contexts_path = tmp_path / "shared_contexts_32k.jsonl"
    contexts_path.write_text(
        json.dumps(
            {
                "ctx-1": [
                    {"role": "user", "content": "User: hi"},
                    {"role": "assistant", "content": "Assistant: hello"},
                ]
            }
        )
        + "\n"
    )
    questions_path = tmp_path / "questions_32k.csv"
    with questions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "user_question_or_message",
                "correct_answer",
                "all_options",
                "shared_context_id",
                "end_index_in_shared_context",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "question_id": "q-0001",
                "user_question_or_message": "What did the user say?",
                "correct_answer": "(a)",
                "all_options": "(a) hi\n(b) bye",
                "shared_context_id": "ctx-1",
                "end_index_in_shared_context": "2",
            }
        )

    for case in personamem_v1.load(questions_path, contexts_path):
        _assert_wire_compatible(case)


def test_personamem_v2_loader_output_satisfies_the_add_wire_contract(tmp_path: Path) -> None:
    data_root = tmp_path
    history_path = data_root / "persona0.json"
    history_path.write_text(
        json.dumps(
            {
                "chat_history": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        )
    )
    benchmark_csv = tmp_path / "benchmark.csv"
    with benchmark_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "persona_id",
                "chat_history_32k_link",
                "user_query",
                "correct_answer",
                "incorrect_answers",
                "preference",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "persona_id": "0",
                "chat_history_32k_link": "persona0.json",
                "user_query": "{'role': 'user', 'content': 'What did I say?'}",
                "correct_answer": "hi",
                "incorrect_answers": '["bye", "later"]',
                "preference": "does not matter",
            }
        )

    for case in personamem_v2.load(benchmark_csv, data_root):
        _assert_wire_compatible(case)
