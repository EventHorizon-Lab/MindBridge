"""Isolation and artifact checks for the local LoCoMo-Refined CLI."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

import pytest

from mindbridge import AnswerResult, AsyncMemory, MemoryRecord, SearchHit
from mindbridge.benchmarks import locomo_refined_cli
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.locomo_refined import (
    LoCoMoRefinedConversation,
    LoCoMoRefinedQuestion,
    LoCoMoRefinedTurn,
)
from mindbridge.benchmarks.locomo_refined_cli import _Arguments
from mindbridge.benchmarks.locomo_refined_runner import LoCoMoRefinedPrediction

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


class _FakeAsyncMemory:
    instances: ClassVar[list[_FakeAsyncMemory]] = []
    active: ClassVar[int] = 0
    peak: ClassVar[int] = 0

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.closed = False
        self.add_requests: list[tuple[str, Mapping[str, object]]] = []
        self.ask_requests: list[tuple[str, int]] = []
        self.records: list[MemoryRecord] = []
        self.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.active = 0
        cls.peak = 0

    async def __aenter__(self) -> _FakeAsyncMemory:
        type(self).active += 1
        type(self).peak = max(type(self).peak, type(self).active)
        return self

    async def __aexit__(self, *_error: object) -> None:
        self.closed = True
        type(self).active -= 1

    async def add(
        self,
        content: str,
        *,
        occurred_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MemoryRecord:
        await asyncio.sleep(0)
        assert occurred_at == NOW
        assert metadata is not None
        self.add_requests.append((content, metadata))
        dialog_id = metadata.get("dialog_id")
        assert isinstance(dialog_id, str)
        record = MemoryRecord(
            id=f"memory:{dialog_id}",
            content=content,
            created_at=NOW,
            occurred_at=occurred_at,
            metadata=metadata,
        )
        self.records.append(record)
        return record

    async def ask(self, question: str, *, limit: int = 5) -> AnswerResult:
        await asyncio.sleep(0)
        self.ask_requests.append((question, limit))
        return AnswerResult(
            answer="Hello",
            hits=tuple(
                SearchHit(
                    id=record.id,
                    content=record.content,
                    score=0.8,
                    created_at=record.created_at,
                    occurred_at=record.occurred_at,
                    metadata=record.metadata,
                )
                for record in self.records
            ),
        )


async def test_parallel_conversations_use_distinct_closed_physical_memories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeAsyncMemory.reset()
    monkeypatch.setattr(locomo_refined_cli, "AsyncMemory", _FakeAsyncMemory)
    arguments = _arguments(tmp_path, unit_concurrency=3)
    conversations = tuple(_conversation(f"sample-{index}") for index in range(6))
    run = BenchmarkRun(arguments.data_root, "locomo-refined", arguments.run_id)

    predictions, unit_dirs = await locomo_refined_cli._run_conversations(
        arguments, conversations, run
    )

    assert _FakeAsyncMemory.peak == 3
    assert len(set(unit_dirs)) == len(conversations)
    assert {instance.data_dir for instance in _FakeAsyncMemory.instances} == set(unit_dirs)
    assert all(path.parent == run.path for path in unit_dirs)
    assert all(instance.closed for instance in _FakeAsyncMemory.instances)
    assert [prediction.qa_id for prediction in predictions] == [
        f"sample-{index}#q0000" for index in range(6)
    ]
    forbidden = {"tenant", "tenant_id", "user", "user_id", "run", "run_id"}
    for instance in _FakeAsyncMemory.instances:
        assert instance.ask_requests == [("What happened?", 13)]
        assert len(instance.add_requests) == 1
        metadata = instance.add_requests[0][1]
        assert set(metadata) == {"benchmark", "sample_id", "dialog_id"}
        assert forbidden.isdisjoint(metadata)
        assert arguments.run_id not in repr((instance.add_requests, instance.ask_requests))


@pytest.mark.parametrize("failure", [RuntimeError("failed"), asyncio.CancelledError()])
async def test_unit_failures_and_cancellation_propagate_after_closing_memory(
    failure: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeAsyncMemory.reset()
    monkeypatch.setattr(locomo_refined_cli, "AsyncMemory", _FakeAsyncMemory)

    async def fail(
        memory: AsyncMemory,
        conversation: LoCoMoRefinedConversation,
        **_options: object,
    ) -> tuple[LoCoMoRefinedPrediction, ...]:
        del memory, conversation
        raise failure

    monkeypatch.setattr(locomo_refined_cli, "run_locomo_refined_conversation", fail)
    arguments = _arguments(tmp_path)
    run = BenchmarkRun(arguments.data_root, "locomo-refined", arguments.run_id)

    with pytest.raises(type(failure)):
        await locomo_refined_cli._run_conversations(arguments, (_conversation("sample"),), run)

    assert _FakeAsyncMemory.instances
    assert all(instance.closed for instance in _FakeAsyncMemory.instances)


def test_artifacts_are_official_reproducible_and_protected_by_default(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_bytes(b"official dataset\n")
    conversation = _conversation("sample")
    prediction = LoCoMoRefinedPrediction(
        qa_id="sample#q0000",
        predicted_answer="Hello",
        mindbridge_answered=True,
        mindbridge_confidence=0.8,
        mindbridge_prediction_context=("D1:1",),
    )
    first = _arguments(tmp_path / "first", dataset=dataset)
    second = _arguments(tmp_path / "second", dataset=dataset)
    first_run = BenchmarkRun(first.data_root, "locomo-refined", first.run_id)
    second_run = BenchmarkRun(second.data_root, "locomo-refined", second.run_id)
    first_dirs = (first_run.unit_dir(conversation.sample_id),)
    second_dirs = (second_run.unit_dir(conversation.sample_id),)

    locomo_refined_cli._write_artifacts(
        first, (conversation,), (prediction,), first_run, first_dirs
    )
    locomo_refined_cli._write_artifacts(
        second, (conversation,), (prediction,), second_run, second_dirs
    )

    first_manifest_path = locomo_refined_cli._manifest_path(first.output)
    second_manifest_path = locomo_refined_cli._manifest_path(second.output)
    assert first.output.read_bytes() == second.output.read_bytes()
    assert first_manifest_path.read_bytes() == second_manifest_path.read_bytes()
    row = json.loads(first.output.read_text(encoding="utf-8"))
    manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    assert row["qa_id"] == "sample#q0000"
    assert row["predicted_answer"] == "Hello"
    assert "mindbridge_trace_id" not in row
    assert manifest["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert manifest["predictions_sha256"] == hashlib.sha256(first.output.read_bytes()).hexdigest()
    assert manifest["run_id"] == "run-01"
    assert manifest["question_count"] == 1
    assert manifest["turn_count"] == 1
    assert manifest["embedding_dimension"] > 0
    assert all(
        manifest[key]
        for key in ("mindbridge_version", "zvec_version", "python_version", "platform")
    )
    assert str(tmp_path) not in json.dumps(manifest)
    assert manifest["relative_layout"]["run"] == first_run.relative_layout.as_posix()
    assert not tuple(first.output.parent.glob(".*.tmp"))

    with pytest.raises(FileExistsError, match="already exists"):
        locomo_refined_cli._write_artifacts(
            first, (conversation,), (prediction,), first_run, first_dirs
        )


def _arguments(
    root: Path,
    *,
    dataset: Path | None = None,
    unit_concurrency: int = 2,
) -> _Arguments:
    return _Arguments(
        dataset=dataset or root / "dataset.json",
        output=root / "artifacts" / "predictions.jsonl",
        data_root=root / "data",
        run_id="run-01",
        limit=None,
        unit_concurrency=unit_concurrency,
        request_concurrency=2,
        recall_limit=13,
        overwrite=False,
        resume=False,
    )


def _conversation(sample_id: str) -> LoCoMoRefinedConversation:
    return LoCoMoRefinedConversation(
        sample_id=sample_id,
        turns=(
            LoCoMoRefinedTurn(
                dialog_id="D1:1",
                speaker="Caroline",
                text="Hello",
                occurred_at=NOW,
            ),
        ),
        questions=(
            LoCoMoRefinedQuestion(
                question_id=f"{sample_id}#q0000",
                question="What happened?",
                reference_answers=("SECRET GOLD",),
                evidence_dialog_ids=("D1:1",),
                category=1,
                is_multi_modality=False,
            ),
        ),
    )
