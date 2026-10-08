from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from mindbridge import (
    Blob,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryContext,
    MemoryKind,
    MemoryRecord,
    ModelError,
    ModelInput,
    ObservationContext,
    ValidationError,
)
from mindbridge.models.openai_sdk import _formation_content, _formation_results


class HistoryFormer:
    formation_capabilities = frozenset(TinyEmbedder.embedding_capabilities)
    formation_model = "history-test"
    formation_space = "history-test:v1"

    def __init__(self) -> None:
        self.inputs: list[FormationInput] = []
        self.before_return: Callable[[], object] | None = None

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        self.inputs.extend(inputs)
        results = []
        for target in inputs:
            historical = next(
                (r for r in target.history if r.content == "Ada is the visitor."), None
            )
            results.append(
                ()
                if historical is None or target.content.text != "She left her notebook."
                else (
                    FormationProposal(
                        kind=MemoryKind.EVENT,
                        content="Ada left her notebook.",
                        subject="Ada",
                        evidence_ids=(target.memory_id, historical.id),
                    ),
                )
            )
        if self.before_return is not None:
            self.before_return()
        return tuple(results)

    def close(self) -> None:
        pass


def test_history_is_read_only_and_joint_sources_require_agreed_metadata(
    tmp_path: Path,
) -> None:
    former = HistoryFormer()
    with Memory(
        tmp_path, embedder=TinyEmbedder(), former=former, formation_history_max_rows=8
    ) as memory:
        original = memory.add("Ada is the visitor.", metadata={"common": 1, "old": 2})
        current = memory.add("She left her notebook.", metadata={"common": 1, "new": 3})
        assert [value.memory_id for value in former.inputs] == [original.id, current.id]
        assert [record.id for record in former.inputs[-1].history] == [original.id]
        event = next(
            record for record in memory.list().items if record.content == "Ada left her notebook."
        )
        assert event.context is not None
        assert event.context.subject == "Ada"
        assert set(event.context.evidence_ids) == {original.id, current.id}
        assert event.metadata == {}


def test_history_reports_truncation_without_skipping_a_long_recent_correction(
    tmp_path: Path,
) -> None:
    former = HistoryFormer()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        formation_history_max_rows=8,
        formation_history_budget_chars=30,
    ) as memory:
        memory.add("Ada is the visitor.")
        memory.add("The earlier name was a guess. " * 5)
        memory.add("She left her notebook.")
        assert former.inputs[-1].history == ()
        assert former.inputs[-1].history_truncated


def test_text_history_does_not_resolve_or_reopen_original_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    former = HistoryFormer()
    with Memory(
        tmp_path, embedder=TinyEmbedder(), former=former, formation_history_max_rows=8
    ) as memory:
        original = memory.add(("Ada is the visitor.", Blob(b"image", "image/png")))

        def unexpected_media(_asset: object) -> None:
            raise AssertionError("text history must not resolve original media")

        monkeypatch.setattr(memory._hydrator, "asset_ref", unexpected_media)
        memory.add("She left her notebook.")
        (history,) = former.inputs[-1].history
        assert history.id == original.id and history.assets == ()


def test_forgetting_a_cited_history_during_model_call_prevents_atomic_commit(
    tmp_path: Path,
) -> None:
    former = HistoryFormer()
    with Memory(
        tmp_path, embedder=TinyEmbedder(), former=former, formation_history_max_rows=8
    ) as memory:
        original = memory.add("Ada is the visitor.")
        former.before_return = lambda: memory.forget((original.id,))
        with pytest.raises(ModelError, match="evidence changed"):
            memory.add("She left her notebook.")
        records = memory.list().items
        assert any(record.content == "She left her notebook." for record in records)
        assert not any(record.content == "Ada left her notebook." for record in records)


def _history_input(*, retired: bool = False) -> FormationInput:
    before = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = MemoryRecord(
        id="old",
        content="Ada is the visitor.",
        created_at=before,
        context=MemoryContext(
            kind=MemoryKind.ENTITY,
            subject="Ada",
            basis=EvidenceBasis.USER_STATEMENT,
            confidence=1.0,
            valid_from=None,
            valid_until=None,
            recorded_at=before,
            retired_at=datetime(2026, 1, 2, tzinfo=timezone.utc) if retired else None,
        ),
    )
    return FormationInput(
        memory_id="current",
        content=ModelInput(text="She left her notebook."),
        context=ObservationContext(),
        history=(history,),
    )


@pytest.mark.parametrize("retired", [False, True])
def test_adapter_exposes_history_status_but_only_accepts_active_witnesses(retired: bool) -> None:
    target = _history_input(retired=retired)
    content = _formation_content((target,))
    assert isinstance(content, str)
    payload = json.loads(content)["observations"][0]
    assert payload["read_only_history"][0]["active"] is not retired
    response = json.dumps(
        {
            "items": [
                {
                    "observation_id": "observation_0",
                    "proposals": [
                        {
                            "kind": "event",
                            "content": "Ada left her notebook.",
                            "subject": "Ada",
                            "confidence": 0.9,
                            "evidence_observation_ids": ["observation_0", "history_0"],
                        }
                    ],
                }
            ]
        }
    )
    (results,) = _formation_results(response, (target,))
    assert len(results) == (0 if retired else 1)
    if results:
        assert results[0].evidence_ids == ("current", "old")


def test_adapter_does_not_accept_another_targets_private_history() -> None:
    first = _history_input()
    second = replace(first, memory_id="second", history=())
    response = json.dumps(
        {
            "items": [
                {"observation_id": "observation_0", "proposals": []},
                {
                    "observation_id": "observation_1",
                    "proposals": [
                        {
                            "kind": "event",
                            "content": "Ada left.",
                            "confidence": 0.9,
                            "evidence_observation_ids": ["observation_1", "history_0"],
                        }
                    ],
                },
            ]
        }
    )
    assert _formation_results(response, (first, second)) == ((), ())


@pytest.mark.parametrize("rows", [-1, 129, True])
def test_history_window_is_validated_before_opening_storage(tmp_path: Path, rows: int) -> None:
    with pytest.raises(ValidationError, match="formation_history_max_rows"):
        Memory(tmp_path / "not-created", embedder=TinyEmbedder(), formation_history_max_rows=rows)
    assert not (tmp_path / "not-created").exists()
