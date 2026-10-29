"""Canonical facts cannot become members of their own new support conjunction."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mindbridge import FormationInput, FormationProposal, Memory, MemoryKind, MemoryNotFoundError


class RestatingFormer:
    formation_capabilities = TinyEmbedder.embedding_capabilities
    formation_model = "self-evidence-test"
    formation_space = "self-evidence-test:v1"

    def __init__(self, *, valid_restatement: bool) -> None:
        self.valid_restatement = valid_restatement
        self.calls = 0

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        assert len(inputs) == 1
        source = inputs[0]
        self.calls += 1
        atlas = FormationProposal(
            kind=MemoryKind.RELATION,
            content="Riley owns an atlas.",
            subject="Riley",
            predicate="owns",
            value="atlas",
            confidence=0.8,
        )
        if self.calls == 1:
            return ((atlas,),)
        previous = next(
            record
            for record in source.history
            if record.context is not None and record.context.kind is MemoryKind.RELATION
        )
        # This declaration is conjunctive; discarding only the self member would falsely
        # authorize a different, independent assessment by the new raw source.
        invalid = replace(atlas, evidence_ids=(source.memory_id, previous.id))
        telescope = replace(atlas, content="Riley owns a telescope.", value="telescope")
        return ((invalid, *((atlas,) if self.valid_restatement else ()), telescope),)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("valid_restatement", [False, True])
def test_automatic_self_evidence_refuses_whole_proposal_without_losing_valid_facts(
    tmp_path: Path, independent: bool, valid_restatement: bool
) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    former = RestatingFormer(valid_restatement=valid_restatement)
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    try:
        with Memory(
            tmp_path,
            embedder=TinyEmbedder(),
            former=former,
            formation_history_max_rows=12,
            independent_evidence=independent,
            tracer=provider.get_tracer("self-evidence-test"),
        ) as memory:
            original = memory.add("Riley owns an atlas.", occurred_at=at)
            atlas = next(
                record
                for record in memory.list().items
                if record.context is not None and record.context.kind is MemoryKind.RELATION
            )
            current = memory.add(
                "Riley also owns a telescope and still has the atlas.",
                occurred_at=at + timedelta(days=1),
            )
            assert memory.get(current.id).content == current.content
            updated = memory.get(atlas.id)
            assert updated.context is not None
            expected_sources = {original.id, current.id} if valid_restatement else {original.id}
            assert set(updated.context.evidence_ids) == expected_sources
            telescope = next(
                record
                for record in memory.list().items
                if record.context is not None and record.context.value == "telescope"
            )
            assert telescope.context is not None
            assert telescope.context.evidence_ids == (current.id,)
            with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
                clauses = connection.execute(
                    "SELECT c.member_count,m.source_memory_id FROM memory_evidence_clauses c "
                    "JOIN memory_evidence_clause_members m USING(memory_id,clause_id) "
                    "WHERE c.memory_id=? AND c.retired_at IS NULL",
                    (atlas.id,),
                ).fetchall()
                assert set(clauses) == {(1, source) for source in expected_sources}
                assert connection.execute(
                    "SELECT COUNT(*) FROM memory_evidence WHERE memory_id=source_memory_id"
                ).fetchone() == (0,)
                assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
                assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
            assert memory.delete(original.id)
            if valid_restatement:
                surviving = memory.get(atlas.id).context
                assert surviving is not None and surviving.visible
            else:
                with pytest.raises(MemoryNotFoundError):
                    memory.get(atlas.id)
            surviving_telescope = memory.get(telescope.id).context
            assert surviving_telescope is not None and surviving_telescope.visible
            assert memory.get(current.id).content == current.content
    finally:
        provider.shutdown()
    adds = [span for span in exporter.get_finished_spans() if span.name == "mindbridge.add"]
    assert [
        (span.attributes or {}).get("mindbridge.formation.refused_proposals") for span in adds
    ] == [0, 1]
