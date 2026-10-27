from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from mindbridge import (
    EmbedTask,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryKind,
    Modality,
    ModelInput,
)
from mindbridge.infrastructure.local import LocalStore, UnsupportedSchemaError


class PredicateFormer:
    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "predicate-fixture-v1"
    formation_space = "predicate-fixture-v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(
            (FormationProposal(content=value.content.text, **json.loads(value.content.text)),)
            for value in inputs
        )

    def close(self) -> None:
        pass


def _add_claim(memory: Memory, predicate: str, value: str, **fields: object) -> str:
    source = memory.add(
        json.dumps(
            {"kind": "state", "subject": "Ana", "predicate": predicate, "value": value, **fields},
            ensure_ascii=False,
        )
    )
    return next(
        record.id
        for record in memory.list().items
        if record.context is not None and source.id in record.context.evidence_ids
    )


_FULLWIDTH_ALIAS = " " + "".join(chr(ord(c) + 0xFEE0) for c in "FAVORITE_BEVERAGE") + " "


@pytest.mark.parametrize("alias", ["favorite_beverage", "喜欢的饮料", _FULLWIDTH_ALIAS])
def test_alias_correction_reaches_the_original_claim(tmp_path: Path, alias: str) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        first_id = _add_claim(memory, "preferred_drink", "tea")
        second_id = _add_claim(memory, alias, "coffee")
        first, second = memory.get(first_id).context, memory.get(second_id).context
        assert first is not None and second is not None
        assert first.lineage_id == second.lineage_id
        assert first.valid_until == second.valid_from
        assert second.supersedes_id == first_id
        assert (first.predicate, second.predicate) == ("preferred_drink", alias.strip())


@pytest.mark.parametrize(
    "first_predicate,second_predicate,fields",
    [
        ("home_city", "current_location", {}),
        ("preferred_drink", "likes", {}),
        ("preferred_drink", "favorite_beverage", {"subject": "Mei"}),
        ("preferred_drink", "favorite_beverage", {"kind": "relation"}),
    ],
)
def test_distinct_slots_do_not_merge(
    tmp_path: Path, first_predicate: str, second_predicate: str, fields: dict[str, str]
) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        first_id = _add_claim(memory, first_predicate, "tea")
        second_id = _add_claim(memory, second_predicate, "coffee", **fields)
        first, second = memory.get(first_id).context, memory.get(second_id).context
        assert first is not None and second is not None
        assert first.lineage_id != second.lineage_id
        assert first.valid_until is None


def test_relation_aliases_keep_multiple_values(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        first_id = _add_claim(memory, "preferred_drink", "tea", kind="relation")
        second_id = _add_claim(memory, "favorite_beverage", "coffee", kind="relation")
        first, second = memory.get(first_id).context, memory.get(second_id).context
        assert first is not None and second is not None
        assert first.lineage_id == second.lineage_id
        assert first.kind is second.kind is MemoryKind.RELATION
        assert first.retired_at is second.retired_at is None


def _history(path: Path) -> dict[str, list[tuple[object, ...]]]:
    with closing(sqlite3.connect(path / "state.sqlite3")) as connection:
        return {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in (
                "memory_versions",
                "memory_evidence_clauses",
                "memory_evidence_clause_versions",
                "embeddings",
                "search_index_queue",
                "memory_operations",
            )
        }


def test_current_predicate_provenance_and_history_survive_reopening(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        first = _add_claim(memory, "preferred_drink", "tea")
        second = _add_claim(memory, "favorite_beverage", "coffee")
    before = _history(tmp_path)
    with LocalStore(tmp_path):
        pass
    assert _history(tmp_path) == before
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        provenance = connection.execute(
            "SELECT predicate, predicate_key, predicate_registry, predicate_match "
            "FROM memory_semantics WHERE memory_id = ?",
            (second,),
        ).fetchone()
        assert provenance == (
            "favorite_beverage",
            "preferred_drink",
            "mindbridge-predicates-v1",
            "alias",
        )
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        original, other = memory.get(first).context, memory.get(second).context
        assert original is not None and other is not None
        assert original.lineage_id == other.lineage_id
        assert original.valid_until == other.valid_from
        third = _add_claim(memory, "喜欢的饮料", "water")
        corrected = memory.get(third).context
        original, other = memory.get(first).context, memory.get(second).context
        assert corrected is not None and original is not None and other is not None
        assert other.valid_until == corrected.valid_from
        assert original.valid_until == other.valid_from


@pytest.mark.parametrize(
    "corruption",
    [
        "ALTER TABLE memory_semantics DROP COLUMN predicate_key",
        "ALTER TABLE memory_semantics ADD COLUMN previous_lineage_id TEXT",
    ],
    ids=["missing-provenance", "obsolete-provenance"],
)
def test_current_schema_requires_exact_predicate_columns(tmp_path: Path, corruption: str) -> None:
    with LocalStore(tmp_path):
        pass
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.execute(corruption)
        connection.commit()
    with pytest.raises(UnsupportedSchemaError, match="predicate"):
        LocalStore(tmp_path)


def test_alias_index_rebuild_reuses_authoritative_vectors(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        first = _add_claim(memory, "preferred_drink", "tea", kind="relation")
        second = _add_claim(memory, "favorite_beverage", "coffee", kind="relation")
    shutil.rmtree(tmp_path / "zvec")

    class StoredVectorsOnly(TinyEmbedder):
        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            assert task is EmbedTask.QUERY, "rebuild must reuse authoritative vectors"
            return super().embed(inputs, task=task)

    with Memory(tmp_path, embedder=StoredVectorsOnly(), minimum_relevance=0) as memory:
        left, right = memory.get(first).context, memory.get(second).context
        assert left is not None and right is not None and left.lineage_id == right.lineage_id
        assert {first, second}.issubset({hit.id for hit in memory.search("drink", limit=20)})


def test_user_stated_alias_overrides_an_inferred_trait(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        inferred = _add_claim(memory, "preferred_drink", "tea", kind="trait", confidence=0.8)
        _add_claim(memory, "preferred_drink", "tea", kind="trait", confidence=0.9)
        stated = _add_claim(
            memory, "favorite_beverage", "coffee", kind="trait", basis="user_statement"
        )
        old, new = memory.get(inferred).context, memory.get(stated).context
        assert old is not None and new is not None
        assert old.lineage_id == new.lineage_id
        assert old.valid_until == new.valid_from


@pytest.mark.parametrize(
    "kind,delete_source", [("state", False), ("state", True), ("trait", False)]
)
def test_removing_a_correction_restores_the_previous_alias_claim(
    tmp_path: Path, kind: str, delete_source: bool
) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        fields = {"kind": kind, "basis": "user_statement" if kind == "trait" else "model_inference"}
        first = _add_claim(memory, "preferred_drink", "tea", **fields)
        second = _add_claim(memory, "favorite_beverage", "coffee", **fields)
        correction = _add_claim(
            memory,
            "喜欢的饮料",
            "water",
            kind=kind,
            basis="user_statement" if kind == "trait" else "model_inference",
        )
        context = memory.get(correction).context
        assert context is not None
        assert memory.delete(context.evidence_ids[0] if delete_source else correction)
        left, right = memory.get(first).context, memory.get(second).context
        assert left is not None and right is not None
        assert left.valid_until == right.valid_from
        assert right.valid_until is None
