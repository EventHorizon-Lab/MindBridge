"""Focused structural tests for the isolated evidence-excerpt experiment."""

from __future__ import annotations

import hashlib
import math
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from mindbridge import (
    ContextBudget,
    ContextExcerpt,
    ContextSymbolCoverage,
    ContextSymbolRole,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryKind,
    MemoryType,
    Modality,
    ObservationContext,
    RetrievalScope,
    TextSpanPiece,
    TextSpanSelector,
)
from mindbridge.api.app import ContextBundleResponse
from mindbridge.api.mcp import ContextBundleResult
from mindbridge.cli import _bundle_value
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.infrastructure.local.store import (
    SCHEMA_VERSION,
    LocalStore,
    StoredTextSelector,
    UnsupportedSchemaError,
)
from mindbridge.infrastructure.local.store.index import IndexOutbox
from mindbridge.kernel.content import contextual_text_key_descriptors, contextual_text_keys
from mindbridge.models.base import EmbedTask, ModelInput

REFERENCE = datetime(2026, 9, 9, tzinfo=timezone.utc)


@pytest.mark.parametrize("score", [True, "0.5", None])
def test_context_excerpt_rejects_non_numeric_scores_with_validation_error(score: object) -> None:
    source = "source"
    piece = TextSpanPiece(
        role="body",
        start_codepoint=0,
        end_codepoint=len(source),
        source_text=source,
        sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    selector = TextSpanSelector(
        parent_content_sha256=hashlib.sha256(source.encode()).hexdigest(),
        embedding_input_sha256=hashlib.sha256(source.encode()).hexdigest(),
        recipe_version="test-v1",
        pieces=(piece,),
    )

    with pytest.raises(ValidationError, match="score must be between zero and one"):
        ContextExcerpt(
            source_memory_id="memory",
            matched_index_id="embedding",
            content=source,
            selector=selector,
            score=cast(Any, score),
            created_at=REFERENCE,
        )


class _PartEmbedder:
    embedding_model = "excerpt-test"
    embedding_space = "excerpt-test:2"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT})

    def __init__(self, *, reject_long_documents: bool = False) -> None:
        self.reject_long_documents = reject_long_documents
        self.calls: list[tuple[EmbedTask, tuple[str, ...]]] = []

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        batch = tuple(inputs)
        self.calls.append((task, tuple(value.text for value in batch)))
        if (
            task is EmbedTask.DOCUMENT
            and self.reject_long_documents
            and any(len(value.text) > 3_000 for value in batch)
        ):
            raise ModelError("fixture document is too large", reason="payload_too_large")
        vectors = []
        for value in batch:
            if task is EmbedTask.QUERY or (
                "TARGET CLAIM" in value.text
                and len(value.text) < 3_000
                and not value.text.startswith("DERIVED TARGET")
            ):
                vectors.append((1.0, 0.0))
            elif value.text.startswith("DERIVED TARGET"):
                vectors.append((0.99, math.sqrt(1.0 - 0.99**2)))
            else:
                vectors.append((0.0, 1.0))
        return tuple(vectors)

    def close(self) -> None:
        return None


class _DerivedFormer:
    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "excerpt-formation"
    formation_space = "excerpt-formation:v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        proposals: list[tuple[FormationProposal, ...]] = []
        for position, value in enumerate(inputs):
            if position == 0:
                proposals.append(())
                continue
            proposals.append(
                (
                    FormationProposal(
                        kind=MemoryKind.STATE,
                        content="DERIVED TARGET says the gate is open",
                        subject="gate",
                        predicate="status",
                        value="open",
                        confidence=0.9,
                        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        evidence_ids=(value.memory_id, inputs[0].memory_id),
                    ),
                )
            )
        return tuple(proposals)

    def close(self) -> None:
        return None


def _long_source() -> str:
    return (
        "Conversation header 🙂 e\u0301\r\n"
        + "a" * 2_350
        + " TARGET CLAIM says the gate is open. "
        + "b" * 2_350
        + " Later clarification: TARGET CLAIM is withdrawn and the gate is closed."
    )


def _memory(data_dir: Path, *, former: _DerivedFormer | None = None) -> Memory:
    return Memory(
        data_dir,
        embedder=_PartEmbedder(),
        former=former,
        minimum_relevance=0.0,
    )


def test_descriptors_preserve_existing_keys_for_unicode_strip_crlf_and_repeats() -> None:
    cases = (
        "short",
        " heading  \r\n" + "a" * 5_000,
        "标题🙂e\u0301\r\n" + (" 重复🙂 " * 1_000),
        "head\n" + " " * 1_800 + "tail " * 1_000,
        "a" * 6_000,
    )

    for source in cases:
        descriptors = contextual_text_key_descriptors(source)
        assert tuple(value.key for value in descriptors) == contextual_text_keys(source)
        for descriptor in descriptors:
            for span in descriptor.spans:
                assert source[span.start_codepoint : span.end_codepoint]
        assert len({value.key for value in descriptors}) <= len(descriptors)


def test_new_raw_parts_store_digest_bound_selectors_without_source_text(tmp_path: Path) -> None:
    source = _long_source()
    with _memory(tmp_path) as memory:
        record = memory.add(source, occurred_at=REFERENCE)

        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection, connection:
            selector_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(embedding_text_selectors)")
            }
            piece_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(embedding_text_span_pieces)")
            }
            selectors = connection.execute(
                """
                SELECT parent_content_sha256, embedding_input_sha256, recipe_version
                FROM embedding_text_selectors
                """
            ).fetchall()
            pieces = connection.execute(
                """
                SELECT role, start_codepoint, end_codepoint, piece_sha256
                FROM embedding_text_span_pieces
                ORDER BY embedding_id, selector_position, piece_position
                """
            ).fetchall()

        assert selectors
        assert pieces
        assert "source_text" not in selector_columns | piece_columns
        assert {str(row[0]) for row in selectors} == {
            hashlib.sha256(record.content.encode("utf-8")).hexdigest()
        }
        assert all(source[int(row[1]) : int(row[2])] for row in pieces)

        assert memory.delete(record.id) is True
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection, connection:
            assert (
                connection.execute("SELECT COUNT(*) FROM embedding_text_selectors").fetchone()[0]
                == 0
            )
            assert (
                connection.execute("SELECT COUNT(*) FROM embedding_text_span_pieces").fetchone()[0]
                == 0
            )


def test_compile_returns_an_excerpt_only_when_the_full_raw_record_cannot_fit(
    tmp_path: Path,
) -> None:
    source = _long_source()
    with _memory(tmp_path) as memory:
        record = memory.add(source, occurred_at=REFERENCE)
        compact = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=3_000, max_items=1),
            reference_at=REFERENCE,
            allow_partial_sources=True,
        )
        roomy = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=8_000, max_items=1),
            reference_at=REFERENCE,
            allow_partial_sources=True,
        )

    assert compact.hits == ()
    assert len(compact.excerpts) == 1
    excerpt = compact.excerpts[0]
    assert isinstance(excerpt, ContextExcerpt)
    assert excerpt.source_memory_id == record.id
    assert "TARGET CLAIM says the gate is open" in excerpt.content
    assert "withdrawn and the gate is closed" not in excerpt.content
    assert "omitted source text" in excerpt.content
    assert "omitted text may qualify it" in compact.render()
    assert "offsets " in compact.render() and "body:" in compact.render()
    assert excerpt.matched_index_id not in compact.render()
    assert compact.chars >= len(compact.render())
    assert compact.chars <= compact.budget.max_chars
    assert compact.omitted == 0
    rendered = compact.render()
    document = compact.document()
    presentation = compact.compact()
    partial = next(symbol for symbol in presentation.symbols if symbol.stable_id == record.id)
    assert partial.coverage is ContextSymbolCoverage.PARTIAL
    assert partial.selector == excerpt.selector
    assert partial.roles == (ContextSymbolRole.EXCERPT,)
    assert presentation.resolve_citation(partial.symbol).selector == excerpt.selector
    assert f"partial source {partial.symbol}" in presentation.text
    assert f"1/{compact.budget.max_items} items" in presentation.text
    assert excerpt.matched_index_id not in presentation.text
    assert compact.render() == rendered
    assert compact.document() == document
    assert len(roomy.hits) == 1
    assert roomy.hits[0].id == record.id
    assert roomy.excerpts == ()


def test_partial_sources_are_explicit_opt_in_and_default_skips_selector_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[tuple[str, ...]] = []
    original = IndexOutbox.read_text_selectors

    def tracked(
        store: IndexOutbox,
        embedding_ids: Sequence[str],
    ) -> dict[str, tuple[StoredTextSelector, ...]]:
        reads.append(tuple(embedding_ids))
        return original(store, embedding_ids)

    monkeypatch.setattr(IndexOutbox, "read_text_selectors", tracked)
    with _memory(tmp_path) as memory:
        memory.add(_long_source(), occurred_at=REFERENCE)
        default = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=3_000, max_items=1),
            reference_at=REFERENCE,
        )
        assert reads == []
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection, connection:
            failures_after_default = connection.execute(
                "SELECT COUNT(*) FROM query_failures"
            ).fetchone()[0]
        enabled = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=3_000, max_items=1),
            reference_at=REFERENCE,
            allow_partial_sources=True,
        )
        with pytest.raises(ValidationError, match="allow_partial_sources must be a boolean"):
            memory.compile("TARGET CLAIM", allow_partial_sources=cast("Any", 1))

    assert default.hits == ()
    assert default.excerpts == ()
    assert default.omitted == 1
    assert failures_after_default == 1
    assert enabled.hits == ()
    assert len(enabled.excerpts) == 1
    assert len(reads) == 1


def test_excerpt_only_bundle_does_not_claim_its_memory_type_is_missing(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        memory.add(_long_source(), occurred_at=REFERENCE)
        bundle = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(
                max_chars=3_000,
                max_items=1,
                memory_types=frozenset({MemoryType.SEMANTIC}),
            ),
            reference_at=REFERENCE,
            allow_partial_sources=True,
        )

    assert bundle.hits == ()
    assert len(bundle.excerpts) == 1
    assert not any(
        item.kind.value == "section_empty"
        and item.detail == "the facts section is empty: no semantic memory was included"
        for item in bundle.unknowns
    )
    assert all("no semantic memory was included" not in item.detail for item in bundle.unknowns)


def test_an_excerpt_cannot_satisfy_a_derived_full_record_closure(tmp_path: Path) -> None:
    with _memory(tmp_path, former=_DerivedFormer()) as memory:
        source, _secondary = memory.add_many(
            (_long_source(), "A second source grounds the derived assertion."),
            occurred_at=(REFERENCE, REFERENCE),
        )
        derived = next(
            hit
            for hit in memory.search("TARGET CLAIM", limit=10, reference_at=REFERENCE)
            if hit.context is not None and hit.context.kind is MemoryKind.STATE
        )
        compact = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=3_000, max_items=2),
            reference_at=REFERENCE,
            allow_partial_sources=True,
        )

    assert source.id not in {hit.id for hit in compact.hits}
    assert [value.source_memory_id for value in compact.excerpts] == [source.id]
    assert derived.id not in {hit.id for hit in compact.hits}
    assert compact.excerpts[0].source_memory_id != derived.id
    assert compact.render().count(source.id) == 1


def test_scope_filtering_happens_before_excerpt_selection(tmp_path: Path) -> None:
    before_write = datetime.now(timezone.utc)
    with _memory(tmp_path) as memory:
        memory.add(
            _long_source(),
            context=ObservationContext(
                place_id="studio",
                valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
                valid_until=datetime(2027, 1, 1, tzinfo=timezone.utc),
            ),
            occurred_at=REFERENCE,
        )
        after_write = datetime.now(timezone.utc)
        excluded = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=3_000, max_items=1),
            reference_at=REFERENCE,
            scope=RetrievalScope(place_id="elsewhere"),
            allow_partial_sources=True,
        )
        included = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=3_000, max_items=1),
            reference_at=REFERENCE,
            scope=RetrievalScope(
                place_id="studio",
                valid_at=REFERENCE,
                known_at=after_write,
            ),
            allow_partial_sources=True,
        )
        before_known = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=3_000, max_items=1),
            reference_at=REFERENCE,
            scope=RetrievalScope(known_at=before_write),
            allow_partial_sources=True,
        )
        after_validity = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=3_000, max_items=1),
            reference_at=REFERENCE,
            scope=RetrievalScope(valid_at=datetime(2027, 2, 1, tzinfo=timezone.utc)),
            allow_partial_sources=True,
        )
        wrong_identity = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=3_000, max_items=1),
            reference_at=REFERENCE,
            scope=RetrievalScope(identity_id="identity_missing"),
            allow_partial_sources=True,
        )

    assert excluded.excerpts == ()
    assert before_known.excerpts == ()
    assert after_validity.excerpts == ()
    assert wrong_identity.excerpts == ()
    assert included.excerpts
    assert included.places == ("studio",)


def test_parent_content_tampering_disables_only_the_excerpt(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        record = memory.add(_long_source(), occurred_at=REFERENCE)
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection, connection:
            connection.execute(
                "UPDATE memory_records SET content = content || ' tampered' WHERE memory_id = ?",
                (record.id,),
            )
            connection.commit()

        bundle = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=3_000, max_items=1),
            reference_at=REFERENCE,
            allow_partial_sources=True,
        )

    assert bundle.hits == ()
    assert bundle.excerpts == ()
    assert bundle.omitted == 1


@pytest.mark.parametrize(
    ("statement", "parameters"),
    (
        (
            "UPDATE embedding_text_selectors SET parent_content_sha256 = ?",
            ("0" * 64,),
        ),
        (
            "UPDATE embedding_text_selectors SET embedding_input_sha256 = ?",
            ("0" * 64,),
        ),
        (
            "UPDATE embedding_text_selectors SET recipe_version = ?",
            ("unknown-recipe",),
        ),
        (
            "UPDATE embedding_text_span_pieces SET piece_sha256 = ?",
            ("0" * 64,),
        ),
    ),
)
def test_selector_tampering_disables_excerpt_but_preserves_full_memory(
    tmp_path: Path,
    statement: str,
    parameters: tuple[str, ...],
) -> None:
    with _memory(tmp_path) as memory:
        record = memory.add(_long_source(), occurred_at=REFERENCE)
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection, connection:
            connection.execute(statement, parameters)
            connection.commit()

        compact = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=3_000, max_items=1),
            reference_at=REFERENCE,
            allow_partial_sources=True,
        )
        roomy = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=8_000, max_items=1),
            reference_at=REFERENCE,
            allow_partial_sources=True,
        )

    assert compact.excerpts == ()
    assert compact.omitted == 1
    assert [hit.id for hit in roomy.hits] == [record.id]
    assert roomy.excerpts == ()


def test_constraint_free_selector_tables_are_rejected_at_the_current_version(
    tmp_path: Path,
) -> None:
    """`_validate_text_selector_schema` is what refuses a table that lost its CHECK constraints.

    The declared version says nothing about the shape, so a store carrying the right names with
    none of the guarantees has to be refused rather than written to.
    """
    with LocalStore(tmp_path):
        pass
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection, connection:
        connection.executescript(
            """
            DROP TABLE embedding_text_span_pieces;
            DROP TABLE embedding_text_selectors;
            CREATE TABLE embedding_text_selectors (
                embedding_id TEXT,
                selector_position INTEGER,
                parent_content_sha256 TEXT,
                embedding_input_sha256 TEXT,
                recipe_version TEXT
            );
            CREATE TABLE embedding_text_span_pieces (
                embedding_id TEXT,
                selector_position INTEGER,
                piece_position INTEGER,
                role TEXT,
                start_codepoint INTEGER,
                end_codepoint INTEGER,
                piece_sha256 TEXT
            );
            """
        )
        before = tuple(
            connection.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE name LIKE 'embedding_text_%' ORDER BY type, name"
            )
        )

    with pytest.raises(UnsupportedSchemaError, match="invalid embedding_text_selectors table"):
        LocalStore(tmp_path)

    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection, connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert (
            tuple(
                connection.execute(
                    "SELECT type, name, sql FROM sqlite_master "
                    "WHERE name LIKE 'embedding_text_%' ORDER BY type, name"
                )
            )
            == before
        )


def test_excerpt_serializes_through_rest_mcp_and_cli_contracts(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        memory.add(_long_source(), occurred_at=REFERENCE)
        bundle = memory.compile(
            "TARGET CLAIM",
            budget=ContextBudget(max_chars=3_000, max_items=1),
            reference_at=REFERENCE,
            allow_partial_sources=True,
        )

    rest = ContextBundleResponse.model_validate(bundle.document()).model_dump(mode="json")
    mcp = ContextBundleResult.model_validate(bundle.document()).model_dump(mode="json")
    cli = cast(
        "dict[str, Any]",
        {name: _bundle_value(value) for name, value in bundle.document().items()},
    )
    for document in (rest, mcp, cli):
        assert len(document["excerpts"]) == 1
        value = document["excerpts"][0]
        assert value["source_memory_id"] == bundle.excerpts[0].source_memory_id
        assert value["selector"]["pieces"][0]["source_text"]
