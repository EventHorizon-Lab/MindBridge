"""Query images may supply transient anchors without becoming stored evidence."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from mindbridge import Blob, Memory, Modality, RetrievalMode
from mindbridge.exceptions import ModelError
from mindbridge.models.base import EmbedTask, ModelInput


class _Embedder:
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
    embedding_model = "query-vision-fixture-v1"
    embedding_space = "query-vision-fixture-v1:3"
    embedding_dimension = 3

    def __init__(self, *, fail_description: bool = False) -> None:
        self.query_inputs: list[ModelInput] = []
        self.fail_description = fail_description

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        if task is EmbedTask.QUERY:
            self.query_inputs.extend(inputs)
            if self.fail_description and any(value.text == "ALPHA-947" for value in inputs):
                raise ModelError("description embedding failed", reason="request_rejected")
        return tuple(
            (0.0, 1.0, 0.0)
            if "ALPHA-947" in value.text
            else (1.0, 0.0, 0.0)
            if task is EmbedTask.QUERY
            else (0.8, 0.0, 0.6)
            for value in inputs
        )

    def close(self) -> None:
        pass


class _Describer:
    vision_capabilities = frozenset({Modality.IMAGE})
    vision_model = "query-vision-describer-v1"
    vision_space = "query-vision-describer-v1"

    def __init__(self, result: object = ("ALPHA-947",)) -> None:
        self.result = result
        self.inputs: list[tuple[ModelInput, ...]] = []

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        self.inputs.append(tuple(inputs))
        if isinstance(self.result, Exception):
            raise self.result
        return cast(tuple[str, ...], self.result)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("mode", tuple(RetrievalMode))
def test_query_vision_recovers_ocr_support_through_independent_routes(
    tmp_path: Path, mode: RetrievalMode
) -> None:
    embedder, describer = _Embedder(), _Describer()
    with Memory(
        tmp_path,
        embedder=embedder,
        vision_describer=describer,
        query_vision=True,
        retrieval_mode=mode,
        ambiguity_margin=0,
    ) as memory:
        target = memory.add("ALPHA-947 registration desk")
        memory.add("an unrelated blue poster")
        hits = memory.search(("Where was this photographed?", Blob(b"query", "image/png")), limit=1)

    assert [hit.id for hit in hits] == [target.id]
    assert len(describer.inputs) == 1
    assert describer.inputs[0][0].text == ""
    assert len(describer.inputs[0][0].assets) == 1
    if mode is RetrievalMode.LEXICAL:
        assert embedder.query_inputs == []
    else:
        assert any(value.assets for value in embedder.query_inputs)
        assert any(value.text == "Where was this photographed?" for value in embedder.query_inputs)
        assert any(
            value.text == "ALPHA-947" and not value.assets for value in embedder.query_inputs
        )


def test_query_descriptions_never_persist_as_assets_captions_or_memories(tmp_path: Path) -> None:
    embedder, describer = _Embedder(), _Describer()
    with Memory(
        tmp_path,
        embedder=embedder,
        vision_describer=describer,
        query_vision=True,
        ambiguity_margin=0,
    ) as memory:
        target = memory.add("ALPHA-947 registration desk")
        before = memory.list().items
        memory.search(Blob(b"query-only-image", "image/png"))
        assert memory.list().items == before
        assert memory.get(target.id).content == target.content
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
            assert connection.execute("SELECT COUNT(*) FROM media_assets").fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM visual_descriptions").fetchone()[0] == 0
        assert all(
            asset.path is not None and not asset.path.exists()
            for value in embedder.query_inputs
            for asset in value.assets
        )


@pytest.mark.parametrize("enabled", (False, True))
def test_query_vision_is_explicit_and_text_queries_cost_no_description(
    tmp_path: Path, enabled: bool
) -> None:
    describer = _Describer()
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        vision_describer=describer,
        query_vision=enabled,
    ) as memory:
        memory.add("a blue poster")
        memory.search("a poster")
        assert describer.inputs == []
        memory.search(Blob(b"query", "image/png"))
        assert len(describer.inputs) == int(enabled)


def test_query_vision_without_a_describer_preserves_native_image_search(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_Embedder(), query_vision=True) as memory:
        target = memory.add("a blue poster")
        assert memory.search(Blob(b"query", "image/png"))[0].id == target.id


@pytest.mark.parametrize(
    "description",
    (("",), (), ("one", "two"), ModelError("private provider body", reason="timeout")),
)
def test_invalid_query_descriptions_fail_open_and_never_log_provider_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, description: object
) -> None:
    describer = _Describer(description)
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        vision_describer=describer,
        query_vision=True,
    ) as memory:
        target = memory.add("a blue poster")
        assert memory.search(Blob(b"query", "image/png"))[0].id == target.id
    assert "private provider body" not in caplog.text


@pytest.mark.parametrize("fail", (False, True))
def test_query_description_cache_lives_for_one_operation_only(tmp_path: Path, fail: bool) -> None:
    describer = _Describer(ModelError("failed", reason="timeout") if fail else ("ALPHA-947",))
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        vision_describer=describer,
        query_vision=True,
    ) as memory:
        memory.add("a blue poster")
        query = Blob(b"query", "image/png")
        with memory._lifecycle.operation(model_work=True) as operation:
            prepared = memory._retrieval._materializer.prepare((query, query), operation)
            for _ in range(2):
                memory._retrieval.search_prepared(
                    prepared,
                    limit=10,
                    operation=operation,
                    memory_types=None,
                    reference_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    temporal_range=None,
                    occurred_from=None,
                    occurred_until=None,
                    scope=None,
                    require_unambiguous=False,
                    capture_trace=False,
                )
            assert len(describer.inputs) == 1
            assert len(describer.inputs[0]) == 1
            assert operation.descriptions == {}
            assert operation.speaker_names == {}
        memory.search(query)
        memory.search(query)
        assert len(describer.inputs) == 3


def test_a_long_caption_does_not_dilute_the_original_lexical_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    describer = _Describer(("a blue sky snowy mountains green trees and distant windows",))
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        vision_describer=describer,
        query_vision=True,
        ambiguity_margin=0,
    ) as memory:
        target = memory.add("ALPHA-947")
        memory.add("a silent grey room")
        monkeypatch.setattr(memory._index, "search", lambda *_args, **_kwargs: ())
        # With no dense candidates, the exact original words must retain their full lexical
        # contribution rather than compete with every unrelated caption term.
        hits = memory.search(("alpha-947", Blob(b"query", "image/png")), limit=1)
        assert hits[0].id == target.id


@pytest.mark.parametrize("mode", (RetrievalMode.DENSE, RetrievalMode.HYBRID))
def test_description_embedding_failure_keeps_native_query_and_lexical_anchors(
    tmp_path: Path, mode: RetrievalMode
) -> None:
    embedder = _Embedder(fail_description=True)
    with Memory(
        tmp_path,
        embedder=embedder,
        vision_describer=_Describer(),
        query_vision=True,
        retrieval_mode=mode,
        ambiguity_margin=0,
    ) as memory:
        target = memory.add("ALPHA-947")
        native = memory.add("an unrelated blue poster")
        hits = memory.search(Blob(b"query", "image/png"))
        assert hits[0].id == (target.id if mode is RetrievalMode.HYBRID else native.id)


def test_visible_text_is_an_independent_query_anchor(tmp_path: Path) -> None:
    embedder = _Embedder()
    caption = (
        "Shown: a blue poster beside a desk\nText: ALPHA-947\nPlace: a large registration hall"
    )
    with Memory(
        tmp_path,
        embedder=embedder,
        vision_describer=_Describer((caption,)),
        query_vision=True,
    ) as memory:
        memory.add("ALPHA-947 registration desk")
        memory.search(Blob(b"query", "image/png"))
    assert ModelInput(text=caption) in embedder.query_inputs
    assert ModelInput(text="ALPHA-947") in embedder.query_inputs


def test_query_description_cannot_change_the_original_occurrence_bounds(tmp_path: Path) -> None:
    caption = "Text: ALPHA-947\nWhen: 2001-06-01"
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        vision_describer=_Describer((caption,)),
        query_vision=True,
        ambiguity_margin=0,
    ) as memory:
        memory.add("ALPHA-947 old desk", occurred_at=datetime(2001, 6, 1, tzinfo=timezone.utc))
        target = memory.add(
            "ALPHA-947 current desk", occurred_at=datetime(2024, 6, 1, tzinfo=timezone.utc)
        )
        hits = memory.search(
            ("Find the desk in 2024", Blob(b"query", "image/png")),
            occurred_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
            occurred_until=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert [hit.id for hit in hits] == [target.id]
