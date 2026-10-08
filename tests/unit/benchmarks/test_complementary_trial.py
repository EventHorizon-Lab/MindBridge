from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx2 as httpx
import pytest
from openai import OpenAI

from mindbridge import AnswerPolicy, AnswerResult, Modality, SearchHit
from mindbridge.models.base import EmbedTask, ModelInput


class _FakeModels:
    embedding_model = "deterministic"
    embedding_space = "deterministic:2"
    embedding_dimension = 2
    embedding_capabilities = generation_capabilities = frozenset({Modality.TEXT})

    def __init__(self) -> None:
        self.answer_calls: list[tuple[ModelInput, tuple[SearchHit, ...]]] = []

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in inputs)

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        self.answer_calls.append((question, tuple(hits)))
        return AnswerResult("A caliper from Niko.", tuple(hits))

    def close(self) -> None:
        pass


def _driver() -> ModuleType:
    path = Path(__file__).parents[3] / "benchmarks/complementary_trial.py"
    spec = importlib.util.spec_from_file_location("complementary_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("order", list(itertools.permutations(("rank", "relevance", "coverage"))))
def test_paired_trial_uses_one_shared_plan_and_keeps_labels_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: tuple[str, ...]
) -> None:
    driver = _driver()
    requests: list[dict[str, object]] = []
    clock = [0.0]
    monkeypatch.setattr(driver, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(driver, "ARMS", order)
    monkeypatch.setattr(
        driver,
        "random",
        SimpleNamespace(Random=lambda seed: SimpleNamespace(shuffle=lambda arms: None)),
    )

    def respond(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        requests.append(data)
        if "response_format" in data:
            clock[0] += 50.0
            payload = json.loads(data["messages"][1]["content"])
            assert "HIDDEN_REFERENCE" not in str(payload)
            text = json.dumps(
                {
                    "facets": ["person", "tool"],
                    "records": [
                        {"index": r["index"], "relevance": 3, "covers": [r["index"] % 2]}
                        for r in payload["records"]
                    ],
                }
            )
        else:
            assert data["chat_template_kwargs"] == {"enable_thinking": False}
            assert [message["role"] for message in data["messages"]] == ["user"]
            text = "Yes"
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 0,
                "model": "mock",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    audit = driver.Audit(tmp_path / "http.jsonl")
    with OpenAI(
        api_key="test",
        base_url="http://mock/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(respond),
            event_hooks={"response": [audit.response]},
        ),
    ) as client:
        models = _FakeModels()
        original_answer = models.answer

        def timed_answer(
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: AnswerPolicy = "strict",
            exhaustive: bool = False,
        ) -> AnswerResult:
            clock[0] += 2.0
            return original_answer(
                question, hits, answer_policy=answer_policy, exhaustive=exhaustive
            )

        monkeypatch.setattr(models, "answer", timed_answer)
        reader = driver.SelectorBackend(
            models,
            client,
            "mock",
            audit,
            {"reader_rows": 2, "reader_chars": 1000, "candidate_rows": 60},
            tmp_path,
        )
        with driver.open_memory(tmp_path / "store", models, reader) as memory:
            memory.add("Mara borrowed the caliper.", metadata={"source_id": "one"})
            memory.add("Niko lent the tool.", metadata={"source_id": "two"})
            rows = driver.evaluate(
                memory,
                reader,
                audit,
                {
                    "id": "independent",
                    "dataset": "constructed-text-diagnostics",
                    "question": "What tool and from whom?",
                    "reference": "HIDDEN_REFERENCE",
                    "gold": ["one", "two"],
                },
                tmp_path,
            )
    assert len(rows) == 3 and all(row["plan_valid"] for row in rows)
    assert sum("response_format" in request for request in requests) == 1
    assert len(models.answer_calls) == 3
    assert len({tuple(row["candidate_ids"]) for row in rows}) == 1
    assert "HIDDEN_REFERENCE" not in repr(models.answer_calls)
    assert all(row["seconds"] == 2.0 for row in rows)
    assert {row["arm"]: row["shared_selector_seconds"] for row in rows} == {
        "rank": 0.0,
        "relevance": 50.0,
        "coverage": 50.0,
    }
    saved = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert [row["shared_selector_seconds"] for row in saved] == [
        row["shared_selector_seconds"] for row in rows
    ]


@pytest.mark.parametrize("field", ["question", "reference_answer"])
def test_trial_checks_pinned_question_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    driver = _driver()
    row = {
        "question_id": "same-id",
        "question": "Who?",
        "reference_answer": "Li",
        "category": "Long-Term Identity Profile Inference",
        "target_character_ids": ["Li"],
        "evidence_video_ids": ["clip_001"],
        "before_clip": None,
    }
    path = tmp_path / "qa_test.jsonl"
    path.write_text(json.dumps(row))
    monkeypatch.setitem(
        driver.TASKS,
        "icm-bench",
        replace(driver.TASKS["icm-bench"], digest=driver.dataset_digest(path)),
    )
    assert driver.frozen_questions(path)["same-id"].reference_answer == "Li"
    row[field] = "changed"
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="question dataset changed"):
        driver.frozen_questions(path)


@pytest.mark.parametrize("changed", [None, "assets/video.mp4", "zvec/index"])
def test_trial_freezes_assets_and_index_with_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str | None
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "benchmarks"))
    tree_manifest = importlib.import_module("paired_replay").tree_manifest

    driver = _driver()
    source = tmp_path / "source"
    for name in ("state.sqlite3", ".mindbridge.lock", "assets/video.mp4", "zvec/index"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original")
    manifest = tree_manifest(source)
    recipe = {
        "source_sqlite_sha256": driver.dataset_digest(source / "state.sqlite3"),
        "source_tree_sha256": manifest["sha256"],
    }
    destination = tmp_path / "copy"
    if changed is not None:
        (source / changed).write_bytes(b"modified")
        assert driver.dataset_digest(source / "state.sqlite3") == recipe["source_sqlite_sha256"]
        with pytest.raises(RuntimeError, match="store tree changed"):
            driver.copy_baseline(source, destination, recipe)
        assert not destination.exists()
    else:
        assert driver.copy_baseline(source, destination, recipe) == manifest
        assert tree_manifest(destination) == manifest


def test_trial_rejects_snapshot_changed_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "benchmarks"))
    tree_manifest = importlib.import_module("paired_replay").tree_manifest

    driver = _driver()
    source = tmp_path / "source"
    source.mkdir()
    (source / "state.sqlite3").write_bytes(b"original")
    (source / ".mindbridge.lock").touch()
    recipe = {
        "source_sqlite_sha256": driver.dataset_digest(source / "state.sqlite3"),
        "source_tree_sha256": tree_manifest(source)["sha256"],
    }
    original_copy = driver.shutil.copytree

    def changed_copy(source: Path, destination: Path) -> None:
        original_copy(source, destination)
        (destination / "state.sqlite3").write_bytes(b"modified")

    monkeypatch.setattr(driver.shutil, "copytree", changed_copy)
    with pytest.raises(RuntimeError, match="changed while copying"):
        driver.copy_baseline(source, tmp_path / "copy", recipe)


def test_media_audit_hashes_payload_without_logging_media_bytes() -> None:
    driver = _driver()
    result = driver.redact_media({"url": "data:image/jpeg;base64,SECRET_FRAME"})
    assert "SECRET_FRAME" not in json.dumps(result)
    assert result["url"]["encoded_chars"] > 0


def test_trial_visual_backend_satisfies_memory_contract(tmp_path: Path) -> None:
    driver = _driver()
    with OpenAI(api_key="test", base_url="http://mock/v1") as client:
        vision = driver.vision_backend(client, "mock")
        models = _FakeModels()
        reader = driver.SelectorBackend(
            models,
            client,
            "mock",
            driver.Audit(tmp_path / "http.jsonl"),
            {"reader_rows": 12, "reader_chars": 24000, "candidate_rows": 60},
            tmp_path,
        )
        with driver.open_memory(tmp_path / "store", models, reader, vision=vision):
            assert vision.vision_capabilities == frozenset({Modality.IMAGE, Modality.VIDEO})


@pytest.mark.parametrize("artifact", ["store/partial.db", "http-audit.jsonl", "plans.jsonl"])
def test_trial_rejects_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    driver = _driver()
    assert driver.__file__ is not None
    recipe = json.loads(Path(driver.__file__).with_suffix(".json").read_text())
    output = tmp_path / recipe["suite_id"]
    partial = output / artifact
    partial.parent.mkdir(parents=True)
    partial.write_text("preserve this evidence")
    support = ModuleType("support_coverage")
    support._env = lambda: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "support_coverage", support)
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="refusing to overwrite a trial"):
        driver.main()
    assert partial.read_text() == "preserve this evidence"
