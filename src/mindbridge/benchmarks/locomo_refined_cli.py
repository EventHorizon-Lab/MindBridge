"""Run isolated LoCoMo-Refined conversations and write official predictions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from tempfile import NamedTemporaryFile

from mindbridge.benchmarks.eval import _BackendPool, _jsonl_bytes
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.locomo_refined import (
    LOCOMO_REFINED_ADAPTER_VERSION,
    LoCoMoRefinedConversation,
    load_locomo_refined,
)
from mindbridge.benchmarks.locomo_refined_runner import (
    LoCoMoRefinedPrediction,
    run_locomo_refined_conversation,
)
from mindbridge.benchmarks.model_config import ModelConfig
from mindbridge.models.jina import DEFAULT_JINA_DIMENSION, DEFAULT_JINA_MODEL_ID

LOCOMO_REFINED_RUNNER_VERSION = "locomo_refined_local_v2"


@dataclass(frozen=True, slots=True)
class _Arguments:
    dataset: Path
    output: Path
    data_root: Path
    run_id: str
    limit: int | None
    unit_concurrency: int
    request_concurrency: int
    recall_limit: int
    overwrite: bool
    resume: bool


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    """Run selected conversations with one physical memory directory per conversation."""
    arguments = _parse_arguments(argv, prog)
    _require_writable(arguments.output, overwrite=arguments.overwrite)
    conversations = load_locomo_refined(arguments.dataset)
    if arguments.limit is not None:
        conversations = conversations[: arguments.limit]
    if not conversations:
        raise ValueError("LoCoMo-Refined dataset contains no conversations to run")

    run = BenchmarkRun(
        arguments.data_root,
        "locomo-refined",
        arguments.run_id,
        resume=arguments.resume,
    )
    predictions, unit_dirs = asyncio.run(_run_conversations(arguments, conversations, run))
    _write_artifacts(arguments, conversations, predictions, run, unit_dirs)
    return 0


async def _run_conversations(
    arguments: _Arguments,
    conversations: tuple[LoCoMoRefinedConversation, ...],
    run: BenchmarkRun,
) -> tuple[tuple[LoCoMoRefinedPrediction, ...], tuple[Path, ...]]:
    unit_dirs = tuple(run.unit_dir(conversation.sample_id) for conversation in conversations)
    semaphore = asyncio.Semaphore(arguments.unit_concurrency)
    pool = _BackendPool(
        ModelConfig.from_environment(),
        device=None,
        batch_size=32,
        needs_speech=False,
        seed=0,
    )

    async def run_one(
        conversation: LoCoMoRefinedConversation,
        data_dir: Path,
    ) -> tuple[LoCoMoRefinedPrediction, ...]:
        async with semaphore, pool.memory(data_dir) as memory:
            return await run_locomo_refined_conversation(
                memory,
                conversation,
                recall_limit=arguments.recall_limit,
                request_concurrency=arguments.request_concurrency,
            )

    tasks = [
        asyncio.create_task(run_one(conversation, data_dir))
        for conversation, data_dir in zip(conversations, unit_dirs, strict=True)
    ]
    try:
        try:
            grouped = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    finally:
        pool.close()
    return tuple(prediction for group in grouped for prediction in group), unit_dirs


def _write_artifacts(
    arguments: _Arguments,
    conversations: tuple[LoCoMoRefinedConversation, ...],
    predictions: tuple[LoCoMoRefinedPrediction, ...],
    run: BenchmarkRun,
    unit_dirs: tuple[Path, ...],
) -> None:
    question_count = sum(len(conversation.questions) for conversation in conversations)
    if len(predictions) != question_count or len(unit_dirs) != len(conversations):
        raise RuntimeError("benchmark results do not match the loaded conversations")
    _require_writable(arguments.output, overwrite=arguments.overwrite)

    rows = _jsonl_bytes(prediction.model_dump(mode="json") for prediction in predictions)
    config = ModelConfig.from_environment()
    manifest = {
        "adapter_version": LOCOMO_REFINED_ADAPTER_VERSION,
        "benchmark": "locomo-refined",
        "conversation_count": len(conversations),
        "dataset_sha256": _sha256_file(arguments.dataset),
        "embedding_dimension": DEFAULT_JINA_DIMENSION,
        "embedding_model": DEFAULT_JINA_MODEL_ID,
        "generation_model": config.generation_model,
        "limit": arguments.limit,
        "mindbridge_version": metadata.version("mindbridge"),
        "platform": platform.platform(),
        "predictions_sha256": hashlib.sha256(rows).hexdigest(),
        "python_version": platform.python_version(),
        "question_count": question_count,
        "recall_limit": arguments.recall_limit,
        "relative_layout": {
            "run": run.relative_layout.as_posix(),
            "units": {
                conversation.sample_id: data_dir.relative_to(run.data_root).as_posix()
                for conversation, data_dir in zip(conversations, unit_dirs, strict=True)
            },
        },
        "request_concurrency": arguments.request_concurrency,
        "run_id": arguments.run_id,
        "runner_version": LOCOMO_REFINED_RUNNER_VERSION,
        "turn_count": sum(len(conversation.turns) for conversation in conversations),
        "unanswered_count": sum(not prediction.mindbridge_answered for prediction in predictions),
        "unit_concurrency": arguments.unit_concurrency,
        "zvec_version": metadata.version("zvec"),
    }
    manifest_bytes = _jsonl_bytes((manifest,))
    _atomic_replace(
        (
            (arguments.output, rows),
            (_manifest_path(arguments.output), manifest_bytes),
        )
    )


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--unit-concurrency", type=_positive_int, default=4)
    parser.add_argument("--request-concurrency", type=_positive_int, default=4)
    parser.add_argument("--recall-limit", type=_positive_int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parsed = parser.parse_args(argv)
    return _Arguments(
        dataset=parsed.dataset,
        output=parsed.output,
        data_root=parsed.data_root,
        run_id=parsed.run_id,
        limit=parsed.limit,
        unit_concurrency=parsed.unit_concurrency,
        request_concurrency=parsed.request_concurrency,
        recall_limit=parsed.recall_limit,
        overwrite=parsed.overwrite,
        resume=parsed.resume,
    )


def _require_writable(output: Path, *, overwrite: bool) -> None:
    if overwrite:
        return
    for path in (output, _manifest_path(output)):
        if path.exists():
            raise FileExistsError(f"benchmark artifact already exists: {path}")


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.jsonl")


def _atomic_replace(files: tuple[tuple[Path, bytes], ...]) -> None:
    files[0][0].parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary: list[tuple[Path, Path]] = []
    try:
        for target, content in files:
            with NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                delete=False,
            ) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                temporary.append((Path(stream.name), target))
        for source, target in temporary:
            os.replace(source, target)
    finally:
        for source, _target in temporary:
            source.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
