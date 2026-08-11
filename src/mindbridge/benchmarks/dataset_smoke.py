"""Reproducible schema smoke for official long-memory benchmark releases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from pydantic import AwareDatetime, Field

from mindbridge.benchmarks.locomo import LOCOMO_ADAPTER_VERSION, load_locomo
from mindbridge.benchmarks.m3_bench import (
    M3_BENCH_ADAPTER_VERSION,
    M3BenchVideo,
    load_m3_bench,
)
from mindbridge.contracts import ContractModel, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file


class BenchmarkDatasetSummary(ContractModel):
    """Exact source identity and parsed size for one benchmark split."""

    benchmark: NonEmptyString
    source_repository: NonEmptyString
    source_revision: NonEmptyString
    source_file: NonEmptyString
    source_sha256: Sha256Hex
    adapter_version: NonEmptyString
    context_count: int = Field(gt=0)
    memory_item_count: int = Field(gt=0)
    question_count: int = Field(gt=0)


class DatasetAdapterSmokeResult(ContractModel):
    """Machine-readable proof that pinned official annotations still parse."""

    created_at: AwareDatetime
    datasets: tuple[BenchmarkDatasetSummary, ...] = Field(min_length=3)
    passed: bool


def run_dataset_adapter_smoke(
    *,
    locomo_path: Path,
    locomo_revision: str,
    m3_robot_path: Path,
    m3_web_path: Path,
    m3_revision: str,
) -> DatasetAdapterSmokeResult:
    """Parse LoCoMo and both M3-Bench splits and record immutable inputs."""
    locomo = load_locomo(locomo_path)
    m3_robot = load_m3_bench(m3_robot_path)
    m3_web = load_m3_bench(m3_web_path)
    return DatasetAdapterSmokeResult(
        created_at=datetime.now(timezone.utc),
        datasets=(
            BenchmarkDatasetSummary(
                benchmark="LoCoMo",
                source_repository="snap-research/locomo",
                source_revision=locomo_revision,
                source_file=locomo_path.name,
                source_sha256=sha256_file(locomo_path),
                adapter_version=LOCOMO_ADAPTER_VERSION,
                context_count=len(locomo),
                memory_item_count=sum(len(item.turns) for item in locomo),
                question_count=sum(len(item.questions) for item in locomo),
            ),
            _m3_summary("M3-Bench-robot", m3_robot_path, m3_revision, m3_robot),
            _m3_summary("M3-Bench-web", m3_web_path, m3_revision, m3_web),
        ),
        passed=True,
    )


def main() -> None:
    """Run the official-data smoke and emit a versioned JSON manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--locomo-revision", required=True)
    parser.add_argument("--m3-robot", type=Path, required=True)
    parser.add_argument("--m3-web", type=Path, required=True)
    parser.add_argument("--m3-revision", required=True)
    arguments = parser.parse_args()
    result = run_dataset_adapter_smoke(
        locomo_path=arguments.locomo,
        locomo_revision=arguments.locomo_revision,
        m3_robot_path=arguments.m3_robot,
        m3_web_path=arguments.m3_web,
        m3_revision=arguments.m3_revision,
    )
    print(result.model_dump_json(indent=2))


def _m3_summary(
    benchmark: str,
    source_path: Path,
    source_revision: str,
    videos: tuple[M3BenchVideo, ...],
) -> BenchmarkDatasetSummary:
    return BenchmarkDatasetSummary(
        benchmark=benchmark,
        source_repository="ByteDance-Seed/m3-agent",
        source_revision=source_revision,
        source_file=source_path.name,
        source_sha256=sha256_file(source_path),
        adapter_version=M3_BENCH_ADAPTER_VERSION,
        context_count=len(videos),
        memory_item_count=len(videos),
        question_count=sum(len(video.questions) for video in videos),
    )


if __name__ == "__main__":
    main()
