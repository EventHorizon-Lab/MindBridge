"""Reproducible schema smoke for official long-memory benchmark releases."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import AwareDatetime, Field

from mindbridge.benchmarks.atm_bench import ATM_BENCH_ADAPTER_VERSION, load_atm_bench
from mindbridge.benchmarks.cli import parser as build_parser
from mindbridge.benchmarks.egolife_qa import EGOLIFE_QA_ADAPTER_VERSION, load_egolife_qa
from mindbridge.benchmarks.egomem_reason import (
    EGOMEM_REASON_ADAPTER_VERSION,
    load_egomem_reason,
)
from mindbridge.benchmarks.egotempo import (
    EGOTEMPO_ADAPTER_VERSION,
    load_egotempo,
)
from mindbridge.benchmarks.locomo_refined import (
    LOCOMO_REFINED_ADAPTER_VERSION,
    load_locomo_refined,
)
from mindbridge.benchmarks.m3_bench import (
    M3_BENCH_ADAPTER_VERSION,
    M3BenchVideo,
    load_m3_bench,
)
from mindbridge.benchmarks.mem_gallery import (
    MEM_GALLERY_ADAPTER_VERSION,
    MemGalleryTopic,
    load_mem_gallery,
)
from mindbridge.benchmarks.memlens import MEMLENS_ADAPTER_VERSION, load_memlens
from mindbridge.benchmarks.mm_lifelong import (
    MM_LIFELONG_ADAPTER_VERSION,
    MMLifelongQuestion,
    MMLifelongSplit,
    load_mm_lifelong,
)
from mindbridge.benchmarks.supermemory_vqa import (
    SUPERMEMORY_VQA_ADAPTER_VERSION,
    load_supermemory_vqa,
)
from mindbridge.benchmarks.video_mme import VIDEO_MME_ADAPTER_VERSION, load_video_mme
from mindbridge.contracts import ContractModel, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file


class BenchmarkDatasetSummary(ContractModel):
    """Source identity and parsed size for one benchmark split."""

    benchmark: NonEmptyString
    source_repository: NonEmptyString
    source_file: NonEmptyString
    source_sha256: Sha256Hex
    adapter_version: NonEmptyString
    context_count: int = Field(gt=0)
    memory_item_count: int = Field(gt=0)
    question_count: int = Field(gt=0)


class DatasetAdapterSmokeResult(ContractModel):
    """Machine-readable proof that the official annotations still parse."""

    created_at: AwareDatetime
    datasets: tuple[BenchmarkDatasetSummary, ...] = Field(min_length=16)
    passed: bool


def run_dataset_adapter_smoke(
    *,
    locomo_refined_path: Path,
    m3_robot_path: Path,
    m3_web_path: Path,
    video_mme_path: Path,
    egolife_path: Path,
    egotempo_path: Path,
    egomem_path: Path,
    memlens_path: Path,
    mm_day_path: Path,
    mm_week_path: Path,
    mm_month_train_path: Path,
    mm_month_val_path: Path,
    supermemory_path: Path,
    atm_path: Path,
    atm_hard_path: Path,
    mem_gallery_dialog_path: Path,
) -> DatasetAdapterSmokeResult:
    """Parse every official benchmark annotation release and record its inputs."""
    locomo_refined = load_locomo_refined(locomo_refined_path)
    m3_robot = load_m3_bench(m3_robot_path)
    m3_web = load_m3_bench(m3_web_path)
    video_mme = load_video_mme(video_mme_path)
    egolife = load_egolife_qa(egolife_path)
    egotempo = load_egotempo(egotempo_path)
    egomem = load_egomem_reason(egomem_path)
    memlens = load_memlens(memlens_path)
    mm_day = load_mm_lifelong(mm_day_path, "day_test")
    mm_week = load_mm_lifelong(mm_week_path, "week_test")
    mm_month_train = load_mm_lifelong(mm_month_train_path, "month_train")
    mm_month_val = load_mm_lifelong(mm_month_val_path, "month_val")
    supermemory = load_supermemory_vqa(supermemory_path)
    atm = load_atm_bench(atm_path)
    atm_hard = load_atm_bench(atm_hard_path)
    mem_gallery = load_mem_gallery(mem_gallery_dialog_path)
    return DatasetAdapterSmokeResult(
        created_at=datetime.now(timezone.utc),
        datasets=(
            BenchmarkDatasetSummary(
                benchmark="LoCoMo-Refined",
                source_repository="mem-eval-suite/LoCoMo_refined",
                source_file=locomo_refined_path.name,
                source_sha256=sha256_file(locomo_refined_path),
                adapter_version=LOCOMO_REFINED_ADAPTER_VERSION,
                context_count=len(locomo_refined),
                memory_item_count=sum(len(item.turns) for item in locomo_refined),
                question_count=sum(len(item.questions) for item in locomo_refined),
            ),
            _m3_summary("M3-Bench-robot", m3_robot_path, m3_robot),
            _m3_summary("M3-Bench-web", m3_web_path, m3_web),
            BenchmarkDatasetSummary(
                benchmark="Video-MME",
                source_repository="lmms-eval/Video-MME",
                source_file=video_mme_path.name,
                source_sha256=sha256_file(video_mme_path),
                adapter_version=VIDEO_MME_ADAPTER_VERSION,
                context_count=len(video_mme),
                memory_item_count=len(video_mme),
                question_count=sum(len(video.questions) for video in video_mme),
            ),
            BenchmarkDatasetSummary(
                benchmark="EgoLifeQA",
                source_repository="lmms-lab/EgoLife",
                source_file=egolife_path.name,
                source_sha256=sha256_file(egolife_path),
                adapter_version=EGOLIFE_QA_ADAPTER_VERSION,
                context_count=1,
                memory_item_count=len({question.query_day for question in egolife}),
                question_count=len(egolife),
            ),
            BenchmarkDatasetSummary(
                benchmark="EgoTempo",
                source_repository="google-research-datasets/egotempo",
                source_file=egotempo_path.name,
                source_sha256=sha256_file(egotempo_path),
                adapter_version=EGOTEMPO_ADAPTER_VERSION,
                context_count=len({question.source_video_id for question in egotempo}),
                memory_item_count=len({question.clip_id for question in egotempo}),
                question_count=len(egotempo),
            ),
            BenchmarkDatasetSummary(
                benchmark="EgoMemReason",
                source_repository="Ted412/EgoMemReason",
                source_file=egomem_path.name,
                source_sha256=sha256_file(egomem_path),
                adapter_version=EGOMEM_REASON_ADAPTER_VERSION,
                context_count=len({question.identity for question in egomem}),
                memory_item_count=len(
                    {(question.identity, question.query_time) for question in egomem}
                ),
                question_count=len(egomem),
            ),
            BenchmarkDatasetSummary(
                benchmark="MEMLENS-32K",
                source_repository="xiyuRenBill/MEMLENS",
                source_file=memlens_path.name,
                source_sha256=sha256_file(memlens_path),
                adapter_version=MEMLENS_ADAPTER_VERSION,
                context_count=len(memlens),
                memory_item_count=sum(len(question.sessions) for question in memlens),
                question_count=len(memlens),
            ),
            _mm_lifelong_summary(
                "MM-Lifelong-day-test",
                mm_day_path,
                "day_test",
                mm_day,
            ),
            _mm_lifelong_summary(
                "MM-Lifelong-week-test",
                mm_week_path,
                "week_test",
                mm_week,
            ),
            _mm_lifelong_summary(
                "MM-Lifelong-month-train",
                mm_month_train_path,
                "month_train",
                mm_month_train,
            ),
            _mm_lifelong_summary(
                "MM-Lifelong-month-val",
                mm_month_val_path,
                "month_val",
                mm_month_val,
            ),
            BenchmarkDatasetSummary(
                benchmark="SuperMemory-VQA",
                source_repository="OSU-AIoT-MLSys-Lab/SuperMemory-VQA",
                source_file=supermemory_path.name,
                source_sha256=sha256_file(supermemory_path),
                adapter_version=SUPERMEMORY_VQA_ADAPTER_VERSION,
                context_count=len({question.subject for question in supermemory}),
                memory_item_count=len(
                    {video_id for question in supermemory for video_id in question.source_video_ids}
                ),
                question_count=len(supermemory),
            ),
            BenchmarkDatasetSummary(
                benchmark="ATM-Bench",
                source_repository="Jingbiao/ATM-Bench",
                source_file=atm_path.name,
                source_sha256=sha256_file(atm_path),
                adapter_version=ATM_BENCH_ADAPTER_VERSION,
                context_count=1,
                memory_item_count=len(
                    {evidence for question in atm for evidence in question.evidence_ids}
                ),
                question_count=len(atm),
            ),
            BenchmarkDatasetSummary(
                benchmark="ATM-Bench-Hard",
                source_repository="Jingbiao/ATM-Bench",
                source_file=atm_hard_path.name,
                source_sha256=sha256_file(atm_hard_path),
                adapter_version=ATM_BENCH_ADAPTER_VERSION,
                context_count=1,
                memory_item_count=len(
                    {evidence for question in atm_hard for evidence in question.evidence_ids}
                ),
                question_count=len(atm_hard),
            ),
            _mem_gallery_summary(mem_gallery_dialog_path, mem_gallery),
        ),
        passed=True,
    )


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run the official-data smoke and emit a versioned JSON manifest."""
    parser = build_parser(prog=prog, description=__doc__)
    parser.add_argument(
        "--locomo-refined",
        type=Path,
        required=True,
        help="official locomo-refined release to parse",
    )
    parser.add_argument(
        "--m3-robot", type=Path, required=True, help="official m3 robot release to parse"
    )
    parser.add_argument(
        "--m3-web", type=Path, required=True, help="official m3 web release to parse"
    )
    parser.add_argument(
        "--video-mme", type=Path, required=True, help="official video mme release to parse"
    )
    parser.add_argument(
        "--egolife", type=Path, required=True, help="official egolife release to parse"
    )
    parser.add_argument(
        "--egotempo", type=Path, required=True, help="official egotempo release to parse"
    )
    parser.add_argument(
        "--egomem", type=Path, required=True, help="official egomem release to parse"
    )
    parser.add_argument(
        "--memlens", type=Path, required=True, help="official memlens release to parse"
    )
    parser.add_argument(
        "--mm-day", type=Path, required=True, help="official mm day release to parse"
    )
    parser.add_argument(
        "--mm-week", type=Path, required=True, help="official mm week release to parse"
    )
    parser.add_argument(
        "--mm-month-train",
        type=Path,
        required=True,
        help="official mm month train release to parse",
    )
    parser.add_argument(
        "--mm-month-val", type=Path, required=True, help="official mm month val release to parse"
    )
    parser.add_argument(
        "--supermemory", type=Path, required=True, help="official supermemory release to parse"
    )
    parser.add_argument(
        "--atm", type=Path, required=True, help="official ATM-Bench release to parse"
    )
    parser.add_argument(
        "--atm-hard", type=Path, required=True, help="official ATM-Bench-Hard release to parse"
    )
    parser.add_argument(
        "--mem-gallery-dialog",
        type=Path,
        required=True,
        help="official Mem-Gallery data/dialog directory to parse",
    )
    arguments = parser.parse_args(argv)
    result = run_dataset_adapter_smoke(
        locomo_refined_path=arguments.locomo_refined,
        m3_robot_path=arguments.m3_robot,
        m3_web_path=arguments.m3_web,
        video_mme_path=arguments.video_mme,
        egolife_path=arguments.egolife,
        egotempo_path=arguments.egotempo,
        egomem_path=arguments.egomem,
        memlens_path=arguments.memlens,
        mm_day_path=arguments.mm_day,
        mm_week_path=arguments.mm_week,
        mm_month_train_path=arguments.mm_month_train,
        mm_month_val_path=arguments.mm_month_val,
        supermemory_path=arguments.supermemory,
        atm_path=arguments.atm,
        atm_hard_path=arguments.atm_hard,
        mem_gallery_dialog_path=arguments.mem_gallery_dialog,
    )
    print(result.model_dump_json(indent=2))


def _m3_summary(
    benchmark: str,
    source_path: Path,
    videos: tuple[M3BenchVideo, ...],
) -> BenchmarkDatasetSummary:
    return BenchmarkDatasetSummary(
        benchmark=benchmark,
        source_repository="ByteDance-Seed/m3-agent",
        source_file=source_path.name,
        source_sha256=sha256_file(source_path),
        adapter_version=M3_BENCH_ADAPTER_VERSION,
        context_count=len(videos),
        memory_item_count=len(videos),
        question_count=sum(len(video.questions) for video in videos),
    )


def _mem_gallery_summary(
    dialog_directory: Path,
    topics: tuple[MemGalleryTopic, ...],
) -> BenchmarkDatasetSummary:
    """Summarize the whole dialog directory, digesting its files in sorted order.

    The release is twenty files, so the digest covers the concatenation of their own
    digests: one number that changes if any topic file changes.
    """
    digest = hashlib.sha256(
        "".join(sha256_file(path) for path in sorted(dialog_directory.glob("*.json"))).encode(
            "utf-8"
        )
    ).hexdigest()
    return BenchmarkDatasetSummary(
        benchmark="Mem-Gallery",
        source_repository="Ethan-Bei/Mem-Gallery",
        source_file=f"{dialog_directory.name}/*.json",
        source_sha256=digest,
        adapter_version=MEM_GALLERY_ADAPTER_VERSION,
        context_count=len(topics),
        memory_item_count=sum(
            len(session.rounds) for topic in topics for session in topic.sessions
        ),
        question_count=sum(len(topic.questions) for topic in topics),
    )


def _mm_lifelong_summary(
    benchmark: str,
    source_path: Path,
    split: MMLifelongSplit,
    questions: tuple[MMLifelongQuestion, ...],
) -> BenchmarkDatasetSummary:
    return BenchmarkDatasetSummary(
        benchmark=benchmark,
        source_repository="MM-Lifelong/MM-Lifelong",
        source_file=f"{split}:{source_path.name}",
        source_sha256=sha256_file(source_path),
        adapter_version=MM_LIFELONG_ADAPTER_VERSION,
        context_count=1,
        memory_item_count=sum(question.clue_interval_count for question in questions),
        question_count=len(questions),
    )


if __name__ == "__main__":
    main()
