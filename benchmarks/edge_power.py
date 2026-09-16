#!/usr/bin/env python3
"""Ask whether an edge could change any question, before paying a reader to find out.

Retrieval only: no generation model, no judge. For each question it takes the window the ranking
earns, asks what each structural edge links that window to, and counts the rows that are gold *and*
absent from the window -- the only rows that can move complete support. A reader arm whose
headroom is a handful of questions cannot be distinguished from noise at these sample sizes, and
this is the cheap way to know that in fifteen minutes rather than three hours.

It exists because the round before it skipped this step: a gate was built on a prediction, run for
three hours, and falsified. The instrument is the lesson.

Endpoints come from the same out-of-tree env file `support_coverage.py` reads. Face recognition
needs the two OpenCV model files named there; without them the identity edge is empty by
construction rather than by measurement, which is a different and much weaker claim.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from support_coverage import _embedder, _env

from mindbridge import Memory, MemoryType, ObservationContext
from mindbridge.benchmarks.atm_bench import (
    atm_capture_time,
    atm_email_block,
    load_atm_bench,
    load_atm_emails,
)
from mindbridge.models.opencv_face import OpenCVFaceAnalyzer

LIMIT = 12
INLINE = 20 * 1024 * 1024 * 3 // 4
EDGES = ("capture", "identity", "place", "lineage", "evidence")
# The shipped default, and measurement says it is right. Lowering it to 0.5 looked like more
# signal -- 384 detections from 400 photographs instead of 19 -- and was the opposite: scored
# against a vision model asked whether a photo shows a face clearly enough to recognise the
# person, YuNet at 0.5 has recall 1.00 and precision 0.21, because only 3 of 60 photographs here
# contain a recognisable face at all. The recognizer then shows what those extra detections are:
# at 0.9 the pairwise cosines top out at 0.384 and one pair of 171 crosses SFace's own 0.363, so
# it correctly declines to merge strangers; at 0.5 the median rises from 0.116 to 0.226 and 15.9 %
# of pairs cross it, which is non-face crops embedding near each other and merging spuriously.
FACE_SCORE_THRESHOLD = 0.9


def _source_id(record: object) -> str:
    """The benchmark's own id for a record, read out of canonical metadata JSON.

    Parsed rather than pattern-matched: canonical JSON is compact, so a split on `"source_id": "`
    matches nothing and silently reports that no edge ever supplies gold. That is exactly what the
    first version of this check claimed.
    """
    return str(json.loads(record.metadata_json).get("source_id", ""))  # type: ignore[attr-defined]


def _ingest(memory: Memory, data: Path) -> tuple[str, ...]:
    """Write the ATM corpus with day captures and return its media record ids."""
    media: list[str] = []
    for email in load_atm_emails(data / "raw_memory" / "email" / "emails.json"):
        memory.add(
            atm_email_block(email),
            occurred_at=email.occurred_at,
            metadata={"source_id": email.email_id},
            memory_type=MemoryType.EPISODIC,
            context=ObservationContext(source_id=email.occurred_at.date().isoformat()),
        )
    for folder, pattern in (("image", "*.jpg"), ("video", "*.mp4")):
        for path in sorted((data / "raw_memory" / folder).glob(pattern)):
            if path.stat().st_size > INLINE:
                continue
            media.append(
                memory.add(
                    path,
                    occurred_at=atm_capture_time(path.stem),
                    metadata={"source_id": path.stem},
                    memory_type=MemoryType.EPISODIC,
                    context=ObservationContext(source_id=path.stem[:8]),
                ).id
            )
    return tuple(media)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument(
        "--store",
        type=Path,
        default=Path.home() / ".cache/mindbridge-edge-power/atm",
        help="kept between runs so the query pass can be repeated without re-ingesting",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = _env()
    data = Path(settings["MINDBRIDGE_BENCH_ROOT"]) / "atm-bench" / "data"
    embedder = _embedder(settings)
    face = OpenCVFaceAnalyzer(
        detector_model=settings["MINDBRIDGE_FACE_DETECTOR"],
        recognizer_model=settings["MINDBRIDGE_FACE_RECOGNIZER"],
        score_threshold=FACE_SCORE_THRESHOLD,
    )
    questions = load_atm_bench(data / "atm-bench" / "atm-bench.json")[: args.limit]
    fresh = not args.store.exists()
    timings: dict[str, float] = {}
    try:
        with Memory(
            args.store,
            embedder=embedder,
            face_analyzer=face,
            minimum_relevance=0.0,
            ambiguity_margin=0.0,
        ) as memory:
            if fresh:
                started = time.perf_counter()
                media = _ingest(memory, data)
                timings["ingest_minutes"] = round((time.perf_counter() - started) / 60, 1)
                started = time.perf_counter()
                for memory_id in media:
                    memory.faces(memory_id)
                timings["face_pass_minutes"] = round((time.perf_counter() - started) / 60, 1)
            with memory._store.recall._connections.connection() as connection:
                identities = connection.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
                spanning = connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                      SELECT fo.identity_id FROM face_observations fo
                      JOIN memory_assets ma ON ma.asset_id = fo.asset_id
                      GROUP BY fo.identity_id HAVING COUNT(DISTINCT ma.memory_id) > 1)
                    """
                ).fetchone()[0]
            records = memory._store.recall.recall_digest().records
            # The bound recall programs apply to a predicate, restated here so a probe and the
            # kernel refuse the same non-selective edge.
            ceiling = max(4 * LIMIT, -(-records // 5))
            edges = {edge: {"fired": 0, "rows": 0, "supplied_gold": 0} for edge in EDGES}
            incomplete = 0
            recoverable = 0
            for question in questions:
                hits = memory.search(question.question, limit=LIMIT)
                window_ids = tuple(hit.id for hit in hits)
                window = {str(hit.metadata.get("source_id")) for hit in hits}
                missing = set(question.evidence_ids) - window
                if not missing:
                    continue
                incomplete += 1
                recovered: set[str] = set()
                for edge in EDGES:
                    read = memory._store.recall.related_memories(
                        window_ids, edges=(edge,), ceiling=ceiling, max_rows=24
                    )
                    found = missing & {_source_id(row) for row in read}
                    edges[edge]["fired"] += 1 if len(read) else 0
                    edges[edge]["rows"] += len(read)
                    edges[edge]["supplied_gold"] += 1 if found else 0
                    recovered |= found
                recoverable += 1 if recovered else 0
    finally:
        embedder.close()
        face.close()

    report = {
        "corpus": "atm",
        "questions": len(questions),
        "recall_limit": LIMIT,
        "face_score_threshold": FACE_SCORE_THRESHOLD,
        "identities": identities,
        "identities_spanning_several_memories": spanning,
        "records": records,
        "edge_ceiling": ceiling,
        "windows_missing_gold": incomplete,
        "windows_an_edge_could_complete": recoverable,
        "edges": edges,
        "timings": timings,
        "note": (
            "headroom, not effect: `windows_an_edge_could_complete` is the most questions any "
            "reader arm built on these edges could move, so it bounds what a reader run could "
            "ever show"
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
