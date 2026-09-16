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
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from support_coverage import _embedder, _env

from mindbridge import Memory, MemoryType, ObservationContext
from mindbridge.benchmarks.atm_bench import (
    atm_capture_time,
    atm_email_block,
    load_atm_bench,
    load_atm_emails,
)
from mindbridge.benchmarks.mem_gallery import load_mem_gallery
from mindbridge.models.openai_sdk import OpenAIModels
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


@dataclass(frozen=True, slots=True)
class Unit:
    """One physically isolated store's worth of a corpus, and the questions asked of it."""

    unit_id: str
    # `(content, occurred_at, source_id, capture_id, is_media)` per record.
    records: tuple[tuple[object, datetime, str, str, bool], ...]
    # `(question text, gold source ids)` per question.
    questions: tuple[tuple[str, frozenset[str]], ...]


def _atm_units(data: Path, limit: int) -> tuple[Unit, ...]:
    """ATM-Bench is one store: every question reads the same photographs, clips and emails."""
    records: list[tuple[object, datetime, str, str, bool]] = []
    for email in load_atm_emails(data / "raw_memory" / "email" / "emails.json"):
        records.append(
            (
                atm_email_block(email),
                email.occurred_at,
                email.email_id,
                email.occurred_at.date().isoformat(),
                False,
            )
        )
    for folder, pattern in (("image", "*.jpg"), ("video", "*.mp4")):
        for path in sorted((data / "raw_memory" / folder).glob(pattern)):
            if path.stat().st_size > INLINE:
                continue
            records.append((path, atm_capture_time(path.stem), path.stem, path.stem[:8], True))
    questions = tuple(
        (question.question, frozenset(question.evidence_ids))
        for question in load_atm_bench(data / "atm-bench" / "atm-bench.json")[:limit]
    )
    return (Unit("atm", tuple(records), questions),)


def _gallery_units(data: Path, limit: int) -> tuple[Unit, ...]:
    """Mem-Gallery is one store per topic: a persona, its dated sessions, and its questions.

    The capture is the session, which is what the release actually groups by, and a round's image
    rides on the same record as its text so a photograph of the persona is reachable both ways.
    """
    units = []
    for topic in load_mem_gallery(data / "dialog"):
        records: list[tuple[object, datetime, str, str, bool]] = []
        for session in topic.sessions:
            for round_ in session.rounds:
                text = (
                    f"[{session.occurred_at.date().isoformat()}] "
                    f"{topic.profile.name}: {round_.user}\nAssistant: {round_.assistant}"
                )
                image = (
                    None
                    if round_.image_path is None
                    else (data / "dialog" / round_.image_path).resolve()
                )
                content: object = text if image is None else (text, image)
                records.append(
                    (
                        content,
                        session.occurred_at,
                        round_.round_id,
                        session.session_id,
                        image is not None,
                    )
                )
        questions = tuple(
            (question.question, frozenset(question.clue_round_ids))
            for question in topic.questions
            if question.clue_round_ids
        )
        if questions:
            units.append(Unit(topic.topic, tuple(records), questions[:limit]))
    return tuple(units)


def _write(memory: Memory, unit: Unit) -> tuple[str, ...]:
    """Write one unit's records and return the ids of those carrying media."""
    media: list[str] = []
    for content, occurred_at, source, capture, is_media in unit.records:
        record = memory.add(
            content,  # type: ignore[arg-type]
            occurred_at=occurred_at,
            metadata={"source_id": source},
            memory_type=MemoryType.EPISODIC,
            context=ObservationContext(source_id=capture),
        )
        if is_media:
            media.append(record.id)
    return tuple(media)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("atm", "mem-gallery"), default="atm")
    parser.add_argument("--limit", type=int, default=120, help="questions per unit")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = _env()
    root = Path(settings["MINDBRIDGE_BENCH_ROOT"])
    if args.corpus == "atm":
        data = root / "atm-bench" / "data"
        units = _atm_units(data, args.limit)
    else:
        data = root / "mem-gallery" / "data"
        units = _gallery_units(data, args.limit)

    def backends() -> tuple[OpenAIModels, OpenCVFaceAnalyzer]:
        """One pair per unit, because `Memory.close()` closes the backends it was handed.

        Mem-Gallery is twenty physically isolated stores, so a shared analyzer is closed by the
        first one and refuses the second -- which is the correct behaviour of a resource a memory
        owns, and the reason this is a function rather than two values.
        """
        return (
            _embedder(settings),
            OpenCVFaceAnalyzer(
                detector_model=settings["MINDBRIDGE_FACE_DETECTOR"],
                recognizer_model=settings["MINDBRIDGE_FACE_RECOGNIZER"],
                score_threshold=FACE_SCORE_THRESHOLD,
            ),
        )

    edges = {edge: {"fired": 0, "rows": 0, "supplied_gold": 0} for edge in EDGES}
    totals = {
        "units": 0,
        "records": 0,
        "media_records": 0,
        "questions": 0,
        "identities": 0,
        "identities_spanning_several_memories": 0,
        "windows_missing_gold": 0,
        "windows_an_edge_could_complete": 0,
    }
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="edge-power-") as workspace:
        for unit in units:
            store = Path(workspace) / unit.unit_id
            embedder, face = backends()
            try:
                with Memory(
                    store,
                    embedder=embedder,
                    face_analyzer=face,
                    minimum_relevance=0.0,
                    ambiguity_margin=0.0,
                ) as memory:
                    media = _write(memory, unit)
                    for memory_id in media:
                        memory.faces(memory_id)
                    with memory._store.recall._connections.connection() as connection:
                        totals["identities"] += connection.execute(
                            "SELECT COUNT(*) FROM identities"
                        ).fetchone()[0]
                        totals["identities_spanning_several_memories"] += connection.execute(
                            """
                            SELECT COUNT(*) FROM (
                              SELECT fo.identity_id FROM face_observations fo
                              JOIN memory_assets ma ON ma.asset_id = fo.asset_id
                              GROUP BY fo.identity_id HAVING COUNT(DISTINCT ma.memory_id) > 1)
                            """
                        ).fetchone()[0]
                    records = memory._store.recall.recall_digest().records
                    ceiling = max(4 * LIMIT, -(-records // 5))
                    totals["units"] += 1
                    totals["records"] += records
                    totals["media_records"] += len(media)
                    for question, gold in unit.questions:
                        totals["questions"] += 1
                        hits = memory.search(question, limit=LIMIT)
                        window_ids = tuple(hit.id for hit in hits)
                        window = {str(hit.metadata.get("source_id")) for hit in hits}
                        missing = gold - window
                        if not missing:
                            continue
                        totals["windows_missing_gold"] += 1
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
                        totals["windows_an_edge_could_complete"] += 1 if recovered else 0
            finally:
                embedder.close()
                face.close()
            print(
                f"[{totals['units']}/{len(units)}] {unit.unit_id}: "
                f"{totals['windows_missing_gold']} missing so far, "
                f"{totals['windows_an_edge_could_complete']} completable",
                flush=True,
            )

    report = {
        "corpus": args.corpus,
        "recall_limit": LIMIT,
        "face_score_threshold": FACE_SCORE_THRESHOLD,
        "elapsed_minutes": round((time.perf_counter() - started) / 60, 1),
        "edges": edges,
        "note": (
            "headroom, not effect: `windows_an_edge_could_complete` is the most questions any "
            "reader arm built on these edges could move, so it bounds what a reader run could "
            "ever show"
        ),
        **totals,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
