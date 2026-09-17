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
from pathlib import Path

from corpora import atm_units, gallery_units, source_id, write_unit
from support_coverage import _embedder, _env

from mindbridge import Memory
from mindbridge.kernel.answering import expansion_ceiling
from mindbridge.models.openai_sdk import OpenAIModels
from mindbridge.models.opencv_face import OpenCVFaceAnalyzer

LIMIT = 12
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
        units = atm_units(data, args.limit)
    else:
        data = root / "mem-gallery" / "data"
        units = gallery_units(data, args.limit)

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
                    media = write_unit(memory, unit)
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
                    # The product's own bound, not a restatement of it: a headroom number
                    # measured against a different ceiling is not the bound expansion applies.
                    ceiling = expansion_ceiling(LIMIT, records)
                    totals["units"] += 1
                    totals["records"] += records
                    totals["media_records"] += len(media)
                    for question, gold, _reference in unit.questions:
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
                            found = missing & {source_id(row) for row in read}
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
