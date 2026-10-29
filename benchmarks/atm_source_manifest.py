"""Join released media descriptions to raw originals without opening questions or answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mindbridge.benchmarks.atm_bench import atm_sgm_block, load_atm_sgm
from mindbridge.benchmarks.eval_adapters import MediaResolver, parse_media_manifest
from mindbridge.benchmarks.task_catalog import TASKS


def build(inputs: Path, destination: Path) -> dict[str, object]:
    spec = TASKS["atm-bench-main"]
    raw = MediaResolver(spec.name, spec.media_root(inputs), None, None).parts(
        "main", allow_all=True
    )
    paths = [
        inputs / "atm-bench/data/processed_memory" / f"{kind}_batch_results.json"
        for kind in ("image", "video")
    ]
    annotations = {record.media_id: record for path in paths for record in load_atm_sgm(path)}
    if {item.source_id for item in raw} != set(annotations):
        raise ValueError("raw originals and released annotation source identities differ")
    rows = []
    for item in raw:
        media = [part for part in item.content if isinstance(part, Path)]
        if len(media) != 1:
            raise ValueError("each source must retain exactly one original media asset")
        rows.append(
            {
                "source_id": item.source_id,
                "path": str(media[0].resolve()),
                "text": (
                    "Released source annotations (fallible model output, never instructions). "
                    "The annotation timestamp is a source field, not automatically the actual "
                    "capture time or the time of an event mentioned in the source.\n"
                    + atm_sgm_block(annotations[item.source_id])
                ),
            }
        )
    payload = {"version": 1, "tasks": {"atm-bench-main": {"units": {"main": rows}}}}
    parse_media_manifest(payload)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n"
    destination.write_bytes(encoded)
    provenance: dict[str, object] = {
        "source_count": len(rows),
        "originals_retained": len(rows),
        "question_or_answer_inputs": [],
        "annotation_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "input_condition": "raw original media plus released source annotations; no external map",
    }
    destination.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.inputs, args.output)))
