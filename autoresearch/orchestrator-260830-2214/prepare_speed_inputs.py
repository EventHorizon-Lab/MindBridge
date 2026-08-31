"""Materialize the answer-blind, pre-registered performance sentinels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SOURCES = {
    "locomo": "1aef6da702087d72515d1b9224f0956a2fbab415c11936253bf7d967d3cf8c17",
    "gallery": "d825f7e3798514144230bb9edc9c8811604c92f8ad16584ad24d1fc18029898b",
    "m3": "f43031bf0216a2ef2e7909f20ecd534e0098da17b63cdf94325a07f1bea372c1",
}
OUTPUTS = {
    "locomo": "e4ba8b7266d95cadaed569584fa881e93b9a22959f3f9c237f85bd7c32b83af1",
    "gallery": "eebb7bb10f19c69783f4c603a3706ac6008164cbf8cbfca5558695bdd95d114f",
    "m3": "bfca91c83e2e9693c2ee50d32eaa6e07b80473b8a9f42bac1612cd9160ce32bd",
}
GALLERY_FILE = "Dog_Behavior_Research_Academic_Life.json"


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _write(path: Path, payload: str, expected_sha256: str) -> None:
    actual_sha256 = hashlib.sha256(payload.encode()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"locked payload changed for {path.name}: {actual_sha256}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"refusing to replace different locked input: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _source(path: Path, name: str) -> object:
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != SOURCES[name]:
        raise ValueError(f"locked {name} source changed: {actual_sha256}")
    return _load(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("locomo", type=Path)
    parser.add_argument("gallery", type=Path)
    parser.add_argument("m3", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    locomo = _source(arguments.locomo, "locomo")
    if not isinstance(locomo, list) or len(locomo) != 10:
        raise ValueError("expected ten LoCoMo conversations")
    locomo_record = locomo[3]
    questions = locomo_record.get("qa") if isinstance(locomo_record, dict) else None
    if not isinstance(questions, list):
        raise ValueError("LoCoMo sentinel questions changed")
    selected_record = dict(locomo_record)
    selected_record["qa"] = questions[:20]
    locomo_selected = [selected_record]
    if selected_record.get("sample_id") != "conv-42" or len(questions[:20]) != 20:
        raise ValueError("LoCoMo sentinel membership changed")

    gallery = _source(arguments.gallery, "gallery")
    if not isinstance(gallery, dict):
        raise ValueError("Mem-Gallery sentinel source changed")
    questions = gallery.get("human-annotated QAs")
    if not isinstance(questions, list) or len(questions) < 20:
        raise ValueError("Mem-Gallery sentinel questions changed")
    gallery["human-annotated QAs"] = questions[:20]

    m3 = _source(arguments.m3, "m3")
    if not isinstance(m3, dict) or len(m3) != 100:
        raise ValueError("expected 100 M3 Robot units")
    m3_id = sorted(m3)[5]
    m3_unit = m3[m3_id]
    m3_questions = m3_unit.get("qa_list") if isinstance(m3_unit, dict) else None
    m3_selected = {m3_id: m3_unit}
    if m3_id != "bedroom_06" or not isinstance(m3_questions, list) or len(m3_questions) != 14:
        raise ValueError("M3 sentinel membership changed")

    _write(arguments.output / "locomo-refined.json", _payload(locomo_selected), OUTPUTS["locomo"])
    _write(arguments.output / "mem-gallery" / GALLERY_FILE, _payload(gallery), OUTPUTS["gallery"])
    _write(arguments.output / "m3-robot.json", _payload(m3_selected), OUTPUTS["m3"])
    print(json.dumps({"outputs": OUTPUTS, "question_counts": [20, 20, 14]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
