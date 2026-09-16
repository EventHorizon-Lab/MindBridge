"""The corpora these benchmarks read, shaped so one loop can walk any of them.

A unit is one physically isolated store and the questions asked of it. ATM-Bench is a single unit
-- every question reads the same photographs, clips and emails -- while Mem-Gallery is one unit per
persona. Both scripts in this directory need the same loaders, so they live here rather than in
either of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mindbridge import Memory, MemoryType, ObservationContext
from mindbridge.benchmarks.atm_bench import (
    atm_capture_time,
    atm_email_block,
    load_atm_bench,
    load_atm_emails,
)
from mindbridge.benchmarks.mem_gallery import load_mem_gallery

# The provider adapter refuses an inline media item over 20 MiB base64-encoded, about 15 MiB of
# bytes, and refuses it before the request rather than after -- so the video-sampling retry that
# rescues an over-long clip never runs for an over-large one. Twenty-one of ATM's 533 clips are
# over it; they are skipped, and three of the benchmark's 1,013 questions cite one.
INLINE = 20 * 1024 * 1024 * 3 // 4


def source_id(record: object) -> str:
    """The benchmark's own id for a record, read out of canonical metadata JSON.

    Parsed rather than pattern-matched: canonical JSON is compact, so a split on `"source_id": "`
    matches nothing and silently reports that no edge ever supplies gold. That is exactly what the
    first version of the headroom check claimed.
    """
    return str(json.loads(record.metadata_json).get("source_id", ""))  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class Unit:
    """One physically isolated store's worth of a corpus, and the questions asked of it."""

    unit_id: str
    # `(content, occurred_at, source_id, capture_id, is_media)` per record.
    records: tuple[tuple[object, datetime, str, str, bool], ...]
    # `(question text, gold source ids, reference answer)` per question.
    questions: tuple[tuple[str, frozenset[str], str], ...]


def atm_units(data: Path, limit: int) -> tuple[Unit, ...]:
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
        (question.question, frozenset(question.evidence_ids), question.reference_answer)
        for question in load_atm_bench(data / "atm-bench" / "atm-bench.json")[:limit]
    )
    return (Unit("atm", tuple(records), questions),)


def gallery_units(data: Path, limit: int) -> tuple[Unit, ...]:
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
            (question.question, frozenset(question.clue_round_ids), question.reference_answer)
            for question in topic.questions
            if question.clue_round_ids
        )
        if questions:
            units.append(Unit(topic.topic, tuple(records), questions[:limit]))
    return tuple(units)


def write_unit(memory: Memory, unit: Unit) -> tuple[str, ...]:
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
