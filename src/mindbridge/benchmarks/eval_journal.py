"""Crash-safe, write-only sample journaling for long benchmark attempts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

JOURNAL_SCHEMA_VERSION = 1
JOURNAL_PATH_ENV = "MINDBRIDGE_EVAL_JOURNAL_PATH"
JOURNAL_RUN_ID_ENV = "MINDBRIDGE_EVAL_JOURNAL_RUN_ID"
JOURNAL_ATTEMPT_ID_ENV = "MINDBRIDGE_EVAL_JOURNAL_ATTEMPT_ID"

_RAW_AGREEMENT_FIELDS = (
    "prediction",
    "error_code",
    "error_reason",
    "error_stage",
    "error_cause_type",
)
_UsageSnapshot = Callable[[Mapping[str, object], int], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class JournalRecord:
    """One checksum-verified completed sample from a single execution attempt."""

    payload: Mapping[str, object]
    payload_sha256: str

    @property
    def identity(self) -> tuple[str, str, str, str, str, str]:
        identity = cast(Mapping[str, object], self.payload["identity"])
        return _identity_key(identity)


class DurableResultJournal:
    """Append one fsynced JSONL record per unique completed sample."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        attempt_id: str,
        usage_snapshot: _UsageSnapshot | None = None,
    ) -> None:
        if not run_id or not attempt_id:
            raise ValueError("journal run_id and attempt_id must be non-empty")
        self.path = path
        self.run_id = run_id
        self.attempt_id = attempt_id
        self._usage_snapshot = usage_snapshot
        self._seen: set[tuple[str, str, str, str, str, str]] = set()
        self._counts: dict[tuple[str, str], int] = {}
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        _fsync_directory(path.parent)

    def append(self, sample: Mapping[str, object], *, expected_label: str | None) -> None:
        identity = _identity(self.attempt_id, self.run_id, sample)
        identity_key = _identity_key(identity)
        if identity_key in self._seen:
            raise ValueError(
                f"duplicate journal sample identity: {_display_identity(identity_key)}"
            )
        task_arm = (str(sample["task"]), str(sample["arm"]))
        completed_count = self._counts.get(task_arm, 0) + 1
        usage = (
            None
            if self._usage_snapshot is None
            else dict(self._usage_snapshot(sample, completed_count))
        )
        payload: dict[str, object] = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "sequence": len(self._seen) + 1,
            "identity": identity,
            "expected_label": expected_label,
            "roster": _roster(sample, expected_label),
            "outcome": "error" if sample.get("error_code") is not None else "generation",
            "sample": dict(sample),
            "cumulative_usage": usage,
        }
        checksum = _sha256_payload(payload)
        line = _json_bytes({**payload, "payload_sha256": checksum}) + b"\n"
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            _write_all(descriptor, line)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._seen.add(identity_key)
        self._counts[task_arm] = completed_count


_environment_journal: DurableResultJournal | None = None


def configure_environment_journal(
    *, usage_snapshot: _UsageSnapshot | None = None
) -> DurableResultJournal | None:
    """Create the attempt journal named by the runner environment, if configured."""
    global _environment_journal
    path = os.environ.get(JOURNAL_PATH_ENV)
    if path is None:
        return None
    run_id = os.environ.get(JOURNAL_RUN_ID_ENV)
    attempt_id = os.environ.get(JOURNAL_ATTEMPT_ID_ENV)
    if not run_id or not attempt_id:
        raise RuntimeError("journal path requires run and attempt identifiers")
    if _environment_journal is not None:
        raise RuntimeError("the evaluation journal is already configured")
    _environment_journal = DurableResultJournal(
        Path(path),
        run_id=run_id,
        attempt_id=attempt_id,
        usage_snapshot=usage_snapshot,
    )
    return _environment_journal


def record_environment_sample(sample: Mapping[str, object], *, expected_label: str | None) -> None:
    """Persist one sample when the attempt runner enabled journaling."""
    if _environment_journal is not None:
        _environment_journal.append(sample, expected_label=expected_label)


def reset_environment_journal() -> None:
    """Release process-local journal state after one evaluator invocation."""
    global _environment_journal
    _environment_journal = None


def load_journal(path: Path) -> tuple[JournalRecord, ...]:
    """Read valid records, ignoring at most one unterminated final record."""
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
    records: list[JournalRecord] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for position, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid journal record at line {position}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"invalid journal record at line {position}")
        checksum = raw.pop("payload_sha256", None)
        if not isinstance(checksum, str) or checksum != _sha256_payload(raw):
            raise ValueError(f"journal checksum mismatch at line {position}")
        _validate_payload(raw, position=position)
        record = JournalRecord(payload=raw, payload_sha256=checksum)
        if record.identity in seen:
            raise ValueError(
                f"duplicate journal sample identity at line {position}: "
                f"{_display_identity(record.identity)}"
            )
        seen.add(record.identity)
        records.append(record)
    return tuple(records)


def merge_finalized_samples(
    records: Sequence[JournalRecord],
    finalized: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    attempt_id: str,
) -> tuple[Mapping[str, object], ...]:
    """Prefer finalized rows only when their raw answer and error fields agree."""
    journal = {
        record.identity: cast(Mapping[str, object], record.payload["sample"]) for record in records
    }
    merged = dict(journal)
    finalized_seen: set[tuple[str, str, str, str, str, str]] = set()
    for sample in finalized:
        identity = _identity_key(_identity(attempt_id, run_id, sample))
        if identity in finalized_seen:
            raise ValueError(f"duplicate finalized sample identity: {_display_identity(identity)}")
        finalized_seen.add(identity)
        prior = journal.get(identity)
        if prior is not None and any(
            prior.get(field) != sample.get(field) for field in _RAW_AGREEMENT_FIELDS
        ):
            raise ValueError(
                f"finalized sample conflicts with journal: {_display_identity(identity)}"
            )
        merged[identity] = dict(sample)
    return tuple(merged[identity] for identity in sorted(merged))


def _identity(attempt_id: str, run_id: str, sample: Mapping[str, object]) -> dict[str, str]:
    fields = {
        "attempt_id": attempt_id,
        "run_id": run_id,
        "task": sample.get("task"),
        "unit_id": sample.get("unit_id"),
        "question_id": sample.get("question_id"),
        "arm": sample.get("arm"),
    }
    if any(not isinstance(value, str) or not value for value in fields.values()):
        raise ValueError("journal sample identity fields must be non-empty strings")
    return cast(dict[str, str], fields)


def _identity_key(identity: Mapping[str, object]) -> tuple[str, str, str, str, str, str]:
    return cast(
        tuple[str, str, str, str, str, str],
        tuple(
            str(identity[field])
            for field in ("attempt_id", "run_id", "task", "unit_id", "question_id", "arm")
        ),
    )


def _validate_payload(payload: Mapping[str, object], *, position: int) -> None:
    if payload.get("schema_version") != JOURNAL_SCHEMA_VERSION:
        raise ValueError(f"unsupported journal schema at line {position}")
    if not isinstance(payload.get("sequence"), int) or cast(int, payload["sequence"]) != position:
        raise ValueError(f"non-contiguous journal sequence at line {position}")
    identity = payload.get("identity")
    sample = payload.get("sample")
    if not isinstance(identity, Mapping) or not isinstance(sample, Mapping):
        raise ValueError(f"invalid journal payload at line {position}")
    if _identity_key(identity)[2:] != tuple(
        str(sample.get(field)) for field in ("task", "unit_id", "question_id", "arm")
    ):
        raise ValueError(f"journal identity/sample mismatch at line {position}")
    expected = payload.get("expected_label")
    if expected is not None and not isinstance(expected, str):
        raise ValueError(f"invalid expected label at line {position}")
    roster = payload.get("roster")
    if not isinstance(roster, Mapping) or dict(roster) != _roster(sample, expected):
        raise ValueError(f"journal roster/sample mismatch at line {position}")
    outcome = payload.get("outcome")
    expected_outcome = "error" if sample.get("error_code") is not None else "generation"
    if outcome != expected_outcome:
        raise ValueError(f"journal outcome/sample mismatch at line {position}")


def _sha256_payload(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _roster(sample: Mapping[str, object], expected_label: object) -> dict[str, object]:
    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("journal sample metadata must be a mapping")
    day = metadata.get("day")
    question_type = metadata.get("question_type")
    needs_audio = metadata.get("needs_audio")
    if (
        not isinstance(expected_label, str)
        or expected_label not in {"A", "B", "C", "D"}
        or not isinstance(day, int)
        or isinstance(day, bool)
        or not isinstance(question_type, str)
        or not question_type
        or not isinstance(needs_audio, bool)
    ):
        raise ValueError("journal sample lacks canonical EgoLife scoring roster fields")
    return {
        "question_id": sample["question_id"],
        "expected_label": expected_label,
        "day": day,
        "question_type": question_type,
        "needs_audio": needs_audio,
    }


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("journal write made no progress")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _display_identity(identity: Sequence[str]) -> str:
    return "/".join(identity)
