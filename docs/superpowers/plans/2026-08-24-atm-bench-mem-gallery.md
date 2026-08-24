# ATM-Bench and Mem-Gallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay ATM-Bench and Mem-Gallery through MindBridge's public SDK the way the other
nine official benchmarks are replayed, and prove both write paths run end to end on a subset.

**Architecture:** Each benchmark gets the established trio — a pydantic adapter over the
pinned release, a runner that ingests through `observe`/`remember` and answers through
`recall`, and a CLI that pins run identity into a manifest. No answer-quality metric is
computed anywhere; predictions go to the official scorers and `mindbridge-bench score`
records their verdict.

**Tech Stack:** Python 3.10–3.14, pydantic v2 contracts, `asyncio`, pytest, `uv`.

**Spec:** [docs/superpowers/specs/2026-08-24-atm-bench-mem-gallery-design.md](../specs/2026-08-24-atm-bench-mem-gallery-design.md)

## Global Constraints

- `benchmarks/` may only call the public SDK and contracts. No product module may import it,
  and it may not import `mindbridge.infrastructure` or `mindbridge.application`.
- Quality gates, all five, before every commit: `uv run ruff format --check .`,
  `uv run ruff check .`, `uv run mypy`, `uv run pytest -W error`, `git diff --check`.
- Markdown changes also pass the pinned Docker gates from `CONTRIBUTING.md`:
  `davidanson/markdownlint-cli2:v0.23.0` and `lycheeverse/lychee:0.23.0`. `ruff format` also
  formats Python inside Markdown fences.
- `RememberRequest.summary` is `NonEmptyString`, max 2048 characters. Anything longer must be
  chunked; measured maxima are in the tasks that need them.
- `ObserveRequest.media_objects` has `min_length=1`, `max_length=8`. A text-only unit is a
  `remember` write.
- `MediaObjectInput.media_object_id` is caller-assigned and returned on every `EvidenceView`.
  Both benchmarks set it to the official ID so evidence maps back with no side table.
- One clock: the media filename stem, read as UTC, is capture time for every ATM modality.
  Never the video record's `timestamp` field — it is true UTC and would sit an hour off the
  images for half the year.
- No new dependency and no new optional extra. Both releases are plain JSON.
- Adapter versions: `atm_bench_official_v1`, `mem_gallery_official_v1`. Runner versions:
  `atm_bench_production_api_v1`, `mem_gallery_production_api_v1`.
- Pinned corpora, already downloaded:
  `.benchmarks/atm-bench` at `Jingbiao/ATM-Bench` rev `78e826dc07e97466b2f54443831ef9a83ab8b27c`,
  `.benchmarks/mem-gallery` at `Ethan-Bei/Mem-Gallery` rev `af912daba984e896e253016b7c7e334ef92c2a6f`.

---

### Task 1: ATM-Bench adapter

**Files:**

- Create: `src/mindbridge/benchmarks/atm_bench.py`
- Modify: `tests/unit/benchmarks/test_dataset_adapters.py` (append cases)

**Interfaces:**

- Consumes: nothing from earlier tasks.
- Produces:
  - `ATM_BENCH_ADAPTER_VERSION: str = "atm_bench_official_v1"`
  - `AtmQuestionType = Literal["number", "list_recall", "open_end"]`
  - `class AtmBenchQuestion(ContractModel)`: `question_id: Identifier`,
    `question: NonEmptyString`, `reference_answer: NonEmptyString`,
    `qtype: AtmQuestionType`, `evidence_ids: tuple[Identifier, ...]` (min 1),
    `niah_evidence_ids: tuple[Identifier, ...] = ()`
  - `class AtmEmail(ContractModel)`: `email_id: Identifier`, `occurred_at: AwareDatetime`,
    `summary: NonEmptyString`, `body: str`
  - `class AtmSgmRecord(ContractModel)`: `media_id: Identifier`, `media_kind: MediaKind`,
    `occurred_at: AwareDatetime`, `location_name: str`, `city: str`,
    `short_caption: str`, `caption: str`, `ocr_text: str`,
    `tags: tuple[NonEmptyString, ...]`, `duration_seconds: float | None`,
    `size_bytes: int`
  - `def load_atm_bench(dataset_path: Path) -> tuple[AtmBenchQuestion, ...]`
  - `def load_atm_niah_pool(pool_path: Path) -> tuple[AtmBenchQuestion, ...]`
  - `def load_atm_emails(emails_path: Path) -> tuple[AtmEmail, ...]`
  - `def load_atm_sgm(batch_results_path: Path) -> tuple[AtmSgmRecord, ...]`
  - `def atm_evidence_kind(evidence_id: str) -> Literal["email", "media"]`
  - `def atm_capture_time(media_name: str) -> datetime`
  - `def atm_sgm_block(record: AtmSgmRecord) -> str`
  - `def atm_email_block(email: AtmEmail) -> str`
  - `def atm_memory_chunks(block: str, evidence_id: str) -> tuple[str, ...]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/benchmarks/test_dataset_adapters.py`. Add
`from mindbridge.benchmarks.atm_bench import (...)` to the imports at the top of the file,
alongside the existing adapter imports.

```python
def _atm_question(**overrides: object) -> dict[str, object]:
    question = {
        "id": "1defb7d5-aab4-4244-8b3c-971a36376b04",
        "question": "How much did I pay for accommodation for BMVC 2024?",
        "answer": "£799.74",
        "notes": "",
        "evidence_ids": ["email202411160004", "20250223_130249"],
        "qtype": "number",
    }
    return question | overrides


def test_atm_bench_adapter_reads_questions_and_classifies_evidence(tmp_path: Path) -> None:
    dataset_path = tmp_path / "atm-bench.json"
    dataset_path.write_text(json.dumps([_atm_question()]), encoding="utf-8")

    questions = load_atm_bench(dataset_path)

    assert len(questions) == 1
    assert questions[0].question_id == "1defb7d5-aab4-4244-8b3c-971a36376b04"
    assert questions[0].qtype == "number"
    assert questions[0].evidence_ids == ("email202411160004", "20250223_130249")
    assert questions[0].niah_evidence_ids == ()
    assert atm_evidence_kind("email202411160004") == "email"
    assert atm_evidence_kind("20250223_130249") == "media"


def test_atm_bench_adapter_refuses_duplicate_ids_and_missing_evidence(tmp_path: Path) -> None:
    duplicated = tmp_path / "duplicated.json"
    duplicated.write_text(json.dumps([_atm_question(), _atm_question()]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate question IDs"):
        load_atm_bench(duplicated)

    empty_evidence = tmp_path / "empty_evidence.json"
    empty_evidence.write_text(json.dumps([_atm_question(evidence_ids=[])]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_atm_bench(empty_evidence)

    empty_release = tmp_path / "empty.json"
    empty_release.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        load_atm_bench(empty_release)


def test_atm_niah_pool_requires_every_gold_evidence_in_the_pool(tmp_path: Path) -> None:
    complete = tmp_path / "niah25.json"
    complete.write_text(
        json.dumps(
            [
                _atm_question(
                    niah_evidence_ids=[
                        "email202411160004",
                        "20250223_130249",
                        "20220430_132212",
                    ]
                )
            ]
        ),
        encoding="utf-8",
    )
    pool = load_atm_niah_pool(complete)
    assert len(pool[0].niah_evidence_ids) == 3

    truncated = tmp_path / "broken.json"
    truncated.write_text(
        json.dumps([_atm_question(niah_evidence_ids=["20250223_130249"])]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pool must contain every gold evidence"):
        load_atm_niah_pool(truncated)


def test_atm_email_adapter_reads_naive_timestamps_as_utc(tmp_path: Path) -> None:
    emails_path = tmp_path / "emails.json"
    emails_path.write_text(
        json.dumps(
            [
                {
                    "id": "email202411160004",
                    "timestamp": "2024-11-16 09:12:00",
                    "short_summary": "Hotel confirmation",
                    "detail": "Total £799.74 for four nights.",
                }
            ]
        ),
        encoding="utf-8",
    )

    emails = load_atm_emails(emails_path)

    assert emails[0].email_id == "email202411160004"
    assert emails[0].occurred_at == datetime(2024, 11, 16, 9, 12, tzinfo=timezone.utc)
    assert emails[0].summary == "Hotel confirmation"


def test_atm_sgm_adapter_takes_capture_time_from_the_stem_not_the_timestamp(
    tmp_path: Path,
) -> None:
    batch_path = tmp_path / "video_batch_results.json"
    batch_path.write_text(
        json.dumps(
            [
                {
                    "video_path": "data/raw_memory/video/20220502_172850.mp4",
                    "timestamp": "2022-05-02 16:28:54+00:00",
                    "location_name": "Fellows' Garden, Cambridge, United Kingdom",
                    "city": "Cambridge, United Kingdom",
                    "short_caption": "A blackbird forages on a lawn.",
                    "caption": "A solitary blackbird moves across a green lawn.",
                    "ocr_text": "",
                    "tags": ["blackbird", "garden"],
                    "duration": 3.300756,
                    "file_size": 790569,
                    "entities": [{"entity": "bird", "type": "other"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_atm_sgm(batch_path)

    assert records[0].media_id == "20220502_172850"
    assert records[0].media_kind is MediaKind.VIDEO
    # The stem is local wall clock; the record's own timestamp is UTC and an hour earlier.
    assert records[0].occurred_at == datetime(2022, 5, 2, 17, 28, 50, tzinfo=timezone.utc)
    assert records[0].duration_seconds == pytest.approx(3.300756)
    assert records[0].size_bytes == 790569
    assert records[0].tags == ("blackbird", "garden")


def test_atm_sgm_block_reproduces_the_official_field_order(tmp_path: Path) -> None:
    batch_path = tmp_path / "image_batch_results.json"
    batch_path.write_text(
        json.dumps(
            [
                {
                    "image_path": "data/raw_memory/image/20220703_210745.jpg",
                    "timestamp": "2022-07-03 21:07:45",
                    "location_name": "West Quay Road, Southampton, United Kingdom",
                    "city": "Southampton, United Kingdom",
                    "short_caption": "A small airplane against a clear sky.",
                    "caption": "A solitary small aircraft streaks across a cloudless sky.",
                    "ocr_text": "There is no text visible in the image.",
                    "tags": ["airplane", "sky"],
                    "file_size": 100686,
                }
            ]
        ),
        encoding="utf-8",
    )

    block = atm_sgm_block(load_atm_sgm(batch_path)[0])

    assert block.splitlines() == [
        "ID: 20220703_210745",
        "Type: image",
        "Timestamp: 2022-07-03 21:07:45",
        "Location: West Quay Road, Southampton, United Kingdom",
        "Short Caption: A small airplane against a clear sky.",
        "Caption: A solitary small aircraft streaks across a cloudless sky.",
        "OCR: There is no text visible in the image.",
        "Tags: airplane, sky",
    ]


def test_atm_memory_chunks_keep_every_chunk_addressable_and_within_the_limit() -> None:
    short_block = "ID: 20220703_210745\nType: image\n"
    assert atm_memory_chunks(short_block, "20220703_210745") == (short_block,)

    long_block = "ID: 20220703_210745\n" + "x" * 9_000
    chunks = atm_memory_chunks(long_block, "20220703_210745")

    assert len(chunks) == 5
    assert all(len(chunk) <= 2_048 for chunk in chunks)
    assert all(chunk.startswith("ID: 20220703_210745\n") for chunk in chunks)
    assert "Part 1/5" in chunks[0]
```

Add the imports these cases need to the top of the file:

```python
from datetime import datetime, timezone

from mindbridge.benchmarks.atm_bench import (
    atm_capture_time,
    atm_email_block,
    atm_evidence_kind,
    atm_memory_chunks,
    atm_sgm_block,
    load_atm_bench,
    load_atm_emails,
    load_atm_niah_pool,
    load_atm_sgm,
)
from mindbridge.core import MediaKind
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/benchmarks/test_dataset_adapters.py -k atm -v`
Expected: collection error, `ModuleNotFoundError: No module named 'mindbridge.benchmarks.atm_bench'`

- [ ] **Step 3: Write the adapter**

Create `src/mindbridge/benchmarks/atm_bench.py`:

```python
"""Thin adapter for the official ATM-Bench release.

One archive, two QA splits, and two representations of the same media. The release carries
two clocks: an image's `timestamp` is local wall clock and agrees with its filename stem,
while a video's is true UTC and sits an hour off the stem for half the year. This adapter
takes the stem as capture time for every modality so one archive has one timeline.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from mindbridge.contracts import ContractModel, Identifier, NonEmptyString
from mindbridge.core import MediaKind

ATM_BENCH_ADAPTER_VERSION = "atm_bench_official_v1"

AtmQuestionType = Literal["number", "list_recall", "open_end"]
AtmEvidenceKind = Literal["email", "media"]

_STEM_PATTERN = re.compile(r"^(\d{8})_(\d{6})$")
_EMAIL_PREFIX = "email"
_MEMORY_CHARACTER_LIMIT = 2_048
_CHUNK_BODY_CHARACTERS = 1_800
_MEDIA_KIND_BY_FIELD = {"image_path": MediaKind.IMAGE, "video_path": MediaKind.VIDEO}


class AtmBenchQuestion(ContractModel):
    """One official question and the evidence a correct answer must rest on."""

    question_id: Identifier
    question: NonEmptyString
    reference_answer: NonEmptyString
    qtype: AtmQuestionType
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1)
    niah_evidence_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def require_consistent_evidence(self) -> AtmBenchQuestion:
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("ATM-Bench evidence IDs must not repeat")
        if len(set(self.niah_evidence_ids)) != len(self.niah_evidence_ids):
            raise ValueError("ATM-Bench NIAH evidence IDs must not repeat")
        return self


class AtmEmail(ContractModel):
    """One email in the archive, cited by 430 of the release's evidence references."""

    email_id: Identifier
    occurred_at: AwareDatetime
    summary: NonEmptyString
    body: str


class AtmSgmRecord(ContractModel):
    """One official schema-guided memory record for an image or a video."""

    media_id: Identifier
    media_kind: MediaKind
    occurred_at: AwareDatetime
    raw_timestamp: str
    location_name: str
    city: str
    short_caption: str
    caption: str
    ocr_text: str
    tags: tuple[NonEmptyString, ...] = ()
    duration_seconds: float | None = None
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def require_duration_for_video_only(self) -> AtmSgmRecord:
        if self.media_kind is MediaKind.VIDEO and self.duration_seconds is None:
            raise ValueError("ATM-Bench video records must carry a duration")
        if self.media_kind is MediaKind.IMAGE and self.duration_seconds is not None:
            raise ValueError("ATM-Bench image records must not carry a duration")
        return self


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    answer: str
    qtype: str
    evidence_ids: list[str] = Field(min_length=1)
    niah_evidence_ids: list[str] = Field(default_factory=list)


class _RawEmail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    timestamp: str
    short_summary: str
    detail: str


class _RawSgmRecord(BaseModel):
    # Not `extra="ignore"`: the media path arrives under `image_path` or `video_path`, and the
    # kind is read from which one is present.
    model_config = ConfigDict(extra="allow")

    timestamp: str
    location_name: str = ""
    city: str = ""
    short_caption: str = ""
    caption: str = ""
    ocr_text: str = ""
    tags: list[str] = Field(default_factory=list)
    duration: float | None = None
    file_size: int = 0


def atm_evidence_kind(evidence_id: str) -> AtmEvidenceKind:
    """Say which store an evidence ID belongs to, once, so nothing can disagree later."""
    if evidence_id.startswith(_EMAIL_PREFIX):
        return "email"
    if _STEM_PATTERN.match(evidence_id):
        return "media"
    raise ValueError(f"unrecognised ATM-Bench evidence ID: {evidence_id}")


def atm_capture_time(media_name: str) -> datetime:
    """Read capture time out of a `YYYYMMDD_HHMMSS` stem, as UTC.

    The stem is the camera's local wall clock at the place of capture. Reading it as UTC
    keeps images, videos, and email on one timeline; see the design note on the two clocks.
    """
    stem = Path(media_name).stem
    matched = _STEM_PATTERN.match(stem)
    if matched is None:
        raise ValueError(f"ATM-Bench media name carries no capture time: {media_name}")
    return datetime.strptime(matched.group(1) + matched.group(2), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


def load_atm_bench(dataset_path: Path) -> tuple[AtmBenchQuestion, ...]:
    """Load `atm-bench.json` or `atm-bench-hard.json`."""
    return _questions(dataset_path, require_pool=False)


def load_atm_niah_pool(pool_path: Path) -> tuple[AtmBenchQuestion, ...]:
    """Load one NIAH pool, refusing a pool that has lost a gold evidence item."""
    return _questions(pool_path, require_pool=True)


def load_atm_emails(emails_path: Path) -> tuple[AtmEmail, ...]:
    """Load the archive's emails, naive timestamps read as UTC."""
    raw = json.loads(emails_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("ATM-Bench emails must be a non-empty list")
    emails = tuple(_email(_RawEmail.model_validate(item)) for item in raw)
    if len({email.email_id for email in emails}) != len(emails):
        raise ValueError("ATM-Bench emails contain duplicate IDs")
    return emails


def load_atm_sgm(batch_results_path: Path) -> tuple[AtmSgmRecord, ...]:
    """Load one official `batch_results.json`, keyed by media stem."""
    raw = json.loads(batch_results_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("results", raw)
    if not isinstance(raw, list) or not raw:
        raise ValueError("ATM-Bench batch results must be a non-empty list")
    records = tuple(_sgm_record(item) for item in raw)
    if len({record.media_id for record in records}) != len(records):
        raise ValueError("ATM-Bench batch results contain duplicate media IDs")
    return records


def atm_sgm_block(record: AtmSgmRecord) -> str:
    """Serialize one record exactly as the official Oracle baseline serializes it."""
    return "\n".join(
        (
            f"ID: {record.media_id}",
            f"Type: {record.media_kind.value}",
            f"Timestamp: {record.raw_timestamp}",
            f"Location: {record.location_name}",
            f"Short Caption: {record.short_caption}",
            f"Caption: {record.caption}",
            f"OCR: {record.ocr_text}",
            f"Tags: {', '.join(record.tags)}",
        )
    )


def atm_email_block(email: AtmEmail) -> str:
    """Serialize one email exactly as the official Oracle baseline serializes it."""
    return "\n".join(
        (
            f"ID: {email.email_id}",
            f"Timestamp: {email.occurred_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Summary: {email.summary}",
            f"Detail: {email.body}",
        )
    )


def atm_memory_chunks(block: str, evidence_id: str) -> tuple[str, ...]:
    """Split one serialized block into writes that fit `RememberRequest.summary`.

    1,063 of the 4,292 SGM blocks exceed the 2,048-character limit, the longest at 9,212.
    Every chunk repeats the ID line, so retrieving any one of them still names its evidence.
    """
    if len(block) <= _MEMORY_CHARACTER_LIMIT:
        return (block,)
    head = f"ID: {evidence_id}\n"
    body = block[len(head) :] if block.startswith(head) else block
    slices = tuple(
        body[offset : offset + _CHUNK_BODY_CHARACTERS]
        for offset in range(0, len(body), _CHUNK_BODY_CHARACTERS)
    )
    return tuple(
        f"{head}Part {index}/{len(slices)}\n{piece}" for index, piece in enumerate(slices, start=1)
    )


def _questions(path: Path, *, require_pool: bool) -> tuple[AtmBenchQuestion, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("qas", raw)
    if not isinstance(raw, list) or not raw:
        raise ValueError("ATM-Bench annotations must not be empty")
    questions = tuple(_question(_RawQuestion.model_validate(item)) for item in raw)
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("ATM-Bench annotations contain duplicate question IDs")
    for question in questions:
        for evidence_id in question.evidence_ids:
            atm_evidence_kind(evidence_id)
        if not require_pool:
            continue
        if not question.niah_evidence_ids:
            raise ValueError("ATM-Bench NIAH pools must carry niah_evidence_ids")
        if not set(question.evidence_ids) <= set(question.niah_evidence_ids):
            raise ValueError(
                f"ATM-Bench NIAH pool must contain every gold evidence for {question.question_id}"
            )
    return questions


def _question(raw: _RawQuestion) -> AtmBenchQuestion:
    return AtmBenchQuestion(
        question_id=raw.id,
        question=raw.question,
        reference_answer=raw.answer,
        qtype=_question_type(raw.qtype),
        evidence_ids=tuple(raw.evidence_ids),
        niah_evidence_ids=tuple(raw.niah_evidence_ids),
    )


def _question_type(value: str) -> AtmQuestionType:
    if value not in {"number", "list_recall", "open_end"}:
        raise ValueError(f"unknown ATM-Bench question type: {value}")
    return value  # type: ignore[return-value]


def _email(raw: _RawEmail) -> AtmEmail:
    return AtmEmail(
        email_id=raw.id,
        occurred_at=_naive_utc(raw.timestamp),
        summary=raw.short_summary,
        body=raw.detail,
    )


def _sgm_record(item: object) -> AtmSgmRecord:
    if not isinstance(item, dict):
        raise ValueError("ATM-Bench batch results entries must be objects")
    field = next((name for name in _MEDIA_KIND_BY_FIELD if name in item), None)
    if field is None:
        raise ValueError("ATM-Bench batch results entry names no media path")
    kind = _MEDIA_KIND_BY_FIELD[field]
    raw = _RawSgmRecord.model_validate(item)
    return AtmSgmRecord(
        media_id=Path(str(item[field])).stem,
        media_kind=kind,
        occurred_at=atm_capture_time(str(item[field])),
        raw_timestamp=raw.timestamp,
        location_name=raw.location_name,
        city=raw.city,
        short_caption=raw.short_caption,
        caption=raw.caption,
        ocr_text=raw.ocr_text,
        tags=tuple(tag for tag in raw.tags if tag.strip()),
        duration_seconds=raw.duration if kind is MediaKind.VIDEO else None,
        size_bytes=raw.file_size,
    )


def _naive_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
```

Note on `entities`: the release carries it, and this adapter drops it. The official Oracle
serialization's field set is `type, timestamp, location, short_caption, caption, ocr, tags` —
`entities` is not part of the SGM text at all — and all 168,881 image-side entity items are
double-encoded garbage (`{'entity': '{"entity"', 'type': '"airplane", "type": "other"}'}`)
while only the 970 video-side items are clean. Dropping it is fidelity, not a workaround.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/benchmarks/test_dataset_adapters.py -k atm -v`
Expected: 6 passed

- [ ] **Step 5: Parse the real release**

Run:

```bash
uv run python -c "
from pathlib import Path
from mindbridge.benchmarks.atm_bench import load_atm_bench, load_atm_niah_pool, load_atm_emails, load_atm_sgm
root = Path('/home/yons/thomas/MindBridge/.benchmarks/atm-bench/data')
main = load_atm_bench(root / 'atm-bench' / 'atm-bench.json')
hard = load_atm_bench(root / 'atm-bench' / 'atm-bench-hard.json')
pool = load_atm_niah_pool(root / 'atm-bench' / 'niah' / 'atm-bench-hard-niah25.json')
emails = load_atm_emails(root / 'raw_memory' / 'email' / 'emails.json')
images = load_atm_sgm(root / 'processed_memory' / 'image_batch_results.json')
videos = load_atm_sgm(root / 'processed_memory' / 'video_batch_results.json')
assert (len(main), len(hard), len(pool)) == (1013, 31, 31), (len(main), len(hard), len(pool))
assert (len(emails), len(images), len(videos)) == (6742, 3759, 533)
print('parsed', len(main), len(hard), len(pool), len(emails), len(images), len(videos))
"
```

Expected: `parsed 1013 31 31 6742 3759 533`

- [ ] **Step 6: Mutation-check the two load-time refusals**

Temporarily delete the duplicate-ID check in `_questions`, run
`uv run pytest tests/unit/benchmarks/test_dataset_adapters.py -k atm -v`, confirm
`test_atm_bench_adapter_refuses_duplicate_ids_and_missing_evidence` fails, then restore it.
Repeat for the NIAH superset check and
`test_atm_niah_pool_requires_every_gold_evidence_in_the_pool`.

- [ ] **Step 7: Run the gates and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest -W error && git diff --check
git add src/mindbridge/benchmarks/atm_bench.py tests/unit/benchmarks/test_dataset_adapters.py
git commit -m "Add ATM-Bench dataset adapter"
```

---

### Task 2: Mem-Gallery adapter

**Files:**

- Create: `src/mindbridge/benchmarks/mem_gallery.py`
- Modify: `tests/unit/benchmarks/test_dataset_adapters.py` (append cases)

**Interfaces:**

- Consumes: nothing from Task 1.
- Produces:
  - `MEM_GALLERY_ADAPTER_VERSION: str = "mem_gallery_official_v1"`
  - `MemGalleryPoint = Literal["FR", "MR", "TR", "VR", "TTL", "VS", "CD", "KR", "AR"]`
  - `class MemGalleryProfile(ContractModel)`: `name: NonEmptyString`,
    `persona_summary: NonEmptyString`, `traits: tuple[NonEmptyString, ...]`,
    `conversation_style: NonEmptyString`
  - `class MemGalleryRound(ContractModel)`: `round_id: Identifier`, `user: NonEmptyString`,
    `assistant: NonEmptyString`, `image_id: Identifier | None`,
    `image_path: NonEmptyString | None`, `image_caption: NonEmptyString | None`
  - `class MemGallerySession(ContractModel)`: `session_id: Identifier`,
    `occurred_at: AwareDatetime`, `rounds: tuple[MemGalleryRound, ...]`
  - `class MemGalleryQuestion(ContractModel)`: `question_id: Identifier`,
    `point: MemGalleryPoint`, `question: NonEmptyString`,
    `reference_answer: NonEmptyString`, `session_ids: tuple[Identifier, ...]`,
    `clue_round_ids: tuple[Identifier, ...]`,
    `question_image_path: NonEmptyString | None`,
    `question_image_caption: NonEmptyString | None`
  - `class MemGalleryTopic(ContractModel)`: `topic: Identifier`,
    `profile: MemGalleryProfile`, `sessions: tuple[MemGallerySession, ...]`,
    `questions: tuple[MemGalleryQuestion, ...]`
  - `def load_mem_gallery_topic(topic_path: Path) -> MemGalleryTopic`
  - `def load_mem_gallery(dialog_directory: Path) -> tuple[MemGalleryTopic, ...]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/benchmarks/test_dataset_adapters.py`:

```python
def _mem_gallery_topic_payload() -> dict[str, object]:
    return {
        "character_profile": {
            "name": "Maya",
            "persona_summary": "A part-time librarian who took up baking.",
            "traits": ["curious", "earnest"],
            "conversation_style": "Inquisitive and earnest.",
        },
        "multi_session_dialogues": [
            {
                "session_id": "D1",
                "date": "2024-06-24",
                "dialogues": [
                    {
                        "round": "D1:1",
                        "user": "Can you tell me the basics of handmade baking?",
                        "assistant": "Start with an oven of 30 litres or more.",
                    },
                    {
                        "round": "D1:2",
                        "user": "What is in this picture?",
                        "assistant": "A tray of shortbread.",
                        "image_id": ["D1:IMG_001"],
                        "input_image": ["../image/Baking/D1_IMG_001.jpg"],
                        "image_caption": ["A tray of pale shortbread fingers."],
                    },
                ],
            }
        ],
        "human-annotated QAs": [
            {
                "point": "FR",
                "question": "What oven size was recommended?",
                "answer": "30 litres or more.",
                "session_id": ["D1"],
                "clue": ["D1:1"],
            },
            {
                "point": "TTL",
                "question": "What species of plant is shown in the picture?",
                "question_image": "../image/Baking/QA_IMG_001.jpg",
                "answer": "Foxglove",
                "session_id": ["D1"],
                "clue": ["D1:2"],
                "image_caption": "Cluster of purple bell-shaped flowers.",
            },
        ],
    }


def test_mem_gallery_adapter_reads_sessions_rounds_and_question_images(tmp_path: Path) -> None:
    topic_path = tmp_path / "Baking_Dessert_Daily_Life_Skill.json"
    topic_path.write_text(json.dumps(_mem_gallery_topic_payload()), encoding="utf-8")

    topic = load_mem_gallery_topic(topic_path)

    assert topic.topic == "Baking_Dessert_Daily_Life_Skill"
    assert topic.profile.name == "Maya"
    assert topic.sessions[0].session_id == "D1"
    assert topic.sessions[0].occurred_at == datetime(2024, 6, 24, tzinfo=timezone.utc)
    assert topic.sessions[0].rounds[0].image_id is None
    assert topic.sessions[0].rounds[1].image_id == "D1:IMG_001"
    assert topic.sessions[0].rounds[1].image_path == "../image/Baking/D1_IMG_001.jpg"
    assert topic.questions[0].question_id == "Baking_Dessert_Daily_Life_Skill:1"
    assert topic.questions[0].point == "FR"
    assert topic.questions[0].clue_round_ids == ("D1:1",)
    assert topic.questions[1].question_image_path == "../image/Baking/QA_IMG_001.jpg"
    assert topic.questions[1].question_image_caption == "Cluster of purple bell-shaped flowers."


def test_mem_gallery_adapter_refuses_unknown_points_and_dangling_clues(tmp_path: Path) -> None:
    unknown_point = _mem_gallery_topic_payload()
    qas = unknown_point["human-annotated QAs"]
    assert isinstance(qas, list)
    qas[0]["point"] = "ZZ"
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(json.dumps(unknown_point), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown Mem-Gallery point"):
        load_mem_gallery_topic(unknown_path)

    dangling = _mem_gallery_topic_payload()
    dangling_qas = dangling["human-annotated QAs"]
    assert isinstance(dangling_qas, list)
    dangling_qas[0]["clue"] = ["D9:7"]
    dangling_path = tmp_path / "dangling.json"
    dangling_path.write_text(json.dumps(dangling), encoding="utf-8")
    with pytest.raises(ValueError, match="clue names an unknown round"):
        load_mem_gallery_topic(dangling_path)


def test_mem_gallery_adapter_refuses_a_round_carrying_more_than_one_image(
    tmp_path: Path,
) -> None:
    payload = _mem_gallery_topic_payload()
    sessions = payload["multi_session_dialogues"]
    assert isinstance(sessions, list)
    sessions[0]["dialogues"][1]["input_image"] = ["a.jpg", "b.jpg"]
    sessions[0]["dialogues"][1]["image_id"] = ["D1:IMG_001", "D1:IMG_002"]
    path = tmp_path / "two_images.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one image"):
        load_mem_gallery_topic(path)


def test_mem_gallery_directory_loader_keeps_sorted_topic_order(tmp_path: Path) -> None:
    directory = tmp_path / "dialog"
    directory.mkdir()
    for name in ("Zebra_Topic", "Apple_Topic"):
        (directory / f"{name}.json").write_text(
            json.dumps(_mem_gallery_topic_payload()), encoding="utf-8"
        )

    topics = load_mem_gallery(directory)

    assert tuple(topic.topic for topic in topics) == ("Apple_Topic", "Zebra_Topic")
```

Add to the imports at the top of the file:

```python
from mindbridge.benchmarks.mem_gallery import load_mem_gallery, load_mem_gallery_topic
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/benchmarks/test_dataset_adapters.py -k mem_gallery -v`
Expected: collection error, `ModuleNotFoundError: No module named 'mindbridge.benchmarks.mem_gallery'`

- [ ] **Step 3: Write the adapter**

Create `src/mindbridge/benchmarks/mem_gallery.py`:

```python
"""Thin adapter for the official Mem-Gallery release.

Twenty topic files, each one persona's multi-session dialogue plus the questions annotated
over it. Question IDs are not in the release, so they are derived from release order and
pinned, the way the MM-Lifelong adapter pins question indices.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from mindbridge.contracts import ContractModel, Identifier, NonEmptyString

MEM_GALLERY_ADAPTER_VERSION = "mem_gallery_official_v1"

MemGalleryPoint = Literal["FR", "MR", "TR", "VR", "TTL", "VS", "CD", "KR", "AR"]
_POINTS: frozenset[str] = frozenset(("FR", "MR", "TR", "VR", "TTL", "VS", "CD", "KR", "AR"))
_QA_KEY = "human-annotated QAs"


class MemGalleryProfile(ContractModel):
    """The persona the dialogue belongs to, as the release describes it."""

    name: NonEmptyString
    persona_summary: NonEmptyString
    traits: tuple[NonEmptyString, ...] = ()
    conversation_style: NonEmptyString


class MemGalleryRound(ContractModel):
    """One user turn and its assistant reply, with the round's image when it has one."""

    round_id: Identifier
    user: NonEmptyString
    assistant: NonEmptyString
    image_id: Identifier | None = None
    image_path: NonEmptyString | None = None
    image_caption: NonEmptyString | None = None

    @model_validator(mode="after")
    def require_complete_image(self) -> MemGalleryRound:
        present = (self.image_id is not None, self.image_path is not None)
        if any(present) and not all(present):
            raise ValueError("Mem-Gallery round images need both an ID and a path")
        return self


class MemGallerySession(ContractModel):
    """One dated session, rounds in release order."""

    session_id: Identifier
    occurred_at: AwareDatetime
    rounds: tuple[MemGalleryRound, ...] = Field(min_length=1)


class MemGalleryQuestion(ContractModel):
    """One annotated question, its task type, and the rounds that answer it."""

    question_id: Identifier
    point: MemGalleryPoint
    question: NonEmptyString
    reference_answer: NonEmptyString
    session_ids: tuple[Identifier, ...] = Field(min_length=1)
    clue_round_ids: tuple[Identifier, ...] = ()
    question_image_path: NonEmptyString | None = None
    question_image_caption: NonEmptyString | None = None


class MemGalleryTopic(ContractModel):
    """One topic file: a persona, its sessions, and the questions over them."""

    topic: Identifier
    profile: MemGalleryProfile
    sessions: tuple[MemGallerySession, ...] = Field(min_length=1)
    questions: tuple[MemGalleryQuestion, ...] = Field(min_length=1)


class _RawProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    persona_summary: str
    traits: list[str] = Field(default_factory=list)
    conversation_style: str


class _RawRound(BaseModel):
    model_config = ConfigDict(extra="ignore")

    round: str
    user: str
    assistant: str
    image_id: list[str] = Field(default_factory=list)
    input_image: list[str] = Field(default_factory=list)
    image_caption: list[str] = Field(default_factory=list)


class _RawSession(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str
    date: str
    dialogues: list[_RawRound] = Field(min_length=1)


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    point: str
    question: str
    answer: str
    session_id: list[str] = Field(min_length=1)
    clue: list[str] = Field(default_factory=list)
    question_image: str | None = None
    image_caption: str | None = None


def load_mem_gallery_topic(topic_path: Path) -> MemGalleryTopic:
    """Load one official topic file, keyed by its filename."""
    payload = json.loads(topic_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mem-Gallery topic files must be objects")
    sessions = tuple(
        _session(_RawSession.model_validate(item))
        for item in payload.get("multi_session_dialogues", ())
    )
    if not sessions:
        raise ValueError("Mem-Gallery topics must carry at least one session")
    known_rounds = {round_.round_id for session in sessions for round_ in session.rounds}
    known_sessions = {session.session_id for session in sessions}
    topic = topic_path.stem
    questions = tuple(
        _question(_RawQuestion.model_validate(item), topic, index)
        for index, item in enumerate(payload.get(_QA_KEY, ()), start=1)
    )
    for question in questions:
        dangling_rounds = set(question.clue_round_ids) - known_rounds
        if dangling_rounds:
            raise ValueError(
                f"Mem-Gallery clue names an unknown round: {', '.join(sorted(dangling_rounds))}"
            )
        dangling_sessions = set(question.session_ids) - known_sessions
        if dangling_sessions:
            raise ValueError(
                "Mem-Gallery question names an unknown session: "
                f"{', '.join(sorted(dangling_sessions))}"
            )
    return MemGalleryTopic(
        topic=topic,
        profile=_profile(_RawProfile.model_validate(payload["character_profile"])),
        sessions=sessions,
        questions=questions,
    )


def load_mem_gallery(dialog_directory: Path) -> tuple[MemGalleryTopic, ...]:
    """Load every topic file in `data/dialog`, in sorted filename order."""
    paths = sorted(dialog_directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no Mem-Gallery topic files under {dialog_directory}")
    return tuple(load_mem_gallery_topic(path) for path in paths)


def _profile(raw: _RawProfile) -> MemGalleryProfile:
    return MemGalleryProfile(
        name=raw.name,
        persona_summary=raw.persona_summary,
        traits=tuple(trait for trait in raw.traits if trait.strip()),
        conversation_style=raw.conversation_style,
    )


def _session(raw: _RawSession) -> MemGallerySession:
    return MemGallerySession(
        session_id=raw.session_id,
        occurred_at=datetime.strptime(raw.date, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        rounds=tuple(_round(item) for item in raw.dialogues),
    )


def _round(raw: _RawRound) -> MemGalleryRound:
    images = tuple(raw.input_image)
    identifiers = tuple(raw.image_id)
    captions = tuple(raw.image_caption)
    if len(images) > 1 or len(identifiers) > 1:
        raise ValueError(f"Mem-Gallery round {raw.round} must carry exactly one image")
    return MemGalleryRound(
        round_id=raw.round,
        user=raw.user,
        assistant=raw.assistant,
        image_id=identifiers[0] if identifiers else None,
        image_path=images[0] if images else None,
        image_caption=captions[0] if captions else None,
    )


def _question(raw: _RawQuestion, topic: str, index: int) -> MemGalleryQuestion:
    if raw.point not in _POINTS:
        raise ValueError(f"unknown Mem-Gallery point: {raw.point}")
    return MemGalleryQuestion(
        question_id=f"{topic}:{index}",
        point=raw.point,  # type: ignore[arg-type]
        question=raw.question,
        reference_answer=raw.answer,
        session_ids=tuple(raw.session_id),
        clue_round_ids=tuple(raw.clue),
        question_image_path=raw.question_image,
        question_image_caption=raw.image_caption,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/benchmarks/test_dataset_adapters.py -k mem_gallery -v`
Expected: 4 passed

- [ ] **Step 5: Parse the real release and check it against the paper**

Run:

```bash
uv run python -c "
from collections import Counter
from pathlib import Path
from mindbridge.benchmarks.mem_gallery import load_mem_gallery
topics = load_mem_gallery(Path('/home/yons/thomas/MindBridge/.benchmarks/mem-gallery/data/dialog'))
sessions = sum(len(t.sessions) for t in topics)
rounds = sum(len(s.rounds) for t in topics for s in t.sessions)
images = sum(1 for t in topics for s in t.sessions for r in s.rounds if r.image_id)
questions = [q for t in topics for q in t.questions]
with_image = sum(1 for q in questions if q.question_image_path)
points = Counter(q.point for q in questions)
assert (len(topics), sessions, rounds, images) == (20, 240, 3962, 1003)
assert (len(questions), with_image) == (1711, 487)
assert points == Counter({'TTL': 337, 'VS': 306, 'FR': 219, 'MR': 206, 'AR': 184, 'VR': 174, 'TR': 123, 'CD': 81, 'KR': 81})
print('parsed', len(topics), sessions, rounds, images, len(questions), with_image)
"
```

Expected: `parsed 20 240 3962 1003 1711 487` — these are the paper's own published counts.

- [ ] **Step 6: Mutation-check the point and clue refusals**

Temporarily accept any `point` in `_question`, run
`uv run pytest tests/unit/benchmarks/test_dataset_adapters.py -k mem_gallery -v`, confirm
`test_mem_gallery_adapter_refuses_unknown_points_and_dangling_clues` fails, restore. Repeat
for the dangling-clue check and the same test.

- [ ] **Step 7: Run the gates and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest -W error && git diff --check
git add src/mindbridge/benchmarks/mem_gallery.py tests/unit/benchmarks/test_dataset_adapters.py
git commit -m "Add Mem-Gallery dataset adapter"
```

---

### Task 3: Register both releases in the dataset smoke

**Files:**

- Modify: `src/mindbridge/benchmarks/dataset_smoke.py`
- Modify: `docs/benchmarking.md:249-305` (the "Benchmark dataset smoke" section)

**Interfaces:**

- Consumes: `load_atm_bench`, `ATM_BENCH_ADAPTER_VERSION` (Task 1); `load_mem_gallery`,
  `MEM_GALLERY_ADAPTER_VERSION` (Task 2).
- Produces: `run_dataset_adapter_smoke(..., atm_path: Path, atm_hard_path: Path,
  mem_gallery_dialog_path: Path)` — three new keyword parameters, and
  `DatasetAdapterSmokeResult.datasets` with `min_length=16`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/benchmarks/test_manifests.py`:

```python
def test_dataset_smoke_result_requires_all_sixteen_benchmark_summaries() -> None:
    """A benchmark dropped out of the smoke must fail the contract, not pass quietly."""
    from mindbridge.benchmarks.dataset_smoke import DatasetAdapterSmokeResult

    assert DatasetAdapterSmokeResult.model_fields["datasets"].metadata[0].min_length == 16
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/benchmarks/test_manifests.py -k sixteen -v`
Expected: FAIL, `assert 13 == 16`

- [ ] **Step 3: Extend the smoke**

In `src/mindbridge/benchmarks/dataset_smoke.py`:

**Edit 1** — add the imports beside the existing adapter imports:

```python
from mindbridge.benchmarks.atm_bench import ATM_BENCH_ADAPTER_VERSION, load_atm_bench
from mindbridge.benchmarks.mem_gallery import (
    MEM_GALLERY_ADAPTER_VERSION,
    MemGalleryTopic,
    load_mem_gallery,
)
```

**Edit 2** — change `datasets: tuple[BenchmarkDatasetSummary, ...] = Field(min_length=13)` to
   `min_length=16`.

**Edit 3** — add three parameters to `run_dataset_adapter_smoke`, after `supermemory_path`:

```python
    atm_path: Path,
    atm_hard_path: Path,
    mem_gallery_dialog_path: Path,
```

**Edit 4** — load them beside the existing loads:

```python
    atm = load_atm_bench(atm_path)
    atm_hard = load_atm_bench(atm_hard_path)
    mem_gallery = load_mem_gallery(mem_gallery_dialog_path)
```

**Edit 5** — append three summaries to the `datasets` tuple, after the SuperMemory entry:

```python
(
    BenchmarkDatasetSummary(
        benchmark="ATM-Bench",
        source_repository="Jingbiao/ATM-Bench",
        source_file=atm_path.name,
        source_sha256=sha256_file(atm_path),
        adapter_version=ATM_BENCH_ADAPTER_VERSION,
        context_count=1,
        memory_item_count=len({evidence for question in atm for evidence in question.evidence_ids}),
        question_count=len(atm),
    ),
)
(
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
)
(_mem_gallery_summary(mem_gallery_dialog_path, mem_gallery),)
```

**Edit 6** — add the summary helper beside `_m3_summary`:

```python
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
```

**Edit 7** — add `import hashlib` to the imports. `mindbridge.file_integrity` exports only
   `sha256_file`, and `benchmarks/` must not grow the product's API to gain a one-line
   helper, so the directory digest hashes the concatenated per-file digests inline.

**Edit 8** — add the three CLI arguments beside `--supermemory`:

```python
    parser.add_argument("--atm", type=Path, required=True, help="official ATM-Bench release to parse")
    parser.add_argument(
        "--atm-hard", type=Path, required=True, help="official ATM-Bench-Hard release to parse"
    )
    parser.add_argument(
        "--mem-gallery-dialog",
        type=Path,
        required=True,
        help="official Mem-Gallery data/dialog directory to parse",
    )
```

**Edit 9** — pass them through in `main`:

```python
atm_path = (arguments.atm,)
atm_hard_path = (arguments.atm_hard,)
mem_gallery_dialog_path = (arguments.mem_gallery_dialog,)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/benchmarks/test_manifests.py -k sixteen -v`
Expected: PASS

- [ ] **Step 5: Run the smoke against the real releases**

Run:

```bash
BENCH=/home/yons/thomas/MindBridge/.benchmarks
uv run --extra benchmarks mindbridge-bench datasets \
  --locomo-refined $BENCH/locomo-refined/data/raw/locomo_refined.json \
  --m3-robot $BENCH/m3-agent/data/annotations/robot.json \
  --m3-web $BENCH/m3-agent/data/annotations/web.json \
  --video-mme $BENCH/video-mme/videomme/test-00000-of-00001.parquet \
  --egolife $BENCH/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json \
  --egotempo $BENCH/egotempo/egotempo_openQA.json \
  --egomem $BENCH/egomem-reason/annotations_public.jsonl \
  --memlens $BENCH/memlens/dataset_32k.json \
  --mm-day $BENCH/mm-lifelong/day/test.json \
  --mm-week $BENCH/mm-lifelong/week/test.json \
  --mm-month-train $BENCH/mm-lifelong/month/train.json \
  --mm-month-val $BENCH/mm-lifelong/month/val.json \
  --supermemory $BENCH/supermemory-vqa/data/json/all_qa.json \
  --atm $BENCH/atm-bench/data/atm-bench/atm-bench.json \
  --atm-hard $BENCH/atm-bench/data/atm-bench/atm-bench-hard.json \
  --mem-gallery-dialog $BENCH/mem-gallery/data/dialog
```

Expected: JSON with `"passed": true` and 16 entries; the ATM-Bench entry reports
`"question_count": 1013`, ATM-Bench-Hard `31`, Mem-Gallery `1711` with
`"context_count": 20` and `"memory_item_count": 3962`.

- [ ] **Step 6: Update the smoke section of the benchmarking doc**

In `docs/benchmarking.md`, extend the "Benchmark dataset smoke" section: add ATM-Bench and
Mem-Gallery to the sentence naming the adapters, add the two download commands, and add the
three new flags to the `mindbridge-bench datasets` example.

```bash
uvx --from huggingface-hub hf download Jingbiao/ATM-Bench \
  --repo-type dataset \
  --revision 78e826dc07e97466b2f54443831ef9a83ab8b27c \
  --local-dir .benchmarks/atm-bench
uvx --from huggingface-hub hf download Ethan-Bei/Mem-Gallery \
  --repo-type dataset \
  --revision af912daba984e896e253016b7c7e334ef92c2a6f \
  --local-dir .benchmarks/mem-gallery
```

Both are pinned by revision because the digests in this smoke are only meaningful against a
fixed revision. ATM-Bench is 3.2 GB including the raw media; Mem-Gallery is 530 MB.

- [ ] **Step 7: Run the doc gates, the code gates, and commit**

```bash
docker run --rm -v "$PWD:/workdir:ro" davidanson/markdownlint-cli2:v0.23.0 \
  "**/*.md" "!.git/**" "!.venv/**" "!.pytest_cache/**" "!.benchmarks/**"
docker run --rm -v "$PWD:/input:ro" -w /input lycheeverse/lychee:0.23.0 \
  --no-progress --root-dir /input './*.md' './docs/**/*.md'
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest -W error && git diff --check
git add src/mindbridge/benchmarks/dataset_smoke.py tests/unit/benchmarks/test_manifests.py docs/benchmarking.md
git commit -m "Cover ATM-Bench and Mem-Gallery in the dataset smoke"
```

---

### Task 4: Official query wordings

**Files:**

- Modify: `src/mindbridge/benchmarks/prompts.py`
- Modify: `tests/contracts/test_prompt_catalog.py`

**Interfaces:**

- Consumes: nothing.
- Produces: `ATM_BENCH_QUERY_PROMPT`, `MEM_GALLERY_QUERY_PROMPT`,
  `MEM_GALLERY_REFUSAL_PROMPT`, `MEM_GALLERY_CONFLICT_PROMPT`,
  `MEM_GALLERY_SEARCH_PROMPT`, all `PromptSpec`, all appended to `BENCHMARK_PROMPTS`.
  `def mem_gallery_format_constraint(point: str) -> str` — returns the constraint already
  prefixed with `\n\n`, or empty text where the task type has none.

- [ ] **Step 1: Write the failing test**

In `tests/contracts/test_prompt_catalog.py`, add the five new fingerprints to
`_EXPECTED_BENCHMARK_FINGERPRINTS` with a placeholder digest of 64 zeros each:

```python
    "atm_bench_query_v1": "0" * 64,
    "mem_gallery_query_v1": "0" * 64,
    "mem_gallery_refusal_v1": "0" * 64,
    "mem_gallery_conflict_v1": "0" * 64,
    "mem_gallery_search_v1": "0" * 64,
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/contracts/test_prompt_catalog.py -v`
Expected: FAIL — the catalog has no prompt with those names

- [ ] **Step 3: Add the wordings**

In `src/mindbridge/benchmarks/prompts.py`, append before `BENCHMARK_PROMPTS`:

```python
ATM_BENCH_QUERY_PROMPT = PromptSpec(
    name="atm_bench_query",
    version="atm_bench_query_v1",
    purpose="Apply the official ATM-Bench per-type answer instruction.",
    used_by="mindbridge.benchmarks.atm_bench_runner._question_query",
    text="{question}\n{format_constraint}",
)

MEM_GALLERY_QUERY_PROMPT = PromptSpec(
    name="mem_gallery_query",
    version="mem_gallery_query_v1",
    purpose="Apply the official Mem-Gallery concise-answer instruction.",
    used_by="mindbridge.benchmarks.mem_gallery_runner._question_query",
    text=(
        "Your task is to answer the question about the conversation between {speaker_a} "
        "and {speaker_b} in a concise manner with the help of memory content.\n"
        "Please only provide the content of the answer, without including introductory "
        "phrases like 'answer:'.\n"
        "For questions that require answering a date or time, strictly follow the format "
        "and provide a specific date or time whenever possible.\n"
        "Generate answers primarily concise, yet complete enough to accurately answer the "
        "questions.\n\n"
        "The current question is as follows:\n{question} {format_constraint}"
    ),
)

MEM_GALLERY_REFUSAL_PROMPT = PromptSpec(
    name="mem_gallery_refusal",
    version="mem_gallery_refusal_v1",
    purpose="Carry the official Mem-Gallery answer-refusal constraint for `AR` questions.",
    used_by="mindbridge.benchmarks.prompts.mem_gallery_format_constraint",
    text=(
        "Provide your answer based on the information in the conversation. Only if the "
        "information about the question is not present in the conversation, reply with: "
        "“Not mentioned.”"
    ),
)

MEM_GALLERY_CONFLICT_PROMPT = PromptSpec(
    name="mem_gallery_conflict",
    version="mem_gallery_conflict_v1",
    purpose="Carry the official Mem-Gallery conflict-detection constraint for `CD` questions.",
    used_by="mindbridge.benchmarks.prompts.mem_gallery_format_constraint",
    text=(
        "Please check whether this information conflicts with the conversation, and reply "
        "strictly with either “Yes.” or “No.”"
    ),
)

MEM_GALLERY_SEARCH_PROMPT = PromptSpec(
    name="mem_gallery_search",
    version="mem_gallery_search_v1",
    purpose="Carry the official Mem-Gallery visual-search constraint for `VS` questions.",
    used_by="mindbridge.benchmarks.prompts.mem_gallery_format_constraint",
    text=(
        "Return the image_id of the image(s). If there are multiple images, sort them in "
        "ascending order and separate them by commas. Format example: “D2:IMG_003, "
        "D2:IMG_010, D10:IMG_002” (for format reference only)."
    ),
)

_MEM_GALLERY_CONSTRAINTS = {
    "AR": MEM_GALLERY_REFUSAL_PROMPT,
    "CD": MEM_GALLERY_CONFLICT_PROMPT,
    "VS": MEM_GALLERY_SEARCH_PROMPT,
}


def mem_gallery_format_constraint(point: str) -> str:
    """Return one task type's official constraint, already separated, or empty text.

    The official runner applies a constraint file for `AR`, `CD` and `VS` only; the other
    six task types are asked without one, and adding one would change the task. It also
    prefixes the constraint with a blank line -- `format_constraint_str = "\\n\\n" +
    format_constraint` -- and interpolates it after a literal space, so a constrained
    question renders as its own paragraph. The separator belongs here rather than in the
    template, because this is the only place that knows whether a constraint exists at all,
    and it keeps each `PromptSpec.text` byte-identical to the upstream `.txt` file it
    reproduces.
    """
    prompt = _MEM_GALLERY_CONSTRAINTS.get(point.upper())
    return f"\n\n{prompt.text}" if prompt is not None else ""
```

Then extend the tuple:

```python
BENCHMARK_PROMPTS = (
    EGOMEM_REASON_QUERY_PROMPT,
    MEMLENS_QUERY_PROMPT,
    VIDEO_MME_QUERY_PROMPT,
    EGOTEMPO_QUERY_PROMPT,
    ATM_BENCH_QUERY_PROMPT,
    MEM_GALLERY_QUERY_PROMPT,
    MEM_GALLERY_REFUSAL_PROMPT,
    MEM_GALLERY_CONFLICT_PROMPT,
    MEM_GALLERY_SEARCH_PROMPT,
)
```

- [ ] **Step 4: Record the real fingerprints**

Run: `uv run pytest tests/contracts/test_prompt_catalog.py -v`
The failure prints the computed digests. Replace each `"0" * 64` placeholder with the digest
the test reports for that prompt, then run it again.
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest -W error && git diff --check
git add src/mindbridge/benchmarks/prompts.py tests/contracts/test_prompt_catalog.py
git commit -m "Add ATM-Bench and Mem-Gallery official query wordings"
```

---

### Task 5: ATM-Bench runner

**Files:**

- Create: `src/mindbridge/benchmarks/atm_bench_runner.py`
- Create: `tests/unit/benchmarks/test_atm_runner.py`

**Interfaces:**

- Consumes: everything Task 1 produces, plus `ATM_BENCH_QUERY_PROMPT` (Task 4),
  and from `mindbridge.benchmarks.runtime`: `benchmark_tenant_id`, `ingest_media`.
- Produces:
  - `AtmMediaSource = Literal["raw", "sgm"]`
  - `class AtmPreparedMedia(ContractModel)`: `media_id: Identifier`,
    `media_object: MediaObjectInput`
  - `class AtmPreparedArchive(ContractModel)`: `media: tuple[AtmPreparedMedia, ...]`
  - `def load_prepared_atm(path: Path) -> AtmPreparedArchive`
  - `def validate_prepared_atm(questions, prepared, *, media_source) -> None`
  - `class AtmQuestionResult(ContractModel)`: `question_id`, `question`, `qtype`,
    `reference_answer`, `prediction`, `evidence_ids`, `mindbridge_confidence`,
    `mindbridge_memory_ids`, `mindbridge_media_object_ids`, `mindbridge_trace_id`,
    `retrieved_gold_evidence_count`, `mindbridge_ingest_failure_count`
  - `async def ingest_atm_archive(memory, *, tenant_id, ...) -> int` (returns failures)
  - `async def answer_atm_question(memory, question, *, tenant_id, ...) -> AtmQuestionResult`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/benchmarks/test_atm_runner.py`:

```python
"""Production-contract checks for the single-archive ATM-Bench runner."""

from datetime import datetime, timezone
from typing import cast

import pytest

from mindbridge import MindBridge
from mindbridge.benchmarks.atm_bench import (
    AtmBenchQuestion,
    AtmEmail,
    AtmSgmRecord,
)
from mindbridge.benchmarks.atm_bench_runner import (
    AtmPreparedArchive,
    AtmPreparedMedia,
    answer_atm_question,
    ingest_atm_archive,
    validate_prepared_atm,
)
from mindbridge.contracts import (
    EvidenceView,
    MediaObjectInput,
    MemoryView,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
    RecallMode,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import JobState, MediaKind, MemoryType

NOW = datetime(2025, 2, 23, 13, 2, 49, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self, *, evidence: tuple[str, ...] = ()) -> None:
        self.observe_requests: list[ObserveRequest] = []
        self.remember_requests: list[RememberRequest] = []
        self.recall_requests: list[RecallRequest] = []
        self._evidence = evidence

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self.observe_requests.append(request)
        return ObservationReceipt(
            observation_id="observation_01",
            processing_job_id="job_01",
            evidence_ids=("evidence_01",),
            idempotency_key=request.idempotency_key or "generated",
            status=ObservationStatus.ACCEPTED,
            trace_id="trace_observe",
        )

    async def get_observation_job(
        self, tenant_id: str, job_id: str
    ) -> ObservationProcessingJobView:
        return ObservationProcessingJobView(
            job_id=job_id,
            observation_id="observation_01",
            state=JobState.SUCCEEDED,
            attempt=1,
            error_code=None,
            created_at=NOW,
            updated_at=NOW,
            trace_id="trace_job",
        )

    async def remember(self, request: RememberRequest) -> object:
        self.remember_requests.append(request)
        return object()

    async def recall(self, request: RecallRequest) -> RecallResult:
        self.recall_requests.append(request)
        return RecallResult(
            answer="£799.74",
            confidence=0.82,
            memories=(
                MemoryView(
                    memory_id="memory_01",
                    memory_type=MemoryType.EPISODIC,
                    summary="ID: email202411160004",
                    evidence_ids=(),
                    occurred_at=NOW,
                    ended_at=NOW,
                    created_at=NOW,
                ),
            ),
            evidence=tuple(
                EvidenceView(
                    evidence_id=f"evidence_{index}",
                    media_object_id=media_object_id,
                    start_ms=0,
                    end_ms=0,
                    media_url="https://example.invalid/signed",
                )
                for index, media_object_id in enumerate(self._evidence, start=1)
            ),
            trace_id="trace_recall",
        )


def _question(**overrides: object) -> AtmBenchQuestion:
    fields: dict[str, object] = {
        "question_id": "question_01",
        "question": "How much did I pay for accommodation?",
        "reference_answer": "£799.74",
        "qtype": "number",
        "evidence_ids": ("email202411160004", "20250223_130249"),
    }
    return AtmBenchQuestion.model_validate(fields | overrides)


def _prepared() -> AtmPreparedArchive:
    return AtmPreparedArchive(
        media=(
            AtmPreparedMedia(
                media_id="20250223_130249",
                media_object=MediaObjectInput(
                    media_object_id="20250223_130249",
                    kind=MediaKind.IMAGE,
                    uri="s3://mindbridge-media/atm-bench/20250223_130249.jpg",
                    sha256="a" * 64,
                    size_bytes=100_686,
                    created_at=NOW,
                ),
            ),
        )
    )


def _sgm_record() -> AtmSgmRecord:
    return AtmSgmRecord(
        media_id="20250223_130249",
        media_kind=MediaKind.IMAGE,
        occurred_at=NOW,
        raw_timestamp="2025-02-23 13:02:49",
        location_name="Porto, Portugal",
        city="Porto, Portugal",
        short_caption="A steel bridge over a river.",
        caption="A wide steel arch bridge spans the Douro.",
        ocr_text="",
        tags=("bridge", "porto"),
        size_bytes=100_686,
    )


def _email() -> AtmEmail:
    return AtmEmail(
        email_id="email202411160004",
        occurred_at=NOW,
        summary="Hotel confirmation",
        body="Total £799.74 for four nights.",
    )


async def test_raw_arm_observes_one_media_object_per_observation_named_by_its_stem() -> None:
    api = RecordingMemoryApi()

    failures = await ingest_atm_archive(
        cast(MindBridge, api),
        tenant_id="benchmark_atm_archive_run1",
        device_id="atm_archive",
        media_source="raw",
        prepared=_prepared(),
        sgm_records=(_sgm_record(),),
        emails=(_email(),),
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    assert failures == 0
    assert len(api.observe_requests) == 1
    request = api.observe_requests[0]
    assert [item.media_object_id for item in request.media_objects] == ["20250223_130249"]
    assert request.occurred_at == NOW
    # Emails are written in both arms; the raw arm writes no SGM text.
    summaries = [item.summary for item in api.remember_requests]
    assert any(summary.startswith("ID: email202411160004") for summary in summaries)
    assert not any(summary.startswith("ID: 20250223_130249") for summary in summaries)


async def test_sgm_arm_writes_official_blocks_and_observes_nothing() -> None:
    api = RecordingMemoryApi()

    failures = await ingest_atm_archive(
        cast(MindBridge, api),
        tenant_id="benchmark_atm_archive_run1",
        device_id="atm_archive",
        media_source="sgm",
        prepared=None,
        sgm_records=(_sgm_record(),),
        emails=(_email(),),
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    assert failures == 0
    assert api.observe_requests == []
    summaries = [item.summary for item in api.remember_requests]
    assert any(summary.startswith("ID: 20250223_130249\nType: image") for summary in summaries)
    assert any(summary.startswith("ID: email202411160004") for summary in summaries)


async def test_list_recall_questions_enumerate_and_others_answer() -> None:
    api = RecordingMemoryApi()

    await answer_atm_question(
        cast(MindBridge, api),
        _question(qtype="list_recall"),
        tenant_id="benchmark_atm_archive_run1",
        recall_limit=20,
    )
    await answer_atm_question(
        cast(MindBridge, api),
        _question(qtype="open_end"),
        tenant_id="benchmark_atm_archive_run1",
        recall_limit=20,
    )

    assert api.recall_requests[0].mode is RecallMode.ENUMERATE
    assert api.recall_requests[1].mode is RecallMode.ANSWER


async def test_retrieval_recall_counts_only_gold_evidence_the_recall_returned() -> None:
    api = RecordingMemoryApi(evidence=("20250223_130249", "20220430_132212"))

    result = await answer_atm_question(
        cast(MindBridge, api),
        _question(),
        tenant_id="benchmark_atm_archive_run1",
        recall_limit=20,
    )

    assert result.prediction == "£799.74"
    assert result.mindbridge_media_object_ids == ("20250223_130249", "20220430_132212")
    # One of the two gold evidence items came back; the distractor does not count.
    assert result.retrieved_gold_evidence_count == 1
    assert result.mindbridge_confidence == pytest.approx(0.82)


def test_raw_arm_refuses_to_start_without_every_cited_media_item() -> None:
    with pytest.raises(ValueError, match="missing prepared ATM-Bench media"):
        validate_prepared_atm(
            (_question(evidence_ids=("20250223_130249", "20991231_235959")),),
            _prepared(),
            media_source="raw",
        )

    # The SGM arm needs no prepared media at all.
    validate_prepared_atm((_question(),), None, media_source="sgm")
```

Add `pytest.ini`-level asyncio config already exists in this repository; these coroutine
tests follow the same style as `tests/unit/benchmarks/test_memlens_runner.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/benchmarks/test_atm_runner.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'mindbridge.benchmarks.atm_bench_runner'`

- [ ] **Step 3: Write the runner**

Create `src/mindbridge/benchmarks/atm_bench_runner.py`:

```python
"""Run ATM-Bench through public MindBridge ingestion and recall contracts.

One archive, one tenant: the release is a single person's three and a half years, and every
question is asked of the whole of it. The two media arms differ only in what is written —
`raw` sends the bytes through MindBridge's own perception, `sgm` writes the official
schema-guided text — and emails are written in both.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from mindbridge.benchmarks.atm_bench import (
    ATM_BENCH_ADAPTER_VERSION,
    AtmBenchQuestion,
    AtmEmail,
    AtmQuestionType,
    AtmSgmRecord,
    atm_email_block,
    atm_evidence_kind,
    atm_memory_chunks,
    atm_sgm_block,
)
from mindbridge.benchmarks.prompts import ATM_BENCH_QUERY_PROMPT
from mindbridge.benchmarks.runtime import ingest_media
from mindbridge.contracts import (
    ContractModel,
    Identifier,
    MediaObjectInput,
    NonEmptyString,
    ObserveRequest,
    RecallMode,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import MediaKind, MemoryType, SensorKind
from mindbridge.sdk import MindBridge

AtmMediaSource = Literal["raw", "sgm"]

_FORMAT_CONSTRAINTS: dict[str, str] = {
    "number": "Answer with the number alone, including its unit or currency symbol.",
    "list_recall": (
        "Answer with the matching evidence IDs alone, separated by commas, and nothing else."
    ),
    "open_end": "Answer concisely, using only what the memories support.",
}


class AtmPreparedMedia(ContractModel):
    """One archive item already staged in the object store, keyed by its official stem."""

    media_id: Identifier
    media_object: MediaObjectInput

    @model_validator(mode="after")
    def require_official_media_object_id(self) -> AtmPreparedMedia:
        if self.media_object.media_object_id != self.media_id:
            raise ValueError("ATM-Bench media_object_id must be the official media stem")
        if self.media_object.kind not in (MediaKind.IMAGE, MediaKind.VIDEO):
            raise ValueError("ATM-Bench media must be an image or a video")
        return self


class AtmPreparedArchive(ContractModel):
    """The staged archive one `raw` run reads."""

    media: tuple[AtmPreparedMedia, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_media(self) -> AtmPreparedArchive:
        ids = tuple(item.media_id for item in self.media)
        if len(set(ids)) != len(ids):
            raise ValueError("ATM-Bench prepared media IDs must be unique")
        return self


class AtmQuestionResult(ContractModel):
    """One official-shaped prediction with this run's retrieval diagnostics."""

    question_id: Identifier
    question: NonEmptyString
    qtype: AtmQuestionType
    reference_answer: NonEmptyString
    prediction: str
    evidence_ids: tuple[Identifier, ...]
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_media_object_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier
    retrieved_gold_evidence_count: int = Field(ge=0)
    mindbridge_ingest_failure_count: int = Field(default=0, ge=0)


def load_prepared_atm(path: Path) -> AtmPreparedArchive:
    """Load already staged archive metadata without owning transfer or storage."""
    return AtmPreparedArchive.model_validate_json(path.read_bytes())


def validate_prepared_atm(
    questions: Sequence[AtmBenchQuestion],
    prepared: AtmPreparedArchive | None,
    *,
    media_source: AtmMediaSource,
) -> None:
    """Refuse a `raw` run that cannot ground every media item its questions cite."""
    if media_source == "sgm":
        return
    available = {item.media_id for item in prepared.media} if prepared is not None else set()
    required = {
        evidence_id
        for question in questions
        for evidence_id in question.evidence_ids
        if atm_evidence_kind(evidence_id) == "media"
    }
    missing = required - available
    if missing:
        raise ValueError(f"missing prepared ATM-Bench media: {', '.join(sorted(missing))}")


async def ingest_atm_archive(
    memory: MindBridge,
    *,
    tenant_id: str,
    device_id: str,
    media_source: AtmMediaSource,
    prepared: AtmPreparedArchive | None,
    sgm_records: Sequence[AtmSgmRecord],
    emails: Sequence[AtmEmail],
    request_concurrency: int,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
) -> int:
    """Write the whole archive once, returning how many items failed to land.

    A failure count rather than an exception: an archive of 4,292 media items and 6,742
    emails takes hours, and one bad item must not discard the rest of the run.
    """
    if request_concurrency <= 0:
        raise ValueError("request_concurrency must be positive")
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")
    semaphore = asyncio.Semaphore(request_concurrency)
    by_id = {record.media_id: record for record in sgm_records}
    failures = 0

    if media_source == "raw":
        staged = prepared.media if prepared is not None else ()
        failures += await _gather_units(
            tuple(
                _observe_media(
                    memory,
                    tenant_id=tenant_id,
                    device_id=device_id,
                    item=item,
                    record=by_id.get(item.media_id),
                    sequence=sequence,
                    semaphore=semaphore,
                    poll_interval_seconds=poll_interval_seconds,
                    processing_timeout_seconds=processing_timeout_seconds,
                )
                for sequence, item in enumerate(staged)
            ),
            request_concurrency,
        )
    else:
        failures += await _gather_units(
            tuple(
                _remember_blocks(
                    memory,
                    tenant_id=tenant_id,
                    evidence_id=record.media_id,
                    block=atm_sgm_block(record),
                    occurred_at=record.occurred_at,
                    semaphore=semaphore,
                )
                for record in sgm_records
            ),
            request_concurrency,
        )

    failures += await _gather_units(
        tuple(
            _remember_blocks(
                memory,
                tenant_id=tenant_id,
                evidence_id=email.email_id,
                block=atm_email_block(email),
                occurred_at=email.occurred_at,
                semaphore=semaphore,
            )
            for email in emails
        ),
        request_concurrency,
    )
    return failures


async def answer_atm_question(
    memory: MindBridge,
    question: AtmBenchQuestion,
    *,
    tenant_id: str,
    recall_limit: int,
    ingest_failure_count: int = 0,
) -> AtmQuestionResult:
    """Ask one question of the whole archive and record what came back."""
    if not 1 <= recall_limit <= 100:
        raise ValueError("recall_limit must be between 1 and 100")
    mode = RecallMode.ENUMERATE if question.qtype == "list_recall" else RecallMode.ANSWER
    recalled = await memory.recall(
        RecallRequest(
            tenant_id=tenant_id,
            query=RecallQuery(text=_question_query(question)),
            mode=mode,
            limit=recall_limit,
        )
    )
    media_object_ids = tuple(item.media_object_id for item in recalled.evidence)
    return AtmQuestionResult(
        question_id=question.question_id,
        question=question.question,
        qtype=question.qtype,
        reference_answer=question.reference_answer,
        prediction=recalled.answer or "",
        evidence_ids=question.evidence_ids,
        mindbridge_confidence=recalled.confidence,
        mindbridge_memory_ids=tuple(item.memory_id for item in recalled.memories),
        mindbridge_media_object_ids=media_object_ids,
        mindbridge_trace_id=recalled.trace_id,
        retrieved_gold_evidence_count=len(set(question.evidence_ids) & set(media_object_ids)),
        mindbridge_ingest_failure_count=ingest_failure_count,
    )


def _question_query(question: AtmBenchQuestion) -> str:
    return ATM_BENCH_QUERY_PROMPT.text.format(
        question=question.question,
        format_constraint=_FORMAT_CONSTRAINTS[question.qtype],
    )


async def _gather_units(units: tuple[object, ...], request_concurrency: int) -> int:
    """Await coroutines in bounded batches, counting failures instead of raising them."""
    failures = 0
    for offset in range(0, len(units), request_concurrency):
        outcomes = await asyncio.gather(
            *units[offset : offset + request_concurrency], return_exceptions=True
        )
        failures += sum(isinstance(outcome, BaseException) for outcome in outcomes)
    return failures


async def _observe_media(
    memory: MindBridge,
    *,
    tenant_id: str,
    device_id: str,
    item: AtmPreparedMedia,
    record: AtmSgmRecord | None,
    sequence: int,
    semaphore: asyncio.Semaphore,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
) -> None:
    """Observe exactly one media object, so returned evidence names one archive item."""
    duration = record.duration_seconds if record is not None else None
    started = item.media_object.created_at
    ended = started if duration is None else started + timedelta(seconds=duration)
    async with semaphore:
        await ingest_media(
            memory,
            ObserveRequest(
                tenant_id=tenant_id,
                device_id=device_id,
                boot_id=ATM_BENCH_ADAPTER_VERSION,
                sequence=sequence,
                sensor=SensorKind.CAMERA,
                media_objects=(item.media_object,),
                occurred_at=started,
                ended_at=ended,
                observed_at=ended,
                idempotency_key=f"{ATM_BENCH_ADAPTER_VERSION}:media:{item.media_id}",
            ),
            poll_interval_seconds=poll_interval_seconds,
            processing_timeout_seconds=processing_timeout_seconds,
        )


async def _remember_blocks(
    memory: MindBridge,
    *,
    tenant_id: str,
    evidence_id: str,
    block: str,
    occurred_at: datetime,
    semaphore: asyncio.Semaphore,
) -> None:
    """Write one serialized block, chunked where it exceeds the summary limit."""
    for index, chunk in enumerate(atm_memory_chunks(block, evidence_id)):
        async with semaphore:
            await memory.remember(
                RememberRequest(
                    tenant_id=tenant_id,
                    summary=chunk,
                    memory_type=MemoryType.EPISODIC,
                    occurred_at=occurred_at,
                    idempotency_key=(f"{ATM_BENCH_ADAPTER_VERSION}:text:{evidence_id}:{index}"),
                )
            )
```

Add the two imports the helpers need at the top: `from datetime import datetime, timedelta`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/benchmarks/test_atm_runner.py -v`
Expected: 5 passed

- [ ] **Step 5: Mutation-check the arm switch and the recall-mode switch**

Change `media_source == "raw"` to `media_source == "sgm"` in `ingest_atm_archive`, run the
tests, confirm `test_raw_arm_observes_one_media_object_per_observation_named_by_its_stem`
and `test_sgm_arm_writes_official_blocks_and_observes_nothing` both fail, then restore.
Change the `RecallMode.ENUMERATE` branch to always use `RecallMode.ANSWER`, confirm
`test_list_recall_questions_enumerate_and_others_answer` fails, then restore.

- [ ] **Step 6: Run the gates and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest -W error && git diff --check
git add src/mindbridge/benchmarks/atm_bench_runner.py tests/unit/benchmarks/test_atm_runner.py
git commit -m "Add ATM-Bench runner with raw and SGM media arms"
```

---

### Task 6: ATM-Bench CLI

**Files:**

- Create: `src/mindbridge/benchmarks/atm_cli.py`
- Create: `tests/unit/benchmarks/test_atm_cli.py`
- Modify: `src/mindbridge/benchmarks/cli.py:80-118` (the `RUNNERS` table)

**Interfaces:**

- Consumes: Task 5's runner surface, Task 1's loaders, `ATM_BENCH_QUERY_PROMPT`.
- Produces: `def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None`,
  `class AtmRunManifest(MediaBenchmarkRunManifest)` with
  `benchmark: Literal["ATM-Bench"]`, `split: Literal["main", "hard"]`,
  `media_source: AtmMediaSource`, `dataset_repository`, `evaluator_repository`,
  `prepared_media_manifest_sha256`, `emails_sha256`, `sgm_sha256`,
  `perception_prompt_version`, `query_prompt_version`, `question_ids`,
  `media_item_count`, `email_count`, and `ATM_BENCH_RUNNER_VERSION`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/benchmarks/test_atm_cli.py`:

```python
"""Argument and artifact checks for the ATM-Bench CLI."""

import json
from pathlib import Path

import pytest

from mindbridge.benchmarks import atm_cli
from tests.unit.benchmarks.benchmark_deployment import write_deployment_snapshot


def _write_release(directory: Path) -> Path:
    dataset_path = directory / "atm-bench.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "question_01",
                    "question": "How much did I pay?",
                    "answer": "£799.74",
                    "notes": "",
                    "evidence_ids": ["email202411160004"],
                    "qtype": "number",
                }
            ]
        ),
        encoding="utf-8",
    )
    return dataset_path


def test_raw_run_requires_a_prepared_media_manifest(tmp_path: Path) -> None:
    dataset_path = _write_release(tmp_path)
    deployment = write_deployment_snapshot(tmp_path)
    emails = tmp_path / "emails.json"
    emails.write_text(
        json.dumps(
            [
                {
                    "id": "email202411160004",
                    "timestamp": "2024-11-16 09:12:00",
                    "short_summary": "Hotel",
                    "detail": "Total £799.74.",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw ATM-Bench runs require prepared media"):
        atm_cli.main(
            [
                "--dataset",
                str(dataset_path),
                "--emails",
                str(emails),
                "--output",
                str(tmp_path / "predictions.json"),
                "--api-base-url",
                "http://localhost:8000",
                "--deployment-config",
                str(deployment),
                "--run-id",
                "run1",
                "--split",
                "main",
                "--media-source",
                "raw",
            ],
            prog="mindbridge-bench atm",
        )


def test_sgm_run_requires_the_official_batch_results(tmp_path: Path) -> None:
    dataset_path = _write_release(tmp_path)
    deployment = write_deployment_snapshot(tmp_path, worker=False)
    emails = tmp_path / "emails.json"
    emails.write_text(
        json.dumps(
            [
                {
                    "id": "email202411160004",
                    "timestamp": "2024-11-16 09:12:00",
                    "short_summary": "Hotel",
                    "detail": "Total £799.74.",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sgm ATM-Bench runs require batch results"):
        atm_cli.main(
            [
                "--dataset",
                str(dataset_path),
                "--emails",
                str(emails),
                "--output",
                str(tmp_path / "predictions.json"),
                "--api-base-url",
                "http://localhost:8000",
                "--deployment-config",
                str(deployment),
                "--run-id",
                "run1",
                "--split",
                "main",
                "--media-source",
                "sgm",
            ],
            prog="mindbridge-bench atm",
        )


def test_cli_table_dispatches_atm() -> None:
    from mindbridge.benchmarks.cli import RUNNERS

    assert RUNNERS["atm"].module == "mindbridge.benchmarks.atm_cli"
    assert RUNNERS["atm"].extra is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/benchmarks/test_atm_cli.py -v`
Expected: collection error, `ImportError: cannot import name 'atm_cli'`

- [ ] **Step 3: Write the CLI**

Create `src/mindbridge/benchmarks/atm_cli.py`:

```python
"""Reproducible ATM-Bench runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from mindbridge.benchmarks.artifacts import (
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
)
from mindbridge.benchmarks.atm_bench import (
    ATM_BENCH_ADAPTER_VERSION,
    AtmBenchQuestion,
    load_atm_bench,
    load_atm_emails,
    load_atm_sgm,
)
from mindbridge.benchmarks.atm_bench_runner import (
    AtmMediaSource,
    AtmQuestionResult,
    answer_atm_question,
    ingest_atm_archive,
    load_prepared_atm,
    validate_prepared_atm,
)
from mindbridge.benchmarks.cli_common import (
    MediaArguments,
    MediaBenchmarkRunManifest,
    add_media_arguments,
    connected_memory,
    core_parser,
    media_arguments,
    media_manifest,
    report,
    report_unit,
    select_by_id,
    write_run_artifacts,
)
from mindbridge.benchmarks.prompts import ATM_BENCH_QUERY_PROMPT
from mindbridge.benchmarks.runtime import benchmark_tenant_id
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT

ATM_BENCH_RUNNER_VERSION = "atm_bench_production_api_v1"
AtmSplit = Literal["main", "hard"]


class AtmRunManifest(MediaBenchmarkRunManifest):
    """Immutable source, protocol, deployment, model, and prediction identity."""

    benchmark: Literal["ATM-Bench"] = "ATM-Bench"
    split: AtmSplit
    media_source: AtmMediaSource
    dataset_repository: NonEmptyString
    evaluator_repository: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex | None = None
    emails_sha256: Sha256Hex
    sgm_image_sha256: Sha256Hex | None = None
    sgm_video_sha256: Sha256Hex | None = None
    perception_prompt_version: NonEmptyString | None = None
    query_prompt_version: NonEmptyString
    question_ids: tuple[Identifier, ...] = Field(min_length=1)
    media_item_count: int = Field(ge=0)
    email_count: int = Field(gt=0)
    ingest_failure_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    emails_path: Path
    prepared_media_path: Path | None
    sgm_image_path: Path | None
    sgm_video_path: Path | None
    split: AtmSplit
    media_source: AtmMediaSource
    question_ids: tuple[str, ...]


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Ingest the archive once, then answer the selected questions against it."""
    arguments = _parse_arguments(argv, prog)
    questions = select_by_id(
        load_atm_bench(arguments.dataset_path),
        arguments.question_ids,
        key=lambda question: question.question_id,
        label="selected ATM-Bench question IDs",
    )
    if not questions:
        raise ValueError("ATM-Bench selection must not be empty")
    prepared = (
        load_prepared_atm(arguments.prepared_media_path)
        if arguments.prepared_media_path is not None
        else None
    )
    if arguments.media_source == "raw" and prepared is None:
        raise ValueError("raw ATM-Bench runs require prepared media")
    if arguments.media_source == "sgm" and arguments.sgm_image_path is None:
        raise ValueError("sgm ATM-Bench runs require batch results")
    validate_prepared_atm(questions, prepared, media_source=arguments.media_source)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=arguments.media_source == "raw",
    )
    emails = load_atm_emails(arguments.emails_path)
    sgm_records = tuple(
        record
        for path in (arguments.sgm_image_path, arguments.sgm_video_path)
        if path is not None
        for record in load_atm_sgm(path)
    )
    report(f"running {len(questions)} questions", quiet=arguments.quiet)
    failures, results = asyncio.run(_run(arguments, questions, prepared, sgm_records, emails))
    _write_artifacts(arguments, questions, results, deployment, failures, emails, sgm_records)
    report(f"wrote {arguments.output_path}", quiet=arguments.quiet)
```

The `_run`, `_write_artifacts`, and `_parse_arguments` bodies:

```python
async def _run(
    arguments: _Arguments,
    questions: tuple[AtmBenchQuestion, ...],
    prepared: object,
    sgm_records: tuple[object, ...],
    emails: tuple[object, ...],
) -> tuple[int, tuple[AtmQuestionResult, ...]]:
    tenant_id = benchmark_tenant_id(arguments.tenant_prefix, "archive", arguments.run_id)
    async with connected_memory(arguments) as memory:
        failures = await ingest_atm_archive(
            memory,
            tenant_id=tenant_id,
            device_id=arguments.device_id,
            media_source=arguments.media_source,
            prepared=cast(object, prepared),  # typed in the runner's own signature
            sgm_records=cast(tuple, sgm_records),
            emails=cast(tuple, emails),
            request_concurrency=arguments.request_concurrency,
            poll_interval_seconds=arguments.poll_interval_seconds,
            processing_timeout_seconds=arguments.processing_timeout_seconds,
        )
        results = []
        for index, question in enumerate(questions, start=1):
            report_unit(
                f"question {question.question_id}",
                index=index,
                total=len(questions),
                quiet=arguments.quiet,
            )
            results.append(
                await answer_atm_question(
                    memory,
                    question,
                    tenant_id=tenant_id,
                    recall_limit=arguments.recall_limit,
                    ingest_failure_count=failures,
                )
            )
    return failures, tuple(results)


def _write_artifacts(
    arguments: _Arguments,
    questions: tuple[AtmBenchQuestion, ...],
    results: tuple[AtmQuestionResult, ...],
    deployment: LoadedDeployment,
    failures: int,
    emails: tuple[object, ...],
    sgm_records: tuple[object, ...],
) -> None:
    if tuple(result.question_id for result in results) != tuple(
        question.question_id for question in questions
    ):
        raise ValueError("ATM-Bench predictions must match annotation question order")
    # The official evaluator reads a list of {id, question, answer, prediction} objects.
    predictions = (
        json.dumps(
            [
                {
                    "id": result.question_id,
                    "question": result.question,
                    "qtype": result.qtype,
                    "answer": result.reference_answer,
                    "prediction": result.prediction,
                    "evidence_ids": list(result.evidence_ids),
                    "retrieved_evidence_ids": list(result.mindbridge_media_object_ids),
                    "retrieved_gold_evidence_count": result.retrieved_gold_evidence_count,
                    "mindbridge_confidence": result.mindbridge_confidence,
                    "mindbridge_memory_ids": list(result.mindbridge_memory_ids),
                    "mindbridge_trace_id": result.mindbridge_trace_id,
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    manifest = media_manifest(
        AtmRunManifest,
        arguments,
        deployment,
        runner_version=ATM_BENCH_RUNNER_VERSION,
        adapter_version=ATM_BENCH_ADAPTER_VERSION,
        annotation_sha256=sha256_file(arguments.dataset_path),
        predictions=predictions,
        split=arguments.split,
        media_source=arguments.media_source,
        dataset_repository="Jingbiao/ATM-Bench",
        evaluator_repository="JingbiaoMei/ATM-Bench",
        prepared_media_manifest_sha256=(
            sha256_file(arguments.prepared_media_path)
            if arguments.prepared_media_path is not None
            else None
        ),
        emails_sha256=sha256_file(arguments.emails_path),
        sgm_image_sha256=(
            sha256_file(arguments.sgm_image_path) if arguments.sgm_image_path is not None else None
        ),
        sgm_video_sha256=(
            sha256_file(arguments.sgm_video_path) if arguments.sgm_video_path is not None else None
        ),
        perception_prompt_version=(
            PERCEIVE_EVENTS_PROMPT.version if arguments.media_source == "raw" else None
        ),
        query_prompt_version=ATM_BENCH_QUERY_PROMPT.version,
        question_ids=tuple(question.question_id for question in questions),
        media_item_count=len(sgm_records),
        email_count=len(emails),
        ingest_failure_count=failures,
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = add_media_arguments(
        core_parser(tenant_prefix="benchmark_atm", prog=prog, description=__doc__),
        device_id="atm_archive",
    )
    parser.add_argument("--emails", type=Path, required=True, help="official emails.json to ingest")
    parser.add_argument(
        "--prepared-media", type=Path, help="manifest of staged archive media; required for raw"
    )
    parser.add_argument(
        "--sgm-image", type=Path, help="official image_batch_results.json; required for sgm"
    )
    parser.add_argument("--sgm-video", type=Path, help="official video_batch_results.json")
    parser.add_argument(
        "--split",
        choices=("main", "hard"),
        required=True,
        help="official split this dataset file is, recorded in the manifest",
    )
    parser.add_argument(
        "--media-source",
        choices=("raw", "sgm"),
        default="raw",
        help="ingest the archive's own bytes, or the official schema-guided text",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="official question to run; repeatable, default the whole split",
    )
    parsed = parser.parse_args(argv)
    return media_arguments(
        _Arguments,
        parsed,
        emails_path=parsed.emails,
        prepared_media_path=parsed.prepared_media,
        sgm_image_path=parsed.sgm_image,
        sgm_video_path=parsed.sgm_video,
        split=cast(AtmSplit, parsed.split),
        media_source=cast(AtmMediaSource, parsed.media_source),
        question_ids=tuple(parsed.question_id),
    )


if __name__ == "__main__":
    main()
```

If mypy rejects the `cast(object, ...)` placeholders in `_run`, type the parameters properly
as `AtmPreparedArchive | None`, `tuple[AtmSgmRecord, ...]`, and `tuple[AtmEmail, ...]` and
import those names — the casts are only there to keep the snippet short, and typed
parameters are what this codebase uses.

Then add the row to `RUNNERS` in `src/mindbridge/benchmarks/cli.py`, after the
`"mm-lifelong"` entry:

```python
    "atm": Runner("mindbridge.benchmarks.atm_cli", "Run official ATM-Bench"),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/benchmarks/test_atm_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: Check the help text renders**

Run: `uv run mindbridge-bench atm --help`
Expected: usage line reads `mindbridge-bench atm`, `--media-source` shows
`(default: raw)`, and the exit-status block is present.

- [ ] **Step 6: Run the gates and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest -W error && git diff --check
git add src/mindbridge/benchmarks/atm_cli.py src/mindbridge/benchmarks/cli.py tests/unit/benchmarks/test_atm_cli.py
git commit -m "Add the mindbridge-bench atm command"
```

---

### Task 7: Mem-Gallery runner

**Files:**

- Create: `src/mindbridge/benchmarks/mem_gallery_runner.py`
- Create: `tests/unit/benchmarks/test_mem_gallery_runner.py`

**Interfaces:**

- Consumes: Task 2's contracts, Task 4's `MEM_GALLERY_QUERY_PROMPT` and
  `mem_gallery_format_constraint`, `benchmark_tenant_id`, `ingest_media`.
- Produces:
  - `class MemGalleryPreparedImage(ContractModel)`: `image_key: NonEmptyString`,
    `media_object: MediaObjectInput`
  - `class MemGalleryPreparedImages(ContractModel)`: `images: tuple[...]`
  - `def load_prepared_mem_gallery(path: Path) -> MemGalleryPreparedImages`
  - `def validate_mem_gallery_images(topics, prepared) -> None`
  - `class MemGalleryQuestionResult(ContractModel)`: `question_id`, `topic`, `point`,
    `question`, `reference_answer`, `prediction`, `clue_round_ids`,
    `mindbridge_confidence`, `mindbridge_memory_ids`, `mindbridge_round_ids`,
    `mindbridge_media_object_ids`, `mindbridge_trace_id`,
    `retrieved_clue_round_count`, `mindbridge_ingest_failure_count`
  - `async def run_mem_gallery_topic(memory, topic, *, run_id, prepared, ...) -> tuple[MemGalleryQuestionResult, ...]`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/benchmarks/test_mem_gallery_runner.py`:

```python
"""Production-contract checks for the per-topic Mem-Gallery runner."""

from datetime import datetime, timezone
from typing import cast

import pytest

from mindbridge import MindBridge
from mindbridge.benchmarks.mem_gallery import (
    MemGalleryProfile,
    MemGalleryQuestion,
    MemGalleryRound,
    MemGallerySession,
    MemGalleryTopic,
)
from mindbridge.benchmarks.mem_gallery_runner import (
    MemGalleryPreparedImage,
    MemGalleryPreparedImages,
    run_mem_gallery_topic,
    validate_mem_gallery_images,
)
from mindbridge.contracts import (
    EvidenceView,
    MediaObjectInput,
    MemoryView,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import JobState, MediaKind, MemoryType

NOW = datetime(2024, 6, 24, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self) -> None:
        self.observe_requests: list[ObserveRequest] = []
        self.remember_requests: list[RememberRequest] = []
        self.recall_requests: list[RecallRequest] = []

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self.observe_requests.append(request)
        return ObservationReceipt(
            observation_id="observation_01",
            processing_job_id="job_01",
            evidence_ids=("evidence_01",),
            idempotency_key=request.idempotency_key or "generated",
            status=ObservationStatus.ACCEPTED,
            trace_id="trace_observe",
        )

    async def get_observation_job(
        self, tenant_id: str, job_id: str
    ) -> ObservationProcessingJobView:
        return ObservationProcessingJobView(
            job_id=job_id,
            observation_id="observation_01",
            state=JobState.SUCCEEDED,
            attempt=1,
            error_code=None,
            created_at=NOW,
            updated_at=NOW,
            trace_id="trace_job",
        )

    async def remember(self, request: RememberRequest) -> object:
        self.remember_requests.append(request)
        return object()

    async def recall(self, request: RecallRequest) -> RecallResult:
        self.recall_requests.append(request)
        return RecallResult(
            answer="30 litres or more.",
            confidence=0.7,
            memories=(
                MemoryView(
                    memory_id="memory_01",
                    memory_type=MemoryType.EPISODIC,
                    summary="D1:1 User asked: Can you tell me the basics?",
                    evidence_ids=(),
                    occurred_at=NOW,
                    ended_at=NOW,
                    created_at=NOW,
                ),
            ),
            evidence=(
                EvidenceView(
                    evidence_id="evidence_01",
                    media_object_id="D1:IMG_001",
                    start_ms=0,
                    end_ms=0,
                    media_url="https://example.invalid/signed",
                ),
            ),
            trace_id="trace_recall",
        )


def _topic() -> MemGalleryTopic:
    return MemGalleryTopic(
        topic="Baking",
        profile=MemGalleryProfile(
            name="Maya",
            persona_summary="A librarian who bakes.",
            traits=("curious",),
            conversation_style="Earnest.",
        ),
        sessions=(
            MemGallerySession(
                session_id="D1",
                occurred_at=NOW,
                rounds=(
                    MemGalleryRound(
                        round_id="D1:1",
                        user="Can you tell me the basics?",
                        assistant="Start with a 30 litre oven.",
                    ),
                    MemGalleryRound(
                        round_id="D1:2",
                        user="What is in this picture?",
                        assistant="A tray of shortbread.",
                        image_id="D1:IMG_001",
                        image_path="../image/Baking/D1_IMG_001.jpg",
                        image_caption="Pale shortbread fingers.",
                    ),
                ),
            ),
        ),
        questions=(
            MemGalleryQuestion(
                question_id="Baking:1",
                point="FR",
                question="What oven size was recommended?",
                reference_answer="30 litres or more.",
                session_ids=("D1",),
                clue_round_ids=("D1:1",),
            ),
            MemGalleryQuestion(
                question_id="Baking:2",
                point="VS",
                question="Which image shows shortbread?",
                reference_answer="D1:IMG_001",
                session_ids=("D1",),
                clue_round_ids=("D1:2",),
                question_image_path="../image/Baking/QA_IMG_001.jpg",
                question_image_caption="A tray of biscuits.",
            ),
        ),
    )


def _prepared() -> MemGalleryPreparedImages:
    return MemGalleryPreparedImages(
        images=(
            MemGalleryPreparedImage(
                image_key="../image/Baking/D1_IMG_001.jpg",
                media_object=MediaObjectInput(
                    media_object_id="D1:IMG_001",
                    kind=MediaKind.IMAGE,
                    uri="s3://mindbridge-media/mem-gallery/Baking/D1_IMG_001.jpg",
                    sha256="b" * 64,
                    size_bytes=52_144,
                    created_at=NOW,
                ),
            ),
            MemGalleryPreparedImage(
                image_key="../image/Baking/QA_IMG_001.jpg",
                media_object=MediaObjectInput(
                    media_object_id="Baking:QA_IMG_001",
                    kind=MediaKind.IMAGE,
                    uri="s3://mindbridge-media/mem-gallery/Baking/QA_IMG_001.jpg",
                    sha256="c" * 64,
                    size_bytes=41_002,
                    created_at=NOW,
                ),
            ),
        )
    )


async def test_rounds_are_written_per_speaker_and_images_observed_with_their_round_text() -> None:
    api = RecordingMemoryApi()

    results = await run_mem_gallery_topic(
        cast(MindBridge, api),
        _topic(),
        run_id="run1",
        prepared=_prepared(),
        tenant_prefix="benchmark_mem_gallery",
        device_id="mem_gallery_conversation",
        recall_limit=20,
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    # One tenant for the whole topic, not one per question.
    assert {request.tenant_id for request in api.recall_requests} == {
        request.tenant_id for request in api.remember_requests
    }
    # The image round is observed with its official image_id as the media object ID.
    assert [item.media_object_id for item in api.observe_requests[0].media_objects] == [
        "D1:IMG_001"
    ]
    # Two rounds, two speakers each, and the image round's text is written too.
    assert len(api.remember_requests) == 4
    assert all(request.summary.startswith(("D1:1 ", "D1:2 ")) for request in api.remember_requests)
    assert len(results) == 2


async def test_a_question_image_is_sent_as_a_recall_query_object() -> None:
    api = RecordingMemoryApi()

    await run_mem_gallery_topic(
        cast(MindBridge, api),
        _topic(),
        run_id="run1",
        prepared=_prepared(),
        tenant_prefix="benchmark_mem_gallery",
        device_id="mem_gallery_conversation",
        recall_limit=20,
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    assert api.recall_requests[0].query.media_object_ids == ()
    assert api.recall_requests[1].query.media_object_ids == ("Baking:QA_IMG_001",)


async def test_official_constraints_are_applied_only_to_ar_cd_and_vs() -> None:
    api = RecordingMemoryApi()

    await run_mem_gallery_topic(
        cast(MindBridge, api),
        _topic(),
        run_id="run1",
        prepared=_prepared(),
        tenant_prefix="benchmark_mem_gallery",
        device_id="mem_gallery_conversation",
        recall_limit=20,
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    factual_query = api.recall_requests[0].query.text or ""
    search_query = api.recall_requests[1].query.text or ""
    assert "Return the image_id" not in factual_query
    assert "Return the image_id" in search_query
    # The official wording names the speakers, and the constraint arrives as its own
    # paragraph rather than trailing the question on one line.
    assert "between user (Maya) and assistant" in factual_query
    assert search_query.endswith(
        "\n\nReturn the image_id of the image(s). If there are "
        "multiple images, sort them in ascending order and separate "
        "them by commas. Format example: \u201cD2:IMG_003, "
        "D2:IMG_010, D10:IMG_002\u201d (for format reference only)."
    )


async def test_clue_recall_counts_rounds_the_recall_actually_returned() -> None:
    api = RecordingMemoryApi()

    results = await run_mem_gallery_topic(
        cast(MindBridge, api),
        _topic(),
        run_id="run1",
        prepared=_prepared(),
        tenant_prefix="benchmark_mem_gallery",
        device_id="mem_gallery_conversation",
        recall_limit=20,
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    assert results[0].mindbridge_round_ids == ("D1:1",)
    assert results[0].retrieved_clue_round_count == 1
    assert results[1].retrieved_clue_round_count == 0
    assert results[0].mindbridge_confidence == pytest.approx(0.7)


def test_a_run_refuses_to_start_without_every_referenced_image() -> None:
    prepared = MemGalleryPreparedImages(images=_prepared().images[:1])

    with pytest.raises(ValueError, match="missing prepared Mem-Gallery images"):
        validate_mem_gallery_images((_topic(),), prepared)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/benchmarks/test_mem_gallery_runner.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'mindbridge.benchmarks.mem_gallery_runner'`

- [ ] **Step 3: Write the runner**

Create `src/mindbridge/benchmarks/mem_gallery_runner.py`:

```python
"""Run Mem-Gallery through public MindBridge ingestion and recall contracts.

One tenant per topic: the release is twenty independent personas, and a shared store would
leak one persona's memory into another's questions. Rounds are written per speaker and keyed
by their official round ID, which is what makes the release's `clue` annotation measurable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field, model_validator

from mindbridge.benchmarks.mem_gallery import (
    MEM_GALLERY_ADAPTER_VERSION,
    MemGalleryPoint,
    MemGalleryQuestion,
    MemGalleryRound,
    MemGallerySession,
    MemGalleryTopic,
)
from mindbridge.benchmarks.prompts import (
    MEM_GALLERY_QUERY_PROMPT,
    mem_gallery_format_constraint,
)
from mindbridge.benchmarks.runtime import benchmark_tenant_id, ingest_media
from mindbridge.contracts import (
    ContractModel,
    Identifier,
    MediaObjectInput,
    NonEmptyString,
    ObserveRequest,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import MediaKind, MemoryType, SensorKind
from mindbridge.sdk import MindBridge


class MemGalleryPreparedImage(ContractModel):
    """One staged image, keyed by the release-relative path that references it."""

    image_key: NonEmptyString
    media_object: MediaObjectInput

    @model_validator(mode="after")
    def require_image(self) -> MemGalleryPreparedImage:
        if self.media_object.kind is not MediaKind.IMAGE:
            raise ValueError("Mem-Gallery prepared media objects must be images")
        return self


class MemGalleryPreparedImages(ContractModel):
    """Staged image lookup shared by every topic in one run."""

    images: tuple[MemGalleryPreparedImage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_images(self) -> MemGalleryPreparedImages:
        keys = tuple(image.image_key for image in self.images)
        object_ids = tuple(image.media_object.media_object_id for image in self.images)
        if len(set(keys)) != len(keys):
            raise ValueError("Mem-Gallery prepared image keys must be unique")
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("Mem-Gallery prepared media_object_ids must be unique")
        return self


class MemGalleryQuestionResult(ContractModel):
    """One official-shaped prediction with this run's retrieval diagnostics."""

    question_id: Identifier
    topic: Identifier
    point: MemGalleryPoint
    question: NonEmptyString
    reference_answer: NonEmptyString
    prediction: str
    clue_round_ids: tuple[Identifier, ...]
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_round_ids: tuple[Identifier, ...]
    mindbridge_media_object_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier
    retrieved_clue_round_count: int = Field(ge=0)
    mindbridge_ingest_failure_count: int = Field(default=0, ge=0)


def load_prepared_mem_gallery(path: Path) -> MemGalleryPreparedImages:
    """Load already staged image metadata without owning transfer or storage."""
    return MemGalleryPreparedImages.model_validate_json(path.read_bytes())


def validate_mem_gallery_images(
    topics: Sequence[MemGalleryTopic],
    prepared: MemGalleryPreparedImages,
) -> None:
    """Refuse a run that cannot ground every image its topics and questions reference."""
    available = {image.image_key for image in prepared.images}
    required = {
        round_.image_path
        for topic in topics
        for session in topic.sessions
        for round_ in session.rounds
        if round_.image_path is not None
    } | {
        question.question_image_path
        for topic in topics
        for question in topic.questions
        if question.question_image_path is not None
    }
    missing = required - available
    if missing:
        raise ValueError(f"missing prepared Mem-Gallery images: {', '.join(sorted(missing))}")


async def run_mem_gallery_topic(
    memory: MindBridge,
    topic: MemGalleryTopic,
    *,
    run_id: str,
    prepared: MemGalleryPreparedImages,
    tenant_prefix: str = "benchmark_mem_gallery",
    device_id: str = "mem_gallery_conversation",
    recall_limit: int = 20,
    request_concurrency: int = 4,
    poll_interval_seconds: float = 1.0,
    processing_timeout_seconds: float = 1_800.0,
) -> tuple[MemGalleryQuestionResult, ...]:
    """Ingest one persona's whole dialogue, then answer every question over it."""
    if not 1 <= recall_limit <= 100 or request_concurrency <= 0:
        raise ValueError(
            "recall_limit must be between 1 and 100; request_concurrency must be positive"
        )
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")
    validate_mem_gallery_images((topic,), prepared)
    by_key = {image.image_key: image.media_object for image in prepared.images}
    tenant_id = benchmark_tenant_id(tenant_prefix, topic.topic, run_id)
    semaphore = asyncio.Semaphore(request_concurrency)

    # Sessions are dated and ordered; rounds inside one session share that date, so only
    # insertion order records which came first. Sessions run in order, rounds strictly serial.
    failures = 0
    sequence = 0
    for session in topic.sessions:
        for round_ in session.rounds:
            try:
                await _ingest_round(
                    memory,
                    tenant_id=tenant_id,
                    device_id=device_id,
                    session=session,
                    round_=round_,
                    sequence=sequence,
                    media_object=(
                        by_key[round_.image_path] if round_.image_path is not None else None
                    ),
                    semaphore=semaphore,
                    poll_interval_seconds=poll_interval_seconds,
                    processing_timeout_seconds=processing_timeout_seconds,
                )
            except Exception:  # noqa: BLE001 - a bad round must not discard the topic
                failures += 1
            sequence += 1

    results = []
    for question in topic.questions:
        results.append(
            await _answer_question(
                memory,
                topic,
                question,
                tenant_id=tenant_id,
                recall_limit=recall_limit,
                question_image=(
                    by_key[question.question_image_path]
                    if question.question_image_path is not None
                    else None
                ),
                semaphore=semaphore,
                ingest_failure_count=failures,
            )
        )
    return tuple(results)


async def _ingest_round(
    memory: MindBridge,
    *,
    tenant_id: str,
    device_id: str,
    session: MemGallerySession,
    round_: MemGalleryRound,
    sequence: int,
    media_object: MediaObjectInput | None,
    semaphore: asyncio.Semaphore,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
) -> None:
    """Write one round: its image as an observation, then its two speaker turns."""
    evidence_ids: tuple[str, ...] = ()
    if media_object is not None:
        async with semaphore:
            evidence_ids = await ingest_media(
                memory,
                ObserveRequest(
                    tenant_id=tenant_id,
                    device_id=device_id,
                    boot_id=MEM_GALLERY_ADAPTER_VERSION,
                    sequence=sequence,
                    sensor=SensorKind.CAMERA,
                    media_objects=(media_object,),
                    occurred_at=session.occurred_at,
                    ended_at=session.occurred_at,
                    observed_at=session.occurred_at,
                    idempotency_key=(f"{MEM_GALLERY_ADAPTER_VERSION}:media:{round_.round_id}"),
                ),
                poll_interval_seconds=poll_interval_seconds,
                processing_timeout_seconds=processing_timeout_seconds,
            )
    for role, content in (("User", round_.user), ("Assistant", round_.assistant)):
        summary = f"{round_.round_id} {role} said: {content}"
        if round_.image_caption is not None and role == "User" and round_.image_id is not None:
            summary = (
                f"{round_.round_id} {role} said: {content} "
                f"[image {round_.image_id}: {round_.image_caption}]"
            )
        async with semaphore:
            await memory.remember(
                RememberRequest(
                    tenant_id=tenant_id,
                    summary=summary[:2_048],
                    memory_type=MemoryType.EPISODIC,
                    occurred_at=session.occurred_at,
                    evidence_ids=evidence_ids,
                    idempotency_key=(
                        f"{MEM_GALLERY_ADAPTER_VERSION}:text:{round_.round_id}:{role.lower()}"
                    ),
                )
            )


async def _answer_question(
    memory: MindBridge,
    topic: MemGalleryTopic,
    question: MemGalleryQuestion,
    *,
    tenant_id: str,
    recall_limit: int,
    question_image: MediaObjectInput | None,
    semaphore: asyncio.Semaphore,
    ingest_failure_count: int,
) -> MemGalleryQuestionResult:
    async with semaphore:
        recalled = await memory.recall(
            RecallRequest(
                tenant_id=tenant_id,
                query=RecallQuery(
                    text=_question_query(topic, question),
                    media_object_ids=(
                        () if question_image is None else (question_image.media_object_id,)
                    ),
                ),
                limit=recall_limit,
            )
        )
    round_ids = tuple(
        dict.fromkeys(
            summary.split(" ", 1)[0]
            for summary in (item.summary for item in recalled.memories)
            if ":" in summary.split(" ", 1)[0]
        )
    )
    return MemGalleryQuestionResult(
        question_id=question.question_id,
        topic=topic.topic,
        point=question.point,
        question=question.question,
        reference_answer=question.reference_answer,
        prediction=recalled.answer or "",
        clue_round_ids=question.clue_round_ids,
        mindbridge_confidence=recalled.confidence,
        mindbridge_memory_ids=tuple(item.memory_id for item in recalled.memories),
        mindbridge_round_ids=round_ids,
        mindbridge_media_object_ids=tuple(item.media_object_id for item in recalled.evidence),
        mindbridge_trace_id=recalled.trace_id,
        retrieved_clue_round_count=len(set(question.clue_round_ids) & set(round_ids)),
        mindbridge_ingest_failure_count=ingest_failure_count,
    )


def _question_query(topic: MemGalleryTopic, question: MemGalleryQuestion) -> str:
    """Reproduce the official query, including the speaker framing it names.

    Upstream resolves `speaker_a` to `user (<persona name>)` and `speaker_b` to
    `assistant`, so which persona the dialogue belongs to is part of what the model is
    asked. The adapter already carries that name, so dropping the clause would tell the
    model less than the benchmark tells its own baselines.
    """
    return MEM_GALLERY_QUERY_PROMPT.text.format(
        speaker_a=f"user ({topic.profile.name})",
        speaker_b="assistant",
        question=question.question,
        format_constraint=mem_gallery_format_constraint(question.point),
    )
```

The round ID is recovered from the memory summary's first token because `RememberRequest`
has no external-ID field. Writing it as the summary's first token is what makes the
release's `clue` annotation measurable at all; keep the two in step.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/benchmarks/test_mem_gallery_runner.py -v`
Expected: 5 passed

- [ ] **Step 5: Mutation-check the tenancy, the query image, and the constraint switch**

Change `benchmark_tenant_id(tenant_prefix, topic.topic, run_id)` to use
`question.question_id`, confirm
`test_rounds_are_written_per_speaker_and_images_observed_with_their_round_text` fails,
restore. Drop `media_object_ids` from the recall query, confirm
`test_a_question_image_is_sent_as_a_recall_query_object` fails, restore. Make
`mem_gallery_format_constraint` return the search constraint for every point, confirm
`test_official_constraints_are_applied_only_to_ar_cd_and_vs` fails, restore.

- [ ] **Step 6: Run the gates and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest -W error && git diff --check
git add src/mindbridge/benchmarks/mem_gallery_runner.py tests/unit/benchmarks/test_mem_gallery_runner.py
git commit -m "Add Mem-Gallery runner with image-as-query recall"
```

---

### Task 8: Mem-Gallery CLI

**Files:**

- Create: `src/mindbridge/benchmarks/mem_gallery_cli.py`
- Create: `tests/unit/benchmarks/test_mem_gallery_cli.py`
- Modify: `src/mindbridge/benchmarks/cli.py` (the `RUNNERS` table)

**Interfaces:**

- Consumes: Task 7's runner surface, Task 2's `load_mem_gallery`, Task 4's prompts.
- Produces: `main(argv, *, prog)`, `MEM_GALLERY_RUNNER_VERSION =
  "mem_gallery_production_api_v1"`, `class MemGalleryRunManifest(MediaBenchmarkRunManifest)`
  with `benchmark: Literal["Mem-Gallery"]`, `dataset_repository`,
  `evaluator_repository`, `prepared_images_manifest_sha256`, `perception_prompt_version`,
  `query_prompt_version`, `topics`, `question_ids`, `session_count`, `round_count`,
  `image_reference_count`, `question_image_count`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/benchmarks/test_mem_gallery_cli.py`:

```python
"""Argument and artifact checks for the Mem-Gallery CLI."""

import json
from pathlib import Path

import pytest

from mindbridge.benchmarks import mem_gallery_cli
from tests.unit.benchmarks.benchmark_deployment import write_deployment_snapshot


def _write_dialog_directory(root: Path) -> Path:
    directory = root / "dialog"
    directory.mkdir()
    (directory / "Baking.json").write_text(
        json.dumps(
            {
                "character_profile": {
                    "name": "Maya",
                    "persona_summary": "A librarian who bakes.",
                    "traits": ["curious"],
                    "conversation_style": "Earnest.",
                },
                "multi_session_dialogues": [
                    {
                        "session_id": "D1",
                        "date": "2024-06-24",
                        "dialogues": [
                            {
                                "round": "D1:1",
                                "user": "Basics?",
                                "assistant": "A 30 litre oven.",
                            }
                        ],
                    }
                ],
                "human-annotated QAs": [
                    {
                        "point": "FR",
                        "question": "What oven size?",
                        "answer": "30 litres.",
                        "session_id": ["D1"],
                        "clue": ["D1:1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_run_requires_a_prepared_images_manifest(tmp_path: Path) -> None:
    directory = _write_dialog_directory(tmp_path)
    deployment = write_deployment_snapshot(tmp_path)

    with pytest.raises(SystemExit):
        mem_gallery_cli.main(
            [
                "--dataset",
                str(directory),
                "--output",
                str(tmp_path / "predictions.json"),
                "--api-base-url",
                "http://localhost:8000",
                "--deployment-config",
                str(deployment),
                "--run-id",
                "run1",
            ],
            prog="mindbridge-bench mem-gallery",
        )


def test_unknown_topic_selection_is_refused(tmp_path: Path) -> None:
    directory = _write_dialog_directory(tmp_path)
    deployment = write_deployment_snapshot(tmp_path)
    prepared = tmp_path / "prepared.json"
    prepared.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_key": "../image/Baking/D1_IMG_001.jpg",
                        "media_object": {
                            "media_object_id": "D1:IMG_001",
                            "kind": "image",
                            "uri": "s3://mindbridge-media/mem-gallery/D1_IMG_001.jpg",
                            "sha256": "b" * 64,
                            "size_bytes": 1,
                            "created_at": "2024-06-24T00:00:00Z",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown selected Mem-Gallery topics"):
        mem_gallery_cli.main(
            [
                "--dataset",
                str(directory),
                "--prepared-images",
                str(prepared),
                "--output",
                str(tmp_path / "predictions.json"),
                "--api-base-url",
                "http://localhost:8000",
                "--deployment-config",
                str(deployment),
                "--run-id",
                "run1",
                "--topic",
                "Nope",
            ],
            prog="mindbridge-bench mem-gallery",
        )


def test_cli_table_dispatches_mem_gallery() -> None:
    from mindbridge.benchmarks.cli import RUNNERS

    assert RUNNERS["mem-gallery"].module == "mindbridge.benchmarks.mem_gallery_cli"
    assert RUNNERS["mem-gallery"].extra is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/benchmarks/test_mem_gallery_cli.py -v`
Expected: collection error, `ImportError: cannot import name 'mem_gallery_cli'`

- [ ] **Step 3: Write the CLI**

Create `src/mindbridge/benchmarks/mem_gallery_cli.py` following the same structure as
`atm_cli.py`:

```python
"""Reproducible Mem-Gallery runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from mindbridge.benchmarks.artifacts import (
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
)
from mindbridge.benchmarks.cli_common import (
    MediaArguments,
    MediaBenchmarkRunManifest,
    add_media_arguments,
    connected_memory,
    core_parser,
    media_arguments,
    media_manifest,
    report,
    report_unit,
    select_by_id,
    write_run_artifacts,
)
from mindbridge.benchmarks.mem_gallery import (
    MEM_GALLERY_ADAPTER_VERSION,
    MemGalleryTopic,
    load_mem_gallery,
)
from mindbridge.benchmarks.mem_gallery_runner import (
    MemGalleryPreparedImages,
    MemGalleryQuestionResult,
    load_prepared_mem_gallery,
    run_mem_gallery_topic,
    validate_mem_gallery_images,
)
from mindbridge.benchmarks.prompts import MEM_GALLERY_QUERY_PROMPT
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT

MEM_GALLERY_RUNNER_VERSION = "mem_gallery_production_api_v1"


class MemGalleryRunManifest(MediaBenchmarkRunManifest):
    """Immutable source, protocol, deployment, model, and prediction identity."""

    benchmark: Literal["Mem-Gallery"] = "Mem-Gallery"
    dataset_repository: NonEmptyString
    evaluator_repository: NonEmptyString
    prepared_images_manifest_sha256: Sha256Hex
    perception_prompt_version: NonEmptyString
    query_prompt_version: NonEmptyString
    topics: tuple[Identifier, ...] = Field(min_length=1)
    question_ids: tuple[Identifier, ...] = Field(min_length=1)
    session_count: int = Field(gt=0)
    round_count: int = Field(gt=0)
    image_reference_count: int = Field(ge=0)
    question_image_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_images_path: Path
    topics: tuple[str, ...]


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run one tenant per topic over the official dialogue directory."""
    arguments = _parse_arguments(argv, prog)
    topics = select_by_id(
        load_mem_gallery(arguments.dataset_path),
        arguments.topics,
        key=lambda topic: topic.topic,
        label="selected Mem-Gallery topics",
    )
    if not topics:
        raise ValueError("Mem-Gallery selection must not be empty")
    prepared = load_prepared_mem_gallery(arguments.prepared_images_path)
    validate_mem_gallery_images(topics, prepared)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(arguments.deployment_config_path, require_worker=True)
    report(f"running {len(topics)} topics", quiet=arguments.quiet)
    results = asyncio.run(_run(arguments, topics, prepared))
    _write_artifacts(arguments, topics, results, deployment)
    report(f"wrote {arguments.output_path}", quiet=arguments.quiet)


async def _run(
    arguments: _Arguments,
    topics: tuple[MemGalleryTopic, ...],
    prepared: MemGalleryPreparedImages,
) -> tuple[MemGalleryQuestionResult, ...]:
    async with connected_memory(arguments) as memory:
        results: list[MemGalleryQuestionResult] = []
        for index, topic in enumerate(topics, start=1):
            report_unit(
                f"topic {topic.topic}", index=index, total=len(topics), quiet=arguments.quiet
            )
            results.extend(
                await run_mem_gallery_topic(
                    memory,
                    topic,
                    run_id=arguments.run_id,
                    prepared=prepared,
                    tenant_prefix=arguments.tenant_prefix,
                    device_id=arguments.device_id,
                    recall_limit=arguments.recall_limit,
                    request_concurrency=arguments.request_concurrency,
                    poll_interval_seconds=arguments.poll_interval_seconds,
                    processing_timeout_seconds=arguments.processing_timeout_seconds,
                )
            )
        return tuple(results)


def _write_artifacts(
    arguments: _Arguments,
    topics: tuple[MemGalleryTopic, ...],
    results: tuple[MemGalleryQuestionResult, ...],
    deployment: LoadedDeployment,
) -> None:
    expected = tuple(question.question_id for topic in topics for question in topic.questions)
    if tuple(result.question_id for result in results) != expected:
        raise ValueError("Mem-Gallery predictions must match annotation question order")
    # The official evaluator reads a list of per-question objects with `point` as category.
    predictions = (
        json.dumps(
            [
                {
                    "question_id": result.question_id,
                    "topic": result.topic,
                    "point": result.point,
                    "question": result.question,
                    "answer": result.reference_answer,
                    "prediction": result.prediction,
                    "clue": list(result.clue_round_ids),
                    "retrieved_ids": list(result.mindbridge_round_ids),
                    "retrieved_clue_round_count": result.retrieved_clue_round_count,
                    "retrieved_image_ids": list(result.mindbridge_media_object_ids),
                    "mindbridge_confidence": result.mindbridge_confidence,
                    "mindbridge_memory_ids": list(result.mindbridge_memory_ids),
                    "mindbridge_trace_id": result.mindbridge_trace_id,
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    manifest = media_manifest(
        MemGalleryRunManifest,
        arguments,
        deployment,
        runner_version=MEM_GALLERY_RUNNER_VERSION,
        adapter_version=MEM_GALLERY_ADAPTER_VERSION,
        annotation_sha256=_dialog_digest(arguments.dataset_path),
        predictions=predictions,
        dataset_repository="Ethan-Bei/Mem-Gallery",
        evaluator_repository="YuanchenBei/Mem-Gallery",
        prepared_images_manifest_sha256=sha256_file(arguments.prepared_images_path),
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        query_prompt_version=MEM_GALLERY_QUERY_PROMPT.version,
        topics=tuple(topic.topic for topic in topics),
        question_ids=expected,
        session_count=sum(len(topic.sessions) for topic in topics),
        round_count=sum(len(session.rounds) for topic in topics for session in topic.sessions),
        image_reference_count=sum(
            1
            for topic in topics
            for session in topic.sessions
            for round_ in session.rounds
            if round_.image_id is not None
        ),
        question_image_count=sum(
            1
            for topic in topics
            for question in topic.questions
            if question.question_image_path is not None
        ),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _dialog_digest(dialog_directory: Path) -> str:
    """Digest the concatenated digests of every topic file, in sorted order."""
    import hashlib

    joined = "".join(sha256_file(path) for path in sorted(dialog_directory.glob("*.json")))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = add_media_arguments(
        core_parser(tenant_prefix="benchmark_mem_gallery", prog=prog, description=__doc__),
        device_id="mem_gallery_conversation",
    )
    parser.add_argument(
        "--prepared-images",
        type=Path,
        required=True,
        help="manifest of staged dialogue and question images",
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        help="official topic to run; repeatable, default all twenty",
    )
    parsed = parser.parse_args(argv)
    return media_arguments(
        _Arguments,
        parsed,
        prepared_images_path=parsed.prepared_images,
        topics=tuple(parsed.topic),
    )


if __name__ == "__main__":
    main()
```

`--dataset` here is the `data/dialog` directory rather than a file; `core_parser` already
declares it as a `Path` and adds no `is_file` check, so this needs no change to the shared
parser. `_dialog_digest` pins all twenty files as one number.

Then add the row to `RUNNERS` in `src/mindbridge/benchmarks/cli.py`, after `"atm"`:

```python
    "mem-gallery": Runner("mindbridge.benchmarks.mem_gallery_cli", "Run official Mem-Gallery"),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/benchmarks/test_mem_gallery_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: Check both new commands appear in the table**

Run: `uv run mindbridge-bench --help`
Expected: the benchmark list contains `atm` and `mem-gallery` with their summaries.

- [ ] **Step 6: Run the gates and commit**

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest -W error && git diff --check
git add src/mindbridge/benchmarks/mem_gallery_cli.py src/mindbridge/benchmarks/cli.py tests/unit/benchmarks/test_mem_gallery_cli.py
git commit -m "Add the mindbridge-bench mem-gallery command"
```

---

### Task 9: Documentation

**Files:**

- Modify: `docs/benchmarking.md` (add two runner sections and update the adapter list)
- Modify: `docs/benchmarks-sota.md` (add sections 3.10 and 3.11)

**Interfaces:**

- Consumes: the CLI surfaces from Tasks 6 and 8, verbatim.
- Produces: no code.

- [ ] **Step 1: Add the ATM-Bench runner section to `docs/benchmarking.md`**

Place it after the MEMLENS section, matching that section's shape: what the benchmark is,
the staging manifest example, the run command, and how to score it. The manifest example:

```json
{
  "media": [
    {
      "media_id": "20250223_130249",
      "media_object": {
        "media_object_id": "20250223_130249",
        "kind": "image",
        "uri": "s3://mindbridge-media/atm-bench/20250223_130249.jpg",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "size_bytes": 100686,
        "created_at": "2025-02-23T13:02:49Z"
      }
    }
  ]
}
```

The run command:

```bash
uv run mindbridge-bench atm \
  --dataset .benchmarks/atm-bench/data/atm-bench/atm-bench-hard.json \
  --split hard \
  --media-source raw \
  --emails .benchmarks/atm-bench/data/raw_memory/email/emails.json \
  --prepared-media .benchmarks/atm-prepared-media.json \
  --sgm-image .benchmarks/atm-bench/data/processed_memory/image_batch_results.json \
  --sgm-video .benchmarks/atm-bench/data/processed_memory/video_batch_results.json \
  --output .benchmarks/results/atm-hard-raw.json \
  --api-base-url http://localhost:8000 \
  --deployment-config .benchmarks/deployment.json \
  --run-id atm-hard-raw-01
```

Document the two clocks and the arm distinction in prose: capture time comes from the
filename stem for every modality, and MindBridge's `raw` arm sends media through its own
perception rather than into an answerer's context, so the comparable published column is
ATM's SGM column, not its Raw column.

- [ ] **Step 2: Add the Mem-Gallery runner section to `docs/benchmarking.md`**

```bash
uv run mindbridge-bench mem-gallery \
  --dataset .benchmarks/mem-gallery/data/dialog \
  --prepared-images .benchmarks/mem-gallery-prepared-images.json \
  --topic Baking_Dessert_Daily_Life_Skill \
  --output .benchmarks/results/mem-gallery-baking.json \
  --api-base-url http://localhost:8000 \
  --deployment-config .benchmarks/deployment.json \
  --run-id mem-gallery-baking-01
```

State that `--dataset` is the directory, that one tenant is created per topic, that the
prepared-images manifest must cover both dialogue images and the 487 question images, and
that `media_object_id` must be the official `image_id` because `VS` answers name it.

- [ ] **Step 3: Add sections 3.10 and 3.11 to `docs/benchmarks-sota.md`**

Follow the existing per-benchmark table shape. The published baselines to record, all from
the two projects' own materials:

ATM-Bench memory systems, `Qwen3-VL-8B-Instruct-FP8` answerer, `gpt-5-mini` judge:

| System | ATM-Bench QS | ATM-Bench Recall@10 | ATM-Bench-Hard QS | Hard Recall@10 |
| --- | --- | --- | --- | --- |
| Memexa (DeepSeek-V4-flash judge, not gpt-5-mini) | 68.0 | 79.1 | 47.9 | 44.7 |
| MemPalace | 56.8 | 76.4 | 9.7 | 28.3 |
| ATM-RAG (paper's own) | 51.0 | 68.7 | 8.4 | 28.8 |
| MemoryOS | 47.2 | 59.2 | 13.7 | 32.7 |
| A-Mem | 44.8 | 66.4 | 9.9 | 31.7 |
| mem0 | 43.5 | 61.9 | 9.2 | 23.7 |
| HippoRAG2 | 42.9 | 66.4 | 9.4 | 31.9 |
| SimpleMem | 27.3 | 23.3 | 3.2 | 7.0 |

General-purpose coding agents on the 31-question hard split reach higher — 58.8% (GPT-5.6
Sol, medium) and 58.4% (Claude Opus 5, xhigh) — and the SGM Oracle ceiling for that split
is 60.5 (MiniMax-M3). Record the Memexa row's judge difference and the agent-versus-memory-
system distinction; they are not the same measurement.

Mem-Gallery, `Qwen2.5-VL-7B` backbone, 13 memory systems:

| System | F1 | LLM judge |
| --- | --- | --- |
| MuRAG (best multimodal) | 0.6966 | 0.8229 |
| UniversalRAG | 0.6827 | 0.8016 |
| A-Mem (best textual) | 0.6228 | 0.7431 |
| Full memory (text) | 0.3625 | — |
| Full memory (multimodal) | 0.3354 | — |

Per-task highs worth recording: MuRAG leads `VS` at 0.8818 F1 and `TTL` at 0.8177 F1, and
FIFO reaches 1.0000 F1 on `AR` — a reminder that a system which abstains freely wins that
task outright, so `AR` must be read next to the other eight.

Both sections state the evaluation-scope warning section 2 already establishes: MindBridge
runs these through its own write path, so a number is only comparable to a published one
when the answerer and judge are named.

- [ ] **Step 4: Run the documentation gates**

```bash
docker run --rm -v "$PWD:/workdir:ro" davidanson/markdownlint-cli2:v0.23.0 \
  "**/*.md" "!.git/**" "!.venv/**" "!.pytest_cache/**" "!.benchmarks/**"
docker run --rm -v "$PWD:/input:ro" -w /input lycheeverse/lychee:0.23.0 \
  --no-progress --root-dir /input './*.md' './docs/**/*.md'
uv run ruff format --check .
```

Expected: `0 error(s)` from markdownlint, no failed links, and ruff clean — it formats the
Python inside these fences too.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmarking.md docs/benchmarks-sota.md
git commit -m "Document the ATM-Bench and Mem-Gallery runners and their published baselines"
```

---

### Task 10: Subset runs against the live deployment

**Files:**

- Create: `.benchmarks/stage_media.py` (operator script, deliberately outside the wheel)
- Create: `.benchmarks/results/atm-hard-raw.json`, `.benchmarks/results/atm-hard-sgm.json`,
  `.benchmarks/results/mem-gallery-baking.json` (run artifacts)

**Interfaces:**

- Consumes: the two CLIs, the prepared-media manifest formats from Tasks 5 and 7.
- Produces: three prediction files with their manifests, and the numbers for the handoff
  summary.

`.benchmarks/` is gitignored; nothing in this task is committed. It exists to prove the
write paths run.

- [ ] **Step 1: Bring the deployment up and confirm it answers**

Follow `docs/deployment.md`. Confirm the API, the worker, PostgreSQL, and the object store
are all up, then:

```bash
curl -sf http://localhost:8000/health | head -20
```

Expected: a healthy response naming the configured plugins. Write the deployment snapshot
JSON the CLIs pin to `.benchmarks/deployment.json` — the shape is in `docs/benchmarking.md`.

- [ ] **Step 2: Write the staging script**

Create `.benchmarks/stage_media.py`. It uploads files to the object store and writes the
prepared-media manifest. It uses `boto3` directly because it is an operator script, not
part of the package — `benchmarks/` may only call the public SDK, and adding boto3 to the
`benchmarks` extra to re-implement `mc mirror` is out of scope by design.

```python
"""Stage benchmark media into the object store and emit a prepared-media manifest.

Operator script. Not part of the wheel: `benchmarks/` may only call the public SDK, and the
manifest formats it writes are inputs to `mindbridge-bench atm` and `... mem-gallery`.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import boto3

STEM = re.compile(r"^(\d{8})_(\d{6})$")
KIND_BY_SUFFIX = {".jpg": "image", ".jpeg": "image", ".png": "image", ".mp4": "video"}


def digest(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def capture_time(path: Path) -> str:
    matched = STEM.match(path.stem)
    if matched is None:
        return "1970-01-01T00:00:00Z"
    stamp = datetime.strptime(matched.group(1) + matched.group(2), "%Y%m%d%H%M%S")
    return stamp.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="directory of media to stage")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True, help="key prefix inside the bucket")
    parser.add_argument("--endpoint-url", default=None, help="set for MinIO")
    parser.add_argument("--output", type=Path, required=True, help="manifest to write")
    parser.add_argument(
        "--layout",
        choices=("atm", "mem-gallery"),
        required=True,
        help="which manifest shape to emit",
    )
    parser.add_argument("--only", action="append", default=[], help="stem to stage; repeatable")
    arguments = parser.parse_args()

    client = boto3.client("s3", endpoint_url=arguments.endpoint_url)
    wanted = set(arguments.only)
    entries = []
    for path in sorted(arguments.source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in KIND_BY_SUFFIX:
            continue
        if wanted and path.stem not in wanted:
            continue
        key = f"{arguments.prefix}/{path.relative_to(arguments.source)}"
        client.upload_file(str(path), arguments.bucket, key)
        checksum, size = digest(path)
        media_object = {
            "kind": KIND_BY_SUFFIX[path.suffix.lower()],
            "uri": f"s3://{arguments.bucket}/{key}",
            "sha256": checksum,
            "size_bytes": size,
            "created_at": capture_time(path),
        }
        if arguments.layout == "atm":
            media_object["media_object_id"] = path.stem
            entries.append({"media_id": path.stem, "media_object": media_object})
        else:
            # Mem-Gallery: `D1_IMG_001.jpg` under `<Topic>/` is image_id `D1:IMG_001`;
            # a `QA_IMG_*` file is a question image, keyed `<Topic>:<stem>`.
            topic = path.parent.name
            if path.stem.startswith("QA_"):
                media_object["media_object_id"] = f"{topic}:{path.stem}"
            else:
                media_object["media_object_id"] = path.stem.replace("_", ":", 1)
            entries.append(
                {
                    "image_key": f"../image/{topic}/{path.name}",
                    "media_object": media_object,
                }
            )
    payload = {"media": entries} if arguments.layout == "atm" else {"images": entries}
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"staged {len(entries)} objects into {arguments.output}")


if __name__ == "__main__":
    main()
```

Verify the Mem-Gallery ID derivation against the release before running it in bulk:

```bash
uv run python -c "
import json, glob
ids = set()
for f in glob.glob('/home/yons/thomas/MindBridge/.benchmarks/mem-gallery/data/dialog/*.json'):
    d = json.load(open(f))
    for s in d['multi_session_dialogues']:
        for r in s['dialogues']:
            for i, p in zip(r.get('image_id', []), r.get('input_image', [])):
                stem = p.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                assert stem.replace('_', ':', 1) == i, (stem, i)
                ids.add(i)
print('derivation holds for', len(ids), 'dialogue images')
"
```

Expected: `derivation holds for 1003 dialogue images`. If any assertion fires, fix the
derivation in the script before staging — a wrong `media_object_id` silently breaks `VS`
scoring and `clue` recall.

- [ ] **Step 3: Stage the ATM subset**

Pick the 31-question hard split — it is the split every published agent number uses, and it
is small. Stage only the media it cites plus the NIAH-25 pool's distractors:

```bash
BENCH=/home/yons/thomas/MindBridge/.benchmarks
uv run python -c "
import json
from pathlib import Path
root = Path('$BENCH/atm-bench/data/atm-bench')
hard = json.loads((root / 'atm-bench-hard.json').read_text())
pool = json.loads((root / 'niah' / 'atm-bench-hard-niah25.json').read_text())
stems = {e for q in hard for e in q['evidence_ids'] if not e.startswith('email')}
stems |= {e for q in pool for e in q['niah_evidence_ids'] if not e.startswith('email')}
Path('$BENCH/atm-subset-stems.txt').write_text('\n'.join(sorted(stems)))
print(len(stems), 'media items to stage')
"
uv run python $BENCH/stage_media.py \
  --source $BENCH/atm-bench/data/raw_memory/image \
  --bucket mindbridge-media --prefix atm-bench/image \
  --endpoint-url http://localhost:9000 \
  --layout atm --output $BENCH/atm-prepared-image.json \
  $(sed 's/^/--only /' $BENCH/atm-subset-stems.txt | tr '\n' ' ')
```

Repeat for `data/raw_memory/video` into `--output $BENCH/atm-prepared-video.json`, then
merge the two `media` arrays into `$BENCH/atm-prepared-media.json`.

- [ ] **Step 4: Run the ATM raw arm**

```bash
uv run mindbridge-bench atm \
  --dataset $BENCH/atm-bench/data/atm-bench/atm-bench-hard.json \
  --split hard --media-source raw \
  --emails $BENCH/atm-bench/data/raw_memory/email/emails.json \
  --prepared-media $BENCH/atm-prepared-media.json \
  --output $BENCH/results/atm-hard-raw.json \
  --api-base-url http://localhost:8000 \
  --deployment-config $BENCH/deployment.json \
  --run-id atm-hard-raw-01 \
  --request-concurrency 4
```

Then assert the run actually wrote and actually retrieved — a run that answered from an
empty store is a failed run, not a low score:

```bash
uv run python -c "
import json
rows = json.load(open('$BENCH/results/atm-hard-raw.json'))
manifest = json.load(open('$BENCH/results/atm-hard-raw.json.manifest.json'))
assert manifest['media_source'] == 'raw' and manifest['split'] == 'hard'
assert manifest['ingest_failure_count'] == 0, manifest['ingest_failure_count']
assert all(r['mindbridge_memory_ids'] for r in rows), 'a question retrieved nothing'
print('questions', len(rows),
      'gold evidence retrieved', sum(r['retrieved_gold_evidence_count'] for r in rows),
      'of', sum(len(r['evidence_ids']) for r in rows))
"
```

- [ ] **Step 5: Run the ATM SGM arm**

```bash
uv run mindbridge-bench atm \
  --dataset $BENCH/atm-bench/data/atm-bench/atm-bench-hard.json \
  --split hard --media-source sgm \
  --emails $BENCH/atm-bench/data/raw_memory/email/emails.json \
  --sgm-image $BENCH/atm-bench/data/processed_memory/image_batch_results.json \
  --sgm-video $BENCH/atm-bench/data/processed_memory/video_batch_results.json \
  --output $BENCH/results/atm-hard-sgm.json \
  --api-base-url http://localhost:8000 \
  --deployment-config $BENCH/deployment.json \
  --run-id atm-hard-sgm-01 \
  --request-concurrency 4
```

Run the same assertion script against `atm-hard-sgm.json` and its
`atm-hard-sgm.json.manifest.json` sidecar, with
`manifest['media_source'] == 'sgm'`. This arm writes all 4,292 SGM blocks and all 6,742
emails, so expect it to take longer than the raw subset even though it uses no GPU.

- [ ] **Step 6: Stage and run one Mem-Gallery topic**

```bash
uv run python $BENCH/stage_media.py \
  --source $BENCH/mem-gallery/data/image/Baking_Dessert_Daily_Life_Skill \
  --bucket mindbridge-media --prefix mem-gallery/Baking_Dessert_Daily_Life_Skill \
  --endpoint-url http://localhost:9000 \
  --layout mem-gallery --output $BENCH/mem-gallery-prepared-images.json
```

The staging script keys images by `../image/<Topic>/<file>`, so run it from a directory
whose `parent.name` is the topic — the command above satisfies that.

```bash
uv run mindbridge-bench mem-gallery \
  --dataset $BENCH/mem-gallery/data/dialog \
  --prepared-images $BENCH/mem-gallery-prepared-images.json \
  --topic Baking_Dessert_Daily_Life_Skill \
  --output $BENCH/results/mem-gallery-baking.json \
  --api-base-url http://localhost:8000 \
  --deployment-config $BENCH/deployment.json \
  --run-id mem-gallery-baking-01 \
  --request-concurrency 4
```

Assert the same three things, plus that the image-as-query path ran:

```bash
uv run python -c "
import json
rows = json.load(open('$BENCH/results/mem-gallery-baking.json'))
manifest = json.load(open('$BENCH/results/mem-gallery-baking.json.manifest.json'))
assert manifest['topics'] == ['Baking_Dessert_Daily_Life_Skill']
assert manifest['round_count'] == 262 and manifest['image_reference_count'] == 57
assert all(r['mindbridge_memory_ids'] for r in rows), 'a question retrieved nothing'
vs = [r for r in rows if r['point'] == 'VS']
assert any(r['retrieved_image_ids'] for r in vs), 'no VS question retrieved an image'
print('questions', len(rows),
      'clue rounds retrieved', sum(r['retrieved_clue_round_count'] for r in rows),
      'of', sum(len(r['clue']) for r in rows))
"
```

The counts 262 rounds and 57 images are this topic's measured values from the release.

- [ ] **Step 7: Report, do not score**

Write the three runs' numbers into the handoff summary: questions answered, ingest failure
count, gold-evidence or clue-round retrieval, and wall clock per run. Do not compute or
quote an accuracy — QS, EM, F1, and the judge scores come from the official scorers, and
`mindbridge-bench score` is how their verdict gets recorded beside these predictions.

Nothing in this task is committed; `.benchmarks/` is gitignored.

---

## Self-Review

**Spec coverage.** Every section of the design has a task: the two adapters (Tasks 1, 2), the
verified release counts as a smoke gate (Task 3), the official wordings (Task 4), both
runners with the ATM arm switch and Mem-Gallery's image-as-query (Tasks 5, 7), both CLIs and
the `RUNNERS` rows (Tasks 6, 8), the docs including the published baselines (Task 9), and the
subset runs with the write-path proof the success criteria demand (Task 10). The spec's
"one clock" rule is implemented in Task 1 and asserted by
`test_atm_sgm_adapter_takes_capture_time_from_the_stem_not_the_timestamp`; its
"no answer-quality metric" rule shows up as the absence of any scoring code and is stated
again in Task 10 Step 7. Out-of-scope items stay out: NIAH pools are loaded and validated
(Task 1) but no run mode consumes them, and the staging script lives in `.benchmarks/`.

**Placeholders.** None. Every code step carries the code; every run step carries the command
and its expected output. Two steps say "if mypy rejects X, do Y" — those are real
alternatives with both branches specified, not deferred decisions.

**Type consistency.** `AtmMediaSource` is defined in the runner (Task 5) and imported by the
CLI (Task 6). `MemGalleryPoint` is defined in the adapter (Task 2) and reused by the runner's
result model (Task 7). `atm_memory_chunks(block, evidence_id)` keeps that argument order in
Task 1's test, Task 1's implementation, and Task 5's `_remember_blocks`.
`validate_prepared_atm(questions, prepared, *, media_source)` and
`validate_mem_gallery_images(topics, prepared)` are called with those exact shapes in their
CLIs. `run_mem_gallery_topic` returns a tuple of results per topic and the CLI extends a flat
list from it, which matches the manifest's `question_ids` built over the same nesting.
