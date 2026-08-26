"""Producing prepared media for the three benchmarks whose clips sit on an absolute clock.

EgoLifeQA, EgoMemReason, and SuperMemory-VQA differ from every other producer in one way: their
media is not positioned relative to the start of its own file. EgoLife's clips carry a `DAYn`
plus `HHMMSSFF` timecode on a seven-day wearer clock, and SuperMemory's segments hang off each
recording's own Unix start. So a producer here cannot number segments -- it has to place them.

EgoLifeQA and EgoMemReason share that clock, that release, and that shape: `load_prepared_egomem`
returns a tuple of the same `EgoLifePreparedStream` `load_prepared_egolife` returns one of. They
are two selections over one stream builder here rather than two derivations of one clock, because
a second derivation of `(day - 1) * 86400` plus a 20 FPS frame divisor is exactly the kind of
plausible-but-wrong duplication that reads fine and scores nothing.

Three things are forced rather than chosen.

**The clock comes from the adapter.** `egolife_qa.egolife_timecode_offset_ms` is what the runner
orders and withholds clips by, so it is what the timecodes written here have to agree with. This
module never converts a timecode itself.

**Only what the run will ingest.** Both runners are causal: a clip whose end crosses the question
time is withheld, and a clip after the last selected question is never read at all. That bound is
what makes `--limit` affordable, because it is also the bound on the download -- A1_JAKE alone is
about 103 GB of 30-second clips, and one SuperMemory participant is about 51 GB.

**Segments split where the questions do.** `run_supermemory_vqa` refuses a manifest before
ingesting anything unless every selected question's end is exactly a segment boundary, which is
the official protocol: a run may see the span the question was asked in and nothing after it.
So the split here is the 30-second grid unioned with those boundaries, not the grid alone.

Every path a release is expected to occupy is one named constant below, and each says what it was
derived from. None of them could be checked against a corpus.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from mindbridge.benchmarks.cli_common import report, select_by_id
from mindbridge.benchmarks.egolife_qa import egolife_timecode_offset_ms
from mindbridge.benchmarks.egolife_runner import EgoLifePreparedClip, EgoLifePreparedStream
from mindbridge.benchmarks.releases import ensure_media
from mindbridge.benchmarks.runtime import benchmark_tenant_id
from mindbridge.benchmarks.staging import (
    SEGMENT_SECONDS,
    PrepareRequest,
    Staging,
    key_component,
    media_duration_ms,
    staging,
    within,
    write_manifest,
)
from mindbridge.benchmarks.supermemory_runner import (
    SuperMemoryPreparedSegment,
    SuperMemoryPreparedSubject,
    SuperMemoryPreparedVideo,
)
from mindbridge.contracts import IdentityObservationInput, MediaObjectInput
from mindbridge.core import IdentityKind, MediaKind

EGOLIFE_TIMELINE_ORIGIN = datetime(2000, 1, 1, tzinfo=timezone.utc)
"""The instant `DAY1 00:00:00:00` stands for.

EgoLife's clock is relative to its own first day, so which instant day one begins at is arbitrary
and only has to be the same instant twice; `docs/benchmarking.md` shows this one. Fixed rather
than the wall clock for the same reason `STAGED_AT` is: a run manifest pins the prepared
manifest's digest, and two preparations of one stream have to agree.
"""

EGOLIFE_MEDIA = "egolife"
"""The media set holding every EgoLife clip, for both EgoLifeQA and EgoMemReason.

Unverified against a corpus. `releases.py` documents the layout as
`egolife/<A#_NAME>/DAY<n>/DAY<n>_<A#_NAME>_<HHMMSSFF>.mp4`, and the day directory is the finest
grain `ensure_media`'s `only` can name: the clips are a 30-second grid with recording gaps in it,
so which file names exist cannot be derived without listing the directory.
"""

EGOLIFE_PUBLIC_SUBJECT = "A1_JAKE"
"""The only wearer EgoLifeQA publishes questions for -- all 500 of them.

`EgoLifeQuestion` carries a day and a timecode but no subject, and `EgoLifePreparedStream` holds
exactly one, so the subject cannot be read off the selection and is not a flag on the runner. The
catalog's dataset path, `egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json`, is the release's only
question file. EgoMemReason, on the same media, names its wearer in every question instead.
"""

SUPERMEMORY_VIDEO = ("supermemory-vqa", "data", "video")
"""Where one SuperMemory-VQA recording sits, under `Person_<subject>/<video_id>.mp4`.

Unverified against a corpus; `releases.py` documents this layout and `docs/benchmarking.md`
fetches the media with `--include 'data/video/*'`. The directory capitalises `Person` while the
transcript sidecars beside it do not, which is why the two are spelled out separately.

82 of the release's 83 sessions have an MP4. The missing one is VRS-only, and a recording with no
file here is prepared from its transcript alone rather than failing the run.
"""

SUPERMEMORY_TRANSCRIPT = ("supermemory-vqa", "data", "transcripts")
"""Where one recording's aligned transcript sits.

`person_<subject>/<video_id lowercased>_gemini_aligned_transcript.json`, per the release's own
`data/transcripts/README.md`: Gemini text and person labels aligned onto Whisper's timings. The
`_whisper_transcript.json` beside it is the raw ASR, whose diarisation labels that README says are
*not* the person labels the questions use, so it is deliberately not read.
"""

SUPERMEMORY_TRANSCRIPT_MODEL_ID = "supermemory_vqa_gemini_aligned_transcript"
"""Provenance for the voice spans below, which are a released annotation and not an edge model."""

_ALIGNMENT_CONFIDENCE = {"high": 1.0, "medium": 0.6, "low": 0.3}
"""The release's three timing-confidence labels as the number an identity span carries.

`confidence` is the detector's confidence in the *span*, and the transcript README says its own
`alignment_confidence` is "confidence in the timing assignment, not in the transcript text" --
the same quantity. A flat 1.0 would claim the lines whose timing was evenly distributed across a
caption segment are as well placed as the ones that matched Whisper exactly.
"""

_TRANSCRIPT_CHARACTERS = 2_048
"""`NonEmptyString`'s own ceiling, applied here so a talkative segment truncates rather than
failing a preparation that has already uploaded gigabytes."""

_MAXIMUM_IDENTITY_SPANS = 512
"""What one observation may carry, per `IdentityObservationInput`'s own field constraint."""


def prepare_egolife(request: PrepareRequest) -> None:
    """Stage the A1_JAKE clips every selected question can causally reach, and nothing later."""
    from mindbridge.benchmarks.egolife_cli import _parse_arguments
    from mindbridge.benchmarks.egolife_qa import load_egolife_qa

    arguments = _parse_arguments(list(request.argv), None)
    questions = select_by_id(
        load_egolife_qa(arguments.dataset_path),
        arguments.question_ids,
        key=lambda question: question.question_id,
        label="EgoLifeQA question IDs",
        limit=arguments.limit,
    )
    if not questions:
        raise ValueError("EgoLifeQA selection must not be empty")
    stream = _egolife_stream(
        request,
        staging(),
        subject_id=EGOLIFE_PUBLIC_SUBJECT,
        horizon_ms=max(question.query_offset_ms for question in questions),
        tenant_id=benchmark_tenant_id(
            arguments.tenant_prefix, EGOLIFE_PUBLIC_SUBJECT, arguments.run_id
        ),
    )
    write_manifest(arguments.prepared_media_path, stream)


def prepare_egomem(request: PrepareRequest) -> None:
    """Stage one EgoLife stream per selected identity, each cut off at its own last question.

    Per identity rather than one horizon for all of them: `run_egomem_reason` is called once per
    identity with only that identity's questions, and refuses a stream belonging to another, so
    media past a wearer's own last query time is an upload nothing reads. The identities keep the
    order the runner takes them in, which is the order the selection first names them.
    """
    from mindbridge.benchmarks.egomem_cli import _parse_arguments
    from mindbridge.benchmarks.egomem_reason import load_egomem_reason

    arguments = _parse_arguments(list(request.argv), None)
    questions = select_by_id(
        load_egomem_reason(arguments.dataset_path),
        arguments.example_ids,
        key=lambda question: question.example_id,
        label="EgoMemReason example IDs",
        limit=arguments.limit,
    )
    if not questions:
        raise ValueError("EgoMemReason selection must not be empty")
    horizons: dict[str, int] = {}
    for question in questions:
        horizons[question.identity] = max(
            horizons.get(question.identity, 0), question.query_offset_ms
        )
    target = staging()
    streams = [
        _egolife_stream(
            request,
            target,
            subject_id=identity,
            horizon_ms=horizon_ms,
            tenant_id=benchmark_tenant_id(arguments.tenant_prefix, identity, arguments.run_id),
        )
        for identity, horizon_ms in horizons.items()
    ]
    write_manifest(arguments.prepared_media_path, streams)


def _egolife_stream(
    request: PrepareRequest,
    target: Staging,
    *,
    subject_id: str,
    horizon_ms: int,
    tenant_id: str,
) -> EgoLifePreparedStream:
    """Stage one wearer's released clips up to `horizon_ms` on the seven-day EgoLife clock.

    The release is already cut into the 30-second clips both shapes want, each named by the
    timecode it starts at, so nothing here re-encodes: the official bytes are staged as they are.
    That keeps the AAC track the release ships, which is the whole audio channel for a corpus
    whose questions are half conversational, and which a re-encode is the documented way to lose.
    """
    subject = key_component(subject_id, label="EgoLife identity")
    clips = [
        EgoLifePreparedClip(
            day=source.day,
            start_timecode=source.timecode,
            media_object=target.stage(
                tenant_id=tenant_id,
                key=f"egolife/{subject}/day{source.day}/{source.timecode}.mp4",
                content=source.path.read_bytes(),
                kind=MediaKind.VIDEO,
                media_object_id=f"egolife_{subject}_day{source.day}_{source.timecode}",
                duration_ms=duration_ms,
            ),
        )
        for source, duration_ms in _egolife_durations(
            _egolife_sources(request, subject, horizon_ms), horizon_ms
        )
    ]
    report(f"  {subject}: {len(clips)} clips -> {tenant_id}", quiet=request.quiet)
    return EgoLifePreparedStream(
        subject_id=subject,
        timeline_origin=EGOLIFE_TIMELINE_ORIGIN,
        clips=tuple(clips),
    )


@dataclass(frozen=True, slots=True)
class _EgoLifeSource:
    """One released clip file, placed on the wearer's own clock by its name alone."""

    day: int
    timecode: str
    start_ms: int
    path: Path


def _egolife_sources(
    request: PrepareRequest,
    subject: str,
    horizon_ms: int,
) -> tuple[_EgoLifeSource, ...]:
    """Every clip of this wearer that starts before the horizon, in chronological order.

    Fetched a day at a time because that is the finest grain the release can be narrowed to, and
    which days those are is arithmetic on the same clock the timecodes use rather than a listing.
    """
    days = range(1, horizon_ms // 86_400_000 + 2)
    # Eager, and deliberately not behind `if not <source>.exists():` the way the producers of
    # fetchable-or-not media are. That form exists only because `ensure_media` refuses an
    # `UNOBTAINABLE` set before it looks at the filesystem, so an eager call there fails a
    # correctly hand-placed corpus; neither release here is one. Eager buys two things instead:
    # a day here is a 30-second grid *with recording gaps*, so which file names exist is only
    # knowable after the fetch and per-day `is_dir()` would skip a half-fetched day and silently
    # shorten the stream -- a wrong score with no error -- and the Hub client's ETag comparison
    # repairs a truncated clip that `is_dir()` cannot see. It costs one request.
    ensure_media(
        EGOLIFE_MEDIA,
        root=request.benchmarks_root,
        only=tuple(f"{subject}/DAY{day}/*.mp4" for day in days),
        announce=None if request.quiet else partial(report, quiet=False),
        download=request.download,
    )
    sources: list[_EgoLifeSource] = []
    for day in days:
        directory = within(request.benchmarks_root, EGOLIFE_MEDIA, subject, f"DAY{day}")
        if not directory.is_dir():
            raise FileNotFoundError(
                f"EgoLife day {directory} is absent; it is part of the lmms-lab/EgoLife release, "
                "which holds about 15 GB per wearer per day"
            )
        prefix = f"DAY{day}_{subject}_"
        for path in directory.glob(f"{prefix}*.mp4"):
            timecode = path.stem[len(prefix) :]
            start_ms = egolife_timecode_offset_ms(day, timecode)
            if start_ms < horizon_ms:
                sources.append(_EgoLifeSource(day, timecode, start_ms, path))
    if not sources:
        raise FileNotFoundError(
            f"no EgoLife clip for {subject} starts before {horizon_ms} ms on its own clock, so "
            "the release on disk cannot answer even the earliest selected question"
        )
    return tuple(sorted(sources, key=lambda source: source.start_ms))


def _egolife_durations(
    sources: Sequence[_EgoLifeSource],
    horizon_ms: int,
) -> Iterator[tuple[_EgoLifeSource, int]]:
    """Pair each clip with the duration its manifest entry may declare.

    A container runs past the grid the release names its files on: a clip named `11100000` with
    `11103000` after it decodes to 30040 ms, because the last frame's own duration is inside it.
    Declaring that measured length makes consecutive clips overlap, which the stream rejects
    outright, so a clip with a successor is declared no longer than the gap to it. Under-declaring
    shortens an evidence span by a frame; over-declaring loses the whole manifest.

    Then the causal bound: `run_egolife_qa` withholds a clip whose end crosses the query time and
    never reaches one after it, so a clip ending past the horizon is an upload nothing reads.
    """
    for index, source in enumerate(sources):
        duration_ms = media_duration_ms(source.path)
        if index + 1 < len(sources):
            duration_ms = min(duration_ms, sources[index + 1].start_ms - source.start_ms)
        if duration_ms > 0 and source.start_ms + duration_ms <= horizon_ms:
            yield source, duration_ms


def prepare_supermemory(request: PrepareRequest) -> None:
    """Cut one participant's recordings where the official protocol says a run may look.

    Every recording of the participant is prepared, not only the ones questions were asked in:
    the answers live across the whole timeline and a run is scoped to one subject. What bounds
    the work is the last selected question -- a segment ending after it is never ingested, and a
    recording starting after it is skipped whole, which is what keeps `--limit` affordable on a
    participant whose media is about 51 GB.
    """
    from mindbridge.benchmarks.supermemory_cli import _parse_arguments, _select_questions
    from mindbridge.benchmarks.supermemory_vqa import load_supermemory_vqa

    arguments = _parse_arguments(list(request.argv), None)
    questions = _select_questions(
        load_supermemory_vqa(arguments.dataset_path),
        arguments.subject,
        arguments.question_ids,
        arguments.limit,
    )
    horizon = max(question.question_ended_at for question in questions)
    started = _supermemory_video_starts(arguments.dataset_path, arguments.subject)
    wanted = sorted(
        (pair for pair in started.items() if pair[1] < horizon),
        key=lambda pair: (pair[1], pair[0]),
    )
    if not wanted:
        raise ValueError(
            f"no SuperMemory-VQA recording of subject {arguments.subject} starts before "
            f"{horizon.isoformat()}, which is when its last selected question ends"
        )
    # Eager, like `_egolife_sources` and unlike the producers whose media may be `UNOBTAINABLE`:
    # those call on absence only because `ensure_media` refuses an unfetchable set before it looks
    # at the filesystem, which would fail a correctly hand-placed corpus. This release is
    # fetchable, and calling every run is what lets the Hub client's ETag comparison repair a
    # recording truncated by an interrupted transfer. On absence that file is present, so it is
    # never repaired, `media_duration_ms` under-reports it, and the manifest comes out quietly
    # short -- which is a wrong score rather than a failure.
    ensure_media(
        SUPERMEMORY_VIDEO[0],
        root=request.benchmarks_root,
        only=tuple(
            f"data/video/Person_{arguments.subject}/{video_id}.mp4" for video_id, _ in wanted
        ),
        announce=None if request.quiet else partial(report, quiet=False),
        download=request.download,
    )
    tenant_id = benchmark_tenant_id(
        arguments.tenant_prefix, str(arguments.subject), arguments.run_id
    )
    target = staging()
    videos: list[SuperMemoryPreparedVideo] = []
    for video_id, started_at in wanted:
        segments = _supermemory_segments(
            request,
            target,
            subject=arguments.subject,
            video_id=video_id,
            tenant_id=tenant_id,
            boundaries_ms=tuple(
                sorted(
                    {
                        _elapsed_ms(started_at, question.question_ended_at)
                        for question in questions
                        if question.question_video_id == video_id
                        and question.question_ended_at > started_at
                    }
                )
            ),
            horizon_ms=_elapsed_ms(started_at, horizon),
        )
        if not segments:
            continue
        report(f"  {video_id}: {len(segments)} segments -> {tenant_id}", quiet=request.quiet)
        videos.append(
            SuperMemoryPreparedVideo(
                video_id=video_id,
                started_at=started_at,
                segments=tuple(segments),
            )
        )
    write_manifest(
        arguments.prepared_media_path,
        SuperMemoryPreparedSubject(subject=arguments.subject, videos=tuple(videos)),
    )


def _supermemory_segments(
    request: PrepareRequest,
    target: Staging,
    *,
    subject: int,
    video_id: str,
    tenant_id: str,
    boundaries_ms: Sequence[int],
    horizon_ms: int,
) -> list[SuperMemoryPreparedSegment]:
    """Cut one recording on the 30-second grid, forced to break at every question end.

    A segment carries whatever of the two channels reaches it. The released MP4s have no audio
    track at all -- the participants' raw audio is withheld, and one session has no MP4 either --
    so the aligned transcript is the only speech this benchmark has, and it is attached twice: as
    the segment's own text, and as the timed voice spans that tell perception who was speaking
    when. Without the second, perception is handed silent video and told to name people only when
    a name is seen or heard, so on a corpus whose questions are about what B said it can never
    name anyone.

    A grid instant carrying neither is dropped rather than emitted, which merges it into the
    segment after it. A question end carrying neither is kept, so the manifest fails where the
    protocol is unsatisfiable instead of quietly moving the boundary a run is checked against.
    """
    key = key_component(video_id, label="SuperMemory-VQA video ID")
    source = within(request.benchmarks_root, *SUPERMEMORY_VIDEO, f"Person_{subject}", f"{key}.mp4")
    duration_ms = media_duration_ms(source) if source.exists() else 0
    lines = _supermemory_lines(request.benchmarks_root, subject, key)
    spoken_ms = max((round(line.end * 1_000) for line in lines), default=0)
    end_limit = min(
        horizon_ms, max(duration_ms, spoken_ms, boundaries_ms[-1] if boundaries_ms else 0)
    )
    content = source.read_bytes() if min(duration_ms, end_limit) > 0 else b""
    segments: list[SuperMemoryPreparedSegment] = []
    start_ms = 0
    for end_ms in _supermemory_cuts(boundaries_ms, end_limit):
        media_ms = max(0, min(end_ms, duration_ms) - start_ms)
        transcript = _supermemory_text(lines, start_ms, end_ms)
        if media_ms <= 0 and transcript is None and end_ms not in boundaries_ms:
            continue
        media_objects: tuple[MediaObjectInput, ...] = ()
        if media_ms > 0:
            media_objects = (
                target.stage(
                    tenant_id=tenant_id,
                    key=f"supermemory/{key}/{start_ms}.mp4",
                    content=_supermemory_clip(content, start_ms, start_ms + media_ms),
                    kind=MediaKind.VIDEO,
                    media_object_id=f"supermemory_{key}_{start_ms}",
                    duration_ms=media_ms,
                ),
            )
        segments.append(
            SuperMemoryPreparedSegment(
                start_seconds=start_ms / 1_000,
                duration_ms=end_ms - start_ms,
                media_objects=media_objects,
                transcript=transcript,
                identity_observations=_supermemory_identities(lines, start_ms, media_ms),
            )
        )
        start_ms = end_ms
    return segments


def _supermemory_cuts(boundaries_ms: Sequence[int], end_limit: int) -> tuple[int, ...]:
    """Every instant a segment may end at, in order: the grid, the question ends, and the tail.

    All of them are whole seconds -- the grid by construction, the question ends because the
    release measures its spans in them -- which is what keeps `started_at + start_seconds +
    duration_ms` land exactly on the `question_ended_at` the runner compares it against.
    """
    span_ms = SEGMENT_SECONDS * 1_000
    wanted = {
        *range(span_ms, end_limit, span_ms),
        *(bound for bound in boundaries_ms if 0 < bound <= end_limit),
        end_limit,
    }
    return tuple(sorted(bound for bound in wanted if bound > 0))


def _supermemory_clip(content: bytes, start_ms: int, end_ms: int) -> bytes:
    """Cut one segment with the encoder the product stores its own evidence with.

    `end_ms - 1` for the reason `staging.video_segments` documents: `cut_clips` keeps the frame at
    the end of its span, which is right for evidence and wrong for a split, and left closed every
    segment would share its last second with the next one.
    """
    from mindbridge.media.clipping import ClipRequest, cut_clips

    clips = cut_clips(
        content,
        ClipRequest(kind=MediaKind.VIDEO, start_ms=start_ms, end_ms=end_ms - 1),
    )
    return clips[0].content


class _TranscriptLine(BaseModel):
    """One line of the release's aligned transcript, as `data/transcripts/README.md` shapes it."""

    model_config = ConfigDict(extra="ignore")

    start: float
    end: float
    person: str | None = None
    text: str
    kind: str = "speech"
    alignment_confidence: str = "low"


class _AlignedTranscript(BaseModel):
    """The file around those lines; its `metadata` is diagnostic and nothing here reads it."""

    model_config = ConfigDict(extra="ignore")

    transcript: list[_TranscriptLine] = Field(default_factory=list)


def _supermemory_lines(root: Path, subject: int, video_id: str) -> tuple[_TranscriptLine, ...]:
    """Read one recording's aligned transcript, or nothing where the release has none for it."""
    path = within(
        root,
        *SUPERMEMORY_TRANSCRIPT,
        f"person_{subject}",
        f"{video_id.lower()}_gemini_aligned_transcript.json",
    )
    if not path.exists():
        return ()
    lines = _AlignedTranscript.model_validate_json(path.read_bytes()).transcript
    return tuple(line for line in lines if line.text.strip() and line.end >= line.start >= 0)


def _supermemory_text(
    lines: Sequence[_TranscriptLine],
    start_ms: int,
    stop_ms: int,
) -> str | None:
    """The segment's speech and sound events, attributed the way the release attributes them."""
    spoken = [
        f"{line.person}: {line.text.strip()}" if line.person else line.text.strip()
        for line in lines
        if _overlaps(line, start_ms, stop_ms)
    ]
    if not spoken:
        return None
    return "\n".join(spoken)[:_TRANSCRIPT_CHARACTERS]


def _supermemory_identities(
    lines: Sequence[_TranscriptLine],
    start_ms: int,
    media_ms: int,
) -> tuple[IdentityObservationInput, ...]:
    """The segment's attributed speech as timed voice spans over its own media.

    Bounded by the media rather than by the segment because that is what the shape requires: a
    span may not end after the media it points into, and media is shorter than its segment
    wherever a question boundary runs past the container's tail. Sound events and unattributed
    lines are left to the segment's text -- an identity span has to belong to somebody.
    """
    if media_ms <= 0:
        return ()
    spans: dict[tuple[str, int, int], IdentityObservationInput] = {}
    for line in lines:
        if line.kind != "speech" or not line.person:
            continue
        if not _overlaps(line, start_ms, start_ms + media_ms):
            continue
        begin = max(0, round(line.start * 1_000) - start_ms)
        finish = min(media_ms, round(line.end * 1_000) - start_ms)
        identity = (line.person, begin, finish)
        if identity in spans:
            continue
        spans[identity] = IdentityObservationInput(
            identity_id=line.person,
            kind=IdentityKind.VOICE,
            start_ms=begin,
            end_ms=finish,
            confidence=_ALIGNMENT_CONFIDENCE.get(line.alignment_confidence, 0.3),
            model_id=SUPERMEMORY_TRANSCRIPT_MODEL_ID,
            transcript=line.text.strip()[:_TRANSCRIPT_CHARACTERS],
        )
    return tuple(spans.values())[:_MAXIMUM_IDENTITY_SPANS]


def _overlaps(line: _TranscriptLine, start_ms: int, stop_ms: int) -> bool:
    """Whether a transcript line reaches into `[start_ms, stop_ms)` of its own recording."""
    return round(line.start * 1_000) < stop_ms and round(line.end * 1_000) > start_ms


def _supermemory_video_starts(dataset_path: Path, subject: int) -> dict[str, datetime]:
    """Every recording of one participant, and the Unix instant the release starts it at.

    Read from the annotation's raw shape because `SuperMemoryQuestion` deliberately does not keep
    it: the adapter turns a question's span into one absolute `question_ended_at` and drops the
    origin it was measured from, while a prepared video has to declare that origin. Every
    recording a participant's questions name is the question video of at least one of them, so
    nothing selectable is left unplaceable.
    """
    from mindbridge.benchmarks.supermemory_vqa import _RawQuestion

    starts: dict[str, datetime] = {}

    def record(video_id: str, unix_seconds: float) -> None:
        moment = datetime.fromtimestamp(unix_seconds, tz=timezone.utc)
        if starts.setdefault(video_id, moment) != moment:
            raise ValueError(
                f"SuperMemory-VQA gives {video_id} two start times, "
                f"{starts[video_id].isoformat()} and {moment.isoformat()}"
            )

    for raw in TypeAdapter(list[_RawQuestion]).validate_json(dataset_path.read_bytes()):
        if raw.subject != subject:
            continue
        evidence = raw.question_evidence
        for span in evidence.time_spans or ():
            record(span.video_id, float(span.video_start_time_unix))
        if evidence.video_id is not None and evidence.start_time is not None:
            record(evidence.video_id, float(evidence.start_time))
    return starts


def _elapsed_ms(started_at: datetime, moment: datetime) -> int:
    """How far into a recording an absolute instant falls, in whole milliseconds."""
    return round((moment - started_at).total_seconds() * 1_000)
