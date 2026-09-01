"""Thin adapter for the official PersonaMem-v3 `backend/{persona_id}/` release.

One persona directory is one retrieval scope: five engagement logs plus a
calendar modification stream on the memory side, and `test.json` on the
question side. Every event and every query carries a Unix timestamp, and the
official protocol masks out any event at or after a query's moment -- so the
timestamps are load-bearing here, not decoration.

`profile.json` is deliberately not read as memory. It is the scorer-side
ground-truth persona, and the released README is explicit that it is "never
shown to the evaluated agent"; the fields a judge needs travel with each
query instead.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

PERSONAMEM_V3_ADAPTER_VERSION = "personamem_v3_official_v1"

# The five engagement logs, in the order their events are merged before the
# chronological sort. `calendar.json` is a modification stream with a
# different shape and is read separately.
PERSONAMEM_V3_APPS: tuple[tuple[str, str], ...] = (
    ("instagram.json", "Instagram"),
    ("facebook.json", "Facebook"),
    ("threads.json", "Threads"),
    ("chatbot.json", "Chatbot"),
    ("ai_studio.json", "AI_Studio"),
)
_CALENDAR_FILE = "calendar.json"
_TEST_FILE = "test.json"

# Task types whose headline is deterministic: the agent returns a ranking over
# a frozen candidate slate and the released instance names the target.
RANKING_TASK_TYPES = frozenset(
    {
        "personalized_recommendation",
        "hidden_persona_recommendation",
        "at_ai_directive_followup",
        "short_vs_long_term_lifecycle",
    }
)

# Scored by a cluster runner that threads each response into the next prompt.
# A harness that answers every question independently cannot reconstruct the
# cluster, so these rows are dropped rather than scored under a protocol that
# is not the official one.
CLUSTER_TASK_TYPES = frozenset(
    {
        "over_personalization_repetition_chatbot",
        "over_personalization_repetition_recsys",
    }
)


class PersonaMemTurn(ContractModel):
    """One turn of a chatbot, companion, or direct-message conversation."""

    role: NonEmptyString
    content: str


class PersonaMemEvent(ContractModel):
    """One engagement event or calendar modification, at its published moment."""

    event_id: Identifier
    app: NonEmptyString
    timestamp: int
    occurred_at: AwareDatetime
    formatted_timestamp: str = ""
    interaction_type: str = ""
    action_label: str = ""
    user_message: str = ""
    content_type: str = ""
    title: str = ""
    caption: str = ""
    media_description: str = ""
    audio_transcript: str = ""
    hashtags: tuple[str, ...] = ()
    conversation: tuple[PersonaMemTurn, ...] = ()
    location: str = ""
    # Calendar only: when the appointment is scheduled for, which is a
    # different moment from `occurred_at` -- that is when the calendar was
    # edited. `details` carries a removal reason or a rendered field diff.
    scheduled_start: AwareDatetime | None = None
    scheduled_end: AwareDatetime | None = None
    details: str = ""
    author: str = ""
    is_dm: bool = False
    is_ad: bool = False
    is_trending: bool = False


class PersonaMemQuery(ContractModel):
    """One benchmark query and the label material kept outside MindBridge."""

    query_id: Identifier
    persona_id: Identifier
    task_family: NonEmptyString
    task_type: NonEmptyString
    query_kind: NonEmptyString
    expected_behavior: NonEmptyString
    timestamp: int
    asked_at: AwareDatetime
    # A bare `str` guarded below: some queries carry a long scenario body.
    user_query: str
    prior_conversation: tuple[PersonaMemTurn, ...] = ()
    example_response: str = ""
    groundtruth_preference: str = ""
    distractor_preferences: tuple[str, ...] = ()
    rubric_tags: tuple[str, ...] = ()
    # Frozen slate for the ranking family. The release names the target
    # differently per task type -- `held_out_idx` (personalized /
    # hidden-persona recommendation), `positive_indices` (@ai directive
    # follow-up), `matching_indices` (lifecycle) -- and names the items that
    # must stay low either `hard_negative_idxs` or `carveout_indices`. They are
    # normalised here so one scorer can read every ranking task.
    candidates: tuple[Mapping[str, object], ...] = ()
    held_out_index: int | None = None
    positive_indexes: tuple[int, ...] = ()
    negative_indexes: tuple[int, ...] = ()
    # The published `instance_full` minus its slate and its build-time quality
    # checks. The official task-specific judges read their evidence straight
    # out of this record, so it is carried verbatim rather than re-modelled.
    judge_evidence: Mapping[str, object] = {}

    @model_validator(mode="after")
    def require_user_query(self) -> PersonaMemQuery:
        if not self.user_query.strip():
            raise ValueError("PersonaMem-v3 queries require a user query")
        return self


class PersonaMemPersona(ContractModel):
    """One persona directory: its whole timeline and its whole query set."""

    persona_id: Identifier
    events: tuple[PersonaMemEvent, ...] = Field(min_length=1)
    queries: tuple[PersonaMemQuery, ...] = Field(min_length=1)


class _RawInteraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    app: str | None = None
    action_label: str | None = None
    user_message: str | None = None


class _RawLocation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    city: str | None = None
    region: str | None = None
    country: str | None = None


class _RawContent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    caption: str | None = None
    overall_description: str | None = None
    audio_transcript: str | None = None


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str


class _RawDirectMessage(BaseModel):
    """A direct-message turn, which names its parts `sender`/`text` rather
    than the `role`/`content` every chat surface uses."""

    model_config = ConfigDict(extra="ignore")

    sender: str
    text: str


class _RawEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_object_id: str
    source_timestamp: int
    formatted_timestamp: str = ""
    source_hashtags: list[str] = Field(default_factory=list)
    source_interaction_type: str = ""
    interaction_format: _RawInteraction | None = None
    content_type: str | None = None
    content: _RawContent | None = None
    conversation: list[_RawTurn] = Field(default_factory=list)
    messages: list[_RawDirectMessage] = Field(default_factory=list)
    event_location: _RawLocation | None = None
    author_id: str | None = None
    relationship: str | None = None
    is_dm: bool = False
    is_ad: bool = False
    is_trending: bool = False


class _RawCalendarEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    type: str | None = None
    start_ts: int | None = None
    end_ts: int | None = None
    location: _RawLocation | None = None


class _RawCalendarModification(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mod_id: str
    ts: int
    formatted_timestamp: str = ""
    action: str
    entry: _RawCalendarEntry | None = None
    removal_reason: str | None = None
    # `updated` modifications carry no `entry` at all -- only the entry ID and
    # a per-field `{"from": ..., "to": ...}` diff.
    entry_id: str | None = None
    diff: dict[str, object] = Field(default_factory=dict)


class _RawCalendar(BaseModel):
    model_config = ConfigDict(extra="ignore")

    modifications: list[_RawCalendarModification] = Field(default_factory=list)


class _RawInstance(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidates: list[dict[str, object]] = Field(default_factory=list)
    held_out_idx: int | None = None
    positive_indices: list[int] = Field(default_factory=list)
    matching_indices: list[int] = Field(default_factory=list)
    hard_negative_idxs: list[int] = Field(default_factory=list)
    carveout_indices: list[int] = Field(default_factory=list)


class _RawPreference(BaseModel):
    model_config = ConfigDict(extra="ignore")

    persona_item: str | None = None


class _RawQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_id: str
    task_family: str
    task_type: str
    query_kind: str
    expected_behavior: str
    ts: int
    user_query: str
    prior_conversation: list[_RawTurn] | None = None
    example_response: str | None = None
    groundtruth_preference: str | None = None
    distractor_preferences: list[_RawPreference] = Field(default_factory=list)
    # Every task family publishes a list here except the sycophancy rows,
    # which publish the bare tag string `"sycophancy_resistance"`.
    rubric_tags: str | list[str] = Field(default_factory=list)
    instance_full: dict[str, object] | None = None


_EVENTS = TypeAdapter(list[_RawEvent])
_QUERIES = TypeAdapter(list[_RawQuery])


def load_personamem_v3(backend_root: Path) -> tuple[PersonaMemPersona, ...]:
    """Load every persona directory under `backend/`, in numeric order."""
    directories = sorted(
        (path for path in backend_root.iterdir() if (path / _TEST_FILE).is_file()),
        key=lambda path: (0, int(path.name), "") if path.name.isdigit() else (1, 0, path.name),
    )
    personas = tuple(persona for persona in map(_persona, directories) if persona is not None)
    if not personas:
        raise ValueError(f"PersonaMem-v3 backend contains no persona directories: {backend_root}")
    return personas


def _persona(directory: Path) -> PersonaMemPersona | None:
    queries = _queries(directory)
    if not queries:
        return None
    events = [
        _event(raw, app)
        for filename, app in PERSONAMEM_V3_APPS
        if (directory / filename).is_file()
        for raw in _EVENTS.validate_json((directory / filename).read_bytes())
    ]
    events.extend(_calendar_events(directory / _CALENDAR_FILE))
    if not events:
        raise ValueError(f"PersonaMem-v3 persona {directory.name} has no engagement history")
    # Sorted by moment, then by ID, so that the causal cutoffs the runner
    # applies see one deterministic order regardless of app-file order.
    events.sort(key=lambda event: (event.timestamp, event.event_id))
    if len({event.event_id for event in events}) != len(events):
        raise ValueError(f"PersonaMem-v3 persona {directory.name} has duplicate event IDs")
    return PersonaMemPersona(persona_id=directory.name, events=tuple(events), queries=queries)


def _queries(directory: Path) -> tuple[PersonaMemQuery, ...]:
    raw_queries = _QUERIES.validate_json((directory / _TEST_FILE).read_bytes())
    queries = tuple(
        _query(directory.name, raw)
        for raw in raw_queries
        if raw.task_type not in CLUSTER_TASK_TYPES and raw.user_query.strip()
    )
    if len({query.query_id for query in queries}) != len(queries):
        raise ValueError(f"PersonaMem-v3 persona {directory.name} has duplicate query IDs")
    return queries


# Dropped from `judge_evidence`: the slate (rendered into the question
# instead) and the build-time quality checks the release ships alongside each
# row, none of which any judge reads.
_EVIDENCE_EXCLUDED = frozenset(
    {
        "candidates",
        "example_response_self_check",
        "example_response_voice_evidence",
        "holistic_test_quality_check",
        "inferior_response_voice_evidence",
        "voice_alignment_check",
        "voice_evidence_smoke_check",
        "voice_evidence_smoke_check_after_regen",
    }
)


def _query(persona_id: str, raw: _RawQuery) -> PersonaMemQuery:
    payload = raw.instance_full or {}
    instance = _RawInstance.model_validate(payload)
    return PersonaMemQuery(
        query_id=raw.query_id,
        persona_id=persona_id,
        task_family=raw.task_family,
        task_type=raw.task_type,
        query_kind=raw.query_kind,
        expected_behavior=raw.expected_behavior,
        timestamp=raw.ts,
        asked_at=_moment(raw.ts),
        user_query=raw.user_query,
        prior_conversation=tuple(
            PersonaMemTurn(role=turn.role, content=turn.content)
            for turn in raw.prior_conversation or ()
        ),
        example_response=(raw.example_response or "").strip(),
        groundtruth_preference=(raw.groundtruth_preference or "").strip(),
        distractor_preferences=tuple(
            preference.persona_item.strip()
            for preference in raw.distractor_preferences
            if preference.persona_item and preference.persona_item.strip()
        ),
        rubric_tags=_rubric_tags(raw.rubric_tags),
        candidates=tuple(instance.candidates),
        held_out_index=instance.held_out_idx,
        positive_indexes=_indexes(
            instance.held_out_idx,
            instance.positive_indices,
            instance.matching_indices,
        ),
        negative_indexes=_indexes(None, instance.hard_negative_idxs, instance.carveout_indices),
        judge_evidence={
            key: value for key, value in payload.items() if key not in _EVIDENCE_EXCLUDED
        },
    )


def _indexes(held_out: int | None, *groups: list[int]) -> tuple[int, ...]:
    values = ([held_out] if held_out is not None else []) + [
        value for group in groups for value in group
    ]
    return tuple(dict.fromkeys(values))


def _rubric_tags(value: str | list[str]) -> tuple[str, ...]:
    tags = (value,) if isinstance(value, str) else tuple(value)
    return tuple(tag.strip() for tag in tags if tag.strip())


def _event(raw: _RawEvent, app: str) -> PersonaMemEvent:
    content = raw.content or _RawContent()
    interaction = raw.interaction_format or _RawInteraction()
    location = raw.event_location
    return PersonaMemEvent(
        event_id=f"{app.casefold()}:{raw.source_object_id}",
        app=interaction.app or app,
        timestamp=raw.source_timestamp,
        occurred_at=_moment(raw.source_timestamp),
        formatted_timestamp=raw.formatted_timestamp,
        interaction_type=raw.source_interaction_type,
        action_label=interaction.action_label or "",
        user_message=interaction.user_message or "",
        content_type=raw.content_type or "",
        title=content.title or "",
        caption=content.caption or "",
        media_description=content.overall_description or "",
        audio_transcript=content.audio_transcript or "",
        hashtags=tuple(raw.source_hashtags),
        conversation=tuple(
            PersonaMemTurn(role=turn.role, content=turn.content) for turn in raw.conversation
        )
        or tuple(
            PersonaMemTurn(role=message.sender, content=message.text) for message in raw.messages
        ),
        location=_location(location),
        author=raw.relationship or raw.author_id or "",
        is_dm=raw.is_dm,
        is_ad=raw.is_ad,
        is_trending=raw.is_trending,
    )


def _calendar_events(path: Path) -> tuple[PersonaMemEvent, ...]:
    """Read the calendar as one event per modification.

    The release stores calendar state as an append-only `added` / `updated` /
    `removed` stream. Folding it into a snapshot would need a moment to fold
    to, and each query has a different one, so each modification is kept as its
    own timestamped memory and the fold falls out of the causal cutoff.

    Every part of an entry that describes the appointment is kept. The
    scheduled window is not derivable from the modification's own timestamp --
    an appointment can be booked well before it starts, and the entry's
    duration appears nowhere else -- and an `updated` modification carries only
    a field diff, so rendering just its action word would store a memory with
    no content in it.
    """
    if not path.is_file():
        return ()
    calendar = _RawCalendar.model_validate_json(path.read_bytes())
    return tuple(_calendar_event(modification) for modification in calendar.modifications)


def _calendar_event(modification: _RawCalendarModification) -> PersonaMemEvent:
    entry = modification.entry or _RawCalendarEntry()
    return PersonaMemEvent(
        event_id=f"calendar:{modification.mod_id}",
        app="Calendar",
        timestamp=modification.ts,
        occurred_at=_moment(modification.ts),
        formatted_timestamp=modification.formatted_timestamp,
        action_label=modification.action,
        title=entry.title or modification.entry_id or "",
        content_type=entry.type or "",
        location=_location(entry.location),
        scheduled_start=None if entry.start_ts is None else _moment(entry.start_ts),
        scheduled_end=None if entry.end_ts is None else _moment(entry.end_ts),
        details=_calendar_details(modification),
    )


def _calendar_details(modification: _RawCalendarModification) -> str:
    """Describe what a modification changed, in one line."""
    if modification.removal_reason:
        return modification.removal_reason
    changes = []
    for field, change in modification.diff.items():
        if isinstance(change, Mapping) and {"from", "to"} <= set(change):
            before, after = _diff_value(field, change["from"]), _diff_value(field, change["to"])
            changes.append(f"{field} {before} -> {after}")
        else:
            changes.append(f"{field} {_diff_value(field, change)}")
    return "; ".join(changes)


def _diff_value(field: str, value: object) -> str:
    """Render a diff value, resolving epoch fields to a readable moment."""
    if field.endswith("_ts") and isinstance(value, int) and not isinstance(value, bool):
        return _moment(value).isoformat()
    return "(none)" if value in ("", None) else str(value)


def _location(location: _RawLocation | None) -> str:
    if location is None:
        return ""
    parts = (location.city, location.region, location.country)
    return ", ".join(part for part in parts if part)


def _moment(timestamp: int) -> datetime:
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OSError, OverflowError, ValueError) as error:
        raise ValueError(f"invalid PersonaMem-v3 timestamp: {timestamp}") from error


def render_event(event: PersonaMemEvent) -> str:
    """Render one event as the text MindBridge remembers."""
    stamp = event.formatted_timestamp or event.occurred_at.isoformat()
    header = f"[{stamp}] {event.app}"
    if event.action_label:
        header = f"{header} · {event.action_label}"
    if event.interaction_type:
        header = f"{header} ({event.interaction_type})"
    lines = [header]
    for label, value in (
        # `content_type` is `image` / `short_video` / `text` on a feed event and
        # the appointment kind on a calendar one; it is parsed for every event,
        # so it is rendered for every event too.
        ("Type", event.content_type),
        ("Title", event.title),
        ("Caption", event.caption),
        ("Media", event.media_description),
        ("Audio", event.audio_transcript),
        ("User message", event.user_message),
        ("Location", event.location),
        ("Author", event.author),
        ("Details", event.details),
    ):
        if value:
            lines.append(f"{label}: {value}")
    if event.scheduled_start is not None:
        window = event.scheduled_start.isoformat()
        if event.scheduled_end is not None:
            window = f"{window} to {event.scheduled_end.isoformat()}"
        lines.append(f"Scheduled: {window}")
    if event.hashtags:
        lines.append("Hashtags: " + " ".join(event.hashtags))
    flags = tuple(
        name
        for name, present in (
            ("direct message", event.is_dm),
            ("ad", event.is_ad),
            ("trending", event.is_trending),
        )
        if present
    )
    if flags:
        lines.append("Flags: " + ", ".join(flags))
    lines.extend(f"{turn.role}: {turn.content}" for turn in event.conversation if turn.content)
    return "\n".join(lines)


# The only candidate fields a slate ever shows. This is a whitelist rather
# than a blocklist because the released slates carry scorer-side keys next to
# the presentable ones: `personalized_recommendation` candidates include
# `_held_out_persona_item` and `_held_out_category`, the identity of the very
# item the agent is being asked to find. Upstream strips its own `_origin`
# labels for the same reason before it builds a prompt. Anything not named
# here is not rendered, so a newly-added scorer-side key cannot leak by
# default -- but a key added *here* would, so add one only after checking what
# the release stores under it.
_VISIBLE_CANDIDATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("content_type", "type"),
    ("title", "title"),
    ("caption", "caption"),
    ("overall_description", "about"),
)


def render_candidates(candidates: Sequence[Mapping[str, object]]) -> str:
    """Render a frozen candidate slate in its published index order."""
    return "\n".join(
        f"- idx {index}: " + " | ".join(_candidate_fields(candidate))
        for index, candidate in enumerate(candidates)
    )


def _candidate_fields(candidate: Mapping[str, object]) -> tuple[str, ...]:
    app = _text(candidate, "source_app") or _text(candidate, "app")
    fields = [f"app={app}" if app else "app=?"]
    for key, label in _VISIBLE_CANDIDATE_FIELDS:
        value = _text(candidate, key)
        if value:
            fields.append(f"{label}={value}" if key == "content_type" else f"{label}={value!r}")
    fields.append(f"hashtags={_hashtags(candidate)}")
    return tuple(fields)


def _text(candidate: Mapping[str, object], key: str) -> str:
    value = candidate.get(key)
    return value if isinstance(value, str) else ""


def _hashtags(candidate: Mapping[str, object]) -> str:
    value = candidate.get("hashtags")
    if not isinstance(value, list):
        return "[]"
    return json.dumps([item for item in value if isinstance(item, str)], ensure_ascii=False)
