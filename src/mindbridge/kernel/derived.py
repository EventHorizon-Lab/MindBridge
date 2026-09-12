"""Text a model derived from media, kept separable from the caller's own content.

Transcripts, visual descriptions, distilled facts, and speech identities are appended behind
per-asset markers, so the caller's text stays byte-identical at the front of a record and every
reader can strip or select the derived sections without a second table.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.infrastructure.local.store import IndexDocument, StoredAsset, StoredMemory
from mindbridge.kernel.content import PreparedContent, PreparedMemory
from mindbridge.kernel.validation import MAX_TEXT_CHARACTERS, validated_text
from mindbridge.models.base import ModelInput
from mindbridge.types import FaceObservation, MemoryType, Modality, SpeakerSegment

# The one line prefix the describer's durable statements arrive under, inside the same string as
# the visible description. See `split_description`.
_FACT_LINE_PREFIX = "Fact:"


# The one fact shape that is not just indexed text but an assertion about a person: it binds a
# diarised speaker label to a name the dialogue itself stated. Tolerant of how the label is
# spelled, because the label is the model's echo of a prompt, and strict about the sentence,
# because anything looser would rename people out of ordinary description. The name itself is
# bounded to a handful of words rather than "everything up to a period": an unbounded run
# swallows whatever clause follows ("speaker_1 is called Lily and works in marketing" named the
# whole clause), and a name that long fails to match at all rather than binding the wrong text.
NAME_BINDING = re.compile(
    r"speakers?[ _-]?(?P<index>\d{1,3})\s+is\s+called\s+"
    r"(?P<name>[^\s.;,]+(?: [^\s.;,]+){0,3})\s*[.;]?",
    re.IGNORECASE,
)


# A describer is shown this write's own transcript as context, under the same labels the index
# prints (`speaker_prose`), so a fact naming a speaker resolves to the same person. Uncapped,
# one long clip's words would dominate the token budget every visual in it pays; cut at a line
# boundary so a turn is never split mid-sentence.
MAX_DESCRIBE_CONTEXT_CHARACTERS = 12_000


def prepared_from_stored(memory: StoredMemory) -> PreparedMemory:
    """Rebuild the prepared form of a captured row so one path enriches add and settle alike."""
    return PreparedMemory(
        memory_id=memory.memory_id,
        content=PreparedContent(
            text=memory.content,
            assets=memory.assets,
            modality=Modality(memory.modality),
            canonical_parts=stored_canonical_parts(memory.content, memory.assets),
            audio_transcript=has_stream_transcript(memory.content, memory.assets),
            visual_description=has_stream_description(memory.content, memory.assets),
        ),
        metadata_json=memory.metadata_json,
        occurred_at=memory.occurred_at,
        occurred_end=memory.occurred_end,
        memory_type=MemoryType(memory.memory_type),
        place_id=memory.place_id,
    )


def with_stream_transcript(
    prepared: PreparedContent,
    transcript: str,
) -> PreparedContent:
    audio = tuple(asset for asset in prepared.assets if asset.modality == Modality.AUDIO.value)
    if len(audio) != 1:
        raise ValidationError("stream transcript requires exactly one audio asset")
    text = validated_text(transcript, "stream transcript")
    section = f"[transcript:{audio[0].asset_id}]\n{text}"
    content = (
        prepared.text
        if section in prepared.text
        else "\n\n".join(value for value in (prepared.text, section) if value)
    )
    if len(content) > MAX_TEXT_CHARACTERS:
        raise ValidationError(f"content text must not exceed {MAX_TEXT_CHARACTERS} characters")
    return replace(
        prepared,
        text=content,
        canonical_parts=(*prepared.canonical_parts, ("audio_transcript", section)),
        audio_transcript=True,
    )


def with_stream_description(
    prepared: PreparedContent,
    description: str,
) -> PreparedContent:
    visual = tuple(
        asset
        for asset in prepared.assets
        if asset.modality in {Modality.IMAGE.value, Modality.VIDEO.value}
    )
    if len(visual) != 1:
        raise ValidationError("stream description requires exactly one visual asset")
    text = validated_text(description, "stream description")
    section = f"[visual description:{visual[0].asset_id}]\n{text}"
    content = (
        prepared.text
        if section in prepared.text
        else "\n\n".join(value for value in (prepared.text, section) if value)
    )
    if len(content) > MAX_TEXT_CHARACTERS:
        raise ValidationError(f"content text must not exceed {MAX_TEXT_CHARACTERS} characters")
    return replace(
        prepared,
        text=content,
        canonical_parts=(*prepared.canonical_parts, ("visual_description", section)),
        visual_description=True,
    )


def stored_canonical_parts(
    text: str,
    assets: Sequence[StoredAsset],
) -> tuple[tuple[str, str], ...]:
    """Split stored content back into the parts the strong `add` path embedded separately.

    A `StreamInput` folds its transcript and description into `content` before the row is
    written, so a capture arrives here already flattened. Recovering each derived section keys
    the same way `add()` did instead of hashing one merged text. Content with no marker for one
    of its own assets is left as the single text part the caller wrote.
    """
    cuts: dict[int, str] = {}
    for asset in assets:
        markers: tuple[tuple[str, str], ...]
        if asset.modality == Modality.AUDIO.value:
            markers = (("audio_transcript", f"[transcript:{asset.asset_id}]\n"),)
        elif asset.modality in {Modality.IMAGE.value, Modality.VIDEO.value}:
            # Both derived visual sections cut, and both under the kind that reaches the text
            # keys: a facts section folded into the description part would key differently on
            # the settle path than the same content did on the add path.
            markers = (
                ("visual_description", f"[visual description:{asset.asset_id}]\n"),
                ("visual_description", f"[facts:{asset.asset_id}]\n"),
            )
        else:
            continue
        for kind, marker in markers:
            found = text.find(f"\n\n{marker}")
            start = found + 2 if found >= 0 else (0 if text.startswith(marker) else -1)
            if start >= 0:
                cuts[start] = kind
    if not cuts:
        return (("text", text),) if text else ()
    ordered = sorted(cuts.items())
    # Each section was joined with "\n\n", so a cut ends two characters before the next one.
    ends = [start - 2 for start, _ in ordered[1:]] + [len(text)]
    head = text[: max(ordered[0][0] - 2, 0)]
    return (
        *((("text", head),) if head else ()),
        *((kind, text[start:end]) for (start, kind), end in zip(ordered, ends, strict=True)),
    )


def has_stream_transcript(text: str, assets: Sequence[StoredAsset]) -> bool:
    return any(
        asset.modality == Modality.AUDIO.value and f"[transcript:{asset.asset_id}]\n" in text
        for asset in assets
    )


def has_stream_description(text: str, assets: Sequence[StoredAsset]) -> bool:
    return any(
        asset.modality in {Modality.IMAGE.value, Modality.VIDEO.value}
        and (
            f"[visual description:{asset.asset_id}]\n" in text
            or f"[facts:{asset.asset_id}]\n" in text
        )
        for asset in assets
    )


_NO_NAMES: Mapping[str, str] = {}


def description_sections(
    asset_id: str, description: str, names: Mapping[str, str] = _NO_NAMES
) -> tuple[str, ...]:
    """Render one caption as the document sections it becomes, what-is-shown first.

    A caption whose visible half is empty still contributes its facts, and one with no facts is
    exactly the single section this produced before facts existed.

    `names` projects the facts half through whatever `speaker_N` -> name binding this write
    already knows, the same way transcript prose is projected (`speech_retrieval_text`) -- see
    `project_fact_labels` for why this projection runs once, here, and never again.
    """
    visible, facts = split_description(description)
    if facts and names:
        facts = project_fact_labels(facts, names)
    return (
        *((f"[visual description:{asset_id}]\n{visible}",) if visible else ()),
        *((f"[facts:{asset_id}]\n{facts}",) if facts else ()),
    )


def project_fact_labels(facts: str, names: Mapping[str, str]) -> str:
    """Replace a speaker label with the name already known for it, at write time only.

    Unlike `[speech identities:]`, which `_retrieval_text` re-projects from the current name on
    every index rebuild, `[facts:]` is rendered once, when the memory is written, from whatever
    names this write already knows -- its own newly staged ones and any identity already named
    before it. A name a *later* clip asserts does not retroactively rewrite an earlier one's
    stored facts; only the identity's own projected name (`identities()`, `[speech identities:]`)
    stays current for a person the store keeps renaming.
    """
    for label, name in names.items():
        facts = re.sub(rf"\b{re.escape(label)}\b", name, facts)
    return facts


def split_description(description: str) -> tuple[str, str]:
    """Separate a caption's visible description from its `Fact:` lines.

    The describer answers with one string per visual because that is the contract that survives
    a video arriving as several stills -- asked for a per-still or per-section reply, a measured
    endpoint returned one item per still and the whole batch was rejected. So the two halves
    travel as labelled lines in one string and are cut apart here, on the write path, which is
    also where the cache stores them as one row per asset.
    """
    lines = description.splitlines()
    facts = "\n".join(
        stripped
        for line in lines
        if line.startswith(_FACT_LINE_PREFIX)
        and (stripped := line[len(_FACT_LINE_PREFIX) :].strip())
    )
    visible = "\n".join(line for line in lines if not line.startswith(_FACT_LINE_PREFIX))
    return visible.strip(), facts


def derived_text(text: str, assets: Sequence[StoredAsset]) -> str:
    sections = [text] if text else []
    seen: set[str] = set()
    for asset in assets:
        # A transcript is only ever cached for an asset the transcriber declared, so its presence
        # is the routing decision; a second modality comparison here would discard video speech.
        transcript = asset.transcript
        if not transcript or asset.asset_id in seen:
            continue
        seen.add(asset.asset_id)
        # Modality-neutral on purpose. This text becomes `memory_records.content`, which is the
        # BM25 document, so every word in the marker is a term any query matches for free -- and a
        # lexical match alone clears `minimum_relevance`. Naming the modality here labelled video
        # speech "audio" and handed every video memory a free match on that word; deriving it from
        # `asset.modality` would only move the free match to the commoner word. The modality is
        # already published on the record's assets, so nothing is lost by leaving it out.
        marker = f"[transcript:{asset.asset_id}]"
        identity_marker = f"[speech identities:{asset.asset_id}]\n"
        if marker not in text and identity_marker not in text:
            sections.append(f"{marker}\n{transcript}")
    return "\n\n".join(sections)


def speech_identity_text(
    text: str,
    assets: Sequence[StoredAsset],
    segments_by_asset: Mapping[str, tuple[SpeakerSegment, ...]],
) -> str:
    sections = [text] if text else []
    for asset in dict.fromkeys(asset.asset_id for asset in assets):
        segments = segments_by_asset[asset]
        if not segments:
            continue
        sections.append(
            f"[speech identities:{asset}]\n"
            + json.dumps(
                speech_evidence(asset, segments), ensure_ascii=False, separators=(",", ":")
            )
        )
    return "\n\n".join(sections)


def speech_evidence(asset_id: str, segments: Sequence[SpeakerSegment]) -> dict[str, object]:
    """Shape one asset's speaker evidence: the form the document holds and the projection reads."""
    return {
        "asset_id": asset_id,
        "segments": [
            {
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "text": segment.text,
                "speaker_id": segment.speaker_id,
                "speaker_name": segment.speaker_name,
                "identity_score": segment.identity_score,
            }
            for segment in segments
        ],
    }


def truncated_describe_context(text: str) -> str:
    """Cap what a describer is shown of a clip's own words at a bounded character budget.

    Cut at a line boundary -- each line is one speaker's turn (see `speaker_prose`) -- so a
    truncated context still ends on a whole turn rather than a sentence sliced in half.
    """
    if len(text) <= MAX_DESCRIBE_CONTEXT_CHARACTERS:
        return text
    cut = text.rfind("\n", 0, MAX_DESCRIBE_CONTEXT_CHARACTERS)
    return text[:cut] if cut > 0 else text[:MAX_DESCRIBE_CONTEXT_CHARACTERS]


def speaker_prose(asset_id: str, segments: Sequence[SpeakerSegment]) -> str | None:
    """Render one asset's turns as the prose the index carries, or None when nothing was said.

    Routed through the stored evidence shape and `speech_retrieval_text` rather than formatted
    here, so the labels a describer is shown are byte-for-byte the labels the searchable document
    prints. Anything else and a fact naming `speaker_2` would name a different person than the
    transcript the reader sees.
    """
    if not segments:
        return None
    return speech_retrieval_text(
        json.dumps(speech_evidence(asset_id, segments), ensure_ascii=False), asset_id
    )


def speaker_labels(segments: Sequence[SpeakerSegment]) -> dict[str, str]:
    """Map each per-run identity ID to the stable `speaker_N` label the projection prints.

    Mirrors the aliasing rule inside `speech_retrieval_text`, which is what writes those labels
    into the document; a distilled fact naming `speaker_2` can only be resolved back to a person
    by the same rule, so a test pins the two against each other.
    """
    labels: dict[str, str] = {}
    for segment in segments:
        speaker_id = segment.speaker_id
        if speaker_id is not None and speaker_id.startswith("identity_"):
            labels.setdefault(speaker_id, f"speaker_{len(labels) + 1}")
    return labels


def face_identity_text(
    text: str,
    assets: Sequence[StoredAsset],
    observations_by_asset: Mapping[str, tuple[FaceObservation, ...]],
) -> str:
    sections = [text] if text else []
    for asset_id in dict.fromkeys(asset.asset_id for asset in assets):
        grouped: dict[str, list[FaceObservation]] = {}
        for observation in observations_by_asset[asset_id]:
            grouped.setdefault(observation.identity_id, []).append(observation)
        if not grouped:
            continue
        identities = []
        for identity_id, observations in grouped.items():
            times = tuple(
                observation.observed_at_ms
                for observation in observations
                if observation.observed_at_ms is not None
            )
            scores = tuple(
                observation.identity_score
                for observation in observations
                if observation.identity_score is not None
            )
            identities.append(
                {
                    "identity_id": identity_id,
                    "identity_name": observations[0].identity_name,
                    "first_observed_at_ms": min(times) if times else None,
                    "last_observed_at_ms": max(times) if times else None,
                    "observation_count": len(observations),
                    "representative_box": observations[0].bounding_box,
                    "max_identity_score": max(scores) if scores else None,
                }
            )
        evidence = {"asset_id": asset_id, "identities": identities}
        sections.append(
            f"[face identities:{asset_id}]\n"
            + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n\n".join(sections)


def has_indexed_speech(memory: StoredMemory) -> bool:
    """Report whether stored content carries a speech projection that names its speakers."""
    return any(
        f"[speech identities:{asset.asset_id}]\n" in memory.content
        for asset in memory.assets
        if asset.modality in {"audio", "video"}
    )


def without_speech_identities(text: str, asset_ids: Sequence[str]) -> str:
    markers = tuple(f"[speech identities:{asset_id}]\n" for asset_id in asset_ids)
    return "\n\n".join(section for section in text.split("\n\n") if not section.startswith(markers))


def retrieval_content(prepared: PreparedContent) -> PreparedContent:
    if "[speech identities:" not in prepared.text:
        return prepared
    asset_ids = frozenset(asset.asset_id for asset in prepared.assets)
    return replace(
        prepared,
        text=_retrieval_text(prepared.text, asset_ids),
        canonical_parts=tuple(
            (kind, _retrieval_text(value, asset_ids))
            if kind == "text" and "[speech identities:" in value
            else (kind, value)
            for kind, value in prepared.canonical_parts
        ),
    )


def retrieval_document(
    document: IndexDocument,
    asset_ids: frozenset[str],
) -> IndexDocument:
    if "[speech identities:" not in document.content:
        return document
    return replace(document, content=_retrieval_text(document.content, asset_ids))


def _retrieval_text(text: str, asset_ids: frozenset[str]) -> str:
    sections = []
    for section in text.split("\n\n"):
        marker, separator, payload = section.partition("\n")
        if separator and marker.startswith("[speech identities:") and marker.endswith("]"):
            asset_id = marker.removeprefix("[speech identities:").removesuffix("]")
            if asset_id in asset_ids:
                projected = speech_retrieval_text(payload, asset_id)
                if projected is not None:
                    sections.append(f"{marker}\n{projected}")
                    continue
        sections.append(section)
    return "\n\n".join(sections)


def speech_retrieval_text(payload: str, asset_id: str) -> str | None:
    """Project stored speech evidence into the prose the derived index and embedding should carry.

    Stored content keeps the JSON, because the answering model reads it as structured evidence and
    needs the timings and scores. The derived projections do not: as JSON, `start_ms`, `end_ms`,
    `speaker_id` and `identity_score` all became BM25 tokens surrounding the words someone
    actually said, and the embedder encoded the schema along with the speech. Index content is the
    only lever that moves the R@20 ceiling, so the projection carries the utterance and who said
    it and nothing else.

    An unstable per-run `identity_*` id is still replaced by a stable `speaker_N` alias, which is
    what this function existed for: without it a recognizer that re-minted a person rewrote every
    document that mentioned them. A named speaker is projected under their name, which is also the
    token a caller would search for.
    """
    try:
        evidence = json.loads(payload)
        if not isinstance(evidence, dict) or evidence.get("asset_id") != asset_id:
            return None
        segments = evidence["segments"]
        if not isinstance(segments, list):
            return None
        aliases: dict[str, str] = {}
        lines = []
        for segment in segments:
            if not isinstance(segment, dict) or "speaker_id" not in segment:
                return None
            speaker_id = segment.get("speaker_id")
            if isinstance(speaker_id, str) and speaker_id.startswith("identity_"):
                speaker_id = aliases.setdefault(speaker_id, f"speaker_{len(aliases) + 1}")
            name = segment.get("speaker_name")
            speaker = name if isinstance(name, str) and name.strip() else speaker_id
            spoken = segment.get("text")
            said = spoken.strip() if isinstance(spoken, str) else ""
            if not isinstance(speaker, str) or not speaker.strip():
                if said:
                    lines.append(said)
                continue
            lines.append(f"{speaker.strip()}: {said}" if said else speaker.strip())
        if not lines:
            # Nothing was said, so there is no prose to carry. Leaving the payload alone keeps an
            # empty analysis byte-identical to what it was before this projection existed.
            return payload
        return "\n".join(lines)
    except (KeyError, TypeError, ValueError):
        return None


def validated_descriptions(
    descriptions: object,
    inputs: Sequence[ModelInput],
) -> tuple[str, ...]:
    """Accept one non-empty, storable caption per input, in order, and nothing else."""
    if not isinstance(descriptions, tuple | list):
        raise ModelError("vision model returned invalid output", reason="response_invalid")
    values = tuple(descriptions)
    if len(values) != len(inputs) or any(
        not isinstance(description, str) or not description.strip() for description in values
    ):
        raise ModelError("vision model returned invalid output", reason="response_invalid")
    normalized = tuple(cast(str, description).strip() for description in values)
    if any(len(description) > MAX_TEXT_CHARACTERS for description in normalized):
        raise ModelError(
            "vision description exceeded the supported text length", reason="payload_too_large"
        )
    return normalized
