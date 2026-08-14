"""Evidence input construction shared by generation pipelines."""

from __future__ import annotations

from collections.abc import Collection

from mindbridge.application.capabilities import (
    InputPart,
    MediaPart,
    TextPart,
)
from mindbridge.application.perception import ResolvedEvidence


def evidence_parts(
    evidence: tuple[ResolvedEvidence, ...],
    *,
    excluded_media_object_ids: Collection[str] = (),
) -> tuple[InputPart, ...]:
    """Label and attach each distinct source media object exactly once."""
    parts: list[InputPart] = []
    seen = set(excluded_media_object_ids)
    for item in evidence:
        media_object_id = item.media_object.media_object_id
        if media_object_id in seen:
            continue
        seen.add(media_object_id)
        parts.extend(
            (
                TextPart(f"Source media_object_id={media_object_id} follows."),
                MediaPart(
                    kind=item.media_object.kind,
                    url=item.media_url,
                    source_uri=item.media_object.uri,
                ),
            )
        )
    return tuple(parts)
