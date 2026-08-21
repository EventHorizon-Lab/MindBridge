"""Evidence input construction shared by generation pipelines."""

from __future__ import annotations

from collections.abc import Collection

from mindbridge.application.capabilities import (
    InputPart,
    MediaPart,
    TextPart,
)
from mindbridge.application.perception import ResolvedEvidence

# Measured against the deployment's endpoint (qwen3.8-27b, vLLM, 262k ctx) with the derived
# clips the write path stores: one clip costs ~1.65k prompt tokens, so 20 clips is ~33k tokens
# in 17 s. The wall being avoided is a 60 s gateway timeout in front of the model, which
# streaming does not dodge -- it fires whether or not `stream` is set. Attaching every distinct
# recalled object with no ceiling is what made a populated recall fail as `model_unavailable`
# regardless of `--recall-limit`. This ceiling is deliberately above a normal recall limit so it
# only ever truncates a pathological fan-out.
DEFAULT_MAX_EVIDENCE_MEDIA_PARTS = 24


def evidence_parts(
    evidence: tuple[ResolvedEvidence, ...],
    *,
    excluded_media_object_ids: Collection[str] = (),
    max_media_parts: int = DEFAULT_MAX_EVIDENCE_MEDIA_PARTS,
) -> tuple[InputPart, ...]:
    """Label and attach each distinct set of evidence bytes at most once.

    Deduplication keys on the attached URL, not on the source media object id: once a span
    is signed to its own derived clip, two spans of one recording are two different clips,
    and keying on the source would drop all but the first. The source id still labels each
    part, because that is the identity a model is asked to cite.
    """
    if max_media_parts < 0:
        raise ValueError("max_media_parts must not be negative")
    parts: list[InputPart] = []
    excluded = set(excluded_media_object_ids)
    seen_urls: set[str] = set()
    for item in evidence:
        media_object_id = item.media_object.media_object_id
        if media_object_id in excluded or item.media_url in seen_urls:
            continue
        if len(seen_urls) >= max_media_parts:
            break
        seen_urls.add(item.media_url)
        parts.extend(
            (
                TextPart(f"Source media_object_id={media_object_id} follows."),
                MediaPart(
                    kind=item.media_object.kind,
                    url=item.media_url,
                    source_uri=item.media_object.uri,
                    frames_per_second=item.sampled_frames_per_second,
                    max_pixels=item.sampled_max_pixels,
                ),
            )
        )
    return tuple(parts)
