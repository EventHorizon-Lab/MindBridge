"""Evidence input construction shared by generation pipelines."""

from __future__ import annotations

from collections.abc import Collection

from mindbridge.application.capabilities import (
    InputPart,
    MediaPart,
    TextPart,
)
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.media.clipping import AUDIO_WINDOW_MS

# Measured against the deployment's endpoint (qwen3.8-27b, vLLM, 262k ctx) with the derived
# clips the write path stores: one clip costs ~1.65k prompt tokens, so 20 clips is ~33k tokens
# in 17 s. The wall being avoided is a 60 s gateway timeout in front of the model, which
# streaming does not dodge -- it fires whether or not `stream` is set. Attaching every distinct
# recalled object with no ceiling is what made a populated recall fail as `model_unavailable`
# regardless of `--recall-limit`. This ceiling is deliberately above a normal recall limit so it
# only ever truncates a pathological fan-out.
DEFAULT_MAX_EVIDENCE_MEDIA_PARTS = 24

# A span's own bytes are the span, give or take the window an encoder rounds them up to: the write
# path cuts one clip per span, and a span shorter than one `AUDIO_WINDOW_MS` window is still stored
# as a whole window. What a span with no usable clip falls back to is the entire recording it was
# cut from, which can be hours -- the same 12.3k-token shape the ceiling above exists to keep out,
# arriving one span at a time and unaffected by any count. Three times the span, or one window,
# whichever is larger, separates the two: a widened clip is inside that and a recording is orders
# of magnitude past it. Raise it only for a deployment whose spans are routinely most of their
# source, and measure the prompt cost when you do.
MAX_ATTACHED_SPAN_RATIO = 3


def evidence_parts(
    evidence: tuple[ResolvedEvidence, ...],
    *,
    excluded_media_urls: Collection[str] = (),
    max_media_parts: int = DEFAULT_MAX_EVIDENCE_MEDIA_PARTS,
) -> tuple[InputPart, ...]:
    """Label and attach each distinct set of evidence bytes at most once.

    Exclusion and deduplication both key on the attached URL, not on the source media object id:
    once a span is signed to its own derived clip, two spans of one recording are two different
    clips, and keying on the source drops all but the first -- and drops every clip cut from an
    object the question itself was asked with. The source id still labels each part, because that
    is the identity a model is asked to cite.

    A span whose attached bytes are not about the span contributes its label and no bytes. That is
    the multi-window audio case: no single stored clip covers such a span, so it falls back to the
    whole recording, and one of those costs more than the entire page of clips around it.
    """
    if max_media_parts < 0:
        raise ValueError("max_media_parts must not be negative")
    parts: list[InputPart] = []
    excluded = set(excluded_media_urls)
    seen_urls: set[str] = set()
    for item in evidence:
        if item.media_url in excluded or item.media_url in seen_urls:
            continue
        if not _is_about_the_span(item):
            continue
        if len(seen_urls) >= max_media_parts:
            break
        seen_urls.add(item.media_url)
        parts.extend(
            (
                TextPart(f"Source media_object_id={item.media_object.media_object_id} follows."),
                MediaPart(
                    kind=item.media_object.kind,
                    url=item.media_url,
                    # Read as the container format of the bytes being sent, so it has to name
                    # them: a span signed to its `.wav` clip is not the `.m4a` it was cut from.
                    source_uri=(item.attached_media_object or item.media_object).uri,
                    frames_per_second=item.sampled_frames_per_second,
                    max_pixels=item.sampled_max_pixels,
                ),
            )
        )
    return tuple(parts)


def _is_about_the_span(item: ResolvedEvidence) -> bool:
    """Whether the attached bytes are the span rather than everything around it."""
    attached = item.attached_media_object
    if attached is None or attached.duration_ms is None:
        return True
    span_ms = item.evidence_span.end_ms - item.evidence_span.start_ms
    return attached.duration_ms <= max(span_ms * MAX_ATTACHED_SPAN_RATIO, AUDIO_WINDOW_MS)
