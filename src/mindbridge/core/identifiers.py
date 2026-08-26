"""Strongly named and deterministic identifiers for MindBridge records."""

import hashlib
import json
from typing import NewType

TenantId = NewType("TenantId", str)
TombstoneId = NewType("TombstoneId", str)
DeviceId = NewType("DeviceId", str)
MediaObjectId = NewType("MediaObjectId", str)
ObservationId = NewType("ObservationId", str)
EvidenceId = NewType("EvidenceId", str)
FeedbackId = NewType("FeedbackId", str)
EventId = NewType("EventId", str)
EntityId = NewType("EntityId", str)
MentionId = NewType("MentionId", str)
ClaimId = NewType("ClaimId", str)
RelationId = NewType("RelationId", str)
EmbeddingId = NewType("EmbeddingId", str)
MemoryId = NewType("MemoryId", str)
JobId = NewType("JobId", str)


def derive_stable_id(prefix: str, *components: object) -> str:
    """Derive one compact retry-stable ID from canonical source identity."""
    if not prefix.strip():
        raise ValueError("prefix must not be empty")
    canonical = json.dumps(components, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:26]
    return f"{prefix}_{digest}"


EMBEDDING_ID_RECIPE_VERSION = 2
"""Which recipe `derive_embedding_id` currently implements, stored beside every vector.

Version 1 omitted `space_id`. That made an `embedding_id` ambiguous about the only thing the
table needs it to distinguish: the vectors table is keyed
`PRIMARY KEY (tenant_id, embedding_id)`, so re-embedding one object into a second search space
derived the same ID and collided, and the second vector could not be written at all. Version 2
hashes `space_id` in every recipe.

Stored rather than inferred. `_adopt_embedding_id` re-keys a row whose ID an older recipe
produced, and it has to know which rows those are: the previous answer read a migration's
`applied_at` and compared it against `EmbeddingRecord.created_at`, which is supplied by the
caller -- `kernel.py` passes the memory's own creation time -- so a replayed or backfilled record
could claim an amnesty it had no right to, and every future recipe change needed another
hard-coded migration number. A version the writer records is a fact about the write.
"""


def derive_embedding_id(
    *identity: object,
    model_id: str,
    space_id: str,
    task: str,
) -> EmbeddingId:
    """Derive the ID for one object's vector in one search space.

    `identity` is what distinguishes the object being embedded, and it differs by caller: most
    pass an object type and an object id, the summary path passes a memory id that already
    encodes its type, and an evidence clip passes the span plus the ordinal of the clip cut from
    it. What must never differ is the tail, so `model_id`, `space_id` and `task` are
    keyword-only and required -- omitting `space_id` was a silent wrong answer five call sites
    made independently, and a hand-written argument list is what let each of them.

    Bump `EMBEDDING_ID_RECIPE_VERSION` when this changes, or stored rows become unreachable
    from the same inputs with nothing able to say so.
    """
    return EmbeddingId(derive_stable_id("embedding", *identity, model_id, space_id, task))


def derive_observation_id(
    tenant_id: str,
    device_id: str,
    boot_id: str,
    sequence: int,
) -> ObservationId:
    """Derive the one cloud/edge identity for a captured device sequence."""
    return ObservationId(derive_stable_id("observation", tenant_id, device_id, boot_id, sequence))
