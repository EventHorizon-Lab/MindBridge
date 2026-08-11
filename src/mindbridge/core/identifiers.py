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
ClaimId = NewType("ClaimId", str)
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


def derive_observation_id(
    tenant_id: str,
    device_id: str,
    boot_id: str,
    sequence: int,
) -> ObservationId:
    """Derive the one cloud/edge identity for a captured device sequence."""
    return ObservationId(derive_stable_id("observation", tenant_id, device_id, boot_id, sequence))
