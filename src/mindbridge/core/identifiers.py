"""Strongly named identifiers shared by MindBridge domain records."""

from typing import NewType

TenantId = NewType("TenantId", str)
DeviceId = NewType("DeviceId", str)
MediaObjectId = NewType("MediaObjectId", str)
ObservationId = NewType("ObservationId", str)
EvidenceId = NewType("EvidenceId", str)
EventId = NewType("EventId", str)
ClaimId = NewType("ClaimId", str)
EmbeddingId = NewType("EmbeddingId", str)
MemoryId = NewType("MemoryId", str)
