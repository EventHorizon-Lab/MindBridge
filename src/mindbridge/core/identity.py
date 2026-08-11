"""Privacy-preserving identity vocabulary shared by edge and cloud."""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from mindbridge.core._validation import require_aware_datetime, require_non_empty
from mindbridge.core.errors import DomainInvariantError
from mindbridge.core.identifiers import EventId, EvidenceId, TenantId
from mindbridge.core.memory import ModelReference


class IdentityKind(str, Enum):
    """Biometric modality used only to create a device-domain anonymous identity."""

    FACE = "face"
    VOICE = "voice"


@dataclass(frozen=True, slots=True)
class AnonymousIdentityObservation:
    """A time-bounded pseudonym with no biometric template or human label."""

    identity_id: str
    kind: IdentityKind
    start_ms: int
    end_ms: int
    confidence: float
    model_reference: ModelReference

    def __post_init__(self) -> None:
        require_non_empty(self.identity_id, "identity_id")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise DomainInvariantError("identity observation time range is invalid")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise DomainInvariantError("identity confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class IdentityMention:
    """An anonymous person occurrence grounded in one event and evidence span."""

    mention_id: str
    tenant_id: TenantId
    identity_id: str
    event_id: EventId
    evidence_id: EvidenceId
    confidence: float
    created_at: datetime

    def __post_init__(self) -> None:
        for value, name in (
            (self.mention_id, "mention_id"),
            (self.tenant_id, "tenant_id"),
            (self.identity_id, "identity_id"),
            (self.event_id, "event_id"),
            (self.evidence_id, "evidence_id"),
        ):
            require_non_empty(value, name)
        require_aware_datetime(self.created_at, "created_at")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise DomainInvariantError("identity mention confidence must be between 0 and 1")
