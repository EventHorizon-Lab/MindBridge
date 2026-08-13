"""Privacy-preserving identity vocabulary shared by edge and cloud."""

import math
from dataclasses import dataclass
from enum import Enum

from mindbridge.core._validation import require_non_empty
from mindbridge.core.errors import DomainInvariantError
from mindbridge.core.memory import ModelReference


class IdentityKind(str, Enum):
    """Biometric modality used only to create a device-domain anonymous identity."""

    FACE = "face"
    VOICE = "voice"


class IdentityScope(str, Enum):
    """How long an anonymous identity is safe to reuse."""

    DEVICE = "device"
    OBSERVATION = "observation"


@dataclass(frozen=True, slots=True)
class AnonymousIdentityObservation:
    """A time-bounded pseudonym with no biometric template or human label."""

    identity_id: str
    kind: IdentityKind
    start_ms: int
    end_ms: int
    confidence: float
    model_reference: ModelReference
    scope: IdentityScope = IdentityScope.DEVICE
    transcript: str | None = None
    visual_bbox_xyxy: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.identity_id, "identity_id")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise DomainInvariantError("identity observation time range is invalid")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise DomainInvariantError("identity confidence must be between 0 and 1")
        if self.transcript is not None:
            if self.kind is not IdentityKind.VOICE:
                raise DomainInvariantError("identity transcript requires a voice identity")
            if not self.transcript.strip() or self.transcript != self.transcript.strip():
                raise DomainInvariantError("identity transcript must not be blank or padded")
        if self.visual_bbox_xyxy is not None:
            if self.kind is not IdentityKind.FACE:
                raise DomainInvariantError("visual bounding boxes require a face identity")
            left, top, right, bottom = self.visual_bbox_xyxy
            if (
                not all(
                    math.isfinite(value) and 0.0 <= value <= 1.0 for value in self.visual_bbox_xyxy
                )
                or right <= left
                or bottom <= top
            ):
                raise DomainInvariantError("visual bounding box must be normalized xyxy")
