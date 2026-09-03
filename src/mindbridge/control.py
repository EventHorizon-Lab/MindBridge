"""Canonical serialization and idempotency identity for control-plane operations.

Nothing here touches storage or models. `dump_operation` and `load_operation` round-trip one
`MemoryOperation` through the append-only operation log so a logged operation can be replayed or
rolled back. `operation_key` derives the idempotency identity the log's unique index enforces; it
deliberately excludes `rationale`, which is logged but never interpreted, so two proposals that
differ only in their prose stay one operation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

from mindbridge.exceptions import StorageError
from mindbridge.types import (
    FormationProposal,
    IdentityClaim,
    MemoryOperation,
    SpatialContext,
)


def dump_operation(operation: MemoryOperation) -> str:
    """Return the canonical JSON the operation log stores for one operation."""
    return _json({**_identity(operation), "rationale": operation.rationale})


def load_operation(payload: str) -> MemoryOperation:
    """Rebuild one operation from a logged payload, validating it like model output."""
    try:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError
        proposal = value.get("proposal")
        claim = value.get("claim")
        return MemoryOperation(
            intent=cast(Any, value["intent"]),
            evidence_ids=tuple(value.get("evidence_ids") or ()),
            target_ids=tuple(value.get("target_ids") or ()),
            proposal=None if proposal is None else _proposal(proposal),
            claim=None if claim is None else _claim(claim),
            rationale=cast(Any, value.get("rationale")),
        )
    except Exception as error:
        raise StorageError("a logged memory operation could not be read") from error


def operation_key(operation: MemoryOperation, *, recipe: str | None) -> str:
    """Return the idempotency identity of one operation under one reasoning recipe."""
    payload = _json({**_identity(operation), "recipe": recipe})
    return hashlib.sha256(f"mindbridge-operation-v1:{payload}".encode()).hexdigest()


def _identity(operation: MemoryOperation) -> dict[str, object]:
    proposal = operation.proposal
    claim = operation.claim
    return {
        "intent": operation.intent.value,
        "evidence_ids": sorted(operation.evidence_ids),
        "target_ids": sorted(operation.target_ids),
        "proposal": None if proposal is None else _proposal_payload(proposal),
        # Part of the idempotency identity: renaming the same person supersedes rather than
        # replays, so two claims that differ only in the name must be two operations.
        "claim": None if claim is None else _claim_payload(claim),
    }


def _claim_payload(claim: IdentityClaim) -> dict[str, object]:
    return {
        "identity_id": claim.identity_id,
        "name": claim.name,
        "relationship": claim.relationship,
    }


def _claim(value: object) -> IdentityClaim:
    if not isinstance(value, dict):
        raise ValueError
    return IdentityClaim(
        identity_id=cast(Any, value["identity_id"]),
        name=cast(Any, value["name"]),
        relationship=cast(Any, value.get("relationship")),
    )


def _proposal_payload(proposal: FormationProposal) -> dict[str, object]:
    spatial = proposal.spatial
    return {
        "kind": proposal.kind.value,
        "content": proposal.content,
        "basis": proposal.basis.value,
        "subject": proposal.subject,
        "predicate": proposal.predicate,
        "value": proposal.value,
        "confidence": proposal.confidence,
        "valid_from": _time_text(proposal.valid_from),
        "valid_until": _time_text(proposal.valid_until),
        "cue_modality": (None if proposal.cue_modality is None else proposal.cue_modality.value),
        "valence": proposal.valence,
        "arousal": proposal.arousal,
        "spatial": None if spatial is None else _spatial_payload(spatial),
    }


def _spatial_payload(spatial: SpatialContext) -> dict[str, object]:
    orientation = spatial.orientation_xyzw
    return {
        "frame_id": spatial.frame_id,
        "anchor": spatial.anchor.value,
        "x": spatial.x,
        "y": spatial.y,
        "z": spatial.z,
        "orientation_xyzw": None if orientation is None else list(orientation),
        "position_uncertainty_m": spatial.position_uncertainty_m,
    }


def _proposal(value: object) -> FormationProposal:
    if not isinstance(value, dict):
        raise ValueError
    spatial = value.get("spatial")
    return FormationProposal(
        kind=cast(Any, value["kind"]),
        content=cast(Any, value["content"]),
        basis=cast(Any, value["basis"]),
        subject=cast(Any, value.get("subject")),
        predicate=cast(Any, value.get("predicate")),
        value=cast(Any, value.get("value")),
        confidence=cast(Any, value["confidence"]),
        valid_from=_time(value.get("valid_from")),
        valid_until=_time(value.get("valid_until")),
        spatial=None if spatial is None else _spatial(spatial),
        cue_modality=cast(Any, value.get("cue_modality")),
        valence=cast(Any, value.get("valence")),
        arousal=cast(Any, value.get("arousal")),
    )


def _spatial(value: object) -> SpatialContext:
    if not isinstance(value, dict):
        raise ValueError
    orientation = value.get("orientation_xyzw")
    return SpatialContext(
        frame_id=cast(Any, value["frame_id"]),
        anchor=cast(Any, value["anchor"]),
        x=cast(Any, value["x"]),
        y=cast(Any, value["y"]),
        z=cast(Any, value.get("z", 0.0)),
        orientation_xyzw=None if orientation is None else cast(Any, tuple(orientation)),
        position_uncertainty_m=cast(Any, value.get("position_uncertainty_m")),
    )


def _time_text(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _time(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError
    return datetime.fromisoformat(value)


def _json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
