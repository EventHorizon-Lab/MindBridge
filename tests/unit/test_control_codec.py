"""Current operation logs round-trip; incomplete formats fail at the decoding boundary."""

import json
from typing import cast

import pytest

from mindbridge import (
    ConsentClaim,
    ConsentState,
    FormationProposal,
    IdentityClaim,
    MemoryIntent,
    MemoryKind,
    MemoryOperation,
    SpatialAnchor,
    SpatialContext,
    StorageError,
)
from mindbridge.control import dump_operation, load_operation


@pytest.mark.parametrize(
    ("operation", "path", "field"),
    [
        (MemoryOperation(intent=MemoryIntent.FORGET, target_ids=("m",)), (), "identity"),
        (
            MemoryOperation(
                intent=MemoryIntent.IDENTIFY,
                claim=IdentityClaim(identity_id="person", name="Ana"),
                evidence_ids=("m",),
            ),
            ("claim",),
            "relationship",
        ),
        (
            MemoryOperation(
                intent=MemoryIntent.CONSENT,
                consent=ConsentClaim(identity_id="person", state=ConsentState.GRANTED),
            ),
            ("consent",),
            "note",
        ),
        *(
            (
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=("m",),
                    proposal=FormationProposal(
                        kind=MemoryKind.EVENT,
                        content="Ana arrived.",
                        confidence=0.9,
                        spatial=SpatialContext(
                            frame_id="room", anchor=SpatialAnchor.SUBJECT, x=1.0, y=2.0
                        ),
                    ),
                ),
                path,
                field,
            )
            for path, field in ((("proposal",), "evidence_ids"), (("proposal", "spatial"), "z"))
        ),
    ],
)
def test_operation_logs_require_current_fields_even_when_the_value_has_a_default(
    operation: MemoryOperation, path: tuple[str, ...], field: str
) -> None:
    payload = json.loads(dump_operation(operation))
    assert load_operation(json.dumps(payload)) == operation
    target = payload
    for key in path:
        target = cast(dict[str, object], target[key])
    del target[field]

    with pytest.raises(StorageError, match="a logged memory operation could not be read"):
        load_operation(json.dumps(payload))


@pytest.mark.parametrize(
    "payload",
    [
        "not JSON",
        "null",
        "[]",
        "{}",
        '{"intent": "unknown"}',
        '{"intent": "forget", "target_ids": 42}',
        '{"intent": "identify", "claim": {"identity_id": "p", "name": 42}}',
        '{"intent": "consent", "consent": {"identity_id": "p", "state": "unknown"}}',
        '{"intent": "merge", "identity": {"identity_id": "p", "moved_ids": 42}}',
        '{"intent": "consolidate", "evidence_ids": ["m"], "proposal": '
        '{"kind": "event", "content": "an event", "basis": "observation", "confidence": true}}',
        '{"intent": "consolidate", "evidence_ids": ["m"], "proposal": '
        '{"kind": "event", "content": "an event", "basis": "observation", "confidence": '
        + str(10**400)
        + "}}",
        '{"intent": "consolidate", "evidence_ids": ["m"], "proposal": '
        '{"kind": "event", "content": "an event", "basis": "observation", "confidence": 1, '
        '"valid_from": "2026-01-01T00:00:00"}}',
        "[" * 2_000 + "]" * 2_000,
    ],
)
def test_malformed_operation_log_raises_storage_error(payload: str) -> None:
    with pytest.raises(StorageError, match="a logged memory operation could not be read"):
        load_operation(payload)
