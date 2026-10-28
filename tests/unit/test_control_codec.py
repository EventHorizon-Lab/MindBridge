"""Malformed operation logs fail at the decoding/value-construction boundary."""

import pytest

from mindbridge import StorageError
from mindbridge.control import load_operation


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
