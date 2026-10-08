"""The public stream acknowledgement allows skipped retrieval only for deferred captures."""

from datetime import datetime, timezone

import pytest

from mindbridge import MemoryRecord, PrefetchResult, StreamCommit, ValidationError


@pytest.mark.parametrize("pending_settlement", [False, True])
@pytest.mark.parametrize("has_prefetch", [False, True])
@pytest.mark.parametrize("retrieval_error", [None, "retrieval_failed"])
def test_stream_commit_retrieval_contract(
    pending_settlement: bool, has_prefetch: bool, retrieval_error: str | None
) -> None:
    record = MemoryRecord(
        id="captured", content="observation", created_at=datetime(2026, 10, 8, tzinfo=timezone.utc)
    )
    prefetch = PrefetchResult(hits=()) if has_prefetch else None
    valid = has_prefetch != (retrieval_error is not None) or (
        pending_settlement and not has_prefetch and retrieval_error is None
    )
    if valid:
        commit = StreamCommit(
            record, prefetch, retrieval_error, pending_settlement=pending_settlement
        )
        assert commit.record == record
        assert commit.prefetch == prefetch
        assert commit.retrieval_error == retrieval_error
        assert commit.pending_settlement is pending_settlement
    else:
        with pytest.raises(ValidationError, match="prefetch or retrieval_error"):
            StreamCommit(record, prefetch, retrieval_error, pending_settlement=pending_settlement)
