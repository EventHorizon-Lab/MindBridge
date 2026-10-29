"""Errors the local store raises across its own boundary."""

from __future__ import annotations

from collections.abc import Iterable


class LocalStoreClosedError(RuntimeError):
    """Raised when a closed local store is used."""


class LocalStoreCapacityError(RuntimeError):
    """Raised before opening a connection beyond the instance's capacity."""


class UnsupportedSchemaError(RuntimeError):
    """Raised when the data directory has an unknown or incomplete schema."""


class ProtectedMemoryError(RuntimeError):
    """Raised when a proposed mutation touches a host policy or its supporting records."""

    def __init__(self, memory_ids: Iterable[str]) -> None:
        super().__init__("model mutation would change a host response policy")
        self.memory_ids = frozenset(memory_ids)


class StaleOperationError(RuntimeError):
    """Raised when a control-plane operation's preconditions moved before it committed.

    The control plane reads and validates records, then applies them in a later transaction. The
    apply transaction re-checks the preconditions the proposal was built on; when a target or
    cited source was forgotten, corrected, deleted, or already linked in between, nothing is
    written and the caller reports the proposal as stale rather than partially applying it.
    """
