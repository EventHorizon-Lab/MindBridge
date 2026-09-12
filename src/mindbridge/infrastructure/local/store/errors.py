"""Errors the local store raises across its own boundary."""

from __future__ import annotations


class LocalStoreClosedError(RuntimeError):
    """Raised when a closed local store is used."""


class UnsupportedSchemaError(RuntimeError):
    """Raised when the data directory has an unknown or incomplete schema."""


class StaleOperationError(RuntimeError):
    """Raised when a control-plane operation's preconditions moved before it committed.

    The control plane reads and validates records, then applies them in a later transaction. The
    apply transaction re-checks the preconditions the proposal was built on; when a target or
    cited source was forgotten, corrected, deleted, or already linked in between, nothing is
    written and the caller reports the proposal as stale rather than partially applying it.
    """
