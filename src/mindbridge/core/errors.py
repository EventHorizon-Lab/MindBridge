"""Errors raised by MindBridge domain rules."""


class DomainInvariantError(ValueError):
    """Raised when domain data would create an invalid memory state."""


class IdempotencyConflictError(DomainInvariantError):
    """Raised when one idempotency key is reused for different content."""
