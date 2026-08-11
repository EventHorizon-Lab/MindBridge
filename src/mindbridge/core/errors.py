"""Errors raised by MindBridge domain rules."""


class DomainInvariantError(ValueError):
    """Raised when domain data would create an invalid memory state."""
