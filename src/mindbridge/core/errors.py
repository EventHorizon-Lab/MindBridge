"""Errors raised by MindBridge domain rules."""


class DomainInvariantError(ValueError):
    """Raised when domain data would create an invalid memory state."""


class IdempotencyConflictError(DomainInvariantError):
    """Raised when one idempotency key is reused for different content."""


class MemoryIntegrityError(RuntimeError):
    """Raised when persisted references violate an internal invariant."""


class ObjectStorageError(RuntimeError):
    """Raised when immutable evidence media cannot be accessed."""


class TaskBrokerError(RuntimeError):
    """Raised when a durable processing job cannot be delivered."""


class ModelUnavailableError(RuntimeError):
    """Raised when a configured frozen model cannot be loaded or called."""


class ModelOutputError(RuntimeError):
    """Raised when model output violates its declared contract."""
