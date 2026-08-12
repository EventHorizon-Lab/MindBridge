"""Errors raised by MindBridge domain rules."""


class DomainInvariantError(ValueError):
    """Raised when domain data would create an invalid memory state."""


class IdempotencyConflictError(DomainInvariantError):
    """Raised when one idempotency key is reused for different content."""


class MemoryIntegrityError(RuntimeError):
    """Raised when persisted references violate an internal invariant."""


class JobNotFoundError(LookupError):
    """Raised when a caller requests an unknown tenant-owned job."""


class MemoryNotFoundError(LookupError):
    """Raised when a caller requests an unknown tenant-owned memory."""


class ForgetTargetNotFoundError(LookupError):
    """Raised when an explicit forget target never existed in the tenant."""


class MemoryDeletedError(RuntimeError):
    """Raised when a tombstone prevents forgotten content from being resurrected."""


class ObjectStorageError(RuntimeError):
    """Raised when immutable evidence media cannot be accessed."""


class TaskBrokerError(RuntimeError):
    """Raised when a durable processing job cannot be delivered."""


class ModelUnavailableError(RuntimeError):
    """Raised when a configured frozen model cannot be loaded or called."""


class ModelRequestError(RuntimeError):
    """Raised when retrying an unchanged model request cannot succeed."""


class ModelOutputError(RuntimeError):
    """Raised when model output violates its declared contract."""


class EnumerationLimitExceededError(DomainInvariantError):
    """Raised when an exact enumeration would be silently truncated."""
