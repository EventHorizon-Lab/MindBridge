"""Stable exceptions raised by the public MindBridge API."""

from typing import ClassVar


class MindBridgeError(Exception):
    """Base class for failures callers may handle."""

    code: ClassVar[str] = "mindbridge_error"


class ValidationError(MindBridgeError, ValueError):
    """Raised when public input is invalid."""

    code = "validation_error"


class MemoryNotFoundError(MindBridgeError, LookupError):
    """Raised when a memory ID does not exist."""

    code = "memory_not_found"


class SpeakerNotFoundError(MindBridgeError, LookupError):
    """Raised when a local speaker identity does not exist."""

    code = "speaker_not_found"


class ModelError(MindBridgeError):
    """Raised when a configured model cannot complete an operation."""

    code = "model_error"


class StorageError(MindBridgeError):
    """Raised when durable memory state cannot be read or written."""

    code = "storage_error"


class IndexUnavailableError(StorageError):
    """Raised when the vector index cannot serve an operation."""

    code = "index_unavailable"
