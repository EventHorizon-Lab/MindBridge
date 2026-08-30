"""Stable exceptions raised by the public MindBridge API."""

from typing import ClassVar

# Reasons an identical retry can succeed. Membership is the whole retry contract: an agent that
# skips a transient retry loses one call, while an agent that retries a permanent failure never
# stops. Anything unclassified stays out.
RETRYABLE_REASONS: frozenset[str] = frozenset(
    {
        "connection_failed",
        "data_dir_in_use",
        "flush_failed",
        "index_missing",
        "rate_limited",
        "timeout",
    }
)


class MindBridgeError(Exception):
    """Base class for failures callers may handle.

    ``code`` is the stable outer taxonomy. ``reason`` narrows it to a closed sub-vocabulary,
    ``stage`` names the pipeline stage that failed, and ``subject`` carries the asset ID, memory ID,
    or batch position the failure is about. All three are optional, so an unclassified raise site
    keeps working unchanged.
    """

    code: ClassVar[str] = "mindbridge_error"
    default_reason: ClassVar[str | None] = None

    def __init__(
        self,
        *args: object,
        reason: str | None = None,
        stage: str | None = None,
        subject: str | None = None,
    ) -> None:
        super().__init__(*args)
        self.reason = self.default_reason if reason is None else reason
        self.stage = stage
        self.subject = subject

    @property
    def retryable(self) -> bool:
        """Whether an identical retry can succeed; a lookup on ``reason``, never a judgement."""
        return self.reason in RETRYABLE_REASONS


class ValidationError(MindBridgeError, ValueError):
    """Raised when public input is invalid."""

    code = "validation_error"
    default_reason = "input_invalid"


class MemoryNotFoundError(MindBridgeError, LookupError):
    """Raised when a memory ID does not exist."""

    code = "memory_not_found"
    default_reason = "memory_not_found"


class SpeakerNotFoundError(MindBridgeError, LookupError):
    """Raised when a local speaker identity does not exist."""

    code = "speaker_not_found"
    default_reason = "speaker_not_found"


class ModelError(MindBridgeError):
    """Raised when a configured model cannot complete an operation."""

    code = "model_error"


class ModelOutputTruncatedError(ModelError):
    """Raised when generation stopped at an output token limit instead of finishing."""

    code = "model_output_truncated"
    default_reason = "output_truncated"


class StorageError(MindBridgeError):
    """Raised when durable memory state cannot be read or written."""

    code = "storage_error"


class IndexUnavailableError(StorageError):
    """Raised when the vector index cannot serve an operation."""

    code = "index_unavailable"
