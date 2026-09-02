"""Default failure messages shared by REST and MCP.

Kept apart from `errors` so the MCP surface, which installs without FastAPI, can reach them.
"""

from __future__ import annotations

from collections.abc import Mapping

from mindbridge.exceptions import MindBridgeError

# One message per code, shared by REST and MCP so the two surfaces cannot drift apart. A raise
# site's own message always wins; this is only what a bare raise says.
MESSAGE_BY_CODE: Mapping[str, str] = {
    "validation_error": "input is invalid",
    "memory_not_found": "memory does not exist",
    "speaker_not_found": "speaker does not exist",
    "identity_not_found": "identity does not exist",
    "model_error": "model operation failed",
    "model_output_truncated": "model operation failed",
    "storage_error": "memory storage is unavailable",
    "index_unavailable": "memory index is unavailable",
}


def error_message(error: MindBridgeError) -> str:
    """Return the author-written message, or the code's default when the raise site gave none."""
    default = MESSAGE_BY_CODE.get(error.code)
    # An unmapped public code is a MindBridge bug, not caller-actionable detail. REST pairs this
    # message with HTTP 500 and MCP emits the same string.
    return "memory operation failed" if default is None else (str(error) or default)
