"""SQLite source of truth for the local MindBridge runtime.

`LocalStore` is the one entry point; the row values and errors here are what its families
accept and return. Everything else in this package is a connection-level function behind one
family and is not imported outside the package.
"""

from mindbridge.infrastructure.local.store._codec import datetime_text, optional_datetime_text
from mindbridge.infrastructure.local.store._identity import CONSENT_PREDICATE, canonical_subject
from mindbridge.infrastructure.local.store._recall import RECALL_MAX_ROWS, RECALL_MAX_TERM_CHARS
from mindbridge.infrastructure.local.store._schema import SCHEMA_VERSION
from mindbridge.infrastructure.local.store.errors import (
    LocalStoreClosedError,
    StaleOperationError,
    UnsupportedSchemaError,
)
from mindbridge.infrastructure.local.store.local_store import LocalStore
from mindbridge.infrastructure.local.store.rows import (
    IdentityLink,
    IndexCandidate,
    IndexDocument,
    IndexOperation,
    RecallDigest,
    RecallRead,
    SpeechRollback,
    StoredAsset,
    StoredCandidate,
    StoredEmbedding,
    StoredEvidenceClauseChange,
    StoredMemory,
    StoredOperation,
    StoredQueryFailure,
    StoredTextSelector,
    StoredTextSpanPiece,
    validate_asset_name,
)

__all__ = [
    "CONSENT_PREDICATE",
    "RECALL_MAX_ROWS",
    "RECALL_MAX_TERM_CHARS",
    "SCHEMA_VERSION",
    "IdentityLink",
    "IndexCandidate",
    "IndexDocument",
    "IndexOperation",
    "LocalStore",
    "LocalStoreClosedError",
    "RecallDigest",
    "RecallRead",
    "SpeechRollback",
    "StaleOperationError",
    "StoredAsset",
    "StoredCandidate",
    "StoredEmbedding",
    "StoredEvidenceClauseChange",
    "StoredMemory",
    "StoredOperation",
    "StoredQueryFailure",
    "StoredTextSelector",
    "StoredTextSpanPiece",
    "UnsupportedSchemaError",
    "canonical_subject",
    "datetime_text",
    "optional_datetime_text",
    "validate_asset_name",
]
