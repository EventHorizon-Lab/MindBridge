"""Agent Memory Leaderboard Add/Search wire contracts."""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from mindbridge.contracts import (
    ContractModel,
    Identifier,
    NonEmptyString,
    UtcDatetime,
)

_TENANT_DIGEST_CHARACTERS = 32


class _PlatformRequest(ContractModel):
    """A request shape owned by AML, tolerant of fields it adds later."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class AmlMessage(ContractModel):
    """One conversation message in an AML add chunk."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    role: NonEmptyString
    content: NonEmptyString
    timestamp: int | None = None


class AmlAddRequest(_PlatformRequest):
    """One synchronously persisted chunk of conversation history."""

    request_id: Identifier
    messages: Annotated[tuple[AmlMessage, ...], Field(min_length=1, max_length=512)]
    user_id: Identifier
    session_id: Identifier


class AmlAddResponse(ContractModel):
    """Byte-exact acknowledgement AML matches against its request."""

    success: Literal[True] = True
    request_id: Identifier
    user_id: Identifier
    session_id: Identifier


class AmlSearchRequest(_PlatformRequest):
    """One retrieval request scoped to a single AML user."""

    query: NonEmptyString
    options: tuple[NonEmptyString, ...] = ()
    user_id: Identifier
    top_k: Annotated[int, Field(ge=1, le=100)]


class AmlMemoryItem(ContractModel):
    """One retrieved memory in AML's required item shape."""

    id: Identifier
    content: NonEmptyString
    created_at: UtcDatetime | None = None


class AmlSearchResponse(ContractModel):
    """Ranked evidence, most relevant first."""

    data: tuple[AmlMemoryItem, ...]


def derive_tenant_id(prefix: str, user_id: str) -> str:
    """Map an AML user onto a tenant inside one authorized namespace.

    The caller never names a tenant, so it cannot reach outside the namespace,
    and the digest keeps the result inside the 255-character Identifier limit
    whatever AML sends.
    """
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:_TENANT_DIGEST_CHARACTERS]}"
