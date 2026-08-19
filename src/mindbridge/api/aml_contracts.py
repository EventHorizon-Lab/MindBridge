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

    role: NonEmptyString = Field(description="Who spoke, in AML's own vocabulary.")
    content: NonEmptyString = Field(description="What was said, verbatim.")
    timestamp: int | None = Field(
        default=None,
        description="AML's own message timestamp, when it supplies one.",
    )


class AmlAddRequest(_PlatformRequest):
    """One synchronously persisted chunk of conversation history."""

    request_id: Identifier = Field(description="Echoed back so AML can match the reply.")
    messages: Annotated[
        tuple[AmlMessage, ...],
        Field(
            min_length=1,
            max_length=512,
            description="The conversation chunk to extract memories from, in order.",
        ),
    ]
    user_id: Identifier = Field(description="AML user, mapped onto one derived tenant.")
    session_id: Identifier = Field(description="AML conversation this chunk belongs to.")


class AmlAddResponse(ContractModel):
    """Byte-exact acknowledgement AML matches against its request."""

    success: Literal[True] = Field(default=True, description="Always true; a failure is an error.")
    request_id: Identifier = Field(description="The `request_id` this acknowledges.")
    user_id: Identifier = Field(description="The `user_id` this acknowledges.")
    session_id: Identifier = Field(description="The `session_id` this acknowledges.")


class AmlSearchRequest(_PlatformRequest):
    """One retrieval request scoped to a single AML user."""

    query: NonEmptyString = Field(description="What to retrieve, in words.")
    options: tuple[NonEmptyString, ...] = Field(
        default=(),
        description="Answer choices AML supplies for a multiple-choice question.",
    )
    user_id: Identifier = Field(description="AML user whose memory to search.")
    top_k: Annotated[
        int,
        Field(ge=1, le=100, description="How many memories to return, most relevant first."),
    ]


class AmlMemoryItem(ContractModel):
    """One retrieved memory in AML's required item shape."""

    id: Identifier = Field(description="Stable memory ID.")
    content: NonEmptyString = Field(description="The remembered content, in words.")
    created_at: UtcDatetime | None = Field(
        default=None,
        description="When MindBridge retained it, when that is known.",
    )


class AmlSearchResponse(ContractModel):
    """Ranked evidence, most relevant first."""

    data: tuple[AmlMemoryItem, ...] = Field(description="Retrieved memories, most relevant first.")


def derive_tenant_id(prefix: str, user_id: str) -> str:
    """Map an AML user onto a tenant inside one authorized namespace.

    The caller never names a tenant, so it cannot reach outside the namespace,
    and the digest keeps the result inside the 255-character Identifier limit
    whatever AML sends.
    """
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:_TENANT_DIGEST_CHARACTERS]}"
