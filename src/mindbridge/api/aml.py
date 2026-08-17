"""Agent Memory Leaderboard Add/Search adapter over the memory kernel."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated

from fastapi import FastAPI, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from mindbridge.api.aml_contracts import (
    AmlAddRequest,
    AmlAddResponse,
    AmlMemoryItem,
    AmlSearchRequest,
    AmlSearchResponse,
    derive_tenant_id,
)
from mindbridge.api.auth import AuthenticationError
from mindbridge.application.aml_extraction import extract_memories
from mindbridge.application.kernel import MemoryKernel
from mindbridge.contracts import RecallMode, RecallQuery, RecallRequest, RememberRequest
from mindbridge.models import Generator

_BEARER = HTTPBearer(auto_error=False)
_MINIMUM_API_KEY_LENGTH = 32


@dataclass(frozen=True, slots=True)
class AmlSettings:
    """One AML key authorizing one tenant namespace."""

    api_key: str
    tenant_prefix: str

    def __post_init__(self) -> None:
        if len(self.api_key) < _MINIMUM_API_KEY_LENGTH:
            raise ValueError(
                f"the AML API key must be at least {_MINIMUM_API_KEY_LENGTH} characters"
            )
        if not self.tenant_prefix.strip():
            raise ValueError("the AML tenant prefix must not be blank")


def register_aml_routes(
    app: FastAPI,
    kernel: MemoryKernel,
    generator: Generator,
    *,
    settings: AmlSettings,
) -> None:
    """Expose AML's two operations without widening the tenant API surface."""
    expected = hashlib.sha256(settings.api_key.encode()).digest()

    async def authorize(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_BEARER)],
    ) -> None:
        if credentials is None:
            raise AuthenticationError(
                status_code=401,
                code="authentication_required",
                message="a valid bearer API key is required",
            )
        candidate = hashlib.sha256(credentials.credentials.encode()).digest()
        if not hmac.compare_digest(candidate, expected):
            raise AuthenticationError(
                status_code=401,
                code="authentication_failed",
                message="the bearer API key is invalid",
            )

    @app.post("/aml/add", response_model=AmlAddResponse, operation_id="amlAdd")
    async def aml_add(
        request: AmlAddRequest,
        _: None = Security(authorize),
    ) -> AmlAddResponse:
        tenant_id = derive_tenant_id(settings.tenant_prefix, request.user_id)
        outcome = await extract_memories(
            generator,
            request.messages,
            now=datetime.now(tz=timezone.utc),
        )
        await asyncio.gather(
            *(
                kernel.remember(
                    RememberRequest(
                        tenant_id=tenant_id,
                        summary=memory.summary,
                        memory_type=memory.memory_type,
                        occurred_at=memory.occurred_at,
                    )
                )
                for memory in outcome.memories
            )
        )
        return AmlAddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    @app.post("/aml/search", response_model=AmlSearchResponse, operation_id="amlSearch")
    async def aml_search(
        request: AmlSearchRequest,
        _: None = Security(authorize),
    ) -> AmlSearchResponse:
        result = await kernel.recall(
            RecallRequest(
                tenant_id=derive_tenant_id(settings.tenant_prefix, request.user_id),
                query=RecallQuery(text=request.query),
                mode=RecallMode.SEARCH,
                limit=request.top_k,
                include_evidence=False,
            )
        )
        return AmlSearchResponse(
            data=tuple(
                AmlMemoryItem(
                    id=memory.memory_id,
                    content=memory.summary,
                    created_at=memory.created_at,
                )
                for memory in result.memories
            )
        )
