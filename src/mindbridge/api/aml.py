"""Agent Memory Leaderboard Add/Search adapter over the memory kernel."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Final

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
from mindbridge.api.errors import ErrorCode, responses
from mindbridge.application.aml_extraction import extract_memories
from mindbridge.application.kernel import MemoryKernel
from mindbridge.contracts import (
    RecallMode,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.models import Generator
from mindbridge.telemetry import set_current_span_attributes

AML_ERRORS: Final[tuple[ErrorCode, ...]] = (
    "authentication_required",
    "authentication_failed",
    "request_validation_failed",
    "database_unavailable",
)
"""What both leaderboard operations return whatever else they do.

Deliberately not `TENANT_ERRORS`: these routes derive their tenant from `user_id` rather
than proving one against an allow-set, so `tenant_access_denied` is unreachable here and
listing it would document a status the server cannot send.
"""

_AML_MODEL_ERRORS: Final[tuple[ErrorCode, ...]] = (
    "model_request_failed",
    "model_output_invalid",
    "model_unavailable",
)
"""Both operations call a model: add extracts memories, search encodes the query."""

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
    """Expose AML's two operations without widening the tenant API surface.

    Precondition: `app` must be built by `mindbridge.api.app.build_app`, which
    registers the handler for `mindbridge.api.auth.AuthenticationError`. On a
    bare `FastAPI()` app that handler is absent, so a 401 raised here escapes
    unhandled instead of becoming an HTTP response.
    """
    expected = hashlib.sha256(settings.api_key.encode()).digest()

    async def authorize(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_BEARER)],
    ) -> None:
        if credentials is None:
            raise AuthenticationError("authentication_required")
        candidate = hashlib.sha256(credentials.credentials.encode()).digest()
        if not hmac.compare_digest(candidate, expected):
            raise AuthenticationError("authentication_failed")

    # The leaderboard adapter documents itself from the same table as /v1: without this the
    # two routes published FastAPI's default `422 -> HTTPValidationError`, whose only field is
    # `detail`, while the app-wide handler actually answers with an ErrorResponse -- and the
    # 401 this module raises below reached callers without appearing in the document at all.
    @app.post(
        "/aml/add",
        response_model=AmlAddResponse,
        operation_id="amlAdd",
        responses=responses(*AML_ERRORS, "domain_invariant_failed", *_AML_MODEL_ERRORS),
    )
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
        set_current_span_attributes(
            {
                "mindbridge.aml.memories_stored": len(outcome.memories),
                "mindbridge.aml.memories_skipped": outcome.skipped,
            }
        )
        await kernel.remember(
            tuple(
                RememberRequest(
                    tenant_id=tenant_id,
                    summary=memory.summary,
                    memory_type=memory.memory_type,
                    occurred_at=memory.occurred_at,
                )
                for memory in outcome.memories
            )
        )
        return AmlAddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    @app.post(
        "/aml/search",
        response_model=AmlSearchResponse,
        operation_id="amlSearch",
        responses=responses(*AML_ERRORS, *_AML_MODEL_ERRORS),
    )
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
