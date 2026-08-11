"""Tenant-bound bearer authentication for the public REST API."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated

from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import TypeAdapter, ValidationError

from mindbridge.contracts import Identifier

_BEARER = HTTPBearer(auto_error=False)
_IDENTIFIER = TypeAdapter(Identifier)
_MINIMUM_API_KEY_LENGTH = 32


@dataclass(frozen=True, slots=True)
class TenantPrincipal:
    """The exact tenant allowlist proven by one API credential."""

    tenant_ids: frozenset[str]


class AuthenticationError(Exception):
    """A sanitized authentication or tenant-authorization failure."""

    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class TenantApiKeyAuthenticator:
    """Authenticate tenant-allowlisted API keys without retaining plaintext secrets."""

    __slots__ = ("_credentials",)

    def __init__(self, tenant_api_keys: Mapping[str, Sequence[str]]) -> None:
        tenant_ids_by_digest: dict[bytes, set[str]] = {}
        if not tenant_api_keys:
            raise ValueError("at least one tenant API key must be configured")
        for raw_tenant_id, api_keys in tenant_api_keys.items():
            tenant_id = _validated_tenant_id(raw_tenant_id)
            if isinstance(api_keys, str) or not api_keys:
                raise ValueError(f"tenant {tenant_id!r} must have a non-empty API key list")
            for api_key in api_keys:
                if not isinstance(api_key, str) or len(api_key) < _MINIMUM_API_KEY_LENGTH:
                    raise ValueError(
                        f"API keys must be strings of at least {_MINIMUM_API_KEY_LENGTH} characters"
                    )
                digest = hashlib.sha256(api_key.encode()).digest()
                tenant_ids_by_digest.setdefault(digest, set()).add(tenant_id)
        self._credentials = tuple(
            (digest, frozenset(tenant_ids)) for digest, tenant_ids in tenant_ids_by_digest.items()
        )

    @classmethod
    def from_json(cls, value: str) -> TenantApiKeyAuthenticator:
        """Parse ``tenant -> [rotatable keys]`` configuration without echoing secrets."""
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("tenant API keys must be a valid JSON object") from error
        if not isinstance(parsed, dict):
            raise ValueError("tenant API keys must be a JSON object")
        return cls(parsed)

    async def __call__(
        self,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_BEARER)],
    ) -> TenantPrincipal:
        if credentials is None:
            raise AuthenticationError(
                status_code=401,
                code="authentication_required",
                message="a valid bearer API key is required",
            )
        candidate = hashlib.sha256(credentials.credentials.encode()).digest()
        tenant_ids = None
        for known_digest, known_tenant_ids in self._credentials:
            if hmac.compare_digest(candidate, known_digest):
                tenant_ids = known_tenant_ids
        if tenant_ids is None:
            raise AuthenticationError(
                status_code=401,
                code="authentication_failed",
                message="the bearer API key is invalid",
            )
        return TenantPrincipal(tenant_ids)


def require_tenant(principal: TenantPrincipal, tenant_id: str) -> None:
    """Reject cross-tenant identifiers before a use case reaches storage."""
    if tenant_id not in principal.tenant_ids:
        raise AuthenticationError(
            status_code=403,
            code="tenant_access_denied",
            message="the authenticated tenant cannot access this resource",
        )


def _validated_tenant_id(value: object) -> str:
    try:
        tenant_id = _IDENTIFIER.validate_python(value)
    except ValidationError as error:
        raise ValueError("tenant API key mapping contains an invalid tenant ID") from error
    if tenant_id != value:
        raise ValueError("tenant API key mapping contains a non-canonical tenant ID")
    return tenant_id
