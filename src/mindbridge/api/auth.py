"""Optional bearer authentication for one MindBridge service."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_BEARER = HTTPBearer(auto_error=False)


class AuthenticationError(Exception):
    """A credential was required but was missing or invalid."""


class ApiKeyAuthenticator:
    """Validate one configured API key without exposing it in representations."""

    __slots__ = ("_api_key",)

    def __init__(self, api_key: str) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be non-empty text")
        self._api_key = api_key.encode()

    async def __call__(
        self,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_BEARER)],
    ) -> None:
        if credentials is None or not self.accepts(
            credentials.scheme,
            credentials.credentials,
        ):
            raise AuthenticationError

    def accepts(self, scheme: str, credentials: str) -> bool:
        """Check parsed bearer credentials in constant time."""
        return scheme.casefold() == "bearer" and hmac.compare_digest(
            credentials.encode(), self._api_key
        )

    def accepts_header(self, authorization: bytes | None) -> bool:
        """Check a raw HTTP authorization header before reading its body."""
        if authorization is None:
            return False
        parts = authorization.split(maxsplit=1)
        return (
            len(parts) == 2
            and parts[0].lower() == b"bearer"
            and hmac.compare_digest(parts[1], self._api_key)
        )
