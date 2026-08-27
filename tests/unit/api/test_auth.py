"""Checks for the optional single-key bearer dependency."""

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from mindbridge.api.auth import ApiKeyAuthenticator, AuthenticationError

API_KEY = "one-private-api-key"


async def test_authenticator_uses_constant_time_comparison_without_exposing_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparisons: list[tuple[bytes, bytes]] = []

    def compare(candidate: bytes, expected: bytes) -> bool:
        comparisons.append((candidate, expected))
        return candidate == expected

    monkeypatch.setattr("mindbridge.api.auth.hmac.compare_digest", compare)
    authenticator = ApiKeyAuthenticator(API_KEY)
    await authenticator(HTTPAuthorizationCredentials(scheme="Bearer", credentials=API_KEY))

    assert comparisons == [(API_KEY.encode(), API_KEY.encode())]
    assert API_KEY not in repr(authenticator)


@pytest.mark.parametrize(
    "credentials",
    [
        None,
        HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong"),
        HTTPAuthorizationCredentials(scheme="Basic", credentials=API_KEY),
    ],
)
async def test_authenticator_rejects_missing_or_invalid_credentials(
    credentials: HTTPAuthorizationCredentials | None,
) -> None:
    with pytest.raises(AuthenticationError):
        await ApiKeyAuthenticator(API_KEY)(credentials)


def test_authenticator_rejects_an_empty_configured_key() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ApiKeyAuthenticator("")
