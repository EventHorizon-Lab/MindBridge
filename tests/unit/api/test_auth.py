"""Checks for strict, secret-safe tenant API key configuration."""

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from mindbridge.api.auth import TenantApiKeyAuthenticator

API_KEY = "tenant-api-key-000000000000000000"


def test_authenticator_parses_rotatable_keys_without_retaining_plaintext() -> None:
    authenticator = TenantApiKeyAuthenticator.from_json(
        f'{{"tenant_01":["{API_KEY}","next-tenant-key-000000000000000000"]}}'
    )

    assert API_KEY not in repr(authenticator)


async def test_one_key_can_have_an_explicit_multi_tenant_allowlist() -> None:
    authenticator = TenantApiKeyAuthenticator(
        {"tenant_01": (API_KEY,), "benchmark_run_01": (API_KEY,)}
    )

    principal = await authenticator(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=API_KEY)
    )

    assert principal.tenant_ids == frozenset({"tenant_01", "benchmark_run_01"})


@pytest.mark.parametrize(
    "configuration",
    [
        "[]",
        "{}",
        '{"tenant_01":[]}',
        '{"tenant_01":["short"]}',
        '{" tenant_01":["tenant-api-key-000000000000000000"]}',
    ],
)
def test_authenticator_rejects_ambiguous_or_weak_configuration(configuration: str) -> None:
    with pytest.raises(ValueError):
        TenantApiKeyAuthenticator.from_json(configuration)
