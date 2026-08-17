"""AML wire contract tests."""

import pytest
from pydantic import ValidationError

from mindbridge.api.aml_contracts import (
    AmlAddRequest,
    AmlSearchRequest,
    derive_tenant_id,
)


def test_add_request_ignores_unknown_platform_fields() -> None:
    """AML owns this contract; an added field must not fail the run."""
    request = AmlAddRequest.model_validate(
        {
            "request_id": "eval:run-1:locomo_refined:conv-0:chunk-0",
            "messages": [{"role": "user", "content": "Rob moved to Sweden."}],
            "user_id": "eval:run-1:locomo:conv-0",
            "session_id": "eval:run-1:sample:0",
            "future_field": "ignored",
        }
    )
    assert request.messages[0].content == "Rob moved to Sweden."


def test_add_request_rejects_empty_messages() -> None:
    with pytest.raises(ValidationError):
        AmlAddRequest.model_validate(
            {
                "request_id": "r",
                "messages": [],
                "user_id": "u",
                "session_id": "s",
            }
        )


def test_search_request_caps_top_k_at_one_hundred() -> None:
    with pytest.raises(ValidationError):
        AmlSearchRequest.model_validate({"query": "q", "user_id": "u", "top_k": 101})


def test_derive_tenant_id_is_stable_bounded_and_collision_free() -> None:
    first = derive_tenant_id("bench_aml", "eval:run-1:locomo:conv-0")
    second = derive_tenant_id("bench_aml", "eval:run-1:locomo:conv-1")
    assert first == derive_tenant_id("bench_aml", "eval:run-1:locomo:conv-0")
    assert first != second
    assert first.startswith("bench_aml:")
    assert len(derive_tenant_id("bench_aml", "x" * 4_000)) <= 255
