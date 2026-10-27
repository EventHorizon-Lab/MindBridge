"""Provider context overflow has one permanent recovery contract across transports."""

import json

from mindbridge import ModelError
from mindbridge.api.errors import exception_response


def test_provider_context_overflow_is_a_permanent_input_size_error() -> None:
    error = ModelError(
        "generation request failed", reason="context_length_exceeded", stage="generate"
    )
    error.__cause__ = RuntimeError("private provider detail")
    response = exception_response(error)
    body = json.loads(bytes(response.body))
    assert response.status_code == 413
    assert body["code"] == "model_error"
    assert body["reason"] == "context_length_exceeded"
    assert body["stage"] == "generate"
    assert body["retryable"] is False
    assert "private" not in body["message"]
    assert "Retry-After" not in response.headers
