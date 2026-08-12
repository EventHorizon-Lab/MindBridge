"""Normalize OpenAI SDK failures without leaking provider details."""

from typing import NoReturn

import openai

from mindbridge.core import ModelRequestError, ModelUnavailableError


def raise_openai_model_error(error: openai.APIError, message: str) -> NoReturn:
    """Raise a retryable error only for transient transport or provider failures."""
    if isinstance(error, openai.APIConnectionError) or (
        isinstance(error, openai.APIStatusError)
        and (error.status_code in {408, 409, 429} or error.status_code >= 500)
    ):
        raise ModelUnavailableError(message) from error
    raise ModelRequestError(message) from error
