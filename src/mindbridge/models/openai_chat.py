"""Shared streaming completion primitive over the official OpenAI SDK."""

from __future__ import annotations

from dataclasses import dataclass

import openai
from openai import AsyncOpenAI, Omit, omit
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import ResponseFormatJSONObject

from mindbridge.core import ModelOutputError
from mindbridge.models.openai_errors import raise_openai_model_error

REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True, slots=True)
class OpenAITextCompletion:
    """Text plus the provider revision fingerprint when one is available."""

    content: str
    system_fingerprint: str | None


async def stream_text_completion(
    client: AsyncOpenAI,
    *,
    model_id: str,
    messages: list[ChatCompletionMessageParam],
    max_output_tokens: int,
    request_timeout_seconds: float,
    json_mode: bool = False,
    reasoning_effort: ReasoningEffort = None,
) -> OpenAITextCompletion:
    """Read one deterministic text stream and normalize SDK failures."""
    parts: list[str] = []
    finish_reason: str | None = None
    system_fingerprint: str | None = None
    response_format: ResponseFormatJSONObject | Omit = (
        {"type": "json_object"} if json_mode else omit
    )
    try:
        stream = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            modalities=["text"],
            max_tokens=max_output_tokens,
            response_format=response_format,
            reasoning_effort=reasoning_effort if reasoning_effort is not None else omit,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
            timeout=request_timeout_seconds,
        )
        async for chunk in stream:
            system_fingerprint = system_fingerprint or chunk.system_fingerprint
            for choice in chunk.choices:
                if choice.index != 0:
                    continue
                if choice.delta.content:
                    parts.append(choice.delta.content)
                finish_reason = finish_reason or choice.finish_reason
    except openai.APIError as error:
        raise_openai_model_error(error, "Omni model request failed")

    if finish_reason in {"length", "content_filter"}:
        raise ModelOutputError(f"Omni model ended with finish reason {finish_reason}")
    return OpenAITextCompletion(
        content="".join(parts),
        system_fingerprint=system_fingerprint,
    )
