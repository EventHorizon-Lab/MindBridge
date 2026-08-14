"""Shared streaming completion primitive over the official OpenAI SDK."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter

import openai
from openai import AsyncOpenAI, Omit, omit
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import ResponseFormatJSONObject

from mindbridge.core import ModelOutputError
from mindbridge.models.openai_errors import raise_openai_model_error
from mindbridge.telemetry import record_stage_duration, set_current_span_attributes, trace_operation

REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_JSON_CODE_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*)\r?\n```[ \t]*\Z",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class OpenAITextCompletion:
    """Text plus the provider revision fingerprint when one is available."""

    content: str
    system_fingerprint: str | None


def unwrap_json_code_fence(content: str) -> str:
    """Remove one provider-added JSON fence while leaving every other shape strict."""
    match = _JSON_CODE_FENCE.fullmatch(content.strip())
    return match.group("body") if match is not None else content


@trace_operation("mindbridge.model.stream_completion")
async def stream_text_completion(
    client: AsyncOpenAI,
    *,
    model_id: str,
    messages: list[ChatCompletionMessageParam],
    max_output_tokens: int,
    request_timeout_seconds: float,
    json_mode: bool = False,
    attempt: int = 1,
    ttft_stage: str = "model.stream_completion.ttft",
    reasoning_effort: ReasoningEffort = None,
) -> OpenAITextCompletion:
    """Read one deterministic text stream and normalize SDK failures."""
    if not 1 <= attempt <= 10:
        raise ValueError("completion attempt must be between 1 and 10")
    set_current_span_attributes(
        {
            "mindbridge.model.id": model_id,
            "mindbridge.model.json_mode": json_mode,
            "mindbridge.model.structured_attempt": attempt,
            "mindbridge.model.reasoning_effort": reasoning_effort or "provider_default",
        }
    )
    started_at = perf_counter()
    parts: list[str] = []
    finish_reason: str | None = None
    system_fingerprint: str | None = None
    first_text_seen = False
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
            if chunk.usage is not None:
                set_current_span_attributes(
                    {
                        "mindbridge.model.input_tokens": chunk.usage.prompt_tokens,
                        "mindbridge.model.output_tokens": chunk.usage.completion_tokens,
                        "mindbridge.model.total_tokens": chunk.usage.total_tokens,
                    }
                )
            system_fingerprint = system_fingerprint or chunk.system_fingerprint
            for choice in chunk.choices:
                if choice.index != 0:
                    continue
                if choice.delta.content:
                    if not first_text_seen:
                        ttft_seconds = max(0.0, perf_counter() - started_at)
                        set_current_span_attributes({"mindbridge.model.ttft_seconds": ttft_seconds})
                        record_stage_duration(ttft_stage, ttft_seconds)
                        first_text_seen = True
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
