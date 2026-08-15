"""Provider-neutral structured generation mechanics."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar

from mindbridge.application.capabilities import GenerateRequest, GenerateResult, Generator
from mindbridge.core import ModelOutputError

_Output = TypeVar("_Output")
_JSON_CODE_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*)\r?\n```[ \t]*\Z",
    re.DOTALL | re.IGNORECASE,
)


async def generate_json(
    generator: Generator,
    request: GenerateRequest,
    parse: Callable[[str], _Output],
) -> tuple[_Output, GenerateResult]:
    """Retry malformed structured output once with provider JSON mode."""
    result = await generator.generate(request)
    try:
        return parse(result.text), result
    except ModelOutputError:
        result = await generator.generate(replace(request, json_mode=True))
        return parse(result.text), result


def unwrap_json_code_fence(content: str) -> str:
    """Remove one provider-added JSON fence while leaving other shapes strict."""
    match = _JSON_CODE_FENCE.fullmatch(content.strip())
    return match.group("body") if match is not None else content
