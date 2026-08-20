"""Provider-neutral structured generation mechanics."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TypeVar

from pydantic import BaseModel

from mindbridge.application.capabilities import (
    GenerateRequest,
    GenerateResult,
    Generator,
    OutputSchema,
)
from mindbridge.core import ModelOutputError
from mindbridge.telemetry import log_fields, logger, set_current_span_attributes

_Output = TypeVar("_Output")
_LOGGER = logger("mindbridge.application.pipelines.structured")
_JSON_CODE_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*)\r?\n```[ \t]*\Z",
    re.DOTALL | re.IGNORECASE,
)
_UNCONSTRAINED_KEYWORDS = frozenset(
    {
        "default",
        "examples",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "pattern",
        "title",
        "uniqueItems",
    }
)
"""Keywords a strict schema drops rather than sends.

Constrained decoding accepts a shape, not a range: providers reject several of these
outright, and the ones they tolerate they ignore. Dropping them loses nothing, because the
same model that derived the schema still validates every bound after parsing -- the schema
buys the field names and types, the validator keeps the limits.
"""
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "allOf",
        "dependentRequired",
        "dependentSchemas",
        "else",
        "if",
        "not",
        "oneOf",
        "patternProperties",
        "prefixItems",
        "then",
    }
)
_NAMESPACE_KEYWORDS = frozenset({"$defs", "definitions", "properties"})
"""Keywords whose keys are names the model chose, not schema keywords of their own.

Filtering these as keywords is how a field called `title` or `format` disappears from
`properties` while `additionalProperties: false` forbids the model from emitting it -- a
shape the validator then rejects on every single call -- and how a field called `then` is
mistaken for a conditional subschema and fails derivation outright.
"""
"""Keywords no strict decoder will compile, so a schema using one must not be built.

Composition and conditional subschemas are excluded by the provider contract rather than
merely ignored, and a fixed-length tuple reaches JSON Schema as `prefixItems`, which has no
strict equivalent either. Naming all of them matters because this guard is the only thing
standing between an output model growing an inexpressible shape and a deployment discovering
it on its first observation.
"""


def output_schema(name: str, model: type[BaseModel]) -> OutputSchema:
    """Derive one strict, provider-ready schema from the model that parses the output.

    Deriving beats hand-writing: the schema and the parser cannot drift apart, because a
    field renamed in the model is renamed in the constraint by construction.
    """
    schema = _strict_schema(model.model_json_schema())
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        # Required by the provider contract: a strict root has to be an object, so a model
        # whose root is an array or a union cannot be constrained at all. Caught here rather
        # than as a rejected request, which the fallback would quietly turn into JSON mode.
        raise ValueError("a strict output schema must have an object at its root")
    return OutputSchema(
        name=name,
        json_schema=json.dumps(schema, separators=(",", ":"), sort_keys=True),
    )


def _strict_schema(node: object) -> object:
    """Rewrite one JSON Schema node into the subset constrained decoding accepts."""
    if isinstance(node, list):
        return [_strict_schema(item) for item in node]
    if not isinstance(node, Mapping):
        return node
    unsupported = _UNSUPPORTED_KEYWORDS.intersection(node)
    if unsupported:
        # Raised at derivation rather than at request time: an output model that grows a
        # shape the provider cannot express should fail this package's own tests, not a
        # deployment's first observation.
        raise ValueError(
            f"output schema uses keywords strict decoding rejects: {sorted(unsupported)}"
        )
    rewritten = {
        key: _strict_subschemas(value) if key in _NAMESPACE_KEYWORDS else _strict_schema(value)
        for key, value in node.items()
        if key not in _UNCONSTRAINED_KEYWORDS
    }
    properties = rewritten.get("properties")
    if isinstance(properties, Mapping):
        # Every property is required and no others are allowed. A field carrying a default is
        # optional to the validator but must still be emitted, because strict decoding has no
        # way to express "may be absent" -- and an absent key is what a default is for.
        rewritten["required"] = sorted(properties)
        rewritten["additionalProperties"] = False
    return rewritten


def _strict_subschemas(node: object) -> object:
    """Rewrite the subschemas one namespace holds, leaving the names it keys them by alone."""
    if not isinstance(node, Mapping):
        return _strict_schema(node)
    return {name: _strict_schema(subschema) for name, subschema in node.items()}


async def generate_json(
    generator: Generator,
    request: GenerateRequest,
    parse: Callable[[str], _Output],
) -> tuple[_Output, GenerateResult]:
    """Retry malformed structured output once with provider JSON mode."""
    result = await generator.generate(request)
    try:
        output = parse(result.text)
    except ModelOutputError as error:
        set_current_span_attributes({"mindbridge.model.structured_retry_count": 1})
        _LOGGER.warning(
            "structured output rejected, retrying once",
            extra=_retry_fields(request, error),
        )
        result = await generator.generate(replace(request, json_mode=True))
        return parse(result.text), result
    set_current_span_attributes({"mindbridge.model.structured_retry_count": 0})
    return output, result


def unwrap_json_code_fence(content: str) -> str:
    """Remove one provider-added JSON fence while leaving other shapes strict."""
    match = _JSON_CODE_FENCE.fullmatch(content.strip())
    return match.group("body") if match is not None else content


def _retry_fields(request: GenerateRequest, error: ModelOutputError) -> dict[str, object]:
    """Report why one generation is being paid for twice, without the content that failed.

    `constrained` is the field worth reading: `json_mode` does not weaken a request that
    still carries a schema, because `_response_format` prefers the schema whenever the
    endpoint accepts one. On such an endpoint the retry repeats the first attempt's
    arguments at temperature 0, so a run showing these lines is paying for two identical
    generations and the bound that failed is the one to look at.
    """
    return log_fields(
        schema=request.output_schema.name if request.output_schema is not None else None,
        constrained=request.output_schema is not None,
        error_type=type(error).__name__,
        error=str(error),
    )
