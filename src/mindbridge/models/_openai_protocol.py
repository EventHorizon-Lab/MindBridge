"""Pure request policy for known OpenAI families and generic compatible models."""

from collections.abc import Mapping
from re import fullmatch

from mindbridge.exceptions import ValidationError


def reasoning_model(model: str) -> bool:
    if model.startswith("gpt-5-chat"):
        return False
    return (
        model in {"gpt-5", "gpt-6"}
        or model.startswith(("gpt-5-", "gpt-5.", "gpt-6-", "gpt-6."))
        or any(model == family or model.startswith(f"{family}-") for family in ("o1", "o3", "o4"))
    )


def sampling_supported(model: str, extra_body: Mapping[str, object] | None) -> bool:
    if not reasoning_model(model):
        return True
    # The original GPT-5 models have no non-reasoning mode. Later families allow temperature
    # only with reasoning explicitly disabled; the provider still validates model availability.
    original = any(
        model == family or model.startswith(f"{family}-")
        for family in ("gpt-5", "gpt-5-mini", "gpt-5-nano", "o1", "o3", "o4")
    )
    if model.startswith(("gpt-6-astra", "gpt-6.1-sol")):
        return False
    default = "none" if fullmatch(r"gpt-5\.[12](?:-\d{4}-\d{2}-\d{2})?", model) else None
    return not original and (extra_body or {}).get("reasoning_effort", default) == "none"


def validate_sampling(
    model: str,
    *,
    temperature: float | None,
    seed: int | None,
    extra_body: Mapping[str, object] | None,
) -> None:
    body = extra_body or {}
    if not sampling_supported(model, body) and any(
        value is not None
        for value in (
            temperature,
            body.get("temperature"),
            body.get("top_p"),
            body.get("logprobs"),
            body.get("top_logprobs"),
        )
    ):
        raise ValidationError(f"{model} does not support the configured sampling controls")
    if reasoning_model(model) and (seed is not None or body.get("seed") is not None):
        raise ValidationError(f"{model} does not support generation_seed")
    reserved = {"model", "messages", "response_format", "stream", "stream_options"} & body.keys()
    if reserved:
        raise ValidationError("generation_extra_body must not override the model request contract")


def output_limit_parameter(model: str) -> str:
    return "max_completion_tokens" if reasoning_model(model) else "max_tokens"
