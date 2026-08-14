"""Small shared helpers for explicit process configuration contracts."""

import argparse
import json
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import NoReturn, cast


def require_environment_value(environ: Mapping[str, str], name: str) -> str:
    """Return one non-blank required value without logging its contents."""
    value = environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} must be configured")
    return value


def optional_environment_value(environ: Mapping[str, str], name: str) -> str | None:
    """Normalize a missing optional value to None and reject blank equivalently."""
    value = environ.get(name)
    return value if value is not None and value.strip() else None


def validate_plugin_name(value: object, name: str = "plugin name") -> str:
    """Require the canonical entry-point spelling used by every process."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or value.lower() != value
    ):
        raise ValueError(f"{name} must be trimmed lowercase text")
    return value


def plugin_configuration(
    environ: Mapping[str, str],
    name: str,
    default: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Read one plugin JSON object, evaluating its fallback only when needed."""
    encoded = optional_environment_value(environ, name)
    if encoded is None:
        if default is None:
            raise ValueError(f"{name} must be configured for the selected plugin")
        return dict(default())
    try:
        value = json.loads(encoded, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} must contain valid JSON") from error
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not key.strip() for key in value
    ):
        raise ValueError(f"{name} must contain a JSON object with non-empty keys")
    return cast(dict[str, object], value)


def copy_plugin_configuration(
    config: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    """Copy a direct plugin configuration after enforcing the JSON contract."""
    if any(not isinstance(key, str) or not key.strip() for key in config):
        raise ValueError(f"{name} keys must be non-empty text")
    try:
        json.dumps(config, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain JSON values") from error
    return dict(config)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number {value!r} is not supported")


def parse_aware_datetime(value: str) -> datetime:
    """Parse an ISO-8601 CLI value that includes an explicit timezone."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an ISO-8601 datetime") from error
    if parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("must include a timezone offset")
    return parsed
