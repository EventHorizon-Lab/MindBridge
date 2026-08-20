"""Small shared helpers for explicit process configuration contracts."""

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    StrictFloat,
    StrictInt,
    StringConstraints,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the 3.10 floor, which the mypy pin is the one that sees
    import tomli as tomllib


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


CONFIG_FILE_VARIABLE = "MINDBRIDGE_CONFIG_FILE"
"""Names the configuration file explicitly. A set-but-missing path is an error."""

DEFAULT_CONFIG_FILE = "mindbridge.toml"
"""Read from the working directory when nothing names a file.

There is deliberately no parent-directory walk and no XDG lookup: a configuration file found
somewhere the operator did not name is worse than no configuration file.
"""


def variable_name(key: str, section: str | None = None) -> str:
    """Derive the one variable a file key configures.

    The mapping is this function rather than a table, so nothing can fall behind the loader:
    key `k` under section `s` configures `MINDBRIDGE_<S>_<K>`, and a key at the top level of
    the document configures `MINDBRIDGE_<K>`.
    """
    if section is None:
        return f"MINDBRIDGE_{key.upper()}"
    return f"MINDBRIDGE_{section.upper()}_{key.upper()}"


def _configuration_document(
    environ: Mapping[str, str],
    path: Path | None,
) -> dict[str, object] | None:
    """Locate and parse the configuration file, or report that there is none."""
    located = path if path is not None else _located_path(environ)
    if located is None or not located.is_file():
        return None
    try:
        text = located.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"{located} could not be read") from error
    try:
        document = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{located} must contain valid TOML: {error}") from error
    return cast(dict[str, object], document)


def _located_path(environ: Mapping[str, str]) -> Path | None:
    """Resolve which file to read without searching anywhere the operator did not name."""
    named = optional_environment_value(environ, CONFIG_FILE_VARIABLE)
    if named is not None:
        located = Path(named)
        if not located.is_file():
            raise ValueError(f"{CONFIG_FILE_VARIABLE} names {named}, which is not a file")
        return located
    default = Path(DEFAULT_CONFIG_FILE)
    return default if default.is_file() else None


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


PluginText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# StrictInt after StrictFloat so a JSON integer is accepted where a number belongs while a
# quoted number or a bool is not. AfterValidator normalizes the accepted member to float.
PluginNumber = Annotated[StrictFloat | StrictInt, AfterValidator(float)]
PluginInteger = StrictInt


class PluginConfigModel(BaseModel):
    """Strict immutable schema for one plugin's JSON configuration object.

    `extra="forbid"` is what fails a factory on any key it would otherwise ignore.
    `protected_namespaces` is cleared because `model_id` and `model_revision` are
    MindBridge's model-identity fields, not pydantic's reserved namespace.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


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
