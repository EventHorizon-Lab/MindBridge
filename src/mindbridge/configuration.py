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


PLUGIN_SECTIONS: tuple[str, ...] = ("generator", "embedder", "media_embedder", "media_sampling")
"""Sections whose body is one plugin's config object rather than a set of named scalars.

`plugin_configuration()` reads a plugin's config as one opaque object whose schema belongs to
the plugin, so these sections serialise to their `*_CONFIG_JSON` variable instead of
contributing one variable per key. `plugin` is the exception: the selector is read separately
from the config it selects.
"""

PLUGIN_SELECTOR_KEY = "plugin"
"""Reserved inside a plugin section: it names the plugin rather than configuring it."""

KNOWN_SCALAR_KEYS: Mapping[str, tuple[str, ...]] = {
    "database": ("max_pool_size",),
    "object_storage": ("bucket", "endpoint_url", "public_endpoint_url"),
    "embedding": ("dimension", "space_id", "space_revision"),
    "aml": ("tenant_prefix",),
}
"""Sections holding named values MindBridge owns, one variable per key.

Spelled out rather than derived because nothing in the code enumerates them, and an unlisted
key has to be an error: a typo that flattens to a variable no reader looks up is a value that
silently reverts to its default, which is the failure `extra="forbid"` already prevents inside
a plugin config. `test_the_known_keys_cannot_fall_behind_what_the_code_reads` is the guard that
keeps this table honest.
"""

TOP_LEVEL_KEYS: tuple[str, ...] = ("minimum_embedding_similarity",)
"""Keys configuring one deployment-wide value that belongs to no section."""

CREDENTIAL_VARIABLES: frozenset[str] = frozenset(
    {
        "MINDBRIDGE_API_KEY",
        "MINDBRIDGE_AML_API_KEY",
        "MINDBRIDGE_DATABASE_URL",
        "MINDBRIDGE_EMBEDDER_API_KEY",
        "MINDBRIDGE_GENERATOR_API_KEY",
        "MINDBRIDGE_TASK_BROKER_URL",
        "MINDBRIDGE_TENANT_API_KEYS_JSON",
    }
)
"""The variables that may never be read from a file.

Keeping credentials out of every file is the property this split exists to preserve, so a
credential key is an error rather than a warning: a warning that is ignored puts a secret on
disk just as effectively as no check at all.
"""


def _flattened_scalars(document: Mapping[str, object]) -> dict[str, str]:
    """Flatten every file key that configures one named variable."""
    flattened: dict[str, str] = {}
    for name, value in document.items():
        if not isinstance(value, dict):
            flattened.update(_top_level(name, value))
        elif name in PLUGIN_SECTIONS:
            flattened.update(_plugin_selector(name, value))
        else:
            flattened.update(_scalar_section(name, value))
    return flattened


def _top_level(name: str, value: object) -> dict[str, str]:
    """Flatten one key that belongs to no section."""
    # The credential check comes first on every path: "put this in the environment" is what an
    # operator needs to read, where "unknown key" would send them hunting for a typo.
    _reject_credential(variable_name(name))
    if name not in TOP_LEVEL_KEYS:
        raise ValueError(f"{name} is not a known top-level configuration key")
    return _scalar(variable_name(name), value)


def _plugin_selector(name: str, body: Mapping[str, object]) -> dict[str, str]:
    """Take only the selector from a plugin section, and refuse its credentials.

    The rest of the body is one opaque object, assembled by `_flattened_plugins`.
    """
    for key in body:
        _reject_credential(variable_name(key, name))
    selector = body.get(PLUGIN_SELECTOR_KEY)
    if selector is None:
        return {}
    return _scalar(variable_name(PLUGIN_SELECTOR_KEY, name), selector)


def _scalar_section(name: str, body: Mapping[str, object]) -> dict[str, str]:
    """Flatten one section of named values, rejecting anything no variable could carry."""
    for key in body:
        _reject_credential(variable_name(key, name))
    known = KNOWN_SCALAR_KEYS.get(name)
    if known is None:
        raise ValueError(f"{name} is not a known configuration section")
    flattened: dict[str, str] = {}
    for key, entry in body.items():
        if isinstance(entry, dict):
            raise ValueError(f"{name}.{key} must not nest another table")
        if key not in known:
            raise ValueError(f"{key} is not a known key of [{name}]")
        flattened.update(_scalar(variable_name(key, name), entry))
    return flattened


def _scalar(variable: str, value: object) -> dict[str, str]:
    """Render one file scalar as the string an environment reader would have received."""
    _reject_credential(variable)
    if isinstance(value, bool):
        return {variable: "true" if value else "false"}
    if isinstance(value, str | int | float):
        return {variable: str(value)}
    raise ValueError(f"{variable} must be text, a number, or a boolean")


def _reject_credential(variable: str) -> None:
    """Keep every credential out of every file, in one place both flatteners call."""
    if variable in CREDENTIAL_VARIABLES:
        raise ValueError(
            f"{variable} is a credential and must not appear in the configuration file. "
            f"Set it in the environment instead."
        )


ENCODER_SECTIONS: tuple[str, ...] = ("embedder", "media_embedder")
"""Plugin sections that must share the one deployment-wide embedding space."""

EMBEDDING_SECTION = "embedding"
"""The section whose keys every encoder section inherits."""


def _flattened_plugins(
    document: Mapping[str, object],
    environ: Mapping[str, str],
) -> dict[str, str]:
    """Serialise each plugin section into the `*_CONFIG_JSON` its factory reads."""
    space = document.get(EMBEDDING_SECTION)
    shared = space if isinstance(space, dict) else {}
    flattened: dict[str, str] = {}
    for section in PLUGIN_SECTIONS:
        body = document.get(section)
        if not isinstance(body, dict):
            continue
        assembled: dict[str, object] = {}
        if section in ENCODER_SECTIONS:
            assembled.update(shared)
        assembled.update({key: value for key, value in body.items() if key != PLUGIN_SELECTOR_KEY})
        assembled.update(_overrides(section, assembled, environ))
        flattened[variable_name("config_json", section)] = json.dumps(
            assembled, sort_keys=True, allow_nan=False
        )
    return flattened


def _overrides(
    section: str,
    assembled: Mapping[str, object],
    environ: Mapping[str, str],
) -> dict[str, object]:
    """Read every individual variable that overrides one key of this plugin section.

    The environment wins per key rather than per section. Splicing in only the credential would
    make every other individual variable silently dead the moment a file existed, because
    `plugin_configuration()` short-circuits on `*_CONFIG_JSON` and never calls the builder that
    reads them.
    """
    prefix = f"MINDBRIDGE_{section.upper()}_"
    reserved = {
        variable_name("config_json", section),
        variable_name(PLUGIN_SELECTOR_KEY, section),
    }
    overrides: dict[str, object] = {}
    for name, text in environ.items():
        if not name.startswith(prefix) or name in reserved:
            continue
        key = name[len(prefix) :].lower()
        overrides[key] = _as_declared(assembled.get(key), text)
    return overrides


def _as_declared(declared: object, text: str) -> object:
    """Read an environment override in the type the file declared for that key.

    A validated plugin config rejects `"1800"` where it wants a number and `"false"` where it
    wants a boolean, so an override cannot arrive as text wherever the file said otherwise. A
    key the file omits has no declared type and arrives as text, which is what every individual
    plugin variable in the contract today already is.

    `bool` precedes `int` because it subclasses it, and `bool("false")` is `True`.
    """
    if isinstance(declared, bool):
        if text.lower() not in {"true", "false"}:
            raise ValueError(f"{text!r} must be 'true' or 'false'")
        return text.lower() == "true"
    if isinstance(declared, int):
        return int(text)
    if isinstance(declared, float):
        return float(text)
    return text


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
