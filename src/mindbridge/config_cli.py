"""The pre-flight configuration report: what one role still needs, in one pass.

Starting a process was the only validator MindBridge had, and it fails on the first missing
value -- so an operator missing five discovered them one restart at a time. This command asks a
role's own settings class what it requires, substituting a placeholder for each complaint and
retrying, so the list of requirements comes from the class rather than from a table here that
would drift away from it.

Values are never printed. A credential is reported as present or missing and nothing more,
because the same code path handles credentials and structure.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from mindbridge.cli import parser
from mindbridge.configuration import (
    CONFIG_FILE_VARIABLE,
    CREDENTIAL_VARIABLES,
    DEFAULT_CONFIG_FILE,
    MissingConfigurationError,
    configuration_source,
    require_environment_value,
)

ROLES: tuple[str, ...] = ("api", "mcp", "worker", "consolidate", "lifecycle", "edge-sync")
"""The six roles `docs/configuration.md` documents, addressable by name."""

PROBE_CEILING = 64
"""Bound on the ask-and-retry loop, so a failure that is not a missing value cannot spin."""

_PLACEHOLDERS: Mapping[str, str] = {
    "MINDBRIDGE_DATABASE_URL": "postgresql://placeholder:placeholder@localhost:5432/placeholder",
    "MINDBRIDGE_TASK_BROKER_URL": "redis://localhost:6379/0",
    "MINDBRIDGE_TENANT_API_KEYS_JSON": '{"placeholder":["' + "0" * 48 + '"]}',
}
"""Values that satisfy a format check without being usable anywhere.

A non-empty string is enough for most variables, but a DSN, a broker URL, and the tenant key
map are parsed as they are read, so without these the probe would stop at a format error
instead of continuing on to the next missing name.
"""

_GENERIC_PLACEHOLDER = "placeholder" * 6
"""Long enough to clear the 32-character floor the API applies to a tenant key."""


ROLE_EXTRAS: Mapping[str, str] = {
    "api": "server",
    "mcp": "server",
    "worker": "server",
    "consolidate": "server",
    "lifecycle": "server",
}
"""The extra each role's settings class needs, for roles whose class is not in the core install.

`edge-sync` is absent because it needs none. This command itself stays a core module so it runs
on a bare install, which is the state an operator reaches for it from -- but a role whose class
cannot be imported has to say which extra is missing rather than surface a bare ImportError,
because `mindbridge.cli` only names an extra for commands that declare one, and this one
deliberately does not.
"""


def _settings_probe(role: str) -> Callable[[Mapping[str, str]], object]:
    """Return the callable that builds one role's settings, imported only when it is asked for.

    The import is deferred for the same reason `mindbridge.cli` defers a subcommand's: this
    command has to run on a bare install, which is exactly when an operator needs it.
    """
    try:
        return _imported_probe(role)
    except ImportError as error:
        extra = ROLE_EXTRAS.get(role)
        remedy = f"; install it with `uv sync --extra {extra}`" if extra else ""
        raise ImportError(f"checking --role {role} needs {error.name}{remedy}") from error


def _imported_probe(role: str) -> Callable[[Mapping[str, str]], object]:
    """Import one role's settings class, letting an absent extra raise."""
    if role == "api":
        from mindbridge.api.runtime import Settings, require_rest_authentication

        # The tenant key map is optional on Settings because MCP does not require REST
        # authentication. Reporting the API as ready without it would name a role that cannot
        # start.
        def _api(source: Mapping[str, str]) -> object:
            settings = Settings.from_environment(source)
            _validate_models(
                settings.generator_plugin,
                settings.generator_config,
                settings.embedder_plugin,
                settings.embedder_config,
            )
            require_rest_authentication(settings)
            return settings

        return _api
    if role == "mcp":
        from mindbridge.api.runtime import Settings

        def _mcp(source: Mapping[str, str]) -> object:
            settings = Settings.from_environment(source)
            _validate_models(
                settings.generator_plugin,
                settings.generator_config,
                settings.embedder_plugin,
                settings.embedder_config,
            )
            return settings

        return _mcp
    if role == "worker":
        from mindbridge.worker import WorkerSettings

        def _worker(source: Mapping[str, str]) -> object:
            settings = WorkerSettings.from_environment(source)
            _validate_models(
                settings.generator_plugin,
                settings.generator_config,
                settings.text_embedder_plugin,
                settings.text_embedder_config,
            )
            _validate_embedder(settings.media_embedder_plugin, settings.media_embedder_config)
            return settings

        return _worker
    if role == "consolidate":
        from mindbridge.consolidation_cli import ConsolidationSettings

        def _consolidate(source: Mapping[str, str]) -> object:
            settings = ConsolidationSettings.from_environment(source)
            _validate_models(
                settings.generator_plugin,
                settings.generator_config,
                settings.embedder_plugin,
                settings.embedder_config,
            )
            return settings

        return _consolidate
    if role == "lifecycle":
        return lambda source: require_environment_value(source, "MINDBRIDGE_DATABASE_URL")
    return lambda source: require_environment_value(source, "MINDBRIDGE_API_KEY")


def _validate_models(
    generator_plugin: str,
    generator_config: Mapping[str, object],
    embedder_plugin: str,
    embedder_config: Mapping[str, object],
) -> None:
    from mindbridge.models.plugins import validate_generator_configuration

    validate_generator_configuration(generator_plugin, generator_config)
    _validate_embedder(embedder_plugin, embedder_config)


def _validate_embedder(plugin: str, config: Mapping[str, object]) -> None:
    from mindbridge.models.plugins import validate_embedder_configuration

    validate_embedder_configuration(plugin, config)


def _missing(role: str, resolved: Mapping[str, str]) -> tuple[list[str], str | None]:
    """List every required variable this role cannot resolve, plus any other failure.

    Each `MissingConfigurationError` names one variable, so substituting a placeholder for it
    and asking again walks the whole requirement set. Anything else is reported as it arrived:
    a format error is a real answer, not something to probe past.
    """
    build = _settings_probe(role)
    probed = dict(resolved)
    missing: list[str] = []
    for _ in range(PROBE_CEILING):
        try:
            build(probed)
        except MissingConfigurationError as error:
            missing.append(error.name)
            probed[error.name] = _PLACEHOLDERS.get(error.name, _GENERIC_PLACEHOLDER)
            continue
        # Broad on purpose: a format error or a bad plugin name is a real answer for the
        # operator to read, not something to probe past.
        except Exception as error:
            return missing, f"{type(error).__name__}: {error}"
        return missing, None
    return missing, f"more than {PROBE_CEILING} settings are missing"


def _report(missing: Sequence[str], resolved: Mapping[str, str], failure: str | None) -> None:
    """Print one line per setting, naming where each resolved value came from."""
    for name in sorted(missing):
        kind = "credential" if name in CREDENTIAL_VARIABLES else "setting"
        print(f"missing  {name}  ({kind}, required)")
    for name in sorted(name for name in resolved if name.startswith("MINDBRIDGE_")):
        if name in missing:
            continue
        origin = (
            "environment"
            if name in os.environ
            else os.environ.get(CONFIG_FILE_VARIABLE, DEFAULT_CONFIG_FILE)
        )
        print(f"present  {name}  (from {origin})")
    if failure is not None:
        print(f"invalid  {failure}")


def main(argv: Sequence[str], *, prog: str) -> int:
    """Report what one role still needs before it could start."""
    built = parser(
        prog=prog,
        description="Report whether one role's configuration is complete.",
        epilog=(
            "Values are never printed: a credential is reported as present or missing and\n"
            "nothing more, because the same code path handles credentials and structure."
        ),
    )
    built.add_argument(
        "--role",
        required=True,
        choices=ROLES,
        help="the process whose configuration to check",
    )
    arguments = built.parse_args(list(argv))
    resolved = configuration_source()
    probe_source = dict(os.environ)
    if (
        not probe_source.get(CONFIG_FILE_VARIABLE, "").strip()
        and Path(DEFAULT_CONFIG_FILE).is_file()
    ):
        probe_source[CONFIG_FILE_VARIABLE] = DEFAULT_CONFIG_FILE
    missing, failure = _missing(arguments.role, probe_source)
    _report(missing, resolved, failure)

    if missing and Path(".env").is_file():
        print("\nFound .env, which nothing here loads. Try `uv run --env-file .env ...`.")
    if not missing and failure is None:
        print(f"\n{arguments.role} is ready.")
        return 0
    print(f"\n{arguments.role} is not ready: {len(missing)} missing.")
    return 1
