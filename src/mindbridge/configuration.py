"""Small shared helpers for explicit process environment contracts."""

from collections.abc import Mapping


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
