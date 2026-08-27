"""Optional protocol adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mindbridge.api.app import create_app

__all__ = ["create_app"]


def __getattr__(name: str) -> object:
    if name == "create_app":
        from mindbridge.api.app import create_app

        return create_app
    raise AttributeError(name)
