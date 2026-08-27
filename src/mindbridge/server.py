"""Compose the public HTTP app with its single-process server."""

from __future__ import annotations

import os
from ipaddress import ip_address
from pathlib import Path

from mindbridge.api import create_app

API_KEY_ENVIRONMENT_VARIABLE = "MINDBRIDGE_API_KEY"


def serve(
    *,
    data_dir: str | Path = ".mindbridge",
    host: str = "127.0.0.1",
    port: int = 8000,
    tls_certfile: str | Path | None = None,
    tls_keyfile: str | Path | None = None,
) -> None:
    """Serve one local data directory from one Uvicorn worker."""
    normalized_host = host.strip()
    if not normalized_host:
        raise ValueError("host must not be empty")
    if isinstance(port, bool) or not 1 <= port <= 65_535:
        raise ValueError("port must be between 1 and 65535")
    if (tls_certfile is None) != (tls_keyfile is None):
        raise ValueError("tls_certfile and tls_keyfile must be provided together")

    cert_path = None if tls_certfile is None else Path(tls_certfile).expanduser().resolve()
    key_path = None if tls_keyfile is None else Path(tls_keyfile).expanduser().resolve()
    for name, path in (("tls_certfile", cert_path), ("tls_keyfile", key_path)):
        if path is not None and not path.is_file():
            raise ValueError(f"{name} must name a readable file: {path}")

    api_key = os.environ.get(API_KEY_ENVIRONMENT_VARIABLE)
    if api_key is not None and not api_key.strip():
        api_key = None
    if not _is_loopback(normalized_host) and (
        api_key is None or cert_path is None or key_path is None
    ):
        raise ValueError(
            f"{API_KEY_ENVIRONMENT_VARIABLE} and TLS certificate/key files are required "
            "when binding outside loopback"
        )

    from uvicorn import run

    app = create_app(data_dir=data_dir, api_key=api_key)
    run(
        app,
        host=normalized_host,
        port=port,
        workers=1,
        ssl_certfile=None if cert_path is None else str(cert_path),
        ssl_keyfile=None if key_path is None else str(key_path),
    )


def _is_loopback(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


__all__ = ["create_app", "serve"]
