"""Focused contracts for the local product CLI and HTTP server launcher."""

from __future__ import annotations

from pathlib import Path

import pytest
import uvicorn

import mindbridge
from mindbridge import cli, server
from mindbridge.api import mcp as mcp_adapter


class _FakeMemory:
    def __init__(self, data_dir: str | Path, opened: list[_FakeMemory]) -> None:
        self.data_dir = Path(data_dir)
        self.operation: str | None = None
        self.closed = False
        opened.append(self)

    def __enter__(self) -> _FakeMemory:
        return self

    def __exit__(self, *_error: object) -> None:
        self.closed = True

    def reindex(self) -> int:
        self.operation = "reindex"
        return 1

    def optimize(self) -> None:
        self.operation = "optimize"


def test_cli_dispatches_only_the_local_product_surface(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opened: list[_FakeMemory] = []
    served: list[tuple[Path, str, int, Path | None, Path | None]] = []
    mcp_directories: list[Path] = []

    def memory(data_dir: str | Path) -> _FakeMemory:
        return _FakeMemory(data_dir, opened)

    def serve(
        *,
        data_dir: str | Path,
        host: str,
        port: int,
        tls_certfile: str | Path | None,
        tls_keyfile: str | Path | None,
    ) -> None:
        served.append(
            (
                Path(data_dir),
                host,
                port,
                None if tls_certfile is None else Path(tls_certfile),
                None if tls_keyfile is None else Path(tls_keyfile),
            )
        )

    monkeypatch.setattr(mindbridge, "Memory", memory)
    monkeypatch.setattr(server, "serve", serve)
    monkeypatch.setattr(
        mcp_adapter,
        "run_mcp",
        lambda data_dir: mcp_directories.append(Path(data_dir)),
    )

    assert cli.main(["reindex", "--data-dir", "first"]) == 0
    assert cli.main(["optimize", "--data-dir", "second"]) == 0
    assert cli.main(["serve", "--data-dir", "third", "--host", "localhost", "--port", "9000"]) == 0
    assert cli.main(["mcp", "--data-dir", "fourth"]) == 0

    assert [(item.data_dir, item.operation, item.closed) for item in opened] == [
        (Path("first"), "reindex", True),
        (Path("second"), "optimize", True),
    ]
    assert served == [(Path("third"), "localhost", 9000, None, None)]
    assert mcp_directories == [Path("fourth")]

    assert cli.main([]) == 0
    help_text = capsys.readouterr().out
    for command in ("serve", "reindex", "optimize", "mcp"):
        assert command in help_text
    for removed in ("consolidate", "lifecycle", "jobs"):
        assert removed not in help_text


def test_server_requires_authentication_before_a_non_loopback_bind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = object()
    created: list[tuple[Path, str | None]] = []
    runs: list[tuple[object, str, int, int, str | None, str | None]] = []

    def create_app(*, data_dir: str | Path, api_key: str | None) -> object:
        created.append((Path(data_dir), api_key))
        return app

    def run(
        application: object,
        *,
        host: str,
        port: int,
        workers: int,
        ssl_certfile: str | None,
        ssl_keyfile: str | None,
    ) -> None:
        runs.append((application, host, port, workers, ssl_certfile, ssl_keyfile))

    monkeypatch.setattr(server, "create_app", create_app)
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.delenv(server.API_KEY_ENVIRONMENT_VARIABLE, raising=False)

    with pytest.raises(ValueError, match="MINDBRIDGE_API_KEY"):
        server.serve(data_dir="remote", host="0.0.0.0")
    assert created == []
    assert runs == []

    monkeypatch.setenv(server.API_KEY_ENVIRONMENT_VARIABLE, "secret")
    with pytest.raises(ValueError, match="TLS"):
        server.serve(data_dir="remote", host="0.0.0.0")

    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.touch()
    key.touch()
    server.serve(
        data_dir="remote",
        host="0.0.0.0",
        port=9000,
        tls_certfile=cert,
        tls_keyfile=key,
    )

    assert created == [(Path("remote"), "secret")]
    assert runs == [(app, "0.0.0.0", 9000, 1, str(cert), str(key))]


def test_cli_has_stable_failure_usage_and_interrupt_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(**_options: object) -> None:
        raise RuntimeError("first line\nsecond line")

    monkeypatch.setattr(server, "serve", fail)
    assert cli.main(["serve"]) == 1
    assert capsys.readouterr().err == "mindbridge serve: error: first line second line\n"

    def interrupt(**_options: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(server, "serve", interrupt)
    assert cli.main(["serve"]) == cli.INTERRUPT_EXIT_CODE
    assert capsys.readouterr().err == "mindbridge serve: interrupted\n"

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["unknown"])
    assert exit_info.value.code == cli.USAGE_EXIT_CODE


@pytest.mark.parametrize("flag", ["-V", "--version", "-h", "--help"])
def test_global_metadata_flags_exit_successfully(
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "installed_version", lambda: "9.8.7")

    with pytest.raises(SystemExit) as exit_info:
        cli.main([flag])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out
