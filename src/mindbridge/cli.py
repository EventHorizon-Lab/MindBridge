"""Command-line entry point for the local MindBridge product."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path

USAGE_EXIT_CODE = 2
INTERRUPT_EXIT_CODE = 130

_DESCRIPTION = "Fast local multimodal memory for Python agents."


def installed_version() -> str:
    """Return the installed MindBridge distribution version."""
    return version("mindbridge")


def parser(
    *,
    prog: str | None,
    description: str | None,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    """Build the shared parser shape used by legacy standalone modules."""
    built = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog=epilog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    built.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"mindbridge {installed_version()}",
        help="show the installed version and exit",
    )
    return built


ExtensionHandler = Callable[[Sequence[str], str], int]


def main(
    argv: Sequence[str] | None = None,
    *,
    extensions: Mapping[str, tuple[str, ExtensionHandler]] | None = None,
) -> int:
    """Run one product command under the documented exit-code contract."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    available = extensions or {}
    root = _root_parser({name: extension[0] for name, extension in available.items()})
    if not arguments:
        root.print_help()
        return 0

    command = arguments[0]
    try:
        if command in available:
            return available[command][1](arguments[1:], f"mindbridge {command}")
        options = root.parse_args(arguments)
        command = options.command
        if command == "serve":
            from mindbridge.server import serve

            serve(
                data_dir=options.data_dir,
                host=options.host,
                port=options.port,
                tls_certfile=options.tls_certfile,
                tls_keyfile=options.tls_keyfile,
            )
        elif command == "mcp":
            from mindbridge.api.mcp import run_mcp

            run_mcp(data_dir=options.data_dir)
        else:
            from mindbridge import Memory

            with Memory(data_dir=options.data_dir) as memory:
                if command == "reindex":
                    memory.reindex()
                else:
                    memory.optimize()
        return 0
    except KeyboardInterrupt:
        print(f"mindbridge {command}: interrupted", file=sys.stderr)
        return INTERRUPT_EXIT_CODE
    except Exception as error:
        print(f"mindbridge {command}: error: {_one_line(error)}", file=sys.stderr)
        return 1


def _root_parser(extensions: Mapping[str, str] | None = None) -> argparse.ArgumentParser:
    root = parser(prog="mindbridge", description=_DESCRIPTION)
    commands = root.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve the HTTP API")
    _data_dir_argument(serve)
    serve.add_argument("--host", default="127.0.0.1", help="address to bind")
    serve.add_argument("--port", type=_port, default=8000, help="TCP port to bind")
    serve.add_argument("--tls-certfile", type=Path, help="TLS certificate chain file")
    serve.add_argument("--tls-keyfile", type=Path, help="TLS private key file")

    reindex = commands.add_parser("reindex", help="rebuild the local search index")
    _data_dir_argument(reindex)

    optimize = commands.add_parser("optimize", help="optimize the local search index")
    _data_dir_argument(optimize)

    mcp = commands.add_parser("mcp", help="serve MCP over stdio")
    _data_dir_argument(mcp)
    for name, summary in (extensions or {}).items():
        commands.add_parser(name, help=summary)
    return root


def _data_dir_argument(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".mindbridge"),
        help="local MindBridge data directory",
    )


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _one_line(error: BaseException) -> str:
    return " ".join(str(error).split()) or type(error).__name__


def mcp() -> int:
    """Run the MCP console-script alias."""
    return main(("mcp", *sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
