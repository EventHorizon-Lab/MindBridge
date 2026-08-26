"""Store one local file as evidence and observe it, without writing a program to do it.

`observe_file()` has done this from Python since the SDK gained it, and that was the whole
answer for a while: hash the file, ask for a presigned upload, send the bytes, submit an
observation naming the object. Three round trips in a documented order, none of which an
operator should be assembling by hand with `curl` -- and until this command, assembling them by
hand or writing an async Python script were the only two ways to give MindBridge a file that was
already on disk.

Deliberately thin. Everything about correctness lives in `sdk.observe_file`: the digest that
names the object, the kind read off the extension, the sequence derived from the digest so two
files cannot collide onto one observation, and the modification time standing in for a clock so
a repeat is idempotent rather than merely retried. This module parses flags and prints a receipt.

What it prints is the receipt, not a result. A returned `processing_job_id` means the
observation is stored and the memory derived from it does not exist yet, which is the one thing
about this API that surprises everybody; `mindbridge jobs` is what reports that job afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from mindbridge.cli import parser as build_parser
from mindbridge.configuration import configuration_source
from mindbridge.contracts import ObservationReceipt
from mindbridge.core import MediaKind, SensorKind

OBSERVE_ENVIRONMENT = """environment:
  MINDBRIDGE_API_KEY  bearer token for --api-base-url; read from the environment so the
                      credential never reaches a process list or a shell history"""


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Upload one file and observe it, printing the receipt the API answered with."""
    parser = build_parser(prog=prog, description=__doc__, epilog=OBSERVE_ENVIRONMENT)
    parser.add_argument("path", type=Path, help="local file to store as evidence and observe")
    parser.add_argument(
        "--tenant-id", required=True, help="tenant this observation is written under"
    )
    parser.add_argument(
        "--api-base-url", default="http://localhost:8000", help="base URL of the MindBridge API"
    )
    parser.add_argument(
        "--kind",
        choices=tuple(kind.value for kind in MediaKind),
        help="what the file holds; read from its extension when omitted, and required for a "
        "container this API does not recognize",
    )
    parser.add_argument(
        "--occurred-at",
        type=datetime.fromisoformat,
        help="when the recording starts, ISO 8601. Defaults to the file's modification time, "
        "which is stable across a repeat where a clock read would not be",
    )
    parser.add_argument(
        "--ended-at",
        type=datetime.fromisoformat,
        help="when the recording ends, ISO 8601. Nothing here decodes the file, so a duration "
        "left unsaid makes the observation cover an instant",
    )
    parser.add_argument(
        "--device-id", default="cli", help="device identity this observation is attributed to"
    )
    parser.add_argument(
        "--boot-id",
        default="cli",
        help="boot identity of that device; with --device-id and --sequence it is what the "
        "observation ID is derived from",
    )
    parser.add_argument(
        "--sequence",
        type=int,
        help="ordinal within this boot; derived from the file's digest when omitted, so two "
        "different files cannot collide onto one observation",
    )
    parser.add_argument(
        "--sensor",
        choices=tuple(sensor.value for sensor in SensorKind),
        help="sensor that produced it; a microphone for audio and a camera otherwise",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="deadline for one API request; the upload itself is not one request",
    )
    arguments = parser.parse_args(argv)
    receipt = asyncio.run(_observe(arguments))
    # JSON on stdout, like `edge sync`: the IDs are what a next command is given, and a shell
    # that has to cut them out of prose is a shell that will cut out the wrong column one day.
    print(
        json.dumps(
            {
                "observation_id": receipt.observation_id,
                "processing_job_id": receipt.processing_job_id,
                "evidence_ids": list(receipt.evidence_ids),
                "status": receipt.status,
            }
        )
    )


async def _observe(arguments: argparse.Namespace) -> ObservationReceipt:
    """Open a client, observe the file, and close the pools whatever happened.

    `sdk` is imported here rather than at module scope: it needs `httpx`, and this module is
    reached by `mindbridge --help`, which must not pay for it.

    The kind and the sensor are converted to their own enums rather than passed as the strings
    argparse collected. `observe_file` compares the kind against the one the extension implies,
    and a string never equals an enum, so the suffix would be dropped from the object key on
    every call that named a kind at all.
    """
    from mindbridge.sdk import MindBridge

    client = MindBridge.connect(
        base_url=arguments.api_base_url,
        api_key=configuration_source().get("MINDBRIDGE_API_KEY"),
        timeout_seconds=arguments.timeout_seconds,
    )
    async with client:
        return await client.observe_file(
            arguments.path,
            tenant_id=arguments.tenant_id,
            kind=MediaKind(arguments.kind) if arguments.kind is not None else None,
            occurred_at=arguments.occurred_at,
            ended_at=arguments.ended_at,
            device_id=arguments.device_id,
            boot_id=arguments.boot_id,
            sequence=arguments.sequence,
            sensor=SensorKind(arguments.sensor) if arguments.sensor is not None else None,
        )


if __name__ == "__main__":
    main()
