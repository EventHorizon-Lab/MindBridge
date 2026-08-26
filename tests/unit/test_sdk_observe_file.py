"""Contract tests for handing MindBridge a local file instead of an object key."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from mindbridge.core import MediaKind, SensorKind
from mindbridge.sdk import _UPLOAD_CHUNK_BYTES, MindBridge, MindBridgeError, _file_chunks

CONTENT = b"\x00\x01\x02" * 64
DIGEST = hashlib.sha256(CONTENT).hexdigest()
MTIME = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


class FakeDeployment:
    """The two endpoints an upload touches: the API, and object storage itself.

    Separate transports because they are separate hosts holding separate credentials, which is
    as much under test here as any single assertion is.
    """

    def __init__(self, *, upload_status: int = 200) -> None:
        self.api_requests: list[tuple[str, dict[str, Any]]] = []
        self.uploads: list[httpx.Request] = []
        self.upload_bodies: list[bytes] = []
        self._upload_status = upload_status

    async def api(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(await request.aread())
        self.api_requests.append((request.url.path, payload))
        if request.url.path == "/v1/media/uploads":
            return httpx.Response(
                200,
                json={
                    "upload_url": f"https://objects.example.test/signed/{payload['sha256']}",
                    "uri": (
                        "s3://memory/tenants/tenant_01/media/"
                        f"{payload['sha256']}{payload['suffix'] or ''}"
                    ),
                    "expires_at": "2026-08-11T12:15:00Z",
                    "trace_id": "trace_upload",
                },
            )
        return httpx.Response(
            202,
            json={
                "observation_id": "observation_01",
                "processing_job_id": "job_01",
                "evidence_ids": [],
                "idempotency_key": "derived-key",
                "status": "accepted",
                "trace_id": "trace_observe",
            },
        )

    async def storage(self, request: httpx.Request) -> httpx.Response:
        self.upload_bodies.append(await request.aread())
        self.uploads.append(request)
        return httpx.Response(self._upload_status)

    def client(self) -> MindBridge:
        return MindBridge(
            httpx.AsyncClient(
                base_url="https://memory.example.test/",
                headers={"Authorization": "Bearer tenant-01-test-key-00000000000000"},
                transport=httpx.MockTransport(self.api),
            ),
            upload_client=httpx.AsyncClient(transport=httpx.MockTransport(self.storage)),
        )

    def sent(self, path: str) -> list[dict[str, Any]]:
        return [payload for seen, payload in self.api_requests if seen == path]

    def presigned(self) -> dict[str, Any]:
        return self.sent("/v1/media/uploads")[0]

    def observed(self) -> dict[str, Any]:
        return self.sent("/v1/observations")[0]


async def _no_delay(_seconds: float) -> None:
    """Retry without the backoff, which is not what any of these tests are about."""


def media_file(directory: Path, name: str = "clip.mp4", content: bytes = CONTENT) -> Path:
    """Write one file whose modification time is fixed, since the request derives from it."""
    path = directory / name
    path.write_bytes(content)
    os.utime(path, (MTIME.timestamp(), MTIME.timestamp()))
    return path


async def test_a_local_file_becomes_stored_evidence_and_an_observation(tmp_path: Path) -> None:
    """The whole point: a path in, a receipt out, and the bytes at a key nobody named."""
    deployment = FakeDeployment()
    client = deployment.client()

    try:
        receipt = await client.observe_file(
            media_file(tmp_path),
            tenant_id="tenant_01",
            occurred_at=MTIME,
            ended_at=datetime(2026, 8, 11, 12, 0, 30, tzinfo=timezone.utc),
        )
    finally:
        await client.close()

    media = deployment.observed()["media_objects"][0]
    assert receipt.observation_id == "observation_01"
    assert deployment.presigned() == {
        "tenant_id": "tenant_01",
        "sha256": DIGEST,
        "size_bytes": len(CONTENT),
        "suffix": ".mp4",
    }
    assert deployment.upload_bodies == [CONTENT]
    assert media["sha256"] == DIGEST
    assert media["uri"] == f"s3://memory/tenants/tenant_01/media/{DIGEST}.mp4"
    assert media["kind"] == "video"
    assert media["size_bytes"] == len(CONTENT)
    assert deployment.observed()["occurred_at"] == "2026-08-11T12:00:00Z"
    assert deployment.observed()["ended_at"] == "2026-08-11T12:00:30Z"


async def test_the_upload_carries_the_signed_length_and_not_the_api_key(tmp_path: Path) -> None:
    """A presigned URL is its own credential; a second one sent there is a leak and a 400."""
    deployment = FakeDeployment()
    client = deployment.client()

    try:
        await client.observe_file(media_file(tmp_path), tenant_id="tenant_01")
    finally:
        await client.close()

    upload = deployment.uploads[0]
    assert str(upload.url) == f"https://objects.example.test/signed/{DIGEST}"
    assert upload.method == "PUT"
    assert "authorization" not in upload.headers
    assert upload.headers["content-length"] == str(len(CONTENT))
    # Chunked framing is neither the length that was signed nor something an S3 PUT accepts.
    assert "transfer-encoding" not in upload.headers


async def test_no_observation_refers_to_bytes_that_never_arrived(tmp_path: Path) -> None:
    deployment = FakeDeployment(upload_status=403)
    client = deployment.client()

    try:
        with pytest.raises(MindBridgeError) as failure:
            await client.observe_file(media_file(tmp_path), tenant_id="tenant_01")
    finally:
        await client.close()

    assert failure.value.code == "media_upload_failed"
    assert failure.value.status_code == 403
    assert deployment.sent("/v1/observations") == []


async def test_timestamps_default_to_the_file_rather_than_to_now(tmp_path: Path) -> None:
    """`now` would differ between two runs over the same file, and the digest keys on it."""
    deployment = FakeDeployment()
    client = deployment.client()

    try:
        await client.observe_file(media_file(tmp_path), tenant_id="tenant_01")
    finally:
        await client.close()

    observed = deployment.observed()
    assert observed["occurred_at"] == "2026-08-11T12:00:00Z"
    assert observed["ended_at"] == "2026-08-11T12:00:00Z"
    assert observed["observed_at"] == "2026-08-11T12:00:00Z"


@pytest.mark.parametrize(
    ("name", "kind", "sensor"),
    [
        ("clip.mp4", "video", SensorKind.CAMERA),
        ("note.WAV", "audio", SensorKind.MICROPHONE),
        ("shot.png", "image", SensorKind.CAMERA),
    ],
)
async def test_kind_and_sensor_are_read_from_the_extension(
    tmp_path: Path,
    name: str,
    kind: str,
    sensor: SensorKind,
) -> None:
    deployment = FakeDeployment()
    client = deployment.client()

    try:
        await client.observe_file(media_file(tmp_path, name), tenant_id="tenant_01")
    finally:
        await client.close()

    assert deployment.observed()["media_objects"][0]["kind"] == kind
    assert deployment.observed()["sensor"] == sensor.value


async def test_an_unknown_container_needs_a_kind_and_keeps_no_extension(tmp_path: Path) -> None:
    """An extension this API has no opinion about is not carried into an object key."""
    deployment = FakeDeployment()
    client = deployment.client()

    try:
        with pytest.raises(ValueError, match="pass kind="):
            await client.observe_file(media_file(tmp_path, "capture.mts"), tenant_id="tenant_01")
        assert deployment.api_requests == []
        await client.observe_file(
            media_file(tmp_path, "capture.mts"),
            tenant_id="tenant_01",
            kind=MediaKind.VIDEO,
        )
    finally:
        await client.close()

    assert deployment.presigned()["suffix"] is None
    assert deployment.observed()["media_objects"][0]["kind"] == "video"


async def test_a_declared_kind_the_extension_contradicts_wins_over_the_extension(
    tmp_path: Path,
) -> None:
    """The override has to work, and the key it produces may not contradict what it declared."""
    deployment = FakeDeployment()
    client = deployment.client()

    try:
        await client.observe_file(
            media_file(tmp_path),
            tenant_id="tenant_01",
            kind=MediaKind.AUDIO,
        )
    finally:
        await client.close()

    media = deployment.observed()["media_objects"][0]
    assert deployment.presigned()["suffix"] is None
    assert media["uri"] == f"s3://memory/tenants/tenant_01/media/{DIGEST}"
    assert media["kind"] == "audio"


async def test_the_same_file_observed_twice_is_the_same_observation(tmp_path: Path) -> None:
    """Everything the request derives comes from the bytes, so a repeat is a duplicate."""
    deployment = FakeDeployment()
    client = deployment.client()
    path = media_file(tmp_path)

    try:
        await client.observe_file(path, tenant_id="tenant_01")
        await client.observe_file(path, tenant_id="tenant_01")
        await client.observe_file(
            media_file(tmp_path, "other.mp4", b"other bytes"),
            tenant_id="tenant_01",
        )
    finally:
        await client.close()

    first, again, other = deployment.sent("/v1/observations")
    assert first == again
    # A fixed default sequence would make this second file a conflicting write against the
    # first file's key rather than a second observation.
    assert other["sequence"] != first["sequence"]


async def test_signing_is_repeated_through_a_restart_although_it_names_no_key(
    tmp_path: Path,
) -> None:
    """`_repeat_is_safe` would refuse this write; it is repeated on an argued exception.

    Losing a whole upload to one 503 from a restarting dependency is the failure that
    exception exists to avoid, and a second signature is not a second anything else.
    """
    deployment = FakeDeployment()
    attempts: list[str] = []

    async def restarting(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        if request.url.path == "/v1/media/uploads" and len(attempts) == 1:
            return httpx.Response(
                503,
                json={
                    "code": "object_storage_unavailable",
                    "message": "evidence media is unavailable",
                    "trace_id": "trace_unavailable",
                    "issues": [],
                },
            )
        return await deployment.api(request)

    client = MindBridge(
        httpx.AsyncClient(
            base_url="https://memory.example.test/",
            transport=httpx.MockTransport(restarting),
        ),
        upload_client=httpx.AsyncClient(transport=httpx.MockTransport(deployment.storage)),
        sleep=_no_delay,
    )

    try:
        receipt = await client.observe_file(media_file(tmp_path), tenant_id="tenant_01")
    finally:
        await client.close()

    assert receipt.observation_id == "observation_01"
    assert attempts == ["/v1/media/uploads", "/v1/media/uploads", "/v1/observations"]


async def test_a_large_file_is_read_a_chunk_at_a_time(tmp_path: Path) -> None:
    """A multi-gigabyte upload is a network cost, not a memory one."""
    path = media_file(tmp_path, "big.mp4", b"x" * (_UPLOAD_CHUNK_BYTES * 2 + 5))

    chunks = [len(chunk) async for chunk in _file_chunks(path)]

    assert chunks == [_UPLOAD_CHUNK_BYTES, _UPLOAD_CHUNK_BYTES, 5]


async def test_closing_releases_the_upload_pool_as_well(tmp_path: Path) -> None:
    deployment = FakeDeployment()
    client = deployment.client()
    await client.observe_file(media_file(tmp_path), tenant_id="tenant_01")

    await client.close()

    assert client._upload_client is not None
    assert client._upload_client.is_closed
