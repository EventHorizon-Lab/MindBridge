"""Tests for the Jina Sentence Transformers boundary."""

import wave
from array import array
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mindbridge.application.perception import ResolvedEvidence
from mindbridge.core import (
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    ModelOutputError,
    ModelReference,
    ObservationId,
    PixelRegion,
    TenantId,
)
from mindbridge.models import jina as jina_module
from mindbridge.models.jina import (
    JinaOmniEmbedder,
    _decode_event_media,
    _prepared_media_attributes,
    _video_frame_array,
)

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)


class Matrix:
    """Minimal ndarray-shaped result used by the adapter boundary."""

    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def tolist(self) -> list[list[float]]:
        return self._values


class RecordingEncoder:
    """Records whether query/document semantics reach the official API."""

    def __init__(self, values: list[list[float]]) -> None:
        self.values = values
        self.calls: list[str] = []
        self.sentences: list[list[object]] = []

    def encode_query(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int,
    ) -> Matrix:
        self.calls.append("query")
        self.sentences.append(sentences)
        return Matrix(self.values)

    def encode_document(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int,
    ) -> Matrix:
        self.calls.append("document")
        self.sentences.append(sentences)
        return Matrix(self.values)


def test_jina_pins_remote_code_to_the_model_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def sentence_transformer(model_path: str, **kwargs: object) -> RecordingEncoder:
        calls.append((model_path, kwargs))
        return RecordingEncoder([[1.0, 0.0]])

    def snapshot_download(**kwargs: object) -> str:
        calls.append(("snapshot_download", kwargs))
        return "/models/pinned"

    monkeypatch.setattr(
        "mindbridge.models.jina.import_module",
        lambda name: (
            SimpleNamespace(SentenceTransformer=sentence_transformer)
            if name == "sentence_transformers"
            else SimpleNamespace(snapshot_download=snapshot_download)
        ),
    )
    monkeypatch.setattr("mindbridge.models.jina.select_torch_device", lambda _device: "cuda")

    JinaOmniEmbedder.load(revision="pinned-revision", dimension=2)

    assert calls == [
        (
            "snapshot_download",
            {
                "repo_id": "jinaai/jina-embeddings-v5-omni-small-retrieval",
                "revision": "pinned-revision",
            },
        ),
        (
            "/models/pinned",
            {
                "revision": "pinned-revision",
                "trust_remote_code": True,
                "device": "cuda",
                "model_kwargs": {
                    "modality": "omni",
                    "code_revision": "pinned-revision",
                },
                "config_kwargs": {"code_revision": "pinned-revision"},
            },
        ),
    ]


async def test_jina_uses_distinct_query_and_document_methods() -> None:
    """Retrieval prefixes remain owned by Sentence Transformers."""
    encoder = RecordingEncoder([[1.0, 0.0]])
    embedder = JinaOmniEmbedder(
        encoder,
        ModelReference(model_id="jina", revision="revision"),
        dimension=2,
    )

    query = await embedder.encode_queries(("Where is the screwdriver?",))
    document = await embedder.encode_documents((b"image-bytes",))

    assert query == ((1.0, 0.0),)
    assert document == query
    assert encoder.calls == ["query", "document"]
    assert (
        embedder.model_reference.revision
        == f"revision+{jina_module.JINA_EVENT_MEDIA_PREPROCESSING_REVISION}"
    )


async def test_jina_rejects_invalid_model_output() -> None:
    """Malformed upstream vectors cannot enter the semantic index."""
    embedder = JinaOmniEmbedder(
        RecordingEncoder([[1.0, 1.0]]),
        ModelReference(model_id="jina", revision="revision"),
        dimension=2,
    )

    with pytest.raises(ModelOutputError, match="L2-normalized"):
        await embedder.encode_queries(("query",))


async def test_jina_skips_model_call_for_empty_batch() -> None:
    """An empty batch has no model cost or ambiguous output shape."""
    encoder = RecordingEncoder([])
    embedder = JinaOmniEmbedder(
        encoder,
        ModelReference(model_id="jina", revision="revision"),
        dimension=2,
    )

    assert await embedder.encode_documents(()) == ()
    assert encoder.calls == []


async def test_jina_materializes_exact_evidence_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _resolved_evidence("/media/source.mp4", start_ms=500, end_ms=1_500)
    monkeypatch.setattr(
        jina_module,
        "_decode_event_media",
        lambda inputs: (b"event-window",) * len(inputs),
    )
    encoder = RecordingEncoder([[1.0, 0.0]])
    embedder = JinaOmniEmbedder(
        encoder,
        ModelReference(model_id="jina", revision="revision"),
        dimension=2,
    )

    await embedder.encode_documents((evidence,))

    assert encoder.sentences == [[b"event-window"]]


async def test_jina_crops_image_evidence_before_encoding(tmp_path: Path) -> None:
    numpy = import_module("numpy")
    image_module = import_module("PIL.Image")
    pixels = numpy.zeros((64, 64, 3), dtype=numpy.uint8)
    pixels[:, :32, 0] = 255
    pixels[:, 32:, 2] = 255
    media_path = tmp_path / "regions.png"
    image_module.fromarray(pixels).save(media_path)
    evidence = _resolved_evidence(
        str(media_path),
        start_ms=0,
        end_ms=0,
        kind=MediaKind.IMAGE,
        duration_ms=0,
        region=PixelRegion(x_min=0, y_min=0, x_max=32, y_max=64),
    )
    encoder = RecordingEncoder([[1.0, 0.0]])
    embedder = JinaOmniEmbedder(
        encoder,
        ModelReference(model_id="jina", revision="revision"),
        dimension=2,
    )

    await embedder.encode_documents((evidence,))

    cropped = cast(Any, encoder.sentences[0][0])
    assert cropped.shape == (64, 32, 3)
    assert float(cropped[..., 0].mean()) > 250
    assert float(cropped[..., 2].mean()) < 5


def test_event_media_decoder_keeps_only_each_time_window(tmp_path: Path) -> None:
    media_path = tmp_path / "events.mp4"
    _write_video(media_path)

    early, late = _decode_event_media(
        (
            _resolved_evidence(str(media_path), start_ms=0, end_ms=400),
            _resolved_evidence(str(media_path), start_ms=1_500, end_ms=1_900),
        )
    )
    early_array = cast(Any, early)
    late_array = cast(Any, late)

    assert early_array.shape == (4, 64, 64, 3)
    assert late_array.shape == (4, 64, 64, 3)
    assert float(early_array.mean()) < 50
    assert float(late_array.mean()) > 140
    assert jina_module._event_video_metadata([early_array]) == [
        {
            "total_num_frames": 4,
            "fps": 7.5,
            "duration": 0.4,
            "frames_indices": [0, 1, 2, 3],
            "height": 64,
            "width": 64,
        }
    ]


def test_event_media_decoder_keeps_only_each_frame_range(tmp_path: Path) -> None:
    media_path = tmp_path / "frame-ranges.mp4"
    _write_video(media_path)

    early, late = _decode_event_media(
        (
            _resolved_evidence(
                str(media_path),
                start_ms=0,
                end_ms=1_900,
                frame_start=0,
                frame_end=3,
            ),
            _resolved_evidence(
                str(media_path),
                start_ms=0,
                end_ms=1_900,
                frame_start=16,
                frame_end=19,
            ),
        )
    )

    assert float(cast(Any, early).mean()) < 50
    assert float(cast(Any, late).mean()) > 140


def test_event_media_decoder_seeks_to_point_and_stops_after_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_path = tmp_path / "long.mp4"
    _write_video(media_path, frame_count=100)
    decoded_frames = 0
    select_targets = jina_module._select_event_video_targets

    def recording_select_targets(*args: object) -> tuple[int, tuple[float, object]]:
        nonlocal decoded_frames
        decoded_frames += 1
        return cast(tuple[int, tuple[float, object]], cast(Any, select_targets)(*args))

    monkeypatch.setattr(jina_module, "_select_event_video_targets", recording_select_targets)

    (point,) = _decode_event_media(
        (
            _resolved_evidence(
                str(media_path),
                start_ms=8_450,
                end_ms=8_450,
                duration_ms=10_000,
            ),
        )
    )

    assert cast(Any, point).shape == (64, 64, 3)
    assert decoded_frames < 20


def test_event_audio_decoder_samples_point_evidence(tmp_path: Path) -> None:
    media_path = tmp_path / "point.wav"
    with wave.open(str(media_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(array("h", [-1_000] * 8_000 + [1_000] * 8_000).tobytes())

    (point,) = _decode_event_media(
        (
            _resolved_evidence(
                str(media_path),
                start_ms=750,
                end_ms=750,
                kind=MediaKind.AUDIO,
                duration_ms=1_000,
            ),
        )
    )

    point_array = cast(Any, point)
    assert point_array.size >= 1
    assert float(point_array.mean()) > 0


def test_event_frame_pixel_budget_caps_total_pixels() -> None:
    class RecordingFrame:
        width = 1_920
        height = 1_080

        def __init__(self) -> None:
            self.reformatted_size: tuple[int, int] | None = None

        def reformat(
            self,
            *,
            width: int,
            height: int,
            format: str,
        ) -> "RecordingFrame":
            assert format == "rgb24"
            self.reformatted_size = (width, height)
            return self

        def to_ndarray(self, *, format: str) -> object:
            assert format == "rgb24"
            return object()

    frame = RecordingFrame()

    _video_frame_array(frame, frame_count=4)

    assert frame.reformatted_size is not None
    width, height = frame.reformatted_size
    assert width * height * 4 <= jina_module._EVENT_VIDEO_MAX_TOTAL_PIXELS
    assert (width, height) != (1_920, 1_080)


def test_event_frame_crop_keeps_only_grounded_region() -> None:
    av = import_module("av")
    numpy = import_module("numpy")
    pixels = numpy.zeros((64, 64, 3), dtype=numpy.uint8)
    pixels[:, :32, 0] = 255
    pixels[:, 32:, 2] = 255
    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")

    cropped = cast(
        Any,
        _video_frame_array(
            frame,
            frame_count=4,
            region=PixelRegion(x_min=0, y_min=0, x_max=32, y_max=64),
        ),
    )

    assert cropped.shape == (64, 32, 3)
    assert float(cropped[..., 0].mean()) > 250
    assert float(cropped[..., 2].mean()) < 5


def test_prepared_media_attributes_report_actual_model_input() -> None:
    numpy = import_module("numpy")
    video = numpy.zeros((4, 32, 64, 3), dtype=numpy.uint8)
    audio = numpy.zeros(8_000, dtype=numpy.float32)

    assert _prepared_media_attributes([(audio, video), "text"]) == {
        "mindbridge.embedding.multimodal_item_count": 1,
        "mindbridge.embedding.video_frame_count": 4,
        "mindbridge.embedding.video_pixel_count": 8_192,
        "mindbridge.embedding.audio_seconds": 0.5,
    }


def test_event_video_metadata_reaches_upstream_processor() -> None:
    numpy = import_module("numpy")
    video = numpy.zeros((4, 32, 64, 3), dtype=numpy.uint8)
    video = video.view(
        numpy.dtype(
            video.dtype,
            metadata={jina_module._EVENT_VIDEO_METADATA_KEY: {"fps": 3.0, "duration": 1.0}},
        )
    )

    class VideoProcessor:
        def preprocess(self, videos: object, **kwargs: object) -> object:
            return kwargs["video_metadata"]

    video_processor = VideoProcessor()
    encoder = [SimpleNamespace(processor=SimpleNamespace(video_processor=video_processor))]
    jina_module._install_event_video_metadata(encoder)

    assert video_processor.preprocess([video], video_metadata=None) == [
        {
            "total_num_frames": 4,
            "fps": 3.0,
            "duration": 1.0,
            "frames_indices": [0, 1, 2, 3],
            "height": 32,
            "width": 64,
        }
    ]


def _write_video(media_path: Path, *, frame_count: int = 20) -> None:
    av = import_module("av")
    numpy = import_module("numpy")
    container = av.open(str(media_path), mode="w")
    stream = container.add_stream("mpeg4", rate=10)
    stream.width = 64
    stream.height = 64
    stream.pix_fmt = "yuv420p"
    for index in range(frame_count):
        pixels = numpy.full((64, 64, 3), (index % 25) * 10, dtype=numpy.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        frame.pts = index
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _resolved_evidence(
    media_url: str,
    *,
    start_ms: int,
    end_ms: int,
    kind: MediaKind = MediaKind.VIDEO,
    duration_ms: int = 2_000,
    region: PixelRegion | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> ResolvedEvidence:
    media_object_id = MediaObjectId("media_01")
    return ResolvedEvidence(
        evidence_span=EvidenceSpan(
            evidence_id=EvidenceId(f"evidence_{start_ms}_{end_ms}_{frame_start}_{frame_end}"),
            tenant_id=TenantId("tenant_01"),
            observation_id=ObservationId("observation_01"),
            media_object_id=media_object_id,
            start_ms=start_ms,
            end_ms=end_ms,
            created_at=NOW,
            frame_start=frame_start,
            frame_end=frame_end,
            region=region,
        ),
        media_object=MediaObject(
            media_object_id=media_object_id,
            tenant_id=TenantId("tenant_01"),
            kind=kind,
            uri=media_url,
            sha256="a" * 64,
            size_bytes=1,
            duration_ms=duration_ms,
            created_at=NOW,
        ),
        media_url=media_url,
        media_url_expires_at=NOW + timedelta(minutes=5),
    )
