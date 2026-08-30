"""FunASR speech analysis with Fun-ASR-Nano as the default recipe."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from itertools import pairwise
from threading import RLock
from typing import Protocol, cast

from mindbridge._telemetry import mark_model_requests, record_unmetered_model_usage
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import SpeakerEmbedding, SpeechAnalysis, SpeechTurn
from mindbridge.types import AssetRef, Modality

DEFAULT_FUNASR_MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
DEFAULT_FUNASR_MODEL_REVISION = "05201c46f1c38592b1567f857c0d56eab3d0d8ef"
DEFAULT_FUNASR_VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
DEFAULT_FUNASR_VAD_REVISION = "f9a8b8274674755d925277e27063869038d41515"
DEFAULT_FUNASR_SPEAKER_MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"
DEFAULT_FUNASR_SPEAKER_REVISION = "a045b2afcaa9c3049c98a9215a2bc274407ab237"
_MAX_TURNS = 10_000
_MAX_TURN_CHARACTERS = 4_096
_SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")


class _Pipeline(Protocol):
    def generate(self, **kwargs: object) -> list[dict[str, object]]: ...


@dataclass(frozen=True, slots=True)
class FunASRRecipe:
    """One ASR model and the FunASR components composed around it."""

    model_id: str
    vad_model: str
    speaker_model: str
    model_revision: str | None = None
    vad_revision: str | None = None
    speaker_revision: str | None = None
    punctuation_model: str | None = None
    punctuation_revision: str | None = None
    vad_max_single_segment_ms: int | None = None
    hub: str = "ms"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        required = (self.model_id, self.vad_model, self.speaker_model, self.hub)
        if any(not isinstance(value, str) or not value.strip() for value in required):
            raise ValidationError("FunASR recipe model and hub values must not be blank")
        optional = (
            self.model_revision,
            self.vad_revision,
            self.speaker_revision,
            self.punctuation_model,
            self.punctuation_revision,
        )
        if any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in optional
        ):
            raise ValidationError("FunASR optional recipe values must not be blank")
        if self.vad_max_single_segment_ms is not None and (
            isinstance(self.vad_max_single_segment_ms, bool) or self.vad_max_single_segment_ms <= 0
        ):
            raise ValidationError("FunASR VAD segment ceiling must be positive")

    def auto_model_arguments(self) -> dict[str, object]:
        """Return the standard FunASR AutoModel composition arguments."""
        values: dict[str, object] = {
            "model": self.model_id,
            "vad_model": self.vad_model,
            "spk_model": self.speaker_model,
            "hub": self.hub,
        }
        optional = {
            "model_revision": self.model_revision,
            "vad_model_revision": self.vad_revision,
            "spk_model_revision": self.speaker_revision,
            "punc_model": self.punctuation_model,
            "punc_model_revision": self.punctuation_revision,
        }
        values.update((key, value) for key, value in optional.items() if value is not None)
        if self.punctuation_model is None:
            values["spk_mode"] = "vad_segment"
        if self.vad_max_single_segment_ms is not None:
            values["vad_kwargs"] = {"max_single_segment_time": self.vad_max_single_segment_ms}
        if self.trust_remote_code:
            values["trust_remote_code"] = True
        return values


DEFAULT_FUNASR_RECIPE = FunASRRecipe(
    model_id=DEFAULT_FUNASR_MODEL_ID,
    model_revision=DEFAULT_FUNASR_MODEL_REVISION,
    vad_model=DEFAULT_FUNASR_VAD_MODEL_ID,
    vad_revision=DEFAULT_FUNASR_VAD_REVISION,
    speaker_model=DEFAULT_FUNASR_SPEAKER_MODEL_ID,
    speaker_revision=DEFAULT_FUNASR_SPEAKER_REVISION,
    vad_max_single_segment_ms=30_000,
)


class FunASRTranscriber:
    """Lazily map FunASR AutoModel results into MindBridge speech semantics."""

    def __init__(
        self,
        recipe: FunASRRecipe = DEFAULT_FUNASR_RECIPE,
        *,
        device: str = "auto",
    ) -> None:
        if not isinstance(recipe, FunASRRecipe):
            raise ValidationError("recipe must be a FunASRRecipe value")
        normalized_device = device.strip().lower() if isinstance(device, str) else ""
        if (
            normalized_device not in {"auto", "cpu"}
            and re.fullmatch(r"cuda(?::\d+)?", normalized_device) is None
        ):
            raise ValidationError("device must be auto, cpu, cuda, or cuda:<index>")
        self._recipe = recipe
        self._device = normalized_device
        self._pipeline: _Pipeline | None = None
        # ponytail: FunASR inference is serialized per loaded model; relax only after its
        # AutoModel composition demonstrates thread safety and concurrent throughput gains.
        self._lock = RLock()
        self._closed = False

    @property
    def transcription_capabilities(self) -> frozenset[Modality]:
        return frozenset({Modality.AUDIO, Modality.VIDEO})

    @property
    def transcription_model(self) -> str:
        return self._recipe.model_id

    @property
    def transcription_space(self) -> str:
        identity = self._recipe.auto_model_arguments()
        if self._recipe.punctuation_model is None:
            # This only makes FunASR's existing no-punctuation fallback explicit. Excluding it
            # keeps stores created with the v3 recipe readable after the runtime fix.
            identity.pop("spk_mode", None)
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(f"funasr-automodel-v3:{payload}".encode()).hexdigest()[:16]
        return f"{self.transcription_model}:speech:{digest}"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        if isinstance(assets, (str, bytes)):
            raise ValidationError("assets must contain resolved audio or video AssetRef values")
        batch = tuple(assets)
        for asset in batch:
            if (
                not isinstance(asset, AssetRef)
                or not asset.is_resolved
                or asset.modality not in self.transcription_capabilities
            ):
                raise ValidationError("assets must contain resolved audio or video AssetRef values")
        # One batch is one AutoModel call, so the request count is calls issued, not assets sent.
        mark_model_requests(1 if batch else 0, token_usage_expected=0)
        if not batch:
            return ()
        with self._lock:
            if self._closed:
                raise ModelError("FunASR transcriber is closed")
            pipeline = self._load()
            analyses, calls = self._analyze_many(pipeline, batch)
        record_unmetered_model_usage(request_count=calls)
        return analyses

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pipeline, self._pipeline = self._pipeline, None
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()

    def _load(self) -> _Pipeline:
        if self._pipeline is not None:
            return self._pipeline
        try:
            funasr = import_module("funasr")
        except ImportError:
            raise ModelError(
                "FunASR is required for local speech; install mindbridge[local]"
            ) from None
        arguments = self._recipe.auto_model_arguments()
        if self._device != "auto":
            arguments["device"] = self._device
        try:
            pipeline = funasr.AutoModel(
                **arguments,
                disable_update=True,
                disable_pbar=True,
            )
        except Exception as error:
            raise ModelError("failed to load the FunASR speech recipe") from error
        self._pipeline = cast(_Pipeline, pipeline)
        return self._pipeline

    @staticmethod
    def _analyze_many(
        pipeline: _Pipeline,
        assets: Sequence[AssetRef],
    ) -> tuple[tuple[SpeechAnalysis, ...], int]:
        """Analyze one official batch, returning the analyses and the AutoModel calls made."""
        paths = tuple(_speech_path(asset) for asset in assets)
        output = _pipeline_output(pipeline, list(paths) if len(paths) > 1 else paths[0])
        if (
            not isinstance(output, list)
            or len(output) != len(paths)
            or any(not isinstance(result, dict) for result in output)
        ):
            analyses = tuple(FunASRTranscriber._analyze_one(pipeline, asset) for asset in assets)
            return analyses, 1 + len(assets)
        return tuple(_analysis(result) for result in output), 1

    @staticmethod
    def _analyze_one(pipeline: _Pipeline, asset: AssetRef) -> SpeechAnalysis:
        output = _pipeline_output(pipeline, _speech_path(asset))
        if output == []:
            return SpeechAnalysis(turns=(), speakers=())
        if not isinstance(output, list) or len(output) != 1 or not isinstance(output[0], dict):
            raise ModelError("FunASR returned an invalid transcription batch")
        return _analysis(output[0])


def _speech_path(asset: AssetRef) -> str:
    path = asset.path
    if path is None:  # AssetRef.is_resolved was checked at the boundary.
        raise ValidationError("speech asset path is missing")
    try:
        return str(path.resolve(strict=True))
    except OSError:
        raise ValidationError("speech asset path does not exist") from None


def _pipeline_output(pipeline: _Pipeline, value: str | list[str]) -> object:
    try:
        return pipeline.generate(
            input=value,
            batch_size_s=300,
            return_raw_text=True,
            sentence_timestamp=True,
            return_spk_res=True,
            return_spk_center=True,
            disable_pbar=True,
        )
    except Exception as error:
        raise ModelError("FunASR failed to analyze speech") from error


def _analysis(result: dict[str, object]) -> SpeechAnalysis:
    text = result.get("text")
    if not isinstance(text, str):
        raise ModelError("FunASR returned invalid transcription text")
    if not _SPECIAL_TOKEN.sub("", text).strip():
        return SpeechAnalysis(turns=(), speakers=())
    turns = _turns(result.get("sentence_info"))
    if not turns:
        raise ModelError("FunASR returned text without timed speaker turns")
    speakers = _speakers(result.get("spk_embedding_center"))
    labels = {turn.speaker_label for turn in turns if turn.speaker_label is not None}
    if labels - {speaker.speaker_label for speaker in speakers}:
        raise ModelError("FunASR returned a speaker label without a CAM++ centroid")
    return SpeechAnalysis(turns=turns, speakers=speakers)


def _turns(value: object) -> tuple[SpeechTurn, ...]:
    if not isinstance(value, list) or len(value) > _MAX_TURNS:
        raise ModelError("FunASR returned invalid sentence-level diarization")
    turns = []
    for item in value:
        turn = _turn(item)
        if turn is not None:
            turns.append(turn)
    if any(current.start_ms < previous.start_ms for previous, current in pairwise(turns)):
        raise ModelError("FunASR speaker turns are not chronological")
    return tuple(turns)


def _turn(item: object) -> SpeechTurn | None:
    if not isinstance(item, Mapping):
        raise ModelError("FunASR returned an invalid speaker turn")
    start, end = item.get("start"), item.get("end")
    raw_text = item.get("text")
    if raw_text is None:
        raw_text = item.get("sentence")
    if isinstance(raw_text, str) and not _SPECIAL_TOKEN.sub("", raw_text).strip():
        return None
    text = _sentence_text(raw_text)
    speaker = item.get("spk")
    if (
        not isinstance(start, int | float)
        or isinstance(start, bool)
        or not isinstance(end, int | float)
        or isinstance(end, bool)
        or text is None
        or (speaker is not None and not isinstance(speaker, str | int))
    ):
        raise ModelError("FunASR returned an invalid speaker turn")
    text = _SPECIAL_TOKEN.sub("", text).strip()
    if not text:
        return None
    if len(text) > _MAX_TURN_CHARACTERS:
        raise ModelError("FunASR speaker turn text is too long")
    start_ms, end_ms = round(start), round(end)
    if start_ms < 0 or end_ms <= start_ms:
        raise ModelError("FunASR returned an invalid speaker turn range")
    label = str(speaker).strip() if speaker is not None else None
    if label == "":
        raise ModelError("FunASR returned an invalid speaker label")
    return SpeechTurn(start_ms, end_ms, text, label)


def _sentence_text(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(token, str) for token in value)
    ):
        return "".join(value).strip() or None
    return None


def _speakers(value: object) -> tuple[SpeakerEmbedding, ...]:
    if value is None:
        return ()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, list):
        raise ModelError("FunASR returned invalid speaker centroids")
    rows = [value] if value and all(isinstance(number, int | float) for number in value) else value
    if any(not isinstance(row, list) for row in rows):
        raise ModelError("FunASR returned invalid speaker centroids")
    speakers = []
    for index, row in enumerate(rows):
        if not row or any(
            isinstance(number, bool)
            or not isinstance(number, int | float)
            or not math.isfinite(float(number))
            for number in row
        ):
            raise ModelError("FunASR returned invalid speaker centroids")
        values = tuple(float(number) for number in row)
        magnitude = math.sqrt(math.fsum(number * number for number in values))
        if magnitude == 0.0:
            raise ModelError("FunASR returned a zero speaker centroid")
        speakers.append(
            SpeakerEmbedding(str(index), tuple(number / magnitude for number in values))
        )
    return tuple(speakers)
