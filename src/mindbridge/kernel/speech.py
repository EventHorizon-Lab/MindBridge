"""Speech perception: transcripts, diarised speakers, and their identities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from opentelemetry.trace import Tracer

from mindbridge._telemetry import mark_model_requests
from mindbridge.exceptions import MindBridgeError, ModelError
from mindbridge.infrastructure.local.store import StoredAsset
from mindbridge.kernel.content import PreparedContent
from mindbridge.kernel.contracts import Backends
from mindbridge.kernel.derived import derived_text, speech_identity_text, without_speech_identities
from mindbridge.kernel.hydration import Hydrator
from mindbridge.kernel.lifecycle import OperationAssets
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.settings import Settings
from mindbridge.kernel.tracing import Traced
from mindbridge.kernel.validation import MAX_TEXT_CHARACTERS
from mindbridge.models.base import SpeechAnalysis, SpeechBackend
from mindbridge.types import AssetRef, Modality


class Speech(Traced):
    """Transcription, diarisation, and voice identities over stored audio and video."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        backends: Backends,
        settings: Settings,
        hydrator: Hydrator,
    ) -> None:
        super().__init__(tracer)
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._backends = backends
        self._settings = settings
        self._hydrator = hydrator

    def _transcribable_assets(
        self,
        assets: Sequence[StoredAsset],
    ) -> tuple[StoredAsset, ...]:
        supported = {modality.value for modality in self._backends.transcription_capabilities}
        return tuple(
            {
                asset.asset_id: asset
                for asset in assets
                if asset.modality in {"audio", "video"} and asset.modality in supported
            }.values()
        )

    def answer_speech_assets(
        self,
        assets: Sequence[StoredAsset],
    ) -> tuple[StoredAsset, ...]:
        if not isinstance(self._backends.transcriber, SpeechBackend):
            return ()
        return self._transcribable_assets(assets)

    def transcript_fallback(self, assets: Sequence[StoredAsset]) -> frozenset[Modality]:
        """Modalities a derived transcript can rescue for an embedder that cannot take them.

        Empty when a `SpeechBackend` is configured: that composition indexes the same text through
        the `index_speech` opt-in instead, so claiming the rescue here would let a write past the
        guard and then fail later with no transcript in hand.
        """
        if not self.derives_transcripts(assets):
            return frozenset()
        return self._backends.transcription_capabilities & {Modality.AUDIO, Modality.VIDEO}

    def derives_transcripts(self, assets: Sequence[StoredAsset]) -> bool:
        # A SpeechBackend indexes the same text through the explicit `index_speech` opt-in, so its
        # add-time analysis cost stays behind that flag instead of being taken twice.
        return not isinstance(self._backends.transcriber, SpeechBackend) and bool(
            self._transcribable_assets(assets)
        )

    def with_speech_identities(
        self,
        prepared: PreparedContent,
        operation: OperationAssets,
    ) -> PreparedContent:
        speech_assets = self.answer_speech_assets(prepared.assets)
        self.recognize(speech_assets, operation, reversible=True)
        base = without_speech_identities(
            prepared.text,
            tuple(asset.asset_id for asset in speech_assets),
        )
        text = speech_identity_text(base, speech_assets, operation.speech_segments)
        if len(text) > MAX_TEXT_CHARACTERS:
            raise ModelError(
                "speaker identity evidence exceeded the supported text length",
                reason="payload_too_large",
            )
        return replace(prepared, text=text)

    def recognize(
        self,
        assets: Sequence[StoredAsset],
        operation: OperationAssets,
        *,
        reversible: bool = False,
    ) -> None:
        if not isinstance(self._backends.transcriber, SpeechBackend):
            raise ModelError(
                "configured transcription backend cannot analyze speakers",
                reason="backend_not_configured",
            )
        speech_assets = tuple(
            {
                asset.asset_id: asset
                for asset in assets
                if asset.modality in {"audio", "video"}
                and asset.asset_id not in operation.speech_segments
            }.values()
        )
        missing = []
        with (
            self._trace("mindbridge.storage.lookup", kind="stage"),
            translate_storage_errors("read cached speaker recognition"),
        ):
            for asset in speech_assets:
                segments = self._store.media.read_speech(
                    asset.asset_id,
                    space_id=self._backends.transcription_space,
                )
                if segments is None:
                    missing.append(asset)
                else:
                    operation.speech_segments[asset.asset_id] = segments
        if missing:
            analyses = self._analyze_speech(
                tuple(self._hydrator.asset_ref(asset) for asset in missing)
            )
            with (
                self._trace("mindbridge.storage.write", kind="stage"),
                self._write_lock,
                translate_storage_errors("persist speaker recognition"),
            ):
                for asset, analysis in zip(missing, analyses, strict=True):
                    self._store.media.write_asset(asset)
                    operation.persisted.add(asset.asset_id)
                    if reversible:
                        segments, rollback = self._store.media.write_speech_reversible(
                            asset.asset_id,
                            analysis,
                            model_id=self._backends.transcriber.transcription_model,
                            space_id=self._backends.transcription_space,
                            minimum_similarity=self._settings.speaker_similarity,
                            minimum_margin=self._settings.speaker_margin,
                        )
                        if rollback is not None:
                            operation.speech_rollbacks.append(rollback)
                    else:
                        segments = self._store.media.write_speech(
                            asset.asset_id,
                            analysis,
                            model_id=self._backends.transcriber.transcription_model,
                            space_id=self._backends.transcription_space,
                            minimum_similarity=self._settings.speaker_similarity,
                            minimum_margin=self._settings.speaker_margin,
                        )
                    operation.speech_segments[asset.asset_id] = segments
        analyzed = {asset.asset_id for asset in missing}
        for asset in speech_assets:
            segments = operation.speech_segments[asset.asset_id]
            operation.transcripts[asset.asset_id] = "\n".join(segment.text for segment in segments)
            self._trace_identity_yield(
                "mindbridge.identity.speakers",
                tuple((segment.speaker_id, segment.identity_score) for segment in segments),
                cached=asset.asset_id not in analyzed,
            )

    def with_audio_transcripts(
        self,
        prepared: PreparedContent,
        operation: OperationAssets,
    ) -> PreparedContent:
        self.cache_audio_transcripts(prepared.assets, operation)
        assets = tuple(
            replace(asset, transcript=operation.transcripts[asset.asset_id])
            if asset.asset_id in operation.transcripts
            else asset
            for asset in prepared.assets
        )
        text = derived_text(
            prepared.text,
            tuple(asset for asset in assets if asset.asset_id not in operation.speech_segments),
        )
        if len(text) > MAX_TEXT_CHARACTERS:
            raise ModelError(
                "audio transcription exceeded the supported text length", reason="payload_too_large"
            )
        return replace(prepared, text=text, assets=assets)

    def cache_audio_transcripts(
        self,
        assets: Sequence[StoredAsset],
        operation: OperationAssets,
    ) -> None:
        speech = self._transcribable_assets(assets)
        for asset in speech:
            if asset.transcript is not None:
                operation.transcripts.setdefault(asset.asset_id, asset.transcript)
        missing = tuple(
            dict.fromkeys(
                asset.asset_id for asset in speech if asset.asset_id not in operation.transcripts
            )
        )
        if missing:
            by_id = {asset.asset_id: asset for asset in speech}
            refs = tuple(self._hydrator.asset_ref(by_id[asset_id]) for asset_id in missing)
            try:
                transcribe = getattr(self._backends.transcriber, "transcribe", None)
                if callable(transcribe):
                    model = getattr(self._backends.transcriber, "transcription_model", None)
                    with self._model_trace(
                        "transcription",
                        "transcription",
                        model=model if isinstance(model, str) else None,
                        batch_size=len(refs),
                        modalities=(cast(Modality, asset.modality) for asset in refs),
                    ):
                        mark_model_requests(1)
                        generated = transcribe(refs)
                else:
                    analyses = self._analyze_speech(refs)
                    generated = tuple(
                        "\n".join(turn.text for turn in analysis.turns) for analysis in analyses
                    )
                    operation.speech_updates.update(zip(missing, analyses, strict=True))
            except MindBridgeError:
                raise
            except Exception as error:
                raise ModelError(
                    "failed to transcribe audio input", reason="model_failed"
                ) from error
            if len(generated) != len(refs) or any(
                not isinstance(transcript, str) for transcript in generated
            ):
                raise ModelError(
                    "transcription model returned invalid output", reason="response_invalid"
                )
            cached = tuple(
                (asset_id, transcript.strip())
                for asset_id, transcript in zip(missing, generated, strict=True)
            )
            operation.transcripts.update(cached)
            operation.transcript_updates.update(cached)

    def persist_transcripts(self, operation: OperationAssets) -> None:
        speech_ids = operation.speech_updates.keys()
        updates = tuple(
            (asset_id, transcript)
            for asset_id, transcript in operation.transcript_updates.items()
            if asset_id in operation.persisted and asset_id not in speech_ids
        )
        speech = tuple(
            (asset_id, analysis)
            for asset_id, analysis in operation.speech_updates.items()
            if asset_id in operation.persisted
        )
        if updates or speech:
            with (
                self._trace("mindbridge.storage.write", kind="stage"),
                self._write_lock,
                translate_storage_errors("cache audio transcripts"),
            ):
                if updates:
                    self._store.media.set_asset_transcripts(updates)
                if speech and isinstance(self._backends.transcriber, SpeechBackend):
                    for asset_id, analysis in speech:
                        self._store.media.write_speech(
                            asset_id,
                            analysis,
                            model_id=self._backends.transcriber.transcription_model,
                            space_id=self._backends.transcription_space,
                            minimum_similarity=self._settings.speaker_similarity,
                            minimum_margin=self._settings.speaker_margin,
                        )

    def _analyze_speech(
        self,
        assets: Sequence[AssetRef],
    ) -> tuple[SpeechAnalysis, ...]:
        if not isinstance(self._backends.transcriber, SpeechBackend):
            raise ModelError(
                "configured transcription backend cannot analyze speakers",
                reason="backend_not_configured",
            )
        with self._model_trace(
            "transcription",
            "transcription",
            model=self._backends.transcriber.transcription_model,
            batch_size=len(assets),
            modalities=(cast(Modality, asset.modality) for asset in assets),
        ):
            mark_model_requests(1)
            try:
                analyses = self._backends.transcriber.analyze(assets)
            except MindBridgeError:
                raise
            except Exception as error:
                raise ModelError("failed to analyze speech input", reason="model_failed") from error
            if len(analyses) != len(assets) or any(
                not isinstance(analysis, SpeechAnalysis) for analysis in analyses
            ):
                raise ModelError("speech model returned invalid output", reason="response_invalid")
            return tuple(analyses)
