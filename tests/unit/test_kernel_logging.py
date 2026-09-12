"""The kernel says out loud where it degrades instead of raising.

Observability rode entirely on OpenTelemetry spans, which are no-ops when the SDK is not
installed, and the kernel held no logger at all. Silent degradation is how a capability dies
unnoticed here: video captioning never worked for months, and media encoding silently fell back
to text. One logger, at the points where a reason string already existed and was thrown away.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from mindbridge import (
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryKind,
    Modality,
)


class _AudibleAffectFormer:
    """Proposes an audio affect cue for a text-only observation, which the kernel must refuse."""

    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "affect-former-test"
    formation_space = "affect-former-test:v1"

    def form(
        self,
        inputs: Sequence[FormationInput],
    ) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(
            (
                FormationProposal(
                    kind=MemoryKind.AFFECT,
                    content="the speaker sounded anxious",
                    basis=EvidenceBasis.MODEL_INFERENCE,
                    subject="speaker",
                    value="anxious",
                    confidence=0.6,
                    cue_modality=Modality.AUDIO,
                ),
            )
            for _value in inputs
        )

    def close(self) -> None:
        pass


def test_a_refused_formation_proposal_is_logged_with_its_reason(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        Memory(tmp_path, embedder=TinyEmbedder(), former=_AudibleAffectFormer()) as memory,
        caplog.at_level(logging.WARNING, logger="mindbridge.kernel.formation"),
    ):
        record = memory.add("I waited for the call")

    assert any(
        "formation proposal refused" in message
        and record.id in message
        and "modality present in its source" in message
        for message in caplog.messages
    )
