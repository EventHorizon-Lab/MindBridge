from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from mindbridge import (
    EmbedTask,
    FormationProposal,
    MemoryIntent,
    MemoryKind,
    MemoryOperation,
    MemoryRecord,
    MemoryTrigger,
    Modality,
    ModelInput,
)

ATOMIC_MODALITIES = frozenset(value for value in Modality if value is not Modality.OMNI)


class TinyEmbedder:
    """Small deterministic embedder for public-SDK feature tests."""

    embedding_capabilities = ATOMIC_MODALITIES
    embedding_model = "tiny-test"
    embedding_space = "tiny-test:4:l2-v1"
    embedding_dimension = 4

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        del task
        vectors = []
        for value in inputs:
            material = value.text.encode()
            material += b"".join(asset.id.encode() for asset in value.assets)
            digest = hashlib.sha256(material).digest()
            vector = tuple(1.0 + digest[index] / 255.0 for index in range(4))
            norm = math.sqrt(sum(component * component for component in vector))
            vectors.append(tuple(component / norm for component in vector))
        return tuple(vectors)

    def close(self) -> None:
        pass


class CueConsolidator:
    """Proposes one model-inferred trait citing exactly the evidence it was shown.

    Evidence that already contains a trait gets a second, distinct proposal, so consolidating
    over a derived record builds a dependent trait instead of reinforcing the one cited.
    """

    consolidation_model = "cue-consolidator-test"
    consolidation_recipe = "cue-consolidator-test:v1"

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        del trigger
        over_trait = any(
            record.context is not None and record.context.kind is MemoryKind.TRAIT
            for record in evidence
        )
        return (
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=tuple(record.id for record in evidence),
                proposal=FormationProposal(
                    kind=MemoryKind.TRAIT,
                    content=(
                        "The user needs reassurance"
                        if over_trait
                        else "The user is anxious under stress"
                    ),
                    subject="user",
                    predicate="disposition",
                    value="needs-reassurance" if over_trait else "anxious",
                    confidence=0.6,
                ),
            ),
        )

    def close(self) -> None:
        pass
