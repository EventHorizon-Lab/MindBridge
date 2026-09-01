from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from mindbridge import EmbedTask, Modality, ModelInput

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
