"""Reproducible component smoke for the production Jina Omni adapter."""

from __future__ import annotations

import argparse
import asyncio
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Literal

from pydantic import AwareDatetime

from mindbridge.benchmarks.runtime import dot_product
from mindbridge.contracts import ContractModel
from mindbridge.models import EmbedRequest, EmbedTask, ModelInput, TextPart
from mindbridge.models.jina import JinaEmbedder

QUERY = "Where did the robot put the red screwdriver?"
RELEVANT_DOCUMENT = "The robot put the red screwdriver beside the blue toolbox."
UNRELATED_DOCUMENT = "The weather outside is sunny and warm."


class JinaSmokeResult(ContractModel):
    """Versioned output needed to reproduce an embedding smoke run."""

    created_at: AwareDatetime
    embedder_plugin: Literal["jina"] = "jina"
    model_id: str
    revision: str
    space_id: str
    space_revision: str
    task: Literal["retrieval_query"] = "retrieval_query"
    device: str
    dimension: int
    query_norm: float
    relevant_similarity: float
    unrelated_similarity: float
    python_version: str
    torch_version: str
    transformers_version: str
    sentence_transformers_version: str
    passed: bool


async def run_jina_smoke(*, revision: str, device: str) -> JinaSmokeResult:
    """Load the pinned model and compare one relevant and irrelevant document."""
    embedder = JinaEmbedder.load(revision=revision, device=device)
    query_result = await embedder.embed(
        EmbedRequest(
            inputs=(ModelInput((TextPart(QUERY),)),),
            task=EmbedTask.QUERY,
        )
    )
    document_result = await embedder.embed(
        EmbedRequest(
            inputs=(
                ModelInput((TextPart(RELEVANT_DOCUMENT),)),
                ModelInput((TextPart(UNRELATED_DOCUMENT),)),
            ),
            task=EmbedTask.DOCUMENT,
        )
    )
    query_embedding = query_result.embeddings[0]
    query = query_embedding.values
    relevant, unrelated = (
        document_result.embeddings[0].values,
        document_result.embeddings[1].values,
    )
    query_norm = math.hypot(*query)
    relevant_similarity = dot_product(query, relevant)
    unrelated_similarity = dot_product(query, unrelated)
    return JinaSmokeResult(
        created_at=datetime.now(timezone.utc),
        model_id=query_embedding.model_reference.model_id,
        revision=query_embedding.model_reference.revision,
        space_id=query_embedding.space_reference.space_id,
        space_revision=query_embedding.space_reference.revision,
        device=device,
        dimension=len(query),
        query_norm=query_norm,
        relevant_similarity=relevant_similarity,
        unrelated_similarity=unrelated_similarity,
        python_version=platform.python_version(),
        torch_version=version("torch"),
        transformers_version=version("transformers"),
        sentence_transformers_version=version("sentence-transformers"),
        passed=(
            len(query) == query_embedding.dimension
            and math.isclose(query_norm, 1.0, rel_tol=1e-4, abs_tol=1e-6)
            and relevant_similarity > unrelated_similarity
        ),
    )


def main() -> None:
    """Run the smoke and emit a machine-readable manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    result = asyncio.run(run_jina_smoke(revision=arguments.revision, device=arguments.device))
    print(result.model_dump_json(indent=2))
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
